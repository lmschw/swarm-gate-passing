"""Limited-range, occlusion-aware vision sensor.

Unlike sensor_model.py (idealized quadrant distance/bearing, unlimited
identification within HEBBIAN_SENSING_RADIUS=2.01m and no occlusion) and
sensor_model_thymio.py (raw IR, occlusion-like by construction since each ray
reports only the nearest reflector, but with a very short 0.12m range), this
sensor sits between the two: it keeps the "4 body-relative quadrants" concept
but (a) caps range at config.VISION_RANGE and (b) genuinely checks line-of-
sight -- another agent's body between the sensor and a candidate target blocks
that target from being seen, so a densely-packed swarm has to contend with
mutual occlusion, not just distance/direction. Occluders are agent bodies only
(a gate's own solid wall segments are not modeled as occluders, to keep the
geometry tractable).

This mode also drops gradient-map sensing entirely (see config/simulation
module docstrings for the "forget the gradient map, use vision + occlusion
instead" direction) and replaces it with direct sensing of the environment's
gate(s) -- see environment/gate.py -- as visible landmarks (each gate
contributes its two opening-edge points), occlusion-checked the same way as
neighbors. Only the single NEAREST visible gate landmark is reported (not
broken out by quadrant, unlike neighbors): one landmark is enough for a genome
to home in on a gate, and there's no "identity" to preserve the way there is
for individual neighbors.
"""
import numpy as np

from . import config


def _point_segment_distance_and_t(px, py, ax, ay, bx, by):
    """Distance from point(s) P to segment AB, and the segment parameter t of
    the closest point (0=A, 1=B, clamped) -- used to test whether P's body
    lies close to AND between A and B, i.e. genuinely occludes the segment."""
    abx, aby = bx - ax, by - ay
    apx, apy = px - ax, py - ay
    ab_len_sq = abx * abx + aby * aby
    t = np.where(ab_len_sq > 1e-12, (apx * abx + apy * aby) / max(ab_len_sq, 1e-12), 0.0)
    t_clamped = np.clip(t, 0.0, 1.0)
    closest_x = ax + t_clamped * abx
    closest_y = ay + t_clamped * aby
    dist = np.hypot(px - closest_x, py - closest_y)
    return dist, t


def _occluded(sensor_xy, target_xy, occluders_xy, robot_rad):
    """True if any occluder's body (circle of radius robot_rad) sits close to
    AND strictly between sensor_xy and target_xy."""
    if occluders_xy.shape[0] == 0:
        return False
    dist, t = _point_segment_distance_and_t(
        occluders_xy[:, 0], occluders_xy[:, 1], sensor_xy[0], sensor_xy[1], target_xy[0], target_xy[1])
    return bool(np.any((dist < robot_rad) & (t > 0.02) & (t < 0.98)))


def get_sensor_data(agents, gates=None, vision_range=None, light_intensity=None):
    """agents: (n_agents, 4) array of [x, y, heading, battery].
    gates: optional list of environment.gate.Gate -- every gate's two opening-
        edge points are pooled as candidate landmarks; only the nearest
        visible one (from any gate) is reported.
    light_intensity: accepted for call-signature parity with sensor_model.py/
        sensor_model_thymio.py (simulation.py calls all three sensor modules
        the same way) but IGNORED -- this mode has no gradient sensing.

    Returns a (12, n_agents) array: [front_d, front_b, back_d, back_b,
    right_d, right_b, left_d, left_b] (4 occlusion-aware neighbor quadrants,
    range-limited to vision_range), [gate_d, gate_b] (nearest visible gate
    landmark, or (vision_range, 0) if none visible/no gates given), own
    battery, own heading -- each rescaled to roughly [-1, 1].
    """
    if vision_range is None:
        vision_range = config.VISION_RANGE
    n_agents = agents.shape[0]
    x, y, heading = agents[:, 0], agents[:, 1], agents[:, 2]
    robot_rad = config.ROBOT_RAD

    gate_points = []
    for gate in (gates or []):
        gate_points.append((gate.x_arena, gate.y_lo_arena))
        gate_points.append((gate.x_arena, gate.y_hi_arena))

    front_d = np.full(n_agents, vision_range); front_b = np.zeros(n_agents)
    back_d = np.full(n_agents, vision_range); back_b = np.zeros(n_agents)
    right_d = np.full(n_agents, vision_range); right_b = np.zeros(n_agents)
    left_d = np.full(n_agents, vision_range); left_b = np.zeros(n_agents)
    gate_d = np.full(n_agents, vision_range); gate_b = np.zeros(n_agents)

    for i in range(n_agents):
        sensor_xy = np.array([x[i], y[i]])
        others = [j for j in range(n_agents) if j != i]

        candidates = []  # (dist, dx, dy, kind, agent_index_or_None)
        for j in others:
            dx, dy = x[j] - x[i], y[j] - y[i]
            dist = float(np.hypot(dx, dy))
            if dist < vision_range:
                candidates.append((dist, dx, dy, "agent", j))
        for gx, gy in gate_points:
            dx, dy = gx - x[i], gy - y[i]
            dist = float(np.hypot(dx, dy))
            if dist < vision_range:
                candidates.append((dist, dx, dy, "gate", None))
        candidates.sort(key=lambda c: c[0])

        found = {"front": False, "back": False, "right": False, "left": False, "gate": False}

        for dist, dx, dy, kind, j in candidates:
            if all(found.values()):
                break
            occluder_idx = [k for k in others if k != j]
            occluders_xy = agents[occluder_idx, 0:2] if occluder_idx else np.zeros((0, 2))
            target_xy = np.array([x[i] + dx, y[i] + dy])
            if _occluded(sensor_xy, target_xy, occluders_xy, robot_rad):
                continue

            angle = np.arctan2(dy, dx)
            rel = (angle - np.pi / 2.0 - heading[i] + np.pi) % (2.0 * np.pi) - np.pi

            if kind == "gate":
                if not found["gate"]:
                    gate_d[i], gate_b[i] = dist, rel
                    found["gate"] = True
                continue

            if -np.pi / 4 <= rel <= np.pi / 4 and not found["front"]:
                front_d[i], front_b[i] = dist, rel; found["front"] = True
            elif (rel <= -3 * np.pi / 4 or rel >= 3 * np.pi / 4) and not found["back"]:
                back_d[i], back_b[i] = dist, rel; found["back"] = True
            elif -3 * np.pi / 4 <= rel <= -np.pi / 4 and not found["right"]:
                right_d[i], right_b[i] = dist, rel; found["right"] = True
            elif np.pi / 4 <= rel <= 3 * np.pi / 4 and not found["left"]:
                left_d[i], left_b[i] = dist, rel; found["left"] = True

    inputs = np.stack([
        front_d * 2.0 / vision_range - 1.0, front_b * 4.0 / np.pi - 1.0,
        back_d * 2.0 / vision_range - 1.0, back_b * 4.0 / np.pi - 1.0,
        right_d * 2.0 / vision_range - 1.0, right_b * 4.0 / np.pi - 1.0,
        left_d * 2.0 / vision_range - 1.0, left_b * 4.0 / np.pi - 1.0,
        gate_d * 2.0 / vision_range - 1.0, gate_b * 4.0 / np.pi - 1.0,
        agents[:, 3] / 50.0 - 1.0,
        agents[:, 2] / np.pi,
    ], axis=0)  # (12, n_agents)
    return inputs
