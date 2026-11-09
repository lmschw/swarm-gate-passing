"""7-value Thymio II IR proximity sensing for the Hebbian ABCD controller,
extended with gradient-map light sensing.

The 7 IR channels are vendored from energy_efficient_flocking's
ants26_replication/thymio_ir_variant/sensor_model.py: a simulation of the real
Thymio II's onboard prox.horizontal array. Each of the 7 sensors independently
detects the NEAREST reflecting surface -- wall OR another agent, whichever is
closer -- within its narrow detection cone (config.THYMIO_IR_HALF_APERTURE) and
range (config.THYMIO_IR_RANGE); it has no way to know or care what it's
reflecting off, and cannot report a neighbor's identity or bearing beyond "which
of 7 fixed-angle sensors fired" -- a much less informative sensor than
sensor_model.py's idealized quadrant distance/bearing model.

The 10th input (light) is the same gradient-map addition as sensor_model.py's
11th -- see that module's docstring and environment/sensing.py.
"""
import numpy as np

from . import config


def _wall_ray_distances(x, y, global_dir):
    """Distance from (x, y) to the nearest of the 3 simulated-arena walls along
    the ray pointing in global_dir (a global angle in radians, one per agent), or
    +inf if the ray points away from every wall. Only 3 walls exist -- X_RANGE's
    upper bound and Y_RANGE's upper/lower bounds -- there's no lower X wall since
    agents migrate indefinitely in -x by design. Genuine ray-to-line
    intersection, appropriate since each IR sensor is a narrow, specific ray."""
    cos_d, sin_d = np.cos(global_dir), np.sin(global_dir)
    wall_x_upper = config.X_RANGE[1] - config.ROBOT_RAD
    wall_y_upper = config.Y_RANGE[1] - config.ROBOT_RAD
    wall_y_lower = config.Y_RANGE[0] + config.ROBOT_RAD

    dist = np.full(x.shape, np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (wall_x_upper - x) / cos_d
        valid = (np.abs(cos_d) > 1e-9) & (t > 0)
        dist = np.where(valid, np.minimum(dist, t), dist)

        t = (wall_y_upper - y) / sin_d
        valid = (np.abs(sin_d) > 1e-9) & (t > 0)
        dist = np.where(valid, np.minimum(dist, t), dist)

        t = (wall_y_lower - y) / sin_d
        valid = (np.abs(sin_d) > 1e-9) & (t > 0)
        dist = np.where(valid, np.minimum(dist, t), dist)

    return dist


def get_sensor_data(agents, ir_range=None, light_intensity=None, gates=None):
    """agents: (n_agents, 4) array of [x, y, heading, battery].
    light_intensity: optional (n_agents,) array of raw [0, 255] gradient-map
        readings (see environment.sensing.GradientSensor.read); None feeds a
        neutral 0.0 (see sensor_model.get_sensor_data's own docstring).
    gates: IGNORED -- accepted only so simulation.py can call every sensor
        module's get_sensor_data with the same keyword arguments regardless of
        sensor_mode (see sensor_model_vision.py, which does use it).

    Returns a (10, n_agents) array: 7 IR intensities (config.THYMIO_IR_ANGLES
    order, each in [0, 1]) + own battery + own heading + gradient light.
    """
    if ir_range is None:
        ir_range = config.THYMIO_IR_RANGE
    n_agents = agents.shape[0]
    x, y, heading = agents[:, 0], agents[:, 1], agents[:, 2]

    dx = x[None, :] - x[:, None]
    dy = y[None, :] - y[:, None]
    global_bearing = np.arctan2(dy, dx)
    rel_angle = global_bearing - np.pi / 2.0 - heading[:, None]
    rel_angle = (rel_angle + np.pi) % (2.0 * np.pi) - np.pi
    center_distance = np.hypot(dx, dy)
    # THYMIO_IR_RANGE is a surface gap, not center-to-center; convert.
    surface_gap = np.maximum(center_distance - 2.0 * config.ROBOT_RAD, 0.0)
    not_self = ~np.eye(n_agents, dtype=bool)

    half_ap = config.THYMIO_IR_HALF_APERTURE
    intensities = np.zeros((len(config.THYMIO_IR_ANGLES), n_agents))

    for s, sensor_angle in enumerate(config.THYMIO_IR_ANGLES):
        angle_diff = (rel_angle - sensor_angle + np.pi) % (2.0 * np.pi) - np.pi
        in_cone = not_self & (np.abs(angle_diff) <= half_ap) & (surface_gap <= ir_range)
        neighbor_dist = np.where(in_cone, surface_gap, np.inf).min(axis=1)

        global_dir = heading + np.pi / 2.0 + sensor_angle
        wall_dist = _wall_ray_distances(x, y, global_dir)
        wall_dist = np.where(wall_dist <= ir_range, wall_dist, np.inf)

        nearest = np.minimum(neighbor_dist, wall_dist)
        intensity = np.where(np.isfinite(nearest), 1.0 - nearest / ir_range, 0.0)
        intensities[s] = np.clip(intensity, 0.0, 1.0)

    if light_intensity is None:
        light_row = np.zeros(n_agents)
    else:
        light_row = np.asarray(light_intensity, dtype=float) / 127.5 - 1.0

    inputs = np.concatenate([
        intensities,
        (agents[:, 3] / 50.0 - 1.0)[None, :],
        (agents[:, 2] / np.pi)[None, :],
        light_row[None, :],
    ], axis=0)  # (10, n_agents)
    return inputs
