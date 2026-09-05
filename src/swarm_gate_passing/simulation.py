"""Full episode simulation for the Hebbian ABCD controller.

The physics/control loop (wind, drag, battery, collision bookkeeping) is
vendored from energy_efficient_flocking's ants26_replication/experiment/
simulation_hebbian.py, itself a 1:1 port of simulation_free_global_mod_2.m's
main loop. Layered on top of that:

  - gradient-map light sensing (see environment/sensing.py) and a mean
    "path_alignment" reward, from combining in volcano_gradient's sensing.
  - a choice of sensor module (sensor_model.py's idealized quadrant
    distance/bearing sensor, encoding neighbor POSITIONS, or
    sensor_model_thymio.py's raw 7-channel IR proximity sensor).
  - an optional physical Gate (environment/gate.py) placed at a finish line,
    plus the metrics needed to train crossing it: geometric path deviation,
    mean speed, and an all-agents-crossed success flag.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import config
from . import sensor_model as sensor_model_quadrant
from . import sensor_model_thymio
from .hebbian_controller import init_weights, hebbian_step
from .wind_physics import (
    wrap_to_pi, RayTraceCircularRobots, dragforce, batterydrainage, _spawn_agents,
)

_SENSOR_MODULES = {"quadrant": sensor_model_quadrant, "thymio": sensor_model_thymio}


@dataclass
class EpisodeResult:
    dist_travelled: float
    average_batt: float
    collision_time: float
    wall_collision_time: float       # includes gate-barrier hits, see _move()
    cohesion_dist: float
    proximity_penalty: float
    path_alignment: float            # mean sensed light reading, rescaled to [0, 100]
    path_deviation_m: float          # mean geometric distance to the path centerline [m]
    mean_speed: float                # mean realized linear speed [m/s]
    success: float                   # 1.0 iff every agent had crossed finish_x by episode end
    post_gate_cohesion_dist: Optional[float] = None  # mean pairwise distance, steps-past-last-gate only
    telemetry: Optional[dict] = None


def _proximity_penalty(D, iu):
    """Graduated warning penalty for inter-agent surface gap (unused by the
    default stage weights below, kept for parity with the source project's
    HEBBIAN_PROXIMITY_* mechanism in case a stage wants to enable it)."""
    gap = D[iu] - 2.0 * config.ROBOT_RAD
    outer = 1.0 * config.ROBOT_RAD
    inner = 0.5 * config.ROBOT_RAD
    contact = config.COLLISION_MIN_DIST_SLACK
    mid_frac = np.clip((outer - gap) / (outer - inner), 0.0, 1.0)
    steep_frac = np.clip((inner - gap) / (inner - contact), 0.0, 1.0)
    pair_penalty = mid_frac ** 2 + 10.0 * steep_frac ** 2
    return float(np.sum(pair_penalty))


def _world_frame_position(agents):
    """Translates the flocking arena's [-5, 5] x [-5, 5] coordinate system into
    GradientSensor's (0, 0)-bottom-left-corner world frame."""
    world_x = agents[:, 0] - config.X_RANGE[0]
    world_y = agents[:, 1] - config.Y_RANGE[0]
    return world_x, world_y


def _move(agents, vel, dt, n_agents, min_dist, walls, gates=None, wind_enabled=True):
    """Kinematic integration + collision bookkeeping, mirroring move() in
    simulation_free_global_mod_2.m. Tracks inter-robot and wall collisions
    separately so stage_fitness can weight/use them independently.

    gates: optional list of environment.gate.Gate (see environment.gate.
    evenly_spaced_gates), checked in descending-x (travel) order and enforced
    LAST, after every other position clamp -- see the wind_enabled note below
    for why order matters here, not just tidiness. Any agent whose step would
    cross a gate's x outside its opening is stopped right at that barrier
    instead, and counted as a wall hit (same weight/meaning as an
    arena-boundary hit). Since a single step's displacement is tiny relative to
    gate spacing, an agent crossing more than one gate in the same step is not
    expected in practice, but gates are still processed in travel order (the
    order evenly_spaced_gates returns them in) so the FIRST barrier an agent
    would actually reach takes precedence if it ever did happen.

    wind_enabled: gates the "wind-tracking camera window" x-clamp below (the
    ORIGINAL vendored move() applies this unconditionally). That clamp pulls
    any agent more than WIND_TRACKING_MAX_SPAN behind the swarm's leader
    forward -- its only real purpose is bounding RayTraceCircularRobots' grid
    to the swarm's extent, so it's pointless (and actively harmful) when wind
    is off: with no wind cost, an unblocked leader can travel far enough past
    a Gate for this clamp to drag a STILL-BLOCKED straggler through the
    barrier's x in one step, without that straggler ever having been inside
    the opening -- a real bug found by inspecting a trained genome's video
    (one agent "through", the rest never coordinated) that let CMA-ES farm the
    success bonus via this artifact instead of genuine coordinated passage.
    Applying gate-blocking AFTER this clamp (and after the wall clamps) closes
    that hole even if a future config enables both wind and gates together."""
    vel_actual = np.zeros((n_agents, 3))
    vel_actual[:, 0:2] = vel
    vel_actual[:, 2] = agents[:, 2]

    theta = agents[:, 2]
    dx = -vel[:, 0] * dt * np.sin(theta)
    dy = vel[:, 0] * dt * np.cos(theta)

    agents_old = agents.copy()
    agents[:, 0] += dx
    agents[:, 1] += dy
    agents[:, 2] = wrap_to_pi(agents[:, 2] + vel[:, 1] * dt)

    agents_xy = agents[:, 0:2]
    D = np.linalg.norm(agents_xy[:, None, :] - agents_xy[None, :, :], axis=-1)
    close_agents = (D < min_dist) & (~np.eye(n_agents, dtype=bool))
    pair_collisions = int(np.count_nonzero(np.triu(close_agents, k=1)))
    iu = np.triu_indices(n_agents, k=1)
    mean_pairwise_dist = float(np.mean(D[iu])) if n_agents > 1 else 0.0
    proximity_penalty = _proximity_penalty(D, iu) if n_agents > 1 else 0.0

    wall_margin = config.ROBOT_RAD * config.WALL_MARGIN_FACTOR
    wall_hits = int(np.sum((agents[:, 0] > walls[1] - wall_margin) |
                           (agents[:, 1] > walls[2] - wall_margin) |
                           (agents[:, 1] < walls[3] + wall_margin)))

    min_x = np.min(agents[:, 0])
    max_x = min(np.max(agents[:, 0]), min_x + config.WIND_TRACKING_MAX_SPAN)
    window_width = config.WIND_TRACKING_WINDOW_WIDTH
    xRange = [min_x - (window_width - (max_x - min_x)) / 2.0, max_x + (window_width - (max_x - min_x)) / 2.0]

    if wind_enabled:
        agents[:, 0] = np.minimum(agents[:, 0], max_x)
    agents[:, 1] = np.minimum(agents[:, 1], walls[2])
    agents[:, 1] = np.maximum(agents[:, 1], walls[3])

    gate_hits = 0
    if gates:
        old_x = agents_old[:, 0]
        already_blocked = np.zeros(n_agents, dtype=bool)
        for gate in sorted(gates, key=lambda g: -g.x_arena):
            new_x = agents[:, 0]
            crossed = ((old_x - gate.x_arena) * (new_x - gate.x_arena)) < 0.0
            candidates = crossed & ~already_blocked
            if not np.any(candidates):
                continue
            blocked = candidates & np.array([gate.blocks(y) for y in agents[:, 1]])
            if np.any(blocked):
                sign = np.sign(old_x[blocked] - gate.x_arena)
                agents[blocked, 0] = gate.x_arena + 1e-3 * sign
                gate_hits += int(np.count_nonzero(blocked))
                already_blocked |= blocked
        wall_hits += gate_hits

    x_old, y_old = agents_old[:, 0], agents_old[:, 1]
    x_new, y_new = agents[:, 0], agents[:, 1]
    dist = np.sqrt((x_old - x_new) ** 2 + (y_old - y_new) ** 2)

    vel_actual[:, 2] = np.arctan2((y_new - y_old), (x_new - x_old)) - np.pi / 2.0
    vel_actual[:, 0] = dist / dt
    vel_actual[np.isnan(vel_actual[:, 2]), 2] = 0.0

    return vel_actual, agents, xRange, pair_collisions, wall_hits, mean_pairwise_dist, proximity_penalty


def simulate_hebbian_episode(abcd_rules, seed=None, n_agents=None, wind_enabled=True,
                              max_battery=None, min_battery=None, nx=None, ny=None,
                              use_battery_sensor=True, sensor_mode="quadrant",
                              gradient_sensor=None, gates=None, finish_x=None,
                              record_trajectory=False, record_battery=False):
    """Runs one full episode with the Hebbian ABCD controller, until any agent's
    battery depletes or (if finish_x is set) every agent has crossed finish_x.

    abcd_rules: dict from hebbian_controller.unflatten_abcd(), shared by every agent.
    sensor_mode: "quadrant" (idealized distance/bearing, encodes neighbor
        positions) or "thymio" (raw 7-channel IR proximity) -- see config.py.
    gradient_sensor: optional environment.sensing.GradientSensor; when given,
        each agent's position is sampled every step for a light-intensity
        reading fed into the sensor's last input, and also used to compute
        `path_deviation_m` (the agent's geometric distance from the path
        centerline). None disables both (path metrics are 0.0).
    gates: optional list of environment.gate.Gate (see
        environment.gate.evenly_spaced_gates) -- physical barriers that only
        let agents through their opening (see _move()). If given and finish_x
        is None, the LAST gate's x (gates[-1].x_arena, the furthest along the
        track) is used as the finish line -- passing it requires having
        already passed every earlier gate, since each physically blocks
        crossing outside its own opening.
    finish_x: optional arena-frame x; `success` is 1.0 iff every agent's x has
        crossed below this by the end of the episode (episode ends early, as
        soon as that happens, to save compute). None disables both the early
        exit and success tracking (`success` is always 0.0).

    Returns an EpisodeResult (telemetry populated only if record_trajectory/
    record_battery is set).
    """
    if seed is not None:
        np.random.seed(seed)

    if finish_x is None and gates:
        finish_x = gates[-1].x_arena

    sensor_module = _SENSOR_MODULES[sensor_mode]
    n_inputs = config.n_inputs_for_sensor_mode(sensor_mode)
    battery_idx = config.battery_row(n_inputs)

    dt = config.DT
    n_agents = n_agents if n_agents is not None else config.HEBBIAN_N_AGENTS
    robot_rad = config.ROBOT_RAD
    wind_rad = config.WIND_RAD
    xRange = list(config.X_RANGE)
    yRange = list(config.Y_RANGE)
    v_wind = config.V_WIND

    Uinf, kappa = config.UINF, config.KAPPA
    Nx = nx if nx is not None else config.HEBBIAN_NX
    Ny = ny if ny is not None else config.HEBBIAN_NY
    spawn_square_size = config.SPAWN_SQUARE_SIZE
    midpoint = list(config.SPAWN_MIDPOINT)
    max_battery = max_battery if max_battery is not None else config.HEBBIAN_MAX_BATTERY
    min_battery = min_battery if min_battery is not None else config.HEBBIAN_MIN_BATTERY

    walls = [xRange[0] + robot_rad, xRange[1] - robot_rad, yRange[1] - robot_rad, yRange[0] + robot_rad]
    min_dist = config.COLLISION_MIN_DIST_SLACK + 2.0 * robot_rad
    min_dist_initial = config.SPAWN_MIN_DIST_SLACK + 2.0 * robot_rad

    agents = _spawn_agents(n_agents, midpoint, spawn_square_size, min_dist_initial, max_battery, min_battery)
    weights = [init_weights(n_inputs) for _ in range(n_agents)]

    pair_collision_counter = 0
    wall_collision_counter = 0
    cohesion_dist_sum = 0.0
    path_alignment_sum = 0.0
    path_deviation_sum = 0.0
    speed_sum = 0.0
    steps = 0
    post_gate_cohesion_sum = 0.0
    post_gate_steps = 0
    last_gate_x = gates[-1].x_arena if gates else None
    batteryEmpty = False
    success = False
    positions_log = [agents[:, 0:2].copy()] if record_trajectory else None
    headings_log = [agents[:, 2].copy()] if record_trajectory else None
    battery_log = [agents[:, 3].copy()] if record_battery else None
    vel = np.zeros((n_agents, 2))

    while not (batteryEmpty or success):
        if gradient_sensor is not None:
            world_x, world_y = _world_frame_position(agents)
            light_for_controller = gradient_sensor.read(world_x, world_y, add_noise=True)
            light_clean = gradient_sensor.read(world_x, world_y, add_noise=False)
            path_alignment_sum += float(np.mean(light_clean))
            centerline_world_y = gradient_sensor.centerline_y(world_x)
            centerline_arena_y = centerline_world_y + config.Y_RANGE[0]
            path_deviation_sum += float(np.mean(np.abs(agents[:, 1] - centerline_arena_y)))
        else:
            light_for_controller = None

        sensor_inputs = sensor_module.get_sensor_data(agents, light_intensity=light_for_controller)
        if not use_battery_sensor:
            sensor_inputs[battery_idx, :] = 0.0
        for i in range(n_agents):
            w1, w2, w3 = weights[i]
            v_i, w_i, w1n, w2n, w3n = hebbian_step(sensor_inputs[:, i], w1, w2, w3, abcd_rules)
            vel[i, 0] = v_i
            vel[i, 1] = w_i
            weights[i] = (w1n, w2n, w3n)

        vel_actual, agents, xRange, pair_hits, wall_hits, mean_pairwise_dist, _ = _move(
            agents, vel, dt, n_agents, min_dist, walls, gates=gates, wind_enabled=wind_enabled)
        pair_collision_counter += pair_hits
        wall_collision_counter += wall_hits
        cohesion_dist_sum += mean_pairwise_dist
        speed_sum += float(np.mean(vel_actual[:, 0]))
        steps += 1
        if last_gate_x is not None and np.all(agents[:, 0] < last_gate_x):
            post_gate_cohesion_sum += mean_pairwise_dist
            post_gate_steps += 1
        if record_trajectory:
            positions_log.append(agents[:, 0:2].copy())
            headings_log.append(agents[:, 2].copy())

        if wind_enabled:
            yVals, xVals, powerVals = RayTraceCircularRobots(agents, wind_rad, Uinf, xRange, yRange, Nx, Ny)
            F_drag = dragforce(agents, wind_rad, xVals, yVals, powerVals, n_agents, vel_actual, v_wind, kappa)
        else:
            F_drag = np.zeros((n_agents, 2))
        agents, batt_drain = batterydrainage(agents, vel_actual, F_drag, robot_rad, dt)
        if record_battery:
            battery_log.append(agents[:, 3].copy())

        batteryEmpty = np.any(agents[:, 3] <= 0.0)
        if finish_x is not None:
            success = bool(np.all(agents[:, 0] < finish_x))

    average_batt = np.mean(agents[:, 3])
    dist_travelled = -np.mean(agents[:, 0])
    collision_time = pair_collision_counter * dt
    wall_collision_time = wall_collision_counter * dt
    cohesion_dist = cohesion_dist_sum / steps if steps else 0.0
    path_alignment = (100.0 * path_alignment_sum / 255.0 / steps) if steps else 0.0
    path_deviation_m = path_deviation_sum / steps if steps else 0.0
    mean_speed = speed_sum / steps if steps else 0.0
    proximity_penalty = 0.0  # tracked but unweighted by default; see _proximity_penalty
    post_gate_cohesion_dist = (post_gate_cohesion_sum / post_gate_steps) if post_gate_steps else None

    telemetry = None
    if record_trajectory or record_battery:
        telemetry = {
            "positions": np.array(positions_log) if record_trajectory else None,
            "headings": np.array(headings_log) if record_trajectory else None,
            "battery": np.array(battery_log) if record_battery else None,
        }

    return EpisodeResult(
        dist_travelled=dist_travelled, average_batt=average_batt, collision_time=collision_time,
        wall_collision_time=wall_collision_time, cohesion_dist=cohesion_dist,
        proximity_penalty=proximity_penalty, path_alignment=path_alignment,
        path_deviation_m=path_deviation_m, mean_speed=mean_speed, success=float(success),
        post_gate_cohesion_dist=post_gate_cohesion_dist, telemetry=telemetry)


def stage_fitness(result: EpisodeResult, stage: str) -> float:
    """Per-stage fitness formula:

    eff = distance_w*dist
          + avg_batt/battery_w
          - (wall_col_mult*wall_col_time [+ collision_time]) / collision_w
          - cohesion_dist / cohesion_w
          - proximity_penalty / proximity_w
          - path_deviation_m / path_deviation_w
          + path_alignment / path_w
          + mean_speed / speed_w
          - post_gate_cohesion_dist / post_gate_cohesion_w [if the swarm ever got past the last gate]
          + success_bonus [if success]

    distance_w defaults to config.HEBBIAN_EFF_DISTANCE_WEIGHT (16.0, the value
    calibrated for the energy-efficiency curriculum) but is overridable per
    stage -- see config.HEBBIAN_STAGE_FITNESS_WEIGHTS["gate_passing"]'s much
    lower value: at 16.0, a few meters of raw forward progress dwarfs the
    path-deviation/cohesion penalties (both single-digit meters divided by
    single-digit weights) regardless of how those are tuned, making "charge
    straight ahead, get pinned against a wall, keep going" a good strategy --
    walls only clamp Y, never stop X-progress -- with no incentive to actually
    steer toward the gate. All terms are optional (a missing/None weight
    disables that term entirely).
    """
    weights = config.HEBBIAN_STAGE_FITNESS_WEIGHTS[stage]
    distance_w = weights.get("distance_w", config.HEBBIAN_EFF_DISTANCE_WEIGHT)
    eff = distance_w * result.dist_travelled

    battery_w = weights.get("battery_w")
    if battery_w is not None:
        eff += result.average_batt / battery_w

    collision_w = weights.get("collision_w")
    if collision_w is not None:
        wall_col_mult = weights.get("wall_col_mult", 1.0)
        penalty = wall_col_mult * result.wall_collision_time
        if weights.get("include_inter_robot_collision", False):
            penalty += result.collision_time
        eff -= penalty / collision_w

    cohesion_w = weights.get("cohesion_w")
    if cohesion_w is not None:
        eff -= result.cohesion_dist / cohesion_w

    post_gate_cohesion_w = weights.get("post_gate_cohesion_w")
    if post_gate_cohesion_w is not None and result.post_gate_cohesion_dist is not None:
        eff -= result.post_gate_cohesion_dist / post_gate_cohesion_w

    proximity_w = weights.get("proximity_w")
    if proximity_w is not None:
        eff -= result.proximity_penalty / proximity_w

    path_deviation_w = weights.get("path_deviation_w")
    if path_deviation_w is not None:
        eff -= result.path_deviation_m / path_deviation_w

    path_w = weights.get("path_w")
    if path_w is not None:
        eff += result.path_alignment / path_w

    speed_w = weights.get("speed_w")
    if speed_w is not None:
        eff += result.mean_speed / speed_w

    success_bonus = weights.get("success_bonus")
    if success_bonus is not None and result.success:
        eff += success_bonus

    return eff
