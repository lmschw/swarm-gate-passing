"""Full episode simulation for the Hebbian ABCD controller, extended with
gradient-map path sensing.

The physics/control loop (wind, drag, battery, collision bookkeeping) and the
overall structure are vendored from energy_efficient_flocking's
ants26_replication/experiment/simulation_hebbian.py, itself a 1:1 port of
simulation_free_global_mod_2.m's main loop. What's new for this project:
each step, if a `gradient_sensor` (environment.sensing.GradientSensor) is
supplied, every agent's world position is sampled for a light-intensity
reading fed into the 11th sensor input (see sensor_model.py), and the episode's
mean reading is returned as `path_alignment` for use by stage_fitness's new
"follow_gradient_path" term.
"""
import numpy as np

from . import config
from .sensor_model import get_sensor_data
from .hebbian_controller import init_weights, hebbian_step
from .wind_physics import (
    wrap_to_pi, RayTraceCircularRobots, dragforce, batterydrainage, _spawn_agents,
)


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


def _move(agents, vel, dt, n_agents, min_dist, walls):
    """Kinematic integration + collision bookkeeping, mirroring move() in
    simulation_free_global_mod_2.m. Tracks inter-robot and wall collisions
    separately so stage_fitness can weight/use them independently."""
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

    agents[:, 0] = np.minimum(agents[:, 0], max_x)
    agents[:, 1] = np.minimum(agents[:, 1], walls[2])
    agents[:, 1] = np.maximum(agents[:, 1], walls[3])

    x_old, y_old = agents_old[:, 0], agents_old[:, 1]
    x_new, y_new = agents[:, 0], agents[:, 1]
    dist = np.sqrt((x_old - x_new) ** 2 + (y_old - y_new) ** 2)

    vel_actual[:, 2] = np.arctan2((y_new - y_old), (x_new - x_old)) - np.pi / 2.0
    vel_actual[:, 0] = dist / dt
    vel_actual[np.isnan(vel_actual[:, 2]), 2] = 0.0

    return vel_actual, agents, xRange, pair_collisions, wall_hits, mean_pairwise_dist, proximity_penalty


def simulate_hebbian_episode(abcd_rules, seed=None, n_agents=None, wind_enabled=True,
                              max_battery=None, min_battery=None, nx=None, ny=None,
                              use_battery_sensor=True, gradient_sensor=None,
                              record_trajectory=False, record_battery=False):
    """Runs one full episode with the Hebbian ABCD controller, until any agent's
    battery depletes.

    abcd_rules: dict from hebbian_controller.unflatten_abcd(), shared by every agent.
    gradient_sensor: optional environment.sensing.GradientSensor; when given, each
        agent's position is sampled every step for a light-intensity reading fed
        into the 11th sensor input, and the episode-mean reading (rescaled to
        [0, 100]) is returned as `path_alignment`. None disables gradient sensing
        (the 11th input is always fed a neutral 0.0) and `path_alignment` is 0.0.

    Returns (dist_travelled, average_batt, collision_time, wall_collision_time,
    cohesion_dist, proximity_penalty, path_alignment[, telemetry]).
    """
    if seed is not None:
        np.random.seed(seed)

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
    weights = [init_weights() for _ in range(n_agents)]

    pair_collision_counter = 0
    wall_collision_counter = 0
    cohesion_dist_sum = 0.0
    cohesion_dist_steps = 0
    path_alignment_sum = 0.0
    batteryEmpty = False
    positions_log = [agents[:, 0:2].copy()] if record_trajectory else None
    battery_log = [agents[:, 3].copy()] if record_battery else None
    vel = np.zeros((n_agents, 2))

    while not batteryEmpty:
        if gradient_sensor is not None:
            world_x, world_y = _world_frame_position(agents)
            light_for_controller = gradient_sensor.read(world_x, world_y, add_noise=True)
            light_clean = gradient_sensor.read(world_x, world_y, add_noise=False)
            path_alignment_sum += float(np.mean(light_clean))
        else:
            light_for_controller = None

        sensor_inputs = get_sensor_data(agents, light_intensity=light_for_controller)
        if not use_battery_sensor:
            sensor_inputs[8, :] = 0.0
        for i in range(n_agents):
            w1, w2, w3 = weights[i]
            v_i, w_i, w1n, w2n, w3n = hebbian_step(sensor_inputs[:, i], w1, w2, w3, abcd_rules)
            vel[i, 0] = v_i
            vel[i, 1] = w_i
            weights[i] = (w1n, w2n, w3n)

        vel_actual, agents, xRange, pair_hits, wall_hits, mean_pairwise_dist, _ = _move(
            agents, vel, dt, n_agents, min_dist, walls)
        pair_collision_counter += pair_hits
        wall_collision_counter += wall_hits
        cohesion_dist_sum += mean_pairwise_dist
        cohesion_dist_steps += 1
        if record_trajectory:
            positions_log.append(agents[:, 0:2].copy())

        if wind_enabled:
            yVals, xVals, powerVals = RayTraceCircularRobots(agents, wind_rad, Uinf, xRange, yRange, Nx, Ny)
            F_drag = dragforce(agents, wind_rad, xVals, yVals, powerVals, n_agents, vel_actual, v_wind, kappa)
        else:
            F_drag = np.zeros((n_agents, 2))
        agents, batt_drain = batterydrainage(agents, vel_actual, F_drag, robot_rad, dt)
        if record_battery:
            battery_log.append(agents[:, 3].copy())

        batteryEmpty = np.any(agents[:, 3] <= 0.0)

    average_batt = np.mean(agents[:, 3])
    dist_travelled = -np.mean(agents[:, 0])
    collision_time = pair_collision_counter * dt
    wall_collision_time = wall_collision_counter * dt
    cohesion_dist = cohesion_dist_sum / cohesion_dist_steps if cohesion_dist_steps else 0.0
    path_alignment = (100.0 * path_alignment_sum / 255.0 / cohesion_dist_steps) if cohesion_dist_steps else 0.0
    proximity_penalty = 0.0  # tracked but unweighted by default; see _proximity_penalty

    if record_trajectory or record_battery:
        telemetry = {
            "positions": np.array(positions_log) if record_trajectory else None,
            "battery": np.array(battery_log) if record_battery else None,
        }
        return (dist_travelled, average_batt, collision_time, wall_collision_time, cohesion_dist,
                proximity_penalty, path_alignment, telemetry)
    return dist_travelled, average_batt, collision_time, wall_collision_time, cohesion_dist, proximity_penalty, path_alignment


def stage_fitness(dist_travelled, average_batt, collision_time, wall_collision_time, cohesion_dist,
                   proximity_penalty, path_alignment, stage):
    """Per-stage fitness formula:

    eff = HEBBIAN_EFF_DISTANCE_WEIGHT*dist + avg_batt/battery_w
          - (wall_col_mult*wall_col_time [+ collision_time]) / collision_w
          - cohesion_dist / cohesion_w
          - proximity_penalty / proximity_w
          + path_alignment / path_w

    path_alignment is the episode-mean gradient-map light reading, rescaled to
    [0, 100] (see simulate_hebbian_episode) so it's calibrated on the same scale
    as average_batt/battery_w rather than the raw [0, 255] map intensity.
    """
    battery_w, collision_w, wall_col_mult, include_inter_robot, cohesion_w, proximity_w, path_w = \
        config.HEBBIAN_STAGE_FITNESS_WEIGHTS[stage]
    eff = config.HEBBIAN_EFF_DISTANCE_WEIGHT * dist_travelled
    if battery_w is not None:
        eff += average_batt / battery_w
    if collision_w is not None:
        penalty = wall_col_mult * wall_collision_time
        if include_inter_robot:
            penalty += collision_time
        eff -= penalty / collision_w
    if cohesion_w is not None:
        eff -= cohesion_dist / cohesion_w
    if proximity_w is not None:
        eff -= proximity_penalty / proximity_w
    if path_w is not None:
        eff += path_alignment / path_w
    return eff
