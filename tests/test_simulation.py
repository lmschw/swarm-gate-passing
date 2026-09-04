import numpy as np
import pytest

from swarm_gate_passing import config
from swarm_gate_passing.hebbian_controller import unflatten_abcd, init_weights, hebbian_step
from swarm_gate_passing.sensor_model import get_sensor_data as get_sensor_data_quadrant
from swarm_gate_passing.sensor_model_thymio import get_sensor_data as get_sensor_data_thymio
from swarm_gate_passing.simulation import simulate_hebbian_episode, stage_fitness, EpisodeResult
from swarm_gate_passing.environment import GradientSensor, Gate, evenly_spaced_gates, render_path_map


def _random_genome(sensor_mode="quadrant", seed=0):
    n_inputs = config.n_inputs_for_sensor_mode(sensor_mode)
    n_abcd = config.n_abcd_for(n_inputs)
    rng = np.random.default_rng(seed)
    return rng.uniform(config.HEBBIAN_ABCD_BOUNDS[0], config.HEBBIAN_ABCD_BOUNDS[1], n_abcd)


def test_genome_length_matches_config():
    genome = _random_genome()
    rules = unflatten_abcd(genome, n_inputs=config.HEBBIAN_N_INPUTS)
    assert len(rules) == 12
    assert rules["A1"].shape == (config.HEBBIAN_N_INPUTS, config.HEBBIAN_N_HIDDEN)
    assert rules["A3"].shape == (config.HEBBIAN_N_HIDDEN, config.HEBBIAN_N_OUTPUTS)


def test_thymio_genome_is_smaller_and_shapes_differ():
    n_inputs_thymio = config.n_inputs_for_sensor_mode("thymio")
    assert n_inputs_thymio == 10
    genome = _random_genome(sensor_mode="thymio")
    rules = unflatten_abcd(genome, n_inputs=n_inputs_thymio)
    assert rules["A1"].shape == (10, config.HEBBIAN_N_HIDDEN)
    assert len(genome) < len(_random_genome(sensor_mode="quadrant"))


def test_hebbian_step_output_bounds():
    rules = unflatten_abcd(_random_genome())
    w1, w2, w3 = init_weights()
    x_in = np.random.uniform(-1, 1, config.HEBBIAN_N_INPUTS)
    v, w, w1n, w2n, w3n = hebbian_step(x_in, w1, w2, w3, rules)
    assert abs(v) <= config.HEBBIAN_LINEAR_VEL_MAX + 1e-9
    assert abs(w) <= config.HEBBIAN_ANGULAR_VEL_MAX + 1e-9
    assert np.max(np.abs(w1n)) <= 1.0 + 1e-9


def test_quadrant_sensor_data_shape_and_light_normalization():
    agents = np.array([[0.0, 0.0, 0.0, 100.0], [1.0, 0.0, 0.0, 100.0], [0.0, 1.0, 0.0, 100.0]])
    out_no_light = get_sensor_data_quadrant(agents)
    assert out_no_light.shape == (11, 3)
    assert np.all(out_no_light[10, :] == 0.0)

    out_with_light = get_sensor_data_quadrant(agents, light_intensity=np.array([0.0, 127.5, 255.0]))
    assert np.allclose(out_with_light[10, :], [-1.0, 0.0, 1.0])


def test_thymio_sensor_data_shape_and_bounds():
    # agent 0 faces heading=0 (its front-center sensor, angle 0, points toward +y --
    # see the heading convention in sensor_model_thymio._wall_ray_distances/simulation._move).
    # agent 1 sits directly ahead of it, well within THYMIO_IR_RANGE (0.12m surface gap).
    agents = np.array([[0.0, 0.0, 0.0, 100.0], [0.0, 0.15, 0.0, 100.0], [0.0, -3.0, 0.0, 100.0]])
    out = get_sensor_data_thymio(agents)
    assert out.shape == (10, 3)
    assert np.all(out[:7, :] >= 0.0) and np.all(out[:7, :] <= 1.0)  # IR channels are [0, 1], not [-1, 1]
    assert np.any(out[:7, 0] > 0.0)


@pytest.mark.parametrize("sensor_mode", ["quadrant", "thymio"])
@pytest.mark.parametrize("wind_enabled", [False, True])
def test_short_episode_runs_and_returns_finite_fitness(sensor_mode, wind_enabled):
    rules = unflatten_abcd(_random_genome(sensor_mode=sensor_mode, seed=1), n_inputs=config.n_inputs_for_sensor_mode(sensor_mode))
    result = simulate_hebbian_episode(
        rules, seed=42, n_agents=4, wind_enabled=wind_enabled, sensor_mode=sensor_mode,
        max_battery=5.0, min_battery=5.0, nx=20, ny=20)
    assert isinstance(result, EpisodeResult)
    for value in (result.dist_travelled, result.average_batt, result.collision_time,
                  result.wall_collision_time, result.cohesion_dist, result.path_alignment,
                  result.path_deviation_m, result.mean_speed):
        assert np.isfinite(value)
    eff = stage_fitness(result, "save_battery_avoid_all")
    assert np.isfinite(eff)


def test_episode_with_gradient_sensor_returns_path_metrics():
    grid = render_path_map("sine_curve", 10.0, 10.0, path_kwargs={"freq": 2.0})
    sensor = GradientSensor(grid, world_size_x=10.0, world_size_y=10.0, noise_magnitude=0.5)
    rules = unflatten_abcd(_random_genome(seed=2))

    result = simulate_hebbian_episode(
        rules, seed=7, n_agents=3, wind_enabled=False,
        max_battery=5.0, min_battery=5.0, gradient_sensor=sensor)

    assert 0.0 <= result.path_alignment <= 100.0
    assert result.path_deviation_m >= 0.0
    eff = stage_fitness(result, "follow_gradient_no_gate")
    assert np.isfinite(eff)


def test_gate_blocks_crossing_outside_opening_and_lets_opening_through():
    grid = render_path_map("parabola", 10.0, 10.0)  # centerline near y=2.5m (world frame) at x=0
    sensor = GradientSensor(grid, world_size_x=10.0, world_size_y=10.0, noise_magnitude=0.0)
    gate = Gate.centered_on_path(sensor, x_arena=-1.0, opening_width_m=1.0,
                                  x_range=config.X_RANGE, y_range=config.Y_RANGE)

    assert not gate.blocks((gate.y_lo_arena + gate.y_hi_arena) / 2.0)  # center of opening: passes
    assert gate.blocks(gate.y_hi_arena + 1.0)  # well outside the opening: blocked
    assert gate.blocks(gate.y_lo_arena - 1.0)


def test_evenly_spaced_gates_single_gate_matches_original_single_gate():
    grid = render_path_map("sine_curve", 10.0, 10.0, path_kwargs={"freq": 2.0})
    sensor = GradientSensor(grid, world_size_x=10.0, world_size_y=10.0, noise_magnitude=0.0)
    gates = evenly_spaced_gates(sensor, n_gates=1, finish_x=-3.0, opening_width_m=1.0,
                                 x_range=config.X_RANGE, y_range=config.Y_RANGE)
    single = Gate.centered_on_path(sensor, x_arena=-3.0, opening_width_m=1.0,
                                    x_range=config.X_RANGE, y_range=config.Y_RANGE)
    assert len(gates) == 1
    assert gates[0] == single


def test_evenly_spaced_gates_last_gate_is_at_finish_x_and_ordered_by_travel():
    grid = render_path_map("zigzag", 10.0, 10.0)
    sensor = GradientSensor(grid, world_size_x=10.0, world_size_y=10.0, noise_magnitude=0.0)
    gates = evenly_spaced_gates(sensor, n_gates=4, finish_x=-4.0, opening_width_m=1.0,
                                 x_range=config.X_RANGE, y_range=config.Y_RANGE, start_x=0.0)
    assert len(gates) == 4
    assert gates[-1].x_arena == pytest.approx(-4.0)
    xs = [g.x_arena for g in gates]
    assert xs == sorted(xs, reverse=True)  # descending x = travel order (spawn -> finish)


def test_multi_gate_episode_blocks_and_reports_success_consistently():
    grid = render_path_map("sine_curve", 10.0, 10.0, path_kwargs={"freq": 2.0})
    sensor = GradientSensor(grid, world_size_x=10.0, world_size_y=10.0, noise_magnitude=0.0)
    gates = evenly_spaced_gates(sensor, n_gates=3, finish_x=-3.0, opening_width_m=0.8,
                                 x_range=config.X_RANGE, y_range=config.Y_RANGE)
    rules = unflatten_abcd(_random_genome(seed=4))

    result = simulate_hebbian_episode(
        rules, seed=11, n_agents=4, wind_enabled=False, max_battery=8.0, min_battery=8.0,
        gradient_sensor=sensor, gates=gates)

    assert np.isfinite(result.wall_collision_time)
    # success can only be True if the episode actually terminated with every agent
    # past the LAST gate (the only way past every earlier gate's barrier too)
    if result.success:
        assert result.dist_travelled >= 3.0 - 1e-6
    eff = stage_fitness(result, "gate_passing")
    assert np.isfinite(eff)


def test_post_gate_cohesion_is_none_when_swarm_never_clears_last_gate():
    grid = render_path_map("sine_curve", 10.0, 10.0, path_kwargs={"freq": 2.0})
    sensor = GradientSensor(grid, world_size_x=10.0, world_size_y=10.0, noise_magnitude=0.0)
    # gate placed far away and barely-opened battery budget: swarm should never reach it.
    gates = evenly_spaced_gates(sensor, n_gates=1, finish_x=-4.5, opening_width_m=0.3,
                                 x_range=config.X_RANGE, y_range=config.Y_RANGE)
    rules = unflatten_abcd(_random_genome(seed=5))
    result = simulate_hebbian_episode(
        rules, seed=13, n_agents=3, wind_enabled=False, max_battery=0.5, min_battery=0.5,
        gradient_sensor=sensor, gates=gates)
    assert result.post_gate_cohesion_dist is None


def test_post_gate_cohesion_is_tracked_once_swarm_clears_gate():
    grid = render_path_map("sine_curve", 10.0, 10.0, path_kwargs={"freq": 2.0})
    sensor = GradientSensor(grid, world_size_x=10.0, world_size_y=10.0, noise_magnitude=0.0)
    # a "gate" placed beyond the arena's own X_RANGE upper bound (5.0): every agent's x is
    # trivially already below it from the very first step (the wall clamp alone guarantees
    # this), so post-gate cohesion should be tracked from step 1 regardless of genome behavior.
    gates = evenly_spaced_gates(sensor, n_gates=1, finish_x=10.0, opening_width_m=10.0,
                                 x_range=config.X_RANGE, y_range=config.Y_RANGE)
    rules = unflatten_abcd(_random_genome(seed=6))
    result = simulate_hebbian_episode(
        rules, seed=14, n_agents=3, wind_enabled=False, max_battery=3.0, min_battery=3.0,
        gradient_sensor=sensor, gates=gates, finish_x=-100.0)  # unreachable finish -> full episode runs
    assert result.post_gate_cohesion_dist is not None
    assert result.post_gate_cohesion_dist >= 0.0
    eff = stage_fitness(result, "gate_passing")
    assert np.isfinite(eff)


def test_success_flag_true_only_when_finish_line_crossed():
    rules = unflatten_abcd(_random_genome(seed=3))
    # finish_x far behind spawn (0,0): essentially unreachable in a short/no-wind episode with
    # a tiny battery budget, so success should be False.
    result_far = simulate_hebbian_episode(
        rules, seed=9, n_agents=3, wind_enabled=False, max_battery=1.0, min_battery=1.0,
        finish_x=-100.0)
    assert result_far.success == 0.0

    # finish_x at/above spawn: agents start with x in [-1.5, 1.5] (spawn square), so at least
    # some spawn beyond a very lenient finish line -- but success requires ALL agents past it,
    # so use a finish line clearly behind the whole spawn square.
    result_trivial = simulate_hebbian_episode(
        rules, seed=9, n_agents=3, wind_enabled=False, max_battery=1.0, min_battery=1.0,
        finish_x=10.0)
    assert result_trivial.success == 1.0  # every spawn position is already < 10.0
