import numpy as np
import pytest

from swarm_gate_passing import config
from swarm_gate_passing.hebbian_controller import unflatten_abcd, init_weights, hebbian_step
from swarm_gate_passing.sensor_model import get_sensor_data
from swarm_gate_passing.simulation import simulate_hebbian_episode, stage_fitness
from swarm_gate_passing.environment import GradientSensor, render_path_map


def _random_genome(seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(config.HEBBIAN_ABCD_BOUNDS[0], config.HEBBIAN_ABCD_BOUNDS[1], config.HEBBIAN_N_ABCD)


def test_genome_length_matches_config():
    genome = _random_genome()
    rules = unflatten_abcd(genome)
    assert len(rules) == 12
    assert rules["A1"].shape == (config.HEBBIAN_N_INPUTS, config.HEBBIAN_N_HIDDEN)
    assert rules["A3"].shape == (config.HEBBIAN_N_HIDDEN, config.HEBBIAN_N_OUTPUTS)


def test_hebbian_step_output_bounds():
    rules = unflatten_abcd(_random_genome())
    w1, w2, w3 = init_weights()
    x_in = np.random.uniform(-1, 1, config.HEBBIAN_N_INPUTS)
    v, w, w1n, w2n, w3n = hebbian_step(x_in, w1, w2, w3, rules)
    assert abs(v) <= config.HEBBIAN_LINEAR_VEL_MAX + 1e-9
    assert abs(w) <= config.HEBBIAN_ANGULAR_VEL_MAX + 1e-9
    assert np.max(np.abs(w1n)) <= 1.0 + 1e-9


def test_sensor_data_shape_with_and_without_gradient():
    agents = np.array([[0.0, 0.0, 0.0, 100.0], [1.0, 0.0, 0.0, 100.0], [0.0, 1.0, 0.0, 100.0]])
    out_no_light = get_sensor_data(agents)
    assert out_no_light.shape == (config.HEBBIAN_N_INPUTS, 3)
    assert np.all(out_no_light[10, :] == 0.0)  # neutral when no gradient sensing

    out_with_light = get_sensor_data(agents, light_intensity=np.array([0.0, 127.5, 255.0]))
    assert np.allclose(out_with_light[10, :], [-1.0, 0.0, 1.0])


@pytest.mark.parametrize("wind_enabled", [False, True])
def test_short_episode_runs_and_returns_finite_fitness(wind_enabled):
    rules = unflatten_abcd(_random_genome(seed=1))
    dist, batt, ct, wct, coh, prox, path = simulate_hebbian_episode(
        rules, seed=42, n_agents=4, wind_enabled=wind_enabled,
        max_battery=5.0, min_battery=5.0, nx=20, ny=20)  # tiny battery/grid to keep the test fast
    for value in (dist, batt, ct, wct, coh, prox, path):
        assert np.isfinite(value)
    eff = stage_fitness(dist, batt, ct, wct, coh, prox, path, "save_battery_avoid_all")
    assert np.isfinite(eff)


def test_episode_with_gradient_sensor_returns_path_alignment():
    grid = render_path_map("sine_curve", 10.0, 10.0, path_kwargs={"freq": 2.0})
    sensor = GradientSensor(grid, world_size_x=10.0, world_size_y=10.0, noise_magnitude=0.5)
    rules = unflatten_abcd(_random_genome(seed=2))

    dist, batt, ct, wct, coh, prox, path_alignment = simulate_hebbian_episode(
        rules, seed=7, n_agents=3, wind_enabled=False,
        max_battery=5.0, min_battery=5.0, gradient_sensor=sensor)

    assert np.isfinite(path_alignment)
    assert 0.0 <= path_alignment <= 100.0
    eff = stage_fitness(dist, batt, ct, wct, coh, prox, path_alignment, "follow_gradient_path")
    assert np.isfinite(eff)
