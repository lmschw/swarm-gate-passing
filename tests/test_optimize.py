import numpy as np

from swarm_gate_passing import config
from swarm_gate_passing.optimize import EvalConfig, evaluate_candidate


def _tiny_cfg(**overrides):
    kwargs = dict(
        stage="follow_gradient_no_gate", sensor_mode="quadrant", n_agents=3, n_repeats=1,
        seed_base=0, max_battery=3.0, min_battery=3.0, nx=20, ny=20,
        use_battery_sensor=True, wind_enabled=False,
        freq_choices=(2.0, 4.0), gate_enabled=False, finish_x_choices=(-3.0,),
    )
    kwargs.update(overrides)
    return EvalConfig(**kwargs)


def test_evaluate_candidate_returns_finite_loss():
    genome = np.random.default_rng(0).uniform(*config.HEBBIAN_ABCD_BOUNDS, config.HEBBIAN_N_ABCD)
    loss = evaluate_candidate(genome, candidate_id=0, cfg=_tiny_cfg())
    assert np.isfinite(loss)


def test_evaluate_candidate_gate_passing_stage_with_thymio_sensor():
    n_inputs = config.n_inputs_for_sensor_mode("thymio")
    genome = np.random.default_rng(1).uniform(*config.HEBBIAN_ABCD_BOUNDS, config.n_abcd_for(n_inputs))
    cfg = _tiny_cfg(stage="gate_passing", sensor_mode="thymio", gate_enabled=True,
                     finish_x_choices=(-3.0, -4.0))
    loss = evaluate_candidate(genome, candidate_id=0, cfg=cfg)
    assert np.isfinite(loss)


def test_evaluate_candidate_is_deterministic_given_same_ids():
    genome = np.random.default_rng(2).uniform(*config.HEBBIAN_ABCD_BOUNDS, config.HEBBIAN_N_ABCD)
    cfg = _tiny_cfg()
    loss_a = evaluate_candidate(genome, candidate_id=5, cfg=cfg)
    loss_b = evaluate_candidate(genome, candidate_id=5, cfg=cfg)
    assert loss_a == loss_b
