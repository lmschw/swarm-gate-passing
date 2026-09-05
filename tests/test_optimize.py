import json
import os
import tempfile

import numpy as np

from swarm_gate_passing import config
from swarm_gate_passing.optimize import EvalConfig, evaluate_candidate, run_stage
from swarm_gate_passing.fitness_plot import FitnessPlotter


def _tiny_cfg(**overrides):
    kwargs = dict(
        stage="flock_cohesion", sensor_mode="quadrant", n_agents=3, n_repeats=1,
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


def test_evaluate_candidate_flock_gate_speed_stage_with_thymio_sensor():
    n_inputs = config.n_inputs_for_sensor_mode("thymio")
    genome = np.random.default_rng(1).uniform(*config.HEBBIAN_ABCD_BOUNDS, config.n_abcd_for(n_inputs))
    cfg = _tiny_cfg(stage="flock_gate_speed", sensor_mode="thymio", gate_enabled=True,
                     finish_x_choices=(-3.0, -4.0))
    loss = evaluate_candidate(genome, candidate_id=0, cfg=cfg)
    assert np.isfinite(loss)


def test_evaluate_candidate_is_deterministic_given_same_ids():
    genome = np.random.default_rng(2).uniform(*config.HEBBIAN_ABCD_BOUNDS, config.HEBBIAN_N_ABCD)
    cfg = _tiny_cfg()
    loss_a = evaluate_candidate(genome, candidate_id=5, cfg=cfg)
    loss_b = evaluate_candidate(genome, candidate_id=5, cfg=cfg)
    assert loss_a == loss_b


def test_elitism_final_best_matches_best_ever_seen_across_generations():
    """The whole point of elitism (config.HEBBIAN_ELITISM_COUNT) is that the
    final saved genome can never be worse than any per-generation minimum seen
    during the run -- a genuinely better genome found mid-run would itself
    become (and stay) the elite, so it must show up as the final best_loss."""
    with tempfile.TemporaryDirectory() as tmpdir:
        plotter = FitnessPlotter(path=os.path.join(tmpdir, "curve.png"))
        x0 = np.random.default_rng(0).uniform(*config.HEBBIAN_ABCD_BOUNDS, config.HEBBIAN_N_ABCD)
        cfg_kwargs = dict(
            sensor_mode="quadrant", n_agents=3, n_repeats=1,
            max_battery=3.0, min_battery=3.0, nx=20, ny=20,
            use_battery_sensor=True, wind_enabled=False,
            gradient_map_path=None, freq_choices=(), gate_enabled=False, n_gates=1,
            finish_x_choices=(), gate_opening_width=1.0, post_gate_distance=1.5,
            max_steps=30,
        )
        best_genome = run_stage("flock_cohesion", x0, plotter, popsize=6, maxiter=3,
                                 cfg_kwargs=cfg_kwargs, output_dir=tmpdir, name_suffix="",
                                 n_workers=2, cma_seed=7)
        plotter.close()

        with open(os.path.join(tmpdir, "hebbian_flock_cohesion_history.json")) as f:
            history = json.load(f)
        assert history["best_loss"] <= min(history["loss_curve"]) + 1e-9
        assert np.array_equal(np.array(history["genome"]), best_genome)
