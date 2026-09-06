import json
import os
import tempfile

import numpy as np
import pytest

from swarm_gate_passing import config
from swarm_gate_passing.optimize import EvalConfig, evaluate_candidate, run_stage, _make_episode_environment, _lerp_bounds
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


def test_vision_gate_directed_narrows_x_bounds_and_sets_direction_hint():
    cfg = _tiny_cfg(stage="vision_gate_directed", sensor_mode="vision")
    rng = np.random.default_rng(0)
    for _ in range(20):
        gradient_sensor, gates, finish_x = _make_episode_environment(cfg, rng)
        assert gradient_sensor is None
        assert len(gates) == 1
        assert config.GATE_DIRECTED_X_BOUNDS[0] <= gates[0].x_arena <= config.GATE_DIRECTED_X_BOUNDS[1]
        assert finish_x == config.GATE_DIRECTED_DISTANCE_HINT_X


def test_vision_gate_undirected_uses_full_bounds_and_no_hint():
    cfg = _tiny_cfg(stage="vision_gate", sensor_mode="vision")
    rng = np.random.default_rng(0)
    gradient_sensor, gates, finish_x = _make_episode_environment(cfg, rng)
    assert finish_x is None
    assert config.GATE_RANDOM_X_BOUNDS[0] <= gates[0].x_arena <= config.GATE_RANDOM_X_BOUNDS[1]


def test_lerp_bounds_at_endpoints_and_midpoint():
    easy, hard = (-1.2, -0.9), (-3.0, 3.0)
    assert _lerp_bounds(easy, hard, 0.0) == pytest.approx(easy)
    assert _lerp_bounds(easy, hard, 1.0) == pytest.approx(hard)
    mid = _lerp_bounds(easy, hard, 0.5)
    assert mid[0] == pytest.approx((easy[0] + hard[0]) / 2)
    assert mid[1] == pytest.approx((easy[1] + hard[1]) / 2)


def test_vision_gate_curriculum_uses_easy_bounds_at_difficulty_zero():
    cfg = _tiny_cfg(stage="vision_gate_curriculum", sensor_mode="vision", gate_difficulty=0.0)
    rng = np.random.default_rng(0)
    for _ in range(20):
        _, gates, finish_x = _make_episode_environment(cfg, rng)
        assert config.GATE_CURRICULUM_EASY_X_BOUNDS[0] <= gates[0].x_arena <= config.GATE_CURRICULUM_EASY_X_BOUNDS[1]
        assert config.GATE_CURRICULUM_EASY_Y_BOUNDS[0] <= (gates[0].y_lo_arena + gates[0].y_hi_arena) / 2 <= config.GATE_CURRICULUM_EASY_Y_BOUNDS[1]
        assert finish_x is None  # curriculum stage has no direction hint, unlike "directed"


def test_vision_gate_curriculum_uses_full_bounds_at_difficulty_one():
    cfg = _tiny_cfg(stage="vision_gate_curriculum", sensor_mode="vision", gate_difficulty=1.0)
    rng = np.random.default_rng(0)
    xs = []
    for _ in range(30):
        _, gates, _ = _make_episode_environment(cfg, rng)
        xs.append(gates[0].x_arena)
    # at full difficulty, placements should span well beyond the easy band
    assert min(xs) < config.GATE_CURRICULUM_EASY_X_BOUNDS[0]
    assert max(xs) > config.GATE_CURRICULUM_EASY_X_BOUNDS[1] or max(xs) > -1.0


def test_elitism_final_best_equals_last_generations_fresh_minimum():
    """elite is rebuilt EACH generation purely from that generation's fresh
    evaluations (which always include every previous elite, force-included and
    freshly re-tested -- see run_stage) -- never merged with the outgoing elite
    list's OLD stored scores. That's the fix for a real bug: merging with the
    stale list let a lucky historical score (e.g. from an easier point in a
    difficulty curriculum) out-live its own fresh re-evaluation forever, since
    sort-by-loss keeps whichever of the two is numerically better regardless of
    which is still true. Post-fix, the final best_loss must exactly equal the
    last generation's own minimum -- elitism's value is in shaping the search
    trajectory (keeping good genomes in front of CMA-ES across generations),
    not in memorializing a score that may no longer be accurate."""
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
        assert history["best_loss"] == pytest.approx(history["loss_curve"][-1])
        assert np.array_equal(np.array(history["genome"]), best_genome)
