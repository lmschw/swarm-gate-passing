"""Staged CMA-ES training for the Hebbian ABCD controller.

Vendored and extended from energy_efficient_flocking's ants26_replication/
experiment/optimize_hebbian.py. The staged curriculum and CMA-ES loop are
unchanged in spirit; two things are new:

  - --sensor-mode {quadrant,thymio} and --gradient-map / --path-freq-choices /
    --gate-x-choices / --gate-opening-width, threading sensor choice and
    domain-randomized gradient maps/gate placements into every candidate's
    simulation (see EvalConfig/evaluate_candidate below).
  - candidate evaluation is parallelized across a process pool (each CMA-ES
    candidate's simulate_hebbian_episode calls are independent), since this is
    by far the largest lever on wall-clock training time and the vendored
    version evaluated every candidate serially.

Eight stages across two curricula (each --stages run starts its own fresh genome
unless --init-genome is given; curricula don't chain into each other):

  Energy-efficiency curriculum (Table 2 + the gradient-path stage added when
  this project combined in volcano_gradient's sensing):
    1. walk_left                  -- wind disabled, fitness = distance only
    2. save_battery_avoid_wall    -- + battery term, + wall-collision penalty
    3. save_battery_avoid_all     -- + inter-robot collision penalty
    4. follow_gradient_path       -- + reward for staying on a fixed gradient map

  Gate-passing curriculum (this project's second combination -- see config.py's
  GATE_* constants and simulation.EpisodeResult/stage_fitness) -- an INCREMENTAL
  curriculum: each stage's fitness is the previous stage's plus exactly one new
  term, so CMA-ES only ever has to learn one new skill on top of a genome that
  already has the earlier ones, rather than all four at once from scratch
  (confirmed necessary in practice -- a combined single-shot fitness produced
  genomes that neither followed the gradient nor stayed together):
    1. flock_cohesion   -- fitness = stay close together (cohesion only)
    2. flock_gradient   -- + follow the gradient (path-deviation penalty)
    3. flock_gate       -- + reach/pass a physical gate (capped distance-to-goal,
                             success bonus, post-gate regrouping)
    4. flock_gate_speed -- + reward speed

Each stage runs CMA-ES (population 30, 100 generations, sigma0=0.3 by default)
with every candidate evaluated over 3 random seeds, taking the MEDIAN
efficiency as its fitness.
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Optional, Tuple

import cma
import numpy as np

from . import config
from .hebbian_controller import unflatten_abcd
from .simulation import simulate_hebbian_episode, stage_fitness
from .fitness_plot import FitnessPlotter
from .environment import GradientSensor, evenly_spaced_gates, random_gate, render_path_map


@dataclass(frozen=True)
class EvalConfig:
    stage: str
    sensor_mode: str
    n_agents: int
    n_repeats: int
    seed_base: int
    max_battery: Optional[float]
    min_battery: Optional[float]
    nx: Optional[int]
    ny: Optional[int]
    use_battery_sensor: bool
    wind_enabled: bool
    gradient_map_path: Optional[str] = None   # fixed map (follow_gradient_path stage)
    freq_choices: Tuple[float, ...] = ()       # domain-randomized wavelength choices
    gate_enabled: bool = False
    n_gates: int = 1                           # gates evenly spaced along the track when gate_enabled
    finish_x_choices: Tuple[float, ...] = ()   # domain-randomized gate-x choices (gate_enabled) or
                                                # finish-line choices directly (not gate_enabled)
    gate_opening_width: float = config.GATE_OPENING_WIDTH_M
    post_gate_distance: float = config.GATE_POST_GATE_DISTANCE_M
    max_steps: Optional[int] = None
    gate_difficulty: Optional[float] = None   # 0.0 (easy) .. 1.0 (full random) -- see
                                                # config.GATE_CURRICULUM_STAGES/run_stage.
                                                # None = use the stage's static bounds instead.


_VISION_GATE_BOUNDS = {
    "vision_gate_directed": (config.GATE_DIRECTED_X_BOUNDS, config.GATE_DIRECTED_Y_BOUNDS),
}
_VISION_DIRECTION_HINT = {
    "vision_gate_directed": config.GATE_DIRECTED_DISTANCE_HINT_X,
}


def _lerp_bounds(easy, hard, t):
    """Interpolates a (lo, hi) bounds pair between easy and hard at t in [0, 1]."""
    lo = easy[0] + (hard[0] - easy[0]) * t
    hi = easy[1] + (hard[1] - easy[1]) * t
    return (lo, hi)


def _make_episode_environment(cfg: EvalConfig, rng: np.random.Generator):
    """Builds the (gradient_sensor, gates, finish_x) for one episode, applying
    domain randomization when configured. A fresh map is rendered per call
    (cheap -- a few ms at this resolution) rather than cached, so each repeat
    can get an independently-sampled wavelength."""
    if cfg.stage in config.VISION_STAGES:
        # No gradient map at all -- a single gate "strewn" at a random position,
        # unrelated to any path (see environment.gate.random_gate). Placement
        # bounds and (for "directed") a fixed general-direction reward hint are
        # per-stage -- see config.GATE_RANDOM_*/GATE_DIRECTED_*.
        gates = None
        finish_x = None
        if cfg.stage in config.VISION_GATE_ENABLED_STAGES:
            if cfg.gate_difficulty is not None:
                # Curriculum ramp: interpolate placement bounds by difficulty
                # rather than using a fixed pair for the whole stage.
                x_bounds = _lerp_bounds(config.GATE_CURRICULUM_EASY_X_BOUNDS,
                                         config.GATE_CURRICULUM_HARD_X_BOUNDS, cfg.gate_difficulty)
                y_bounds = _lerp_bounds(config.GATE_CURRICULUM_EASY_Y_BOUNDS,
                                         config.GATE_CURRICULUM_HARD_Y_BOUNDS, cfg.gate_difficulty)
            else:
                x_bounds, y_bounds = _VISION_GATE_BOUNDS.get(
                    cfg.stage, (config.GATE_RANDOM_X_BOUNDS, config.GATE_RANDOM_Y_BOUNDS))
            gates = [random_gate(rng, cfg.gate_opening_width, x_bounds, y_bounds)]
            finish_x = _VISION_DIRECTION_HINT.get(cfg.stage)
        return None, gates, finish_x

    world_w = config.X_RANGE[1] - config.X_RANGE[0]
    world_h = config.Y_RANGE[1] - config.Y_RANGE[0]

    gradient_sensor = None
    if cfg.gradient_map_path is not None:
        gradient_sensor = GradientSensor.from_png(cfg.gradient_map_path, world_w, world_h,
                                                   noise_magnitude=config.GRADIENT_MAP_NOISE_MAGNITUDE)
    elif cfg.freq_choices:
        freq = float(rng.choice(cfg.freq_choices))
        grid = render_path_map("sine_curve", world_w, world_h,
                                meters_per_pixel=config.GRADIENT_MAP_METERS_PER_PIXEL,
                                path_width_m=config.GRADIENT_DEFAULT_PATH_WIDTH_M,
                                path_kwargs={"freq": freq})
        gradient_sensor = GradientSensor(grid, world_w, world_h,
                                          noise_magnitude=config.GRADIENT_MAP_NOISE_MAGNITUDE)

    gates = None
    finish_x = None
    if gradient_sensor is not None and cfg.finish_x_choices:
        gate_x = float(rng.choice(cfg.finish_x_choices))
        if cfg.gate_enabled:
            gates = evenly_spaced_gates(gradient_sensor, cfg.n_gates, gate_x, cfg.gate_opening_width,
                                         config.X_RANGE, config.Y_RANGE)
            # Success is measured some distance PAST the last gate, not at it -- a gate can
            # physically scatter the swarm as agents squeeze through individually, so this is
            # what actually requires (and rewards) regrouping rather than crediting "everyone's
            # individually clear of the barrier" the instant that happens. Travel is in -x, so
            # "past" means further negative.
            finish_x = gate_x - cfg.post_gate_distance
        else:
            finish_x = gate_x

    return gradient_sensor, gates, finish_x


def evaluate_candidate(genome, candidate_id, cfg: EvalConfig):
    """Pure function of (genome, candidate_id, cfg) -- no module-level mutable
    state -- so it's safe to run in a worker process. Returns -median(efficiency)
    over cfg.n_repeats replicate episodes (CMA-ES minimizes)."""
    n_inputs = config.n_inputs_for_sensor_mode(cfg.sensor_mode)
    rules = unflatten_abcd(genome, n_inputs=n_inputs)
    env_rng = np.random.default_rng(cfg.seed_base + candidate_id)

    effs = []
    for r in range(cfg.n_repeats):
        seed = cfg.seed_base + candidate_id * 1000 + r
        try:
            gradient_sensor, gates, finish_x = _make_episode_environment(cfg, env_rng)
            crossing_success = cfg.stage in config.VISION_GATE_ENABLED_STAGES
            result = simulate_hebbian_episode(
                rules, seed=seed, n_agents=cfg.n_agents, wind_enabled=cfg.wind_enabled,
                max_battery=cfg.max_battery, min_battery=cfg.min_battery,
                nx=cfg.nx, ny=cfg.ny, use_battery_sensor=cfg.use_battery_sensor,
                sensor_mode=cfg.sensor_mode, gradient_sensor=gradient_sensor,
                gates=gates, finish_x=finish_x, max_steps=cfg.max_steps,
                crossing_success=crossing_success)
            effs.append(stage_fitness(result, cfg.stage))
        except Exception as e:
            print(f"\n⚠️  Candidate {candidate_id} repeat {r} failed "
                  f"({type(e).__name__}: {e}) -- treating as worst-case for this repeat")
            effs.append(-99999.0)

    return -float(np.median(effs))  # CMA-ES minimizes


def run_stage(stage, x0, plotter, popsize, maxiter, cfg_kwargs, output_dir, name_suffix, n_workers,
              cma_seed=None):
    print(f"\n{'=' * 70}\n🧬 STAGE: {stage}  ({cfg_kwargs})\n{'=' * 70}")

    # cma's own 'seed' option treats 0/None as "use system time" (non-deterministic) --
    # without setting it explicitly, our own --seed only pins the per-episode simulation
    # RNG (via EvalConfig/simulate_hebbian_episode's seed=), NOT which candidate genomes
    # CMA-ES samples each generation, so "the same --seed" alone does not reproduce a run
    # end-to-end (confirmed in practice: two runs with --seed 42 diverged sharply by the
    # gate-passing stage, a harder/more sensitive landscape than the earlier stages).
    es = cma.CMAEvolutionStrategy(x0, config.HEBBIAN_CMAES_SIGMA0, {
        'popsize': popsize,
        'maxiter': maxiter,
        'bounds': list(config.HEBBIAN_ABCD_BOUNDS),
        'seed': cma_seed if cma_seed else 0,
    })
    plotter.reset_run(title=f"stage: {stage}{name_suffix}")

    gen = 0
    fitness_history = []
    elite = []  # up to HEBBIAN_ELITISM_COUNT (loss, genome) pairs, sorted best-first, best ever seen
    t_stage_start = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        while not es.stop():
            gen += 1
            t_gen_start = time.time()
            solutions = es.ask()
            # Force-include the best-ever genomes in this generation's candidate pool
            # (replacing that many CMA-ES-sampled ones) so a rare discovery can never
            # simply be dropped by the search distribution drifting away from it --
            # see config.HEBBIAN_ELITISM_COUNT. Re-evaluated fresh every generation
            # (not reusing the old fitness value), since conditions are randomized
            # per episode and a stale score wouldn't be a fair, up-to-date comparison.
            for k in range(min(len(elite), len(solutions))):
                solutions[-(k + 1)] = elite[k][1].copy()

            gate_difficulty = None
            if stage in config.GATE_CURRICULUM_STAGES:
                ramp_gens = max(1, int(maxiter * config.GATE_CURRICULUM_RAMP_FRACTION))
                gate_difficulty = min(1.0, gen / ramp_gens)

            cfg = EvalConfig(stage=stage, seed_base=(gen - 1) * popsize,
                              gate_difficulty=gate_difficulty, **cfg_kwargs)
            futures = [pool.submit(evaluate_candidate, sol, i, cfg) for i, sol in enumerate(solutions)]
            fitness_values = [f.result() for f in futures]
            es.tell(solutions, fitness_values)

            # Built ONLY from this generation's fresh evaluations -- NOT merged with the
            # previous elite list. Every existing elite was just force-included and
            # re-evaluated above, so it's already present in fitness_values/solutions;
            # carrying the OLD elite entries forward too would let a stale, lucky score
            # from earlier in the run (e.g. an easier point in a difficulty curriculum)
            # outlive its own fresh re-evaluation forever, since sort-by-loss always
            # keeps whichever of the two is numerically better regardless of which is
            # actually still true. A real bug, caught because a curriculum run's
            # reported best never moved past its easy-phase discovery despite 90 more
            # generations at harder difficulty -- elite[0] must always reflect a
            # genome's CURRENT performance, not its best-ever historical one.
            elite = sorted(zip(fitness_values, (s.copy() for s in solutions)),
                            key=lambda t: t[0])[:config.HEBBIAN_ELITISM_COUNT]

            plotter.update(gen, fitness_values)
            fitness_history.append(float(min(fitness_values)))
            difficulty_str = f" | difficulty: {gate_difficulty:.2f}" if gate_difficulty is not None else ""
            print(f"✅ Gen {gen:03d}/{maxiter} | Best Loss (neg eff): {min(fitness_values):.4f} "
                  f"| elite best: {elite[0][0]:.4f}{difficulty_str} | {time.time() - t_gen_start:.1f}s")

    print(f"   (stage wall-clock: {time.time() - t_stage_start:.1f}s)")
    # Use our own elite-tracking (freshly re-evaluated every generation it survives)
    # rather than cma's own internal es.result bookkeeping, whose recorded fitness for
    # the best-ever solution can be stale/lucky from whenever it was first sampled --
    # elite[0] reflects at least one recent, fair re-evaluation.
    best_genome = elite[0][1] if elite else es.result[0]
    best_loss = float(elite[0][0]) if elite else float(es.result[1])

    genome_name = f"hebbian_{stage}{name_suffix}_best.npy"
    history_name = f"hebbian_{stage}{name_suffix}_history.json"
    np.save(os.path.join(output_dir, genome_name), best_genome)
    with open(os.path.join(output_dir, history_name), "w") as f:
        json.dump({"stage": stage, "best_loss": best_loss, "best_efficiency": -best_loss,
                   "loss_curve": fitness_history, "genome": best_genome.tolist(),
                   "cfg": {k: v for k, v in cfg_kwargs.items()}}, f, indent=2)

    print(f"💾 Stage '{stage}' complete. Best efficiency: {-best_loss:.4f}. Saved {genome_name}")
    return best_genome


def train_one_seed(seed, output_dir, stages, popsize, maxiter, n_agents, n_repeats,
                    battery, wind_grid, no_battery_sensor, gradient_map, sensor_mode,
                    freq_choices, gate_x_choices, gate_opening_width, n_gates, post_gate_distance,
                    n_workers, init_genome_path=None):
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(seed)
    if any(s in config.VISION_STAGES for s in stages) and sensor_mode != "vision":
        print(f"⚠️  {[s for s in stages if s in config.VISION_STAGES]} expect --sensor-mode vision "
              f"(the only mode that can perceive a gate at all without a gradient map) -- got "
              f"'{sensor_mode}'. Continuing, but the gate will likely be effectively invisible.")
    name_suffix = "_nosensor" if no_battery_sensor else ""
    if sensor_mode != "quadrant":
        name_suffix += f"_{sensor_mode}"
    plotter = FitnessPlotter(path=os.path.join(output_dir, f"hebbian_fitness_curve{name_suffix}.png"))

    n_inputs = config.n_inputs_for_sensor_mode(sensor_mode)
    n_abcd = config.n_abcd_for(n_inputs)

    if init_genome_path is not None:
        genome = np.load(init_genome_path)
        print(f"↳ Seeding '{stages[0]}' from existing genome '{init_genome_path}' "
              f"(instead of a fresh random one).")
    else:
        genome = np.random.uniform(config.HEBBIAN_ABCD_BOUNDS[0], config.HEBBIAN_ABCD_BOUNDS[1], n_abcd)

    for stage_idx, stage in enumerate(stages):
        wind_enabled = config.HEBBIAN_STAGE_WIND_ENABLED[stage]
        gate_enabled = stage in config.GATE_ENABLED_STAGES
        is_gate_task_stage = stage in config.GATE_STAGES
        is_vision_task_stage = stage in config.VISION_STAGES
        this_freq_choices = tuple(freq_choices) if is_gate_task_stage else ()
        this_gate_x_choices = tuple(gate_x_choices) if this_freq_choices else ()
        cfg_kwargs = dict(
            sensor_mode=sensor_mode, n_agents=n_agents, n_repeats=n_repeats,
            max_battery=battery, min_battery=battery, nx=wind_grid, ny=wind_grid,
            use_battery_sensor=not no_battery_sensor, wind_enabled=wind_enabled,
            gradient_map_path=gradient_map if stage == "follow_gradient_path" else None,
            freq_choices=this_freq_choices, gate_enabled=gate_enabled, n_gates=n_gates,
            finish_x_choices=this_gate_x_choices, gate_opening_width=gate_opening_width,
            post_gate_distance=post_gate_distance,
            max_steps=config.GATE_MAX_STEPS if (is_gate_task_stage or is_vision_task_stage) else None,
        )
        genome = run_stage(stage, genome, plotter, popsize, maxiter, cfg_kwargs, output_dir,
                            name_suffix, n_workers, cma_seed=seed + stage_idx + 1)
    plotter.close()


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Staged CMA-ES training of the Hebbian ABCD controller.")
    parser.add_argument("--popsize", type=int, default=config.HEBBIAN_CMAES_POPSIZE)
    parser.add_argument("--maxiter", type=int, default=config.HEBBIAN_CMAES_GEN_MAX)
    parser.add_argument("--n-agents", type=int, default=config.HEBBIAN_N_AGENTS)
    parser.add_argument("--n-repeats", type=int, default=config.HEBBIAN_N_REPEATS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="*", default=None,
                         help="Run the ENTIRE staged curriculum independently for each of these seeds, "
                              "each into its own '<output-dir>/seed_<seed>/' subdirectory. Pass with no "
                              f"values to use this project's canonical seeds ({config.HEBBIAN_BATCH_SEEDS}).")
    parser.add_argument("--output-dir", default="hebbian_results")
    parser.add_argument("--stages", nargs="+", default=list(config.GATE_STAGES),
                         choices=list(config.HEBBIAN_STAGES),
                         help=f"Default is the gate-passing curriculum {config.GATE_STAGES} (see module "
                              "docstring). The energy-efficiency curriculum (walk_left, "
                              "save_battery_avoid_wall, save_battery_avoid_all, follow_gradient_path) "
                              "is a separate, unrelated task -- pass its stage names explicitly to run it.")
    parser.add_argument("--battery", type=float, default=None)
    parser.add_argument("--wind-grid", type=int, default=None)
    parser.add_argument("--no-battery-sensor", action="store_true")
    parser.add_argument("--sensor-mode", default="quadrant", choices=list(config.SENSOR_MODES),
                         help="'quadrant': idealized distance/bearing per neighbor (positions). "
                              "'thymio': raw 7-channel IR proximity, no neighbor identity/bearing.")
    parser.add_argument("--gradient-map", default=None, metavar="PATH",
                         help="Fixed PNG gradient/path map for the 'follow_gradient_path' stage. "
                              "Ignored by every other stage.")
    parser.add_argument("--path-freq-choices", type=float, nargs="+", default=list(config.GATE_FREQ_CHOICES),
                         help=f"Sine-curve wavelengths (as frequency) to sample per episode in any of "
                              f"{config.GATE_STAGES} (domain randomization).")
    parser.add_argument("--gate-x-choices", type=float, nargs="+", default=list(config.GATE_FINISH_X_CHOICES),
                         help="Arena-frame x placements to sample per episode for the finish line "
                              f"(pre-gate stages) / gate ({config.GATE_ENABLED_STAGES}).")
    parser.add_argument("--gate-opening-width", type=float, default=config.GATE_OPENING_WIDTH_M)
    parser.add_argument("--n-gates", type=int, default=1,
                         help="Number of gates evenly spaced along the track (from the spawn area "
                              f"to the randomly-chosen gate placement) in {config.GATE_ENABLED_STAGES}. "
                              "1 (default) reproduces the original single-gate behavior exactly.")
    parser.add_argument("--post-gate-distance", type=float, default=config.GATE_POST_GATE_DISTANCE_M,
                         help=f"How far past the LAST gate (arena meters, {config.GATE_ENABLED_STAGES} only) "
                              "the success/finish line sits -- requires the swarm to keep traveling "
                              "together after the gate, not just individually clear it.")
    parser.add_argument("--init-genome", default=None, metavar="PATH")
    parser.add_argument("--workers", type=int, default=None,
                         help="Process-pool size for parallel candidate evaluation "
                              "(default: min(popsize, cpu_count)).")
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    n_workers = args.workers or min(args.popsize, os.cpu_count() or 1)

    def _run(seed, output_dir):
        train_one_seed(seed, output_dir, args.stages, args.popsize, args.maxiter, args.n_agents,
                        args.n_repeats, args.battery, args.wind_grid, args.no_battery_sensor,
                        args.gradient_map, args.sensor_mode, args.path_freq_choices,
                        args.gate_x_choices, args.gate_opening_width, args.n_gates,
                        args.post_gate_distance, n_workers, init_genome_path=args.init_genome)

    if args.seeds is not None:
        seeds = args.seeds if len(args.seeds) > 0 else list(config.HEBBIAN_BATCH_SEEDS)
        print(f"🚀 {len(seeds)}-seed sweep: seeds={seeds}, stages={args.stages}, "
              f"sensor_mode={args.sensor_mode}, popsize={args.popsize}, maxiter={args.maxiter}, "
              f"n_agents={args.n_agents}, n_repeats={args.n_repeats}, workers={n_workers}")
        for i, seed in enumerate(seeds):
            seed_dir = os.path.join(args.output_dir, f"seed_{seed}")
            print(f"\n{'#' * 70}\n### SEED {seed} ({i + 1}/{len(seeds)}) -> {seed_dir}/\n{'#' * 70}")
            _run(seed, seed_dir)
        print(f"\n🎉 {len(seeds)}-seed sweep complete. Results in '{args.output_dir}/seed_<seed>/'.")
    else:
        print(f"🚀 stages={args.stages}, sensor_mode={args.sensor_mode}, popsize={args.popsize}, "
              f"maxiter={args.maxiter}, n_agents={args.n_agents}, n_repeats={args.n_repeats}, "
              f"workers={n_workers}")
        _run(args.seed, args.output_dir)
        print(f"\n🎉 Staged training complete. Results in '{args.output_dir}/'.")


if __name__ == "__main__":
    main()
