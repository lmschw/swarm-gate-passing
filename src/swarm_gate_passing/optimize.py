"""Staged CMA-ES training for the Hebbian ABCD controller.

Vendored and extended from energy_efficient_flocking's ants26_replication/
experiment/optimize_hebbian.py. The staged curriculum and CMA-ES loop are
unchanged; what's new is the `follow_gradient_path` stage and the
--gradient-map CLI flag that constructs a GradientSensor (ported from
volcano_gradient, see environment/sensing.py) and threads it into every
candidate's simulation.

Four sequential stages of increasing task complexity:
  1. walk_left                  -- wind disabled, fitness = distance only
  2. save_battery_avoid_wall    -- wind enabled, + battery term, + wall-collision penalty
  3. save_battery_avoid_all     -- wind enabled, + battery term, + wall AND inter-robot collision penalty
  4. follow_gradient_path       -- stage 3's terms, + reward for staying on a gradient-mapped path

Each stage runs CMA-ES (population 30, 100 generations, sigma0=0.3) with every
candidate evaluated over 3 random seeds, taking the MEDIAN efficiency as its
fitness. Stage 1 starts from a fresh ABCD_init sampled uniformly from [-5, 5];
each later stage starts from the previous stage's best genome.
"""
import argparse
import json
import os
import sys

import cma
import numpy as np

from . import config
from .hebbian_controller import unflatten_abcd
from .simulation import simulate_hebbian_episode, stage_fitness
from .fitness_plot import FitnessPlotter
from .environment import GradientSensor

current_candidate = 0
total_candidates = 0
active_stage = None
active_n_agents = config.HEBBIAN_N_AGENTS
active_n_repeats = config.HEBBIAN_N_REPEATS
active_seed_base = 0
active_max_battery = None
active_min_battery = None
active_nx = None
active_ny = None
active_use_battery_sensor = True
active_gradient_sensor = None


def fitness_wrapper(genome):
    global current_candidate
    current_candidate += 1
    sys.stdout.write(f"\r   ↳ Evaluating Swarm Candidate: {current_candidate}/{total_candidates} ...")
    sys.stdout.flush()

    rules = unflatten_abcd(genome)
    wind_enabled = config.HEBBIAN_STAGE_WIND_ENABLED[active_stage]

    effs = []
    for r in range(active_n_repeats):
        seed = active_seed_base + current_candidate * 1000 + r  # distinct seed per repeat, per candidate
        try:
            dist, batt, ct, wct, coh, prox, path = simulate_hebbian_episode(
                rules, seed=seed, n_agents=active_n_agents, wind_enabled=wind_enabled,
                max_battery=active_max_battery, min_battery=active_min_battery,
                nx=active_nx, ny=active_ny, use_battery_sensor=active_use_battery_sensor,
                gradient_sensor=active_gradient_sensor)
            effs.append(stage_fitness(dist, batt, ct, wct, coh, prox, path, active_stage))
        except Exception as e:
            print(f"\n⚠️  Candidate {current_candidate} repeat {r} failed "
                  f"({type(e).__name__}: {e}) -- treating as worst-case for this repeat")
            effs.append(-99999.0)

    return -float(np.median(effs))  # CMA-ES minimizes


def run_stage(stage, x0, plotter, popsize, maxiter, n_agents, n_repeats, seed_base, output_dir,
              max_battery, min_battery, nx, ny, use_battery_sensor, gradient_sensor, name_suffix):
    global active_stage, total_candidates, active_n_agents, active_n_repeats, active_seed_base
    global active_max_battery, active_min_battery, active_nx, active_ny, active_use_battery_sensor
    global active_gradient_sensor

    active_stage = stage
    active_n_agents = n_agents
    active_n_repeats = n_repeats
    active_seed_base = seed_base
    active_max_battery = max_battery
    active_min_battery = min_battery
    active_nx = nx
    active_ny = ny
    active_use_battery_sensor = use_battery_sensor
    active_gradient_sensor = gradient_sensor
    total_candidates = popsize

    print(f"\n{'=' * 70}\n🧬 STAGE: {stage}  (wind_enabled={config.HEBBIAN_STAGE_WIND_ENABLED[stage]}, "
          f"battery_sensor={use_battery_sensor}, gradient_map={'yes' if gradient_sensor else 'no'})\n{'=' * 70}")

    es = cma.CMAEvolutionStrategy(x0, config.HEBBIAN_CMAES_SIGMA0, {
        'popsize': popsize,
        'maxiter': maxiter,
        'bounds': list(config.HEBBIAN_ABCD_BOUNDS),
    })
    plotter.reset_run(title=f"stage: {stage}{name_suffix}")

    gen = 0
    fitness_history = []
    while not es.stop():
        gen += 1
        current_candidate_reset()
        solutions = es.ask()
        fitness_values = [fitness_wrapper(sol) for sol in solutions]
        es.tell(solutions, fitness_values)
        plotter.update(gen, fitness_values)
        fitness_history.append(float(min(fitness_values)))
        sys.stdout.write("\r")
        print(f"✅ Gen {gen:03d}/{maxiter} | Best Loss (neg eff): {min(fitness_values):.4f}")

    best_genome = es.result[0]
    best_loss = float(es.result[1])

    genome_name = f"hebbian_{stage}{name_suffix}_best.npy"
    history_name = f"hebbian_{stage}{name_suffix}_history.json"
    np.save(os.path.join(output_dir, genome_name), best_genome)
    with open(os.path.join(output_dir, history_name), "w") as f:
        json.dump({"stage": stage, "battery_sensor": use_battery_sensor, "best_loss": best_loss,
                   "best_efficiency": -best_loss, "loss_curve": fitness_history,
                   "genome": best_genome.tolist(),
                   "n_agents": n_agents, "max_battery": max_battery, "min_battery": min_battery,
                   "wind_grid_nx": nx, "wind_grid_ny": ny}, f, indent=2)

    print(f"💾 Stage '{stage}' complete. Best efficiency: {-best_loss:.4f}. Saved {genome_name}")
    return best_genome


def current_candidate_reset():
    global current_candidate
    current_candidate = 0


def train_one_seed(seed, output_dir, stages, popsize, maxiter, n_agents, n_repeats,
                    battery, wind_grid, no_battery_sensor, gradient_map, init_genome_path=None):
    """The full staged curriculum for a single seed. gradient_map: optional path
    to a PNG (see environment.path_maps) -- used only by stages whose
    HEBBIAN_STAGE_FITNESS_WEIGHTS entry sets a path_w (currently
    follow_gradient_path); harmless to pass for other stages since they ignore
    path_alignment entirely."""
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(seed)
    name_suffix = "_nosensor" if no_battery_sensor else ""
    plotter = FitnessPlotter(path=os.path.join(output_dir, f"hebbian_fitness_curve{name_suffix}.png"))

    gradient_sensor = None
    if gradient_map is not None:
        world_size_x = config.X_RANGE[1] - config.X_RANGE[0]
        world_size_y = config.Y_RANGE[1] - config.Y_RANGE[0]
        gradient_sensor = GradientSensor.from_png(gradient_map, world_size_x, world_size_y,
                                                   noise_magnitude=config.GRADIENT_MAP_NOISE_MAGNITUDE)

    if init_genome_path is not None:
        genome = np.load(init_genome_path)
        print(f"↳ Seeding '{stages[0]}' from existing genome '{init_genome_path}' "
              f"(instead of a fresh random one).")
    else:
        genome = np.random.uniform(config.HEBBIAN_ABCD_BOUNDS[0], config.HEBBIAN_ABCD_BOUNDS[1],
                                    config.HEBBIAN_N_ABCD)
    for stage in stages:
        genome = run_stage(stage, genome, plotter, popsize, maxiter, n_agents,
                            n_repeats, seed, output_dir,
                            max_battery=battery, min_battery=battery,
                            nx=wind_grid, ny=wind_grid,
                            use_battery_sensor=not no_battery_sensor,
                            gradient_sensor=gradient_sensor,
                            name_suffix=name_suffix)
    plotter.close()


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Staged CMA-ES training of the Hebbian ABCD controller.")
    parser.add_argument("--popsize", type=int, default=config.HEBBIAN_CMAES_POPSIZE,
                         help=f"CMA-ES population size (default: {config.HEBBIAN_CMAES_POPSIZE}).")
    parser.add_argument("--maxiter", type=int, default=config.HEBBIAN_CMAES_GEN_MAX,
                         help=f"Generations per stage (default: {config.HEBBIAN_CMAES_GEN_MAX}).")
    parser.add_argument("--n-agents", type=int, default=config.HEBBIAN_N_AGENTS,
                         help=f"Swarm size (default: {config.HEBBIAN_N_AGENTS}).")
    parser.add_argument("--n-repeats", type=int, default=config.HEBBIAN_N_REPEATS,
                         help=f"Random-seed repeats per candidate, fitness=median (default: {config.HEBBIAN_N_REPEATS}).")
    parser.add_argument("--seed", type=int, default=0,
                         help="Base seed for reproducibility (single-run mode; ignored if --seeds is given).")
    parser.add_argument("--seeds", type=int, nargs="*", default=None,
                         help="Run the ENTIRE staged curriculum independently for each of these seeds, "
                              "each into its own '<output-dir>/seed_<seed>/' subdirectory. Pass with no "
                              f"values to use this project's canonical seeds ({config.HEBBIAN_BATCH_SEEDS}).")
    parser.add_argument("--output-dir", default="hebbian_results", help="Where to save genomes/histories.")
    parser.add_argument("--stages", nargs="+", default=list(config.HEBBIAN_STAGES),
                         choices=list(config.HEBBIAN_STAGES),
                         help="Which stages to run, in order (default: all four).")
    parser.add_argument("--battery", type=float, default=None,
                         help=f"Starting battery for all agents (default: {config.HEBBIAN_MAX_BATTERY}).")
    parser.add_argument("--wind-grid", type=int, default=None,
                         help=f"Wind grid resolution, both axes (default: {config.HEBBIAN_NX}). "
                              "The single biggest cost lever: the wake-marching loop is O(Nx) per step.")
    parser.add_argument("--no-battery-sensor", action="store_true",
                         help="Evolve a baseline that cannot sense its own battery level at all.")
    parser.add_argument("--gradient-map", default=None, metavar="PATH",
                         help="PNG gradient/path map (see environment.generate_default_map_set or "
                              "scripts/generate_maps.py) to sense via the 11th input and reward "
                              "following in the 'follow_gradient_path' stage.")
    parser.add_argument("--init-genome", default=None, metavar="PATH",
                         help="Seed the FIRST stage in --stages from this existing genome .npy file "
                              "instead of a fresh random one.")
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)

    if args.seeds is not None:
        seeds = args.seeds if len(args.seeds) > 0 else list(config.HEBBIAN_BATCH_SEEDS)
        print(f"🚀 Launching {len(seeds)}-seed Hebbian ABCD training sweep: seeds={seeds}, "
              f"stages={args.stages}, popsize={args.popsize}, maxiter={args.maxiter}, "
              f"n_agents={args.n_agents}, n_repeats={args.n_repeats}, "
              f"battery_sensor={not args.no_battery_sensor}, gradient_map={args.gradient_map}")
        for i, seed in enumerate(seeds):
            seed_dir = os.path.join(args.output_dir, f"seed_{seed}")
            print(f"\n{'#' * 70}\n### SEED {seed} ({i + 1}/{len(seeds)}) -> {seed_dir}/\n{'#' * 70}")
            train_one_seed(seed, seed_dir, args.stages, args.popsize, args.maxiter, args.n_agents,
                           args.n_repeats, args.battery, args.wind_grid, args.no_battery_sensor,
                           args.gradient_map, init_genome_path=args.init_genome)
        print(f"\n🎉 {len(seeds)}-seed sweep complete. Results in '{args.output_dir}/seed_<seed>/'.")
    else:
        print(f"🚀 Launching staged Hebbian ABCD training: stages={args.stages}, "
              f"popsize={args.popsize}, maxiter={args.maxiter}, n_agents={args.n_agents}, "
              f"n_repeats={args.n_repeats}, battery_sensor={not args.no_battery_sensor}, "
              f"gradient_map={args.gradient_map}")
        train_one_seed(args.seed, args.output_dir, args.stages, args.popsize, args.maxiter,
                       args.n_agents, args.n_repeats, args.battery, args.wind_grid,
                       args.no_battery_sensor, args.gradient_map, init_genome_path=args.init_genome)
        print(f"\n🎉 Staged training complete. Results in '{args.output_dir}/'.")


if __name__ == "__main__":
    main()
