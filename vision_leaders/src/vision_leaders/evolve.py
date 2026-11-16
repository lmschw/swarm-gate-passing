"""Tune the privileged teacher (aiming.Aim) on gate passage with CMA-ES.

    python -m vision_leaders.evolve --out models/aim_evolved.json

No cameras: the teacher itself drives (aiming.AimSimulation, explorers off,
sigma_leader 0.7). Fitness is the mean fraction of the swarm through each
gate. Every generation draws its own training courses, shared by all
candidates: half three-gate slaloms (7 m, offsets 0.5-1.5 m), half five-gate
winding courses (7 m, turns up to 90 deg), on seeds disjoint from
evaluate.py's. Each generation also scores the default Aim() (the rule
aiming at the centre) on the same courses, as a reference for the noise.
A run stops early once no agent has had a gate target for 60 s.

--fix NAME=VALUE holds a parameter fixed (e.g. --fix goal_gain=4 --fix
beta=0.5 keeps the motion model); --init JSON starts the search from a saved
Aim instead of the default. The CMA-ES mean after the last generation is
written to --out.
"""
import argparse
import math
import sys
import time
from dataclasses import asdict, fields
from multiprocessing import Pool

import cma
import numpy as np

from emergent_leaders import random_turns, slalom_course, winding_course

from .aiming import Aim, AimSimulation
from .courses import start

BOUNDS = {"lateral": (-0.5, 4.0), "through": (-2.0, 6.0), "next_margin": (-1.0, 6.0),
          "goal_gain": (1.0, 12.0), "beta": (0.1, 1.0)}
NAMES = [f.name for f in fields(Aim)]
DURATION = {"slalom": 320.0, "winding": 600.0}
TRAIN_SEED = 10_000


def to_aim(x, fixed=None):
    """x in [0, 1]^k over the parameters not in fixed -> Aim."""
    fixed = fixed or {}
    free = [n for n in NAMES if n not in fixed]
    return Aim(**fixed, **{n: BOUNDS[n][0] + float(np.clip(v, 0, 1)) * (BOUNDS[n][1] - BOUNDS[n][0])
                           for n, v in zip(free, x)})


def from_aim(aim, fixed=None):
    return np.array([(getattr(aim, n) - lo) / (hi - lo) for n, (lo, hi) in BOUNDS.items() if n not in (fixed or {})])


def training_course(seed):
    rng = np.random.default_rng(seed)
    if seed % 2 == 0:
        return "slalom", slalom_course(3, spacing=7.0, offset=rng.uniform(0.5, 1.5))
    return "winding", winding_course(random_turns(5, math.radians(90), rng), 7.0)


def score(aim, seed):
    kind, gates = training_course(seed)
    params = aim.model_params()
    sim = AimSimulation(params=params, state=start(seed), gates=gates, explore=False, seed=seed,
                        check_course=False, aim=aim)
    idle, seen = 0.0, False
    for _ in range(round(DURATION[kind] / params.dt)):
        sim.step()
        targets = (sim.state.gate_target >= 0).any()
        seen |= targets
        idle = 0.0 if targets else idle + params.dt
        if seen and idle > 60.0:
            break
    return float(np.isfinite(sim.gate_pass_time).mean())


def _job(args):
    x, fixed, seed = args
    return score(to_aim(x, fixed), seed)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="JSON file for the tuned Aim")
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--popsize", type=int, default=16)
    parser.add_argument("--courses", type=int, default=8, help="training courses per generation")
    parser.add_argument("--sigma0", type=float, default=0.2, help="initial step size (parameters scaled to [0, 1])")
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fix", action="append", default=[], metavar="NAME=VALUE", help="hold a parameter fixed")
    parser.add_argument("--init", help="JSON Aim to start from (default: Aim())")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)

    fixed = {n: float(v) for n, v in (f.split("=") for f in args.fix)}
    init = Aim.load(args.init) if args.init else Aim()
    es = cma.CMAEvolutionStrategy(from_aim(init, fixed), args.sigma0,
                                  {"bounds": [0, 1], "popsize": args.popsize, "seed": args.seed + 1, "verbose": -9})
    reference = from_aim(Aim(**fixed), fixed)
    with Pool(args.workers) as pool:
        for gen in range(args.generations):
            t0 = time.time()
            seeds = [TRAIN_SEED + (args.seed * 1000 + gen) * args.courses + c for c in range(args.courses)]
            xs = es.ask()
            jobs = [(x, fixed, s) for x in [*xs, reference, es.mean] for s in seeds]
            f = np.array(pool.map(_job, jobs)).reshape(len(xs) + 2, len(seeds)).mean(axis=1)
            es.tell(xs, list(-f[:len(xs)]))
            mean = to_aim(es.mean, fixed)
            print(f"gen {gen + 1:3d}: best {f[:len(xs)].max():.3f}  population {f[:len(xs)].mean():.3f}  "
                  f"mean {f[-1]:.3f}  reference {f[-2]:.3f}  step {es.sigma:.3f}  "
                  + " ".join(f"{k} {v:.2f}" for k, v in asdict(mean).items()) + f"  ({time.time() - t0:.0f} s)")
    to_aim(es.mean, fixed).save(args.out)
    print(f"aim -> {args.out}")


if __name__ == "__main__":
    main()
