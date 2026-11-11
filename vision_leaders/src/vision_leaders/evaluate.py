"""Compare gate passing with the rule and with trained networks.

    python -m vision_leaders.evaluate models/goalnet.pt [more models...]

Suites (explorers off, sigma_leader 0.7, swarm facing the course):
- slalom: the default three-gate slalom, seeds 0-9, 320 s;
- winding: emergent_leaders.run's five-gate winding courses for seeds 0-19,
  gates 7 m apart, 600 s.
Reports the fraction of the swarm through each gate and the runs with more
than half of the swarm through every gate.
"""
import argparse
from multiprocessing import Pool

import numpy as np
import torch

from emergent_leaders import ModelParams, Simulation

from .courses import course, start
from .network import GoalNet
from .simulation import VisionSimulation

SUITES = {"slalom": ("slalom", 3, range(10), 320.0), "winding": ("winding", 5, range(20), 600.0)}


def run_one(args):
    model, suite, seed = args
    torch.set_num_threads(1)
    kind, n_gates, _, duration = SUITES[suite]
    params = ModelParams(sigma_leader=0.7)
    common = dict(params=params, state=start(seed), gates=course(kind, n_gates, seed, 7.0, params=params),
                  explore=False, seed=seed, check_course=False)
    if model == "rule":
        sim = Simulation(**common)
    else:
        net = GoalNet.load(model)
        sim = VisionSimulation(net=net, camera=net.cam, **common)
    sim.run(round(duration / params.dt), record_every=1000)
    return model, suite, seed, (~np.isnan(sim.gate_pass_time)).mean(axis=1)


def evaluate(models, suites=tuple(SUITES), workers=20):
    jobs = [(m, s, seed) for m in models for s in suites for seed in SUITES[s][2]]
    with Pool(workers) as pool:
        return pool.map(run_one, jobs)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models", nargs="*", help="GoalNet files; the rule is always included")
    parser.add_argument("--suites", nargs="+", choices=list(SUITES), default=list(SUITES))
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args(argv)
    models = ["rule", *args.models]
    out = evaluate(models, args.suites, args.workers)
    for suite in args.suites:
        for m in models:
            frac = np.array([o[3] for o in out if o[0] == m and o[1] == suite])
            print(f"{suite:8} {m:30}: through {np.round(100 * frac.mean(0)).astype(int)}%  "
                  f"runs > 50% through all {int((frac > 0.5).all(1).sum())} / {len(frac)}")


if __name__ == "__main__":
    main()
