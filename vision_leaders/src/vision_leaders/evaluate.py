"""Compare gate passing with the rule and with trained networks.

    python -m vision_leaders.evaluate models/goalnet.pt [more models...]
    python -m vision_leaders.evaluate models/goalnet.pt --goal-weighting threshold confidence

Suites (explorers off, sigma_leader 0.7, swarm facing the course):
- slalom: the default three-gate slalom, seeds 0-9, 320 s;
- winding: emergent_leaders.run's five-gate winding courses for seeds 0-19,
  gates 7 m apart, 600 s.
Reports the fraction of the swarm through each gate, the runs with more
than half of the swarm through every gate, and for networks the share of
agent-steps with an unsure confidence (0.05 < q < 0.95).
"""
import argparse
from multiprocessing import Pool

import numpy as np
import torch

from emergent_leaders import ModelParams, Simulation

from .courses import course, start
from .network import GoalNet
from .simulation import GOAL_WEIGHTINGS, VisionSimulation

SUITES = {"slalom": ("slalom", 3, range(10), 320.0), "winding": ("winding", 5, range(20), 600.0)}


def run_one(args):
    model, weighting, suite, seed = args
    torch.set_num_threads(1)
    kind, n_gates, _, duration = SUITES[suite]
    params = ModelParams(sigma_leader=0.7)
    common = dict(params=params, state=start(seed), gates=course(kind, n_gates, seed, 7.0, params=params),
                  explore=False, seed=seed, check_course=False)
    unsure = []
    if model == "rule":
        sim = Simulation(**common)
    else:
        net = GoalNet.load(model)
        sim = VisionSimulation(net=net, camera=net.cam, goal_weighting=weighting, **common)
    for _ in range(round(duration / params.dt)):
        sim.step()
        if model != "rule":
            unsure.append(np.mean((sim.net_prob > 0.05) & (sim.net_prob < 0.95)))
    unsure = float(np.mean(unsure)) if unsure else float("nan")
    return model, weighting, suite, seed, (~np.isnan(sim.gate_pass_time)).mean(axis=1), unsure


def evaluate(models, weightings=("threshold",), suites=tuple(SUITES), workers=20):
    policies = [("rule", "-")] + [(m, w) for m in models if m != "rule" for w in weightings]
    jobs = [(m, w, s, seed) for m, w in policies for s in suites for seed in SUITES[s][2]]
    with Pool(workers) as pool:
        return pool.map(run_one, jobs)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models", nargs="*", help="GoalNet files; the rule is always included")
    parser.add_argument("--goal-weighting", nargs="+", choices=GOAL_WEIGHTINGS, default=["threshold"])
    parser.add_argument("--suites", nargs="+", choices=list(SUITES), default=list(SUITES))
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args(argv)
    out = evaluate(args.models, args.goal_weighting, args.suites, args.workers)
    policies = list(dict.fromkeys((o[0], o[1]) for o in out))
    for suite in args.suites:
        for m, w in policies:
            rows = [o for o in out if (o[0], o[1], o[2]) == (m, w, suite)]
            frac = np.array([o[4] for o in rows])
            label = m if m == "rule" else f"{m} ({w})"
            unsure = "" if m == "rule" else f"  unsure q {100 * np.mean([o[5] for o in rows]):.1f}%"
            print(f"{suite:8} {label:40}: through {np.round(100 * frac.mean(0)).astype(int)}%  "
                  f"runs > 50% through all {int((frac > 0.5).all(1).sum())} / {len(frac)}{unsure}")


if __name__ == "__main__":
    main()
