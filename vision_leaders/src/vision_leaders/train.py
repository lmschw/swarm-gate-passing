"""Train GoalNet by imitating the camera-limited gate rule, then fine-tune it
with DAgger on the states the network itself visits.

    python -m vision_leaders.train --out models/goalnet.pt

The model after each stage is also kept, as goalnet_bc.pt and
goalnet_dagger<i>.pt next to --out.

1. Behaviour cloning: the teacher drives the swarm on random courses; every
   `every` steps each agent's image is stored with the teacher's label.
2. DAgger: the network drives; its images are labelled by the teacher, added
   to the data, and the network is retrained on all of it.

Training courses are random: three-gate slaloms (offset 0-1.5 m) and four-gate
winding courses (turns up to 45 deg), at the largest spacing the course rules
allow. Starts face the course (0 +/- 30 deg) in 70% of runs, random otherwise.
Explorers are off. Images without a teacher target are subsampled
(keep_no_gate) to balance the data.
"""
import argparse
import math
import pathlib
import sys
import time
from multiprocessing import Pool

import numpy as np
import torch
from torch import nn

from emergent_leaders import ModelParams, grid_swarm, random_turns, slalom_course, winding_course
from emergent_leaders.gates import max_slalom_spacing, max_spacing

from .camera import CameraParams
from .network import GoalNet
from .simulation import VisionSimulation


def random_scenario(rng, params):
    """(gates, start state) for one training run."""
    r, ahead = params.gate_sense_range, params.gate_lookahead
    if rng.random() < 0.5:
        offset = rng.uniform(0.0, 1.5)
        gates = slalom_course(3, spacing=max_slalom_spacing(r, ahead, offset=offset), offset=offset)
    else:
        turns = random_turns(4, math.radians(45), rng)

        def build(spacing):
            return winding_course(turns, spacing)
        try:
            spacing = max_spacing(build, r, ahead)
        except ValueError:
            spacing = 6.0
        gates = build(spacing)
    start = grid_swarm(25, rng, heading=0.0 if rng.random() < 0.7 else None)
    return gates, start


def rollout(seed, net=None, camera=CameraParams(), params=None, duration=300.0, every=20, keep_no_gate=0.3):
    """Images and teacher labels from one run: (images uint8, lead bool, bearing float32)."""
    params = params or ModelParams(sigma_leader=0.7)
    rng = np.random.default_rng(seed)
    gates, start = random_scenario(rng, params)
    sim = VisionSimulation(params=params, state=start, gates=gates, explore=False, seed=seed, camera=camera,
                           net=net, check_course=False)
    images, lead, bearing = [], [], []
    for step in range(round(duration / params.dt)):
        sim.step()
        if step % every == 0:
            keep = sim.teacher_lead | (rng.random(sim.state.n) < keep_no_gate)
            images.append(sim.images[keep])
            lead.append(sim.teacher_lead[keep])
            bearing.append(sim.teacher_bearing[keep])
    return np.concatenate(images), np.concatenate(lead), np.concatenate(bearing).astype(np.float32)


def _rollout_job(args):
    seed, net_state, camera, duration = args
    torch.set_num_threads(1)
    net = None
    if net_state is not None:
        net = GoalNet(camera)
        net.load_state_dict(net_state)
        net.eval()
    return rollout(seed, net, camera, duration=duration)


def collect(seeds, net=None, camera=CameraParams(), duration=300.0, workers=20):
    state = None if net is None else {k: v.cpu() for k, v in net.state_dict().items()}
    with Pool(workers) as pool:
        parts = pool.map(_rollout_job, [(s, state, camera, duration) for s in seeds])
    return tuple(np.concatenate(x) for x in zip(*parts))


def fit(net, data, epochs=8, batch=512, lr=1e-3, bearing_weight=4.0, seed=0, log=print):
    """Train on (images, lead, bearing); 10% held out for the reported validation numbers."""
    images, lead, bearing = (torch.from_numpy(x) for x in data)
    target = torch.stack([torch.sin(bearing), torch.cos(bearing)], dim=1)
    gen = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(lead), generator=gen)
    n_val = len(order) // 10
    val, train = order[:n_val], order[n_val:]
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()

    def loss_on(idx):
        logit, s, c = net(images[idx])
        pos = lead[idx]
        l_lead = bce(logit, pos.float())
        err = (torch.stack([s, c], 1)[pos] - target[idx][pos]).pow(2).sum(1)
        return l_lead + bearing_weight * (err.mean() if pos.any() else 0.0), logit, s, c

    for epoch in range(epochs):
        net.train()
        for i in range(0, len(train), batch):
            idx = train[i:i + batch]
            opt.zero_grad()
            loss_on(idx)[0].backward()
            opt.step()
        train = train[torch.randperm(len(train), generator=gen)]
        net.eval()
        with torch.no_grad():
            loss, logit, s, c = loss_on(val)
            pos = lead[val]
            acc = ((logit > 0) == pos).float().mean().item()
            err = torch.atan2(s, c)[pos] - bearing[val][pos]
            err = torch.rad2deg(torch.abs((err + math.pi) % (2 * math.pi) - math.pi))
        log(f"  epoch {epoch + 1}: val loss {loss.item():.3f}, lead accuracy {100 * acc:.1f}%, "
            f"bearing error median {err.median().item():.1f} deg, 90th pct {err.quantile(0.9).item():.1f} deg")
    return net


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="model file to write")
    parser.add_argument("--bc-runs", type=int, default=40, help="teacher-driven runs for behaviour cloning")
    parser.add_argument("--dagger-iters", type=int, default=3)
    parser.add_argument("--dagger-runs", type=int, default=20, help="network-driven runs per DAgger iteration")
    parser.add_argument("--duration", type=float, default=300.0, help="simulated time per run [s]")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)   # progress shows up in redirected logs

    torch.manual_seed(0)
    camera = CameraParams()
    t0 = time.time()
    data = collect(range(args.bc_runs), camera=camera, duration=args.duration, workers=args.workers)
    print(f"behaviour cloning: {len(data[1])} images, {100 * data[1].mean():.0f}% with a gate target "
          f"({time.time() - t0:.0f} s)")
    out = pathlib.Path(args.out)
    net = fit(GoalNet(camera), data, epochs=args.epochs)
    net.save(out.with_stem(out.stem + "_bc"))
    seed = args.bc_runs
    for it in range(args.dagger_iters):
        t0 = time.time()
        new = collect(range(seed, seed + args.dagger_runs), net, camera, args.duration, args.workers)
        seed += args.dagger_runs
        data = tuple(np.concatenate([a, b]) for a, b in zip(data, new))
        print(f"DAgger {it + 1}: +{len(new[1])} images, {len(data[1])} in total ({time.time() - t0:.0f} s)")
        net = fit(net, data, epochs=max(2, args.epochs // 2))
        net.save(out.with_stem(f"{out.stem}_dagger{it + 1}"))
    net.save(out)
    print(f"model -> {out}")


if __name__ == "__main__":
    main()
