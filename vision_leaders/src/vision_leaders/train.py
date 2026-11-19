"""Train GoalNet by imitating the gate rule, then fine-tune it with DAgger on
the states the network itself visits.

    python -m vision_leaders.train --out models/goalnet.pt
    python -m vision_leaders.train --out models/goalnet_centre.pt --lone-post-aim centre
    python -m vision_leaders.train --out models/goalnet_aim.pt --aim models/aim_evolved.json

The model after each stage is also kept, as goalnet_bc.pt and
goalnet_dagger<i>.pt next to --out. The teacher is emergent_leaders' rule with
perfect sensing ("full"); --lone-post-aim centre makes agents that sense one
post of a gate aim at its centre rather than at the post; --aim uses the
tuned teacher of aiming.py (and its goal_gain and beta) instead.
--sense-range sets the teacher's gate_sense_range; the training courses stay
those of the default range. The other gate options (options.py) go to the
teacher and the motion model; with --loose-radius the network also predicts
whether the agent is near its target gate's centre (network.py). --memory
gives the network its previous output as input, for --sticky-targets; it
learns to copy it, so don't (see the README). --channels and --hidden size
the network. --save-data DIR keeps the training data (images and labels)
after each stage; --data DIR starts from it instead of collecting
behaviour-cloning runs, e.g. to train another network size on the same data
(with --dagger-iters 0, on that data alone).

1. Behaviour cloning: the teacher drives the swarm on random courses; every
   `every` steps each agent's image is stored with the teacher's label.
2. DAgger: the network drives; its images are labelled by the teacher, added
   to the data, and the network is retrained on all of it.

Training courses are random: three-gate slaloms (offset 0-1.5 m) and winding
courses. With --max-turn 45 (default) these have four gates, turns up to
45 deg and the largest spacing the course rules allow; with a larger
--max-turn, five gates 6-8 m apart (as evaluate.py's, which the course rules
don't allow at sharp turns). With --varied-courses, gates are 2.5-4 m wide
and 5-8 m apart: slaloms of 3-5 gates offset 0-2.5 m (35%) and winding
courses of 4-7 gates with turns up to --max-turn. Starts face the course (0 +/- 30 deg) in 70% of runs, random otherwise.
Explorers are off. Images without a teacher target are subsampled
(keep_no_gate) to balance the data.
"""
import argparse
import json
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

from .aiming import Aim
from .camera import CameraParams
from .network import GoalNet
from .options import add_gate_options, gate_options
from .simulation import VisionSimulation


def random_scenario(rng, params, max_turn=45.0, varied=False):
    """(gates, start state) for one training run."""
    r, ahead = ModelParams().gate_sense_range, params.gate_lookahead   # courses don't follow the teacher's range
    if varied:
        width, spacing = rng.uniform(2.5, 4.0), rng.uniform(5.0, 8.0)
        if rng.random() < 0.35:
            gates = slalom_course(int(rng.integers(3, 6)), spacing=spacing, offset=rng.uniform(0.0, 2.5), width=width)
        else:
            gates = winding_course(random_turns(int(rng.integers(4, 8)), math.radians(max_turn), rng), spacing,
                                   width=width)
    elif rng.random() < 0.5:
        offset = rng.uniform(0.0, 1.5)
        gates = slalom_course(3, spacing=max_slalom_spacing(r, ahead, offset=offset), offset=offset)
    elif max_turn <= 45.0:
        turns = random_turns(4, math.radians(max_turn), rng)

        def build(spacing):
            return winding_course(turns, spacing)
        try:
            spacing = max_spacing(build, r, ahead)
        except ValueError:
            spacing = 6.0
        gates = build(spacing)
    else:
        gates = winding_course(random_turns(5, math.radians(max_turn), rng), rng.uniform(6.0, 8.0))
    start = grid_swarm(25, rng, heading=0.0 if rng.random() < 0.7 else None)
    return gates, start


def rollout(seed, net=None, camera=CameraParams(), params=None, duration=300.0, every=20, keep_no_gate=0.3, aim=None,
            max_turn=45.0, varied=False):
    """Images and teacher labels from one run: (images uint8, previous outputs (n, 3) float32, lead bool,
    bearing float32, near bool)."""
    params = params or ModelParams(sigma_leader=0.7)
    rng = np.random.default_rng(seed)
    gates, start = random_scenario(rng, params, max_turn, varied)
    sim = VisionSimulation(params=params, state=start, gates=gates, explore=False, seed=seed, camera=camera,
                           net=net, check_course=False, aim=aim)
    images, prev, lead, bearing, near = [], [], [], [], []
    for step in range(round(duration / params.dt)):
        sim.step()
        if step % every == 0:
            keep = sim.teacher_lead | (rng.random(sim.state.n) < keep_no_gate)
            images.append(sim.images[keep])
            prev.append(sim.prev_input[keep])
            lead.append(sim.teacher_lead[keep])
            bearing.append(sim.teacher_bearing[keep])
            near.append(sim.teacher_near[keep])
    return (np.concatenate(images), np.concatenate(prev), np.concatenate(lead),
            np.concatenate(bearing).astype(np.float32), np.concatenate(near))


def _rollout_job(args):
    seed, net_state, net_config, camera, params, duration, aim, max_turn, varied = args
    torch.set_num_threads(1)
    net = None
    if net_state is not None:
        net = GoalNet(camera, **net_config)
        net.load_state_dict(net_state)
        net.eval()
    return rollout(seed, net, camera, params, duration=duration, aim=aim, max_turn=max_turn, varied=varied)


def collect(seeds, net=None, camera=CameraParams(), duration=300.0, workers=20, params=None, aim=None, max_turn=45.0,
            varied=False):
    state = None if net is None else {k: v.cpu() for k, v in net.state_dict().items()}
    params = params or ModelParams(sigma_leader=0.7)
    with Pool(workers) as pool:
        parts = pool.map(_rollout_job, [(s, state, getattr(net, "config", None), camera, params, duration, aim,
                                         max_turn, varied) for s in seeds])
    return tuple(np.concatenate(x) for x in zip(*parts))


DATA_FIELDS = ("images", "prev", "lead", "bearing", "near")


def save_data(directory, data, next_seed):
    """The training data as one .npy file per field, and the next free rollout seed."""
    directory = pathlib.Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, x in zip(DATA_FIELDS, data):
        np.save(directory / f"{name}.npy", x)
    (directory / "meta.json").write_text(json.dumps({"next_seed": next_seed}))


def load_data(directory):
    directory = pathlib.Path(directory)
    data = tuple(np.load(directory / f"{name}.npy") for name in DATA_FIELDS)
    return data, json.loads((directory / "meta.json").read_text())["next_seed"]


def new_net(camera, params, memory=False, channels=(16, 32, 32), hidden=64):
    """A GoalNet with the near output if the gate options need it."""
    return GoalNet(camera, memory=memory, near=params.gate_loose_radius > 0, channels=channels, hidden=hidden)


def fit(net, data, epochs=8, batch=512, lr=1e-3, bearing_weight=4.0, seed=0, log=print):
    """Train on (images, previous outputs, lead, bearing, near); 10% held out for the reported
    validation numbers."""
    images, prev, lead, bearing, near = (torch.from_numpy(x) for x in data)
    target = torch.stack([torch.sin(bearing), torch.cos(bearing)], dim=1)
    gen = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(lead), generator=gen)
    n_val = len(order) // 10
    val, train = order[:n_val], order[n_val:]
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()

    def loss_on(idx, out=None):
        logit, s, c, near_logit = net(images[idx], prev[idx]) if out is None else out
        pos = lead[idx]
        loss = bce(logit, pos.float())
        err = (torch.stack([s, c], 1)[pos] - target[idx][pos]).pow(2).sum(1)
        loss = loss + bearing_weight * (err.mean() if pos.any() else 0.0)
        if near_logit is not None:
            loss = loss + bce(near_logit, near[idx].float())
        return loss, logit, s, c, near_logit

    for epoch in range(epochs):
        net.train()
        for i in range(0, len(train), batch):
            idx = train[i:i + batch]
            opt.zero_grad()
            loss_on(idx)[0].backward()
            opt.step()
        train = train[torch.randperm(len(train), generator=gen)]
        net.eval()
        with torch.no_grad():   # in chunks: one pass over all of val would take tens of GB of activations
            parts = [net(images[val[i:i + 4096]], prev[val[i:i + 4096]]) for i in range(0, len(val), 4096)]
            out = [None if p[0] is None else torch.cat(p) for p in zip(*parts)]
            loss, logit, s, c, near_logit = loss_on(val, out)
            pos = lead[val]
            acc = ((logit > 0) == pos).float().mean().item()
            err = torch.atan2(s, c)[pos] - bearing[val][pos]
            err = torch.rad2deg(torch.abs((err + math.pi) % (2 * math.pi) - math.pi))
            near_acc = "" if near_logit is None else \
                f", near accuracy {100 * ((near_logit > 0) == near[val]).float().mean().item():.1f}%"
        log(f"  epoch {epoch + 1}: val loss {loss.item():.3f}, lead accuracy {100 * acc:.1f}%, "
            f"bearing error median {err.median().item():.1f} deg, 90th pct {err.quantile(0.9).item():.1f} deg"
            + near_acc)
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
    parser.add_argument("--lone-post-aim", choices=["post", "centre"], default="post",
                        help="teacher's aim when it senses one post of a gate")
    parser.add_argument("--aim", default=None, help="tuned teacher (JSON from evolve.py) instead of the rule")
    parser.add_argument("--max-turn", type=float, default=45.0, help="largest turn of the winding training courses [deg]")
    parser.add_argument("--varied-courses", action="store_true",
                        help="gate widths 2.5-4 m, spacings 5-8 m, slalom offsets up to 2.5 m, 3-7 gates")
    parser.add_argument("--memory", action="store_true", help="previous output as an input (learns to copy it)")
    parser.add_argument("--channels", type=int, nargs=3, default=(16, 32, 32), help="of the three convolutions")
    parser.add_argument("--hidden", type=int, default=64, help="units of the hidden layer")
    parser.add_argument("--save-data", default=None, help="directory to keep the training data in, after each stage")
    parser.add_argument("--data", default=None,
                        help="start from this saved data instead of collecting behaviour-cloning runs")
    add_gate_options(parser)
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)   # progress shows up in redirected logs

    torch.manual_seed(0)
    camera = CameraParams()
    aim = Aim.load(args.aim) if args.aim else None
    params = aim.model_params() if aim else ModelParams(sigma_leader=0.7, lone_post_aim=args.lone_post_aim)
    params = gate_options(params, args)
    t0 = time.time()
    if args.data:
        data, seed = load_data(args.data)
        print(f"loaded {len(data[2])} images from {args.data}, {100 * data[2].mean():.0f}% with a gate target")
    else:
        data = collect(range(args.bc_runs), camera=camera, duration=args.duration, workers=args.workers,
                       params=params, aim=aim, max_turn=args.max_turn, varied=args.varied_courses)
        seed = args.bc_runs
        print(f"behaviour cloning: {len(data[2])} images, {100 * data[2].mean():.0f}% with a gate target "
              f"({time.time() - t0:.0f} s)")
        if args.save_data:
            save_data(args.save_data, data, seed)
    out = pathlib.Path(args.out)
    net = fit(new_net(camera, params, args.memory, args.channels, args.hidden), data, epochs=args.epochs)
    net.save(out.with_stem(out.stem + "_bc"))
    for it in range(args.dagger_iters):
        t0 = time.time()
        new = collect(range(seed, seed + args.dagger_runs), net, camera, args.duration, args.workers, params, aim,
                      args.max_turn, args.varied_courses)
        seed += args.dagger_runs
        data = tuple(np.concatenate([a, b]) for a, b in zip(data, new))
        print(f"DAgger {it + 1}: +{len(new[2])} images, {len(data[2])} in total ({time.time() - t0:.0f} s)")
        if args.save_data:
            save_data(args.save_data, data, seed)
        net = fit(net, data, epochs=max(2, args.epochs // 2))
        net.save(out.with_stem(f"{out.stem}_dagger{it + 1}"))
    net.save(out)
    print(f"model -> {out}")


if __name__ == "__main__":
    main()
