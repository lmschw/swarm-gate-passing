"""Fine-tune GoalNet with reinforcement learning (PPO), rewarding agents for
gates passed and penalising gates missed.

    python -m vision_leaders.rl --init models/goalnet_r6_gateopts.pt --out models/goalnet_r6_rl.pt \\
        --lone-post-aim centre --sense-range 6 --sticky-targets --loose-radius 3.5 --post-avoid 0.3

The network drives the swarm, deciding every --decide-every steps (0.5 s by
default; see simulation.py) and stochastically: an agent leads with the
predicted probability, aims along the predicted bearing plus Gaussian noise
(standard deviation learned, from --bearing-std) and is near its gate with
the predicted near probability. Every agent shares the network.

Rewards, per agent and decision interval: +1 for each gate passed; -1
(--miss-penalty) for each gate missed, when the agent crosses the gate's line
in the passing direction outside the posts (up to MISS_MARGIN beyond them) or
passes a later gate first; at the end, -1 for each gate neither passed nor
missed. Optional finer terms: --progress-weight per gate spacing (the mean
distance between gate centres) the agent closes towards the centre of its
current gate (its first gate neither passed nor missed, switching only at the
end of an interval, so a pass doesn't jump); --centre-weight at each pass
times 1 - |offset from the centre| / half the width; -(--time-penalty) per
interval while the agent has gates left. --team-weight times the swarm's mean
reward is added to each agent's.
An episode ends when every agent has passed or missed every gate, or after
--duration.

The critic sees privileged state, not the image (critic_features): the next
two gates the agent hasn't passed or missed, in its body frame, the share of
the swarm through the next one, the agent's offset from the swarm's centroid,
its own last decision, and the time and gates left. Advantages are GAE
(--gamma per decision, --lam); the policy loss is PPO's clipped objective.
The first --critic-warmup iterations train only the critic.

Courses are train.py's (--max-turn, --varied-courses). Every --eval-every
iterations the network is evaluated deterministically on evaluate.py's
winding suite with the same decision interval; the model after each
evaluation is kept as <out>_it<i>.pt and the best (agents through every gate)
as --out.
"""
import argparse
import math
import pathlib
import sys
import time
from dataclasses import dataclass
from multiprocessing import Pool

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from emergent_leaders import ModelParams

from .camera import render
from .evaluate import evaluate
from .network import GoalNet
from .options import add_gate_options, gate_options, gate_overrides
from .simulation import VisionSimulation
from .train import random_scenario

MISS_MARGIN = 3.0   # [m] beyond a post, crossing a gate's line counts as missing it
N_FEATURES = 18


def _wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


class Policy(nn.Module):
    """GoalNet as a stochastic policy: lead ~ Bernoulli(q), bearing ~ Normal(predicted, std),
    near ~ Bernoulli(near probability); bearing and near only count for leaders."""

    def __init__(self, net, bearing_std=0.1):
        super().__init__()
        self.net = net
        self.log_std = nn.Parameter(torch.tensor(math.log(bearing_std)))

    def outputs(self, images):
        logit, s, c, near_logit = self.net(images)
        return logit, torch.atan2(s, c), near_logit

    def log_prob(self, images, lead, bearing, near):
        """Log-probabilities of the actions and the entropies of the distributions, each (b,)."""
        logit, mu, near_logit = self.outputs(images)
        lead = lead.float()
        lp = -F.binary_cross_entropy_with_logits(logit, lead, reduction="none")
        z = _wrap(bearing - mu) / self.log_std.exp()
        lp_bearing = -0.5 * z**2 - self.log_std - 0.5 * math.log(2 * math.pi)
        lp_near = -F.binary_cross_entropy_with_logits(near_logit, near.float(), reduction="none")
        q, r = torch.sigmoid(logit), torch.sigmoid(near_logit)
        entropy = (F.binary_cross_entropy_with_logits(logit, q, reduction="none")
                   + q * (self.log_std + 0.5 * math.log(2 * math.pi * math.e)
                          + F.binary_cross_entropy_with_logits(near_logit, r, reduction="none")))
        return lp + lead * (lp_bearing + lp_near), entropy


def critic_features(sim, done, duration):
    """(n, N_FEATURES) privileged state for the critic; done: (n_gates, n) gates each agent passed or missed."""
    s = sim.state
    n_gates = len(sim.gates)
    centres = np.array([g.centre for g in sim.gates])
    headings = np.array([g.heading for g in sim.gates])
    first = np.where((~done).any(axis=0), (~done).argmax(axis=0), n_gates)
    cos, sin = np.cos(s.heading), np.sin(s.heading)
    feats = []
    for ahead in (0, 1):
        k = first + ahead
        valid = (k < n_gates).astype(float)
        kk = np.minimum(k, n_gates - 1)
        d = centres[kk] - s.pos
        rel = headings[kk] - s.heading
        feats += [valid * (cos * d[:, 0] + sin * d[:, 1]) / 10, valid * (-sin * d[:, 0] + cos * d[:, 1]) / 10,
                  valid * np.cos(rel), valid * np.sin(rel), valid]
    kk = np.minimum(first, n_gates - 1)
    through = np.isfinite(sim.gate_pass_time).mean(axis=1)[kk]
    d = s.pos.mean(axis=0) - s.pos
    held = sim._held
    lead = (held[0] >= 0).astype(float) if held is not None else np.zeros(s.n)
    near = (sim.net_near >= 0.5).astype(float) if hasattr(sim, "net_near") else np.zeros(s.n)
    feats += [through, (cos * d[:, 0] + sin * d[:, 1]) / 5, (-sin * d[:, 0] + cos * d[:, 1]) / 5, lead, near,
              np.full(s.n, 1 - sim.step_count * sim.params.dt / duration), (n_gates - first) / n_gates,
              np.full(s.n, s.is_leader.mean())]
    return np.column_stack(feats).astype(np.float32)


class PolicySimulation(VisionSimulation):
    """The policy drives (no teacher), sampling its actions at each decision, and every decision
    is recorded with the critic's features; missed gates are tracked per agent."""

    def __init__(self, *, gates, state, policy, duration, sample=True, action_seed=0, decide_every=10, **kwargs):
        self.policy, self.duration, self.sample = policy, duration, sample
        self.action_rng = np.random.default_rng(action_seed)
        self.records = []   # per decision: (images, lead, bearing, near, log prob, critic features)
        self.missed = np.zeros((len(gates), state.n), dtype=bool)   # the constructor makes the first decision
        self.pass_offset = np.full((len(gates), state.n), np.nan)   # |offset from the centre| / half width, at passing
        super().__init__(gates=gates, state=state, net=policy.net, decide_every=decide_every, **kwargs)

    def decide(self):
        s = self.state
        done = np.isfinite(self.gate_pass_time) | self.missed
        feats = critic_features(self, done, self.duration)   # the state the decision is made in
        self.images, self.post_visibility = render(s.pos, s.heading, self.gates, self.params.post_radius, self.camera)
        images = torch.from_numpy(self.images)
        with torch.no_grad():
            logit, mu, near_logit = self.policy.outputs(images)
            prob, mu, near_prob = torch.sigmoid(logit).numpy(), mu.numpy(), torch.sigmoid(near_logit).numpy()
            if self.sample:
                lead = self.action_rng.random(s.n) < prob
                bearing = _wrap(mu + self.policy.log_std.exp().item() * self.action_rng.standard_normal(s.n))
                near = self.action_rng.random(s.n) < near_prob
            else:
                lead, bearing, near = prob >= 0.5, mu, near_prob >= 0.5
            log_prob, _ = self.policy.log_prob(images, torch.from_numpy(lead), torch.from_numpy(bearing).float(),
                                               torch.from_numpy(near))
        self.net_prob, self.net_bearing, self.net_near = prob, bearing, near.astype(float)
        self.records.append((self.images, lead, bearing.astype(np.float32), near, log_prob.numpy(), feats))
        direction = s.heading + bearing
        target = np.where(lead[:, None], s.pos + np.column_stack([np.cos(direction), np.sin(direction)]), np.nan)
        not_passed = np.isnan(self.gate_pass_time)
        next_gate = np.where(not_passed.any(axis=0), not_passed.argmax(axis=0), len(self.gates) - 1)
        return np.where(lead, next_gate, -1), np.where(lead, 2, 0), target

    def step(self):
        prev = self.state.pos.copy()
        before = np.isfinite(self.gate_pass_time)
        out = super().step()
        passed = np.isfinite(self.gate_pass_time)
        self.missed |= missed_gates(prev, self.state.pos, self.gates, passed)
        for g, i in zip(*np.nonzero(passed & ~before)):
            gate = self.gates[g]
            self.pass_offset[g, i] = abs((self.state.pos[i] - gate.centre) @ gate.tangent) / (gate.width / 2)
        return out


def missed_gates(prev_pos, pos, gates, passed):
    """(n_gates, n) gates missed this step or before: not passed, and either crossed in the passing
    direction outside the posts (up to MISS_MARGIN beyond them) this step, or a later gate is passed."""
    missed = np.zeros_like(passed)
    for g, gate in enumerate(gates):
        c = np.asarray(gate.centre, dtype=float)
        lateral = np.abs((pos - c) @ gate.tangent)
        missed[g] = ((prev_pos - c) @ gate.normal < 0) & ((pos - c) @ gate.normal >= 0) & \
            (lateral > gate.width / 2) & (lateral <= gate.width / 2 + MISS_MARGIN)
    later = np.flip(np.logical_or.accumulate(np.flip(passed, axis=0), axis=0), axis=0)   # this gate or a later one
    missed[:-1] |= later[1:]
    return missed & ~passed


@dataclass(frozen=True)
class Reward:
    """Reward weights; see the module docstring."""
    miss_penalty: float = 1.0
    team_weight: float = 1.0
    progress: float = 0.0   # per gate spacing closed towards the centre of the agent's current gate
    centre: float = 0.0     # at each pass, times 1 - |offset from the centre| / half width
    time: float = 0.0       # per decision interval while the agent has gates left


class RewardTracker:
    """Per-agent rewards of one episode, one decision interval at a time."""

    def __init__(self, gates, pos, weights):
        self.w = weights
        self.centres = np.array([g.centre for g in gates], dtype=float)
        steps = np.linalg.norm(np.diff(self.centres, axis=0), axis=1)
        self.spacing = steps.mean() if len(steps) else 7.0
        self.counted = np.zeros((len(gates), len(pos)), dtype=bool)   # passes and misses already rewarded
        self._aim(pos)

    def _aim(self, pos):
        """The current gate of each agent (its first not yet passed or missed) and the distance to its centre."""
        open_ = ~self.counted
        self.target = np.where(open_.any(axis=0), open_.argmax(axis=0), -1)
        self.dist = np.linalg.norm(self.centres[self.target] - pos, axis=1)

    def interval(self, passed, missed, pass_offset, pos):
        """(n,) own rewards (before the team share) for the interval just ended."""
        new_pass = passed & ~self.counted & ~missed
        new_miss = missed & ~self.counted
        r = new_pass.sum(axis=0) - self.w.miss_penalty * new_miss.sum(axis=0)
        if self.w.centre:
            r = r + self.w.centre * np.where(new_pass, np.clip(1 - pass_offset, 0, 1), 0).sum(axis=0)
        active = self.target >= 0
        if self.w.progress:   # towards the gate that was current during the interval: no jump at a pass
            closer = self.dist - np.linalg.norm(self.centres[self.target] - pos, axis=1)
            r = r + self.w.progress * np.where(active, closer, 0) / self.spacing
        r = r - self.w.time * active
        self.counted |= new_pass | new_miss
        self._aim(pos)
        return r

    def final(self):
        """(n,) penalty for the gates neither passed nor missed."""
        return -self.w.miss_penalty * (~self.counted).sum(axis=0)


def run_episode(policy, params, gates, start, duration, seed, decide_every, weights):
    """One stochastic episode. Returns (images, lead, bearing, near, log prob, features, rewards) with a
    leading decision axis (T, n, ...), and the share of agents through every gate."""
    sim = PolicySimulation(params=params, state=start, gates=gates, explore=False, seed=seed, check_course=False,
                           policy=policy, duration=duration, action_seed=seed, decide_every=decide_every)
    tracker = RewardTracker(gates, sim.state.pos, weights)
    rewards = []

    def collect():
        return tracker.interval(np.isfinite(sim.gate_pass_time), sim.missed, sim.pass_offset, sim.state.pos)

    for i in range(round(duration / params.dt)):
        sim.step()
        if sim.step_count % decide_every == 0:
            if len(sim.records) > 1:
                rewards.append(collect())
            if (np.isfinite(sim.gate_pass_time) | sim.missed).all():
                break
    rewards.append(collect() + tracker.final())
    rewards = np.array(rewards[:len(sim.records)], dtype=np.float32)
    rewards = rewards + weights.team_weight * rewards.mean(axis=1, keepdims=True)
    fields = [np.stack(x) for x in zip(*sim.records)]
    return fields + [rewards], np.isfinite(sim.gate_pass_time).all(axis=0).mean()


def _episode_job(args):
    seed, net_config, state, log_std, camera, params, duration, decide_every, weights, max_turn, varied = args
    torch.set_num_threads(1)
    policy = Policy(GoalNet(camera, **net_config))
    policy.net.load_state_dict(state)
    policy.log_std.data.fill_(log_std)
    policy.eval()
    gates, start = random_scenario(np.random.default_rng(seed), params, max_turn, varied)
    return run_episode(policy, params, gates, start, duration, seed, decide_every, weights)


def gae(values, rewards, gamma, lam):
    """values, rewards: (T, n), the episode ending after the last decision. Returns (advantages, returns)."""
    adv = np.zeros_like(rewards)
    last = np.zeros(rewards.shape[1], dtype=np.float32)
    for t in range(len(rewards) - 1, -1, -1):
        next_value = values[t + 1] if t + 1 < len(rewards) else 0.0
        delta = rewards[t] + gamma * next_value - values[t]
        last = delta + gamma * lam * last
        adv[t] = last
    return adv, adv + values


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--init", required=True, help="GoalNet to start from")
    parser.add_argument("--out", required=True)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--episodes", type=int, default=20, help="per iteration")
    parser.add_argument("--duration", type=float, default=600.0, help="[s] per episode at most")
    parser.add_argument("--decide-every", type=int, default=10)
    parser.add_argument("--miss-penalty", type=float, default=1.0)
    parser.add_argument("--team-weight", type=float, default=1.0)
    parser.add_argument("--progress-weight", type=float, default=0.0,
                        help="reward per gate spacing an agent closes towards its current gate's centre")
    parser.add_argument("--centre-weight", type=float, default=0.0,
                        help="bonus at a pass through the centre, falling linearly to 0 at a post")
    parser.add_argument("--time-penalty", type=float, default=0.0,
                        help="cost per decision interval while an agent has gates left")
    parser.add_argument("--bearing-std", type=float, default=0.1, help="initial exploration noise [rad]")
    parser.add_argument("--gamma", type=float, default=0.995, help="discount per decision")
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--target-kl", type=float, default=0.02,
                        help="stop an update once the KL over the last 20 minibatches exceeds this")
    parser.add_argument("--critic-warmup", type=int, default=2, help="iterations that train only the critic")
    parser.add_argument("--eval-every", type=int, default=3)
    parser.add_argument("--lone-post-aim", choices=["post", "centre"], default="post")
    parser.add_argument("--max-turn", type=float, default=90.0)
    parser.add_argument("--varied-courses", action="store_true")
    parser.add_argument("--workers", type=int, default=20)
    add_gate_options(parser)
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)

    torch.manual_seed(0)
    params = gate_options(ModelParams(sigma_leader=0.7, lone_post_aim=args.lone_post_aim), args)
    net = GoalNet.load(args.init)
    if not net.near and params.gate_loose_radius:
        raise ValueError("--loose-radius needs a network with a near output")
    if net.memory:
        raise ValueError("networks with memory are not supported")
    policy = Policy(net, args.bearing_std)
    critic = nn.Sequential(nn.Linear(N_FEATURES, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 1))
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=1e-3)
    out = pathlib.Path(args.out)
    best = -1.0
    overrides = gate_overrides(args)
    weights = Reward(args.miss_penalty, args.team_weight, args.progress_weight, args.centre_weight,
                     args.time_penalty)

    def evaluate_now(it):
        path = out.with_stem(f"{out.stem}_it{it}")
        policy.net.save(path)
        rows = evaluate([str(path)], suites=("winding",), workers=args.workers, rules=(), overrides=overrides,
                        decide_every=args.decide_every)
        every = np.mean([r[6] for r in rows])
        print(f"  eval winding: through {np.round(100 * np.mean([r[4] for r in rows], axis=0)).astype(int)}%, "
              f"agents through every gate {100 * every:.1f}%, all 25 in {sum(r[6] == 1 for r in rows)} / {len(rows)}")
        return path, every

    path, every = evaluate_now(0)
    best = every
    policy.net.save(out)
    seed = 1_000_000
    for it in range(1, args.iterations + 1):
        t0 = time.time()
        state = {k: v.detach().clone() for k, v in policy.net.state_dict().items()}
        jobs = [(seed + e, policy.net.config, state, policy.log_std.item(), net.cam, params, args.duration,
                 args.decide_every, weights, args.max_turn, args.varied_courses)
                for e in range(args.episodes)]
        seed += args.episodes
        with Pool(args.workers) as pool:
            episodes = pool.map(_episode_job, jobs)
        t_collect = time.time() - t0

        # advantages, per episode, with the critic before this iteration's update
        data, adv, ret = [], [], []
        for fields, _ in episodes:
            images, lead, bearing, near, log_prob, feats, rewards = fields
            with torch.no_grad():
                values = critic(torch.from_numpy(feats)).squeeze(-1).numpy()
            a, r = gae(values, rewards, args.gamma, args.lam)
            data.append((images, lead, bearing, near, log_prob, feats))
            adv.append(a)
            ret.append(r)
        flat = [torch.from_numpy(np.concatenate([d[k].reshape(-1, *d[k].shape[2:]) for d in data])) for k in range(6)]
        images, lead, bearing, near, old_lp, feats = flat
        adv = torch.from_numpy(np.concatenate([a.reshape(-1) for a in adv]))
        ret = torch.from_numpy(np.concatenate([r.reshape(-1) for r in ret]))
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        n = len(adv)

        critic_loss = []
        for _ in range(10 if it <= args.critic_warmup else 4):
            for idx in torch.randperm(n).split(4096):
                loss = (critic(feats[idx]).squeeze(-1) - ret[idx]).pow(2).mean()
                critic_opt.zero_grad()
                loss.backward()
                critic_opt.step()
                critic_loss.append(loss.item())

        kl, clipped, ent = [], [], []
        if it > args.critic_warmup:
            policy.train()
            steps = 0
            for _ in range(args.ppo_epochs):
                epoch_kl = []
                for idx in torch.randperm(n).split(args.batch):
                    if epoch_kl and np.mean(epoch_kl[-20:]) > args.target_kl:
                        break   # stop mid-epoch: an epoch is hundreds of steps
                    steps += 1
                    lp, entropy = policy.log_prob(images[idx], lead[idx], bearing[idx], near[idx])
                    ratio = (lp - old_lp[idx]).exp()
                    loss = -torch.min(ratio * adv[idx], ratio.clamp(1 - args.clip, 1 + args.clip) * adv[idx]).mean()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    with torch.no_grad():
                        epoch_kl.append((old_lp[idx] - lp).mean().item())
                        clipped.append(((ratio - 1).abs() > args.clip).float().mean().item())
                        ent.append(entropy.mean().item())
                kl.append(np.mean(epoch_kl[-20:]))
                if kl[-1] > args.target_kl:
                    break
            policy.eval()

        through = np.mean([e[1] for e in episodes])
        rewards = np.mean([e[0][6].sum(axis=0).mean() for e in episodes])
        print(f"iteration {it}: {n} decisions, return {rewards:.2f}, agents through every gate {100 * through:.1f}% "
              f"(sampled), critic loss {np.mean(critic_loss[-50:]):.3f}"
              + (f", {steps} steps, KL {kl[-1]:.4f}, clipped {100 * np.mean(clipped):.1f}%, entropy {np.mean(ent):.3f}, "
                 f"bearing std {math.degrees(policy.log_std.exp().item()):.1f} deg" if kl else ", critic only")
              + f" ({t_collect:.0f} s collecting, {time.time() - t0 - t_collect:.0f} s updating)")
        if it > args.critic_warmup and it % args.eval_every == 0:
            path, every = evaluate_now(it)
            if every > best:
                best = every
                policy.net.save(out)
    print(f"best: agents through every gate {100 * best:.1f}% -> {out}")


if __name__ == "__main__":
    main()
