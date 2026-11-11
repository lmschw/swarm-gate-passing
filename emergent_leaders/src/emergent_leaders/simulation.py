"""Swarm simulation: motion model + leadership emergence + goal directions.

There are two kinds of leader:

- explorers: followers promoted with probability P(stability) that walk legs
  from the goal model (a random walk by default);
- gate leaders: any agent that senses a gate post in front of it (see
  gates.py). Its goal heading points at the gate centre if it senses both
  posts, at the sensed post otherwise, and is updated every step. It stays a
  leader until it has no gate target in front, e.g. once it has passed
  through. A gate target overrides an explorer's current leg.

One time step:
  1. compute the virtual force and speed commands from the current state;
  2. integrate positions and headings, push agents out of gate posts;
     record gate passages;
  3. on the new configuration, update each agent's stability and quadrant
     coverage;
  4. update gate leaders from what each agent now sees; if
     pause_exploration_at_gates and a gate target exists (or existed less
     than exploration_hold_off ago), explorers step down and steps 5-6 only
     do the bookkeeping;
  5. explorers accumulate travelled distance; one that completes a leg either
     gets a new leg from the goal model or, after legs_per_tenure legs, is
     demoted to follower;
  6. if exploring, eligible followers are promoted to explorers with
     probability P(stability) and get their first leg.
"""
from dataclasses import dataclass, field
import math

import numpy as np

from . import gates as gates_mod
from . import leadership, motion
from .goals import GoalModel, RandomWalkGoal
from .params import ModelParams


@dataclass
class SwarmState:
    pos: np.ndarray            # (n, 2)
    heading: np.ndarray        # (n,)
    is_leader: np.ndarray      # (n,) bool
    goal_heading: np.ndarray   # (n,) psi_i, meaningful for leaders only
    leg_length: np.ndarray     # (n,)
    leg_travelled: np.ndarray  # (n,)
    legs_done: np.ndarray      # (n,) int, completed legs in the current tenure
    stability: np.ndarray      # (n,)
    full_coverage: np.ndarray  # (n,) bool
    gate_target: np.ndarray    # (n,) int, gate a gate leader heads for, -1 otherwise
    posts_seen: np.ndarray     # (n,) int, posts of gate_target sensed (1: aims at it, 2: aims at the centre)

    @property
    def n(self):
        return len(self.heading)

    @classmethod
    def from_poses(cls, pos, heading):
        n = len(heading)
        return cls(pos=np.asarray(pos, dtype=float), heading=np.asarray(heading, dtype=float),
                   is_leader=np.zeros(n, dtype=bool), goal_heading=np.zeros(n),
                   leg_length=np.zeros(n), leg_travelled=np.zeros(n),
                   legs_done=np.zeros(n, dtype=int), stability=np.zeros(n),
                   full_coverage=np.zeros(n, dtype=bool), gate_target=np.full(n, -1),
                   posts_seen=np.zeros(n, dtype=int))


def grid_swarm(n_agents, rng, spacing=0.8, jitter=0.25, heading=None, heading_std=math.radians(30)):
    """Agents on a jittered square lattice. Headings are uniformly random (the
    reference's initial condition) or, if heading is given, normally
    distributed around it."""
    side = math.ceil(math.sqrt(n_agents))
    xx, yy = np.meshgrid(np.arange(side) * spacing, np.arange(side) * spacing)
    pos = np.column_stack([xx.ravel(), yy.ravel()])[:n_agents]
    pos = pos + rng.uniform(-jitter, jitter, pos.shape) * spacing
    if heading is None:
        headings = rng.uniform(-np.pi, np.pi, n_agents)
    else:
        headings = rng.normal(heading, heading_std, n_agents)
    return SwarmState.from_poses(pos, headings)


@dataclass
class History:
    dt: float
    gates: tuple = ()
    post_radius: float = 0.0
    gate_pass_time: np.ndarray = None   # (n_gates, n) time each agent passed each gate, nan if not
    gate_first_target_time: np.ndarray = None  # (n_gates,) time any agent first aimed at each gate, nan if never
    step: list = field(default_factory=list)
    pos: list = field(default_factory=list)
    heading: list = field(default_factory=list)
    is_leader: list = field(default_factory=list)
    goal_heading: list = field(default_factory=list)
    gate_target: list = field(default_factory=list)
    posts_seen: list = field(default_factory=list)
    stability: list = field(default_factory=list)
    promotions: list = field(default_factory=list)   # promotions during each recorded interval
    demotions: list = field(default_factory=list)

    def record(self, step, state, promotions, demotions):
        self.step.append(step)
        self.pos.append(state.pos.copy())
        self.heading.append(state.heading.copy())
        self.is_leader.append(state.is_leader.copy())
        self.goal_heading.append(state.goal_heading.copy())
        self.gate_target.append(state.gate_target.copy())
        self.posts_seen.append(state.posts_seen.copy())
        self.stability.append(state.stability.copy())
        self.promotions.append(promotions)
        self.demotions.append(demotions)

    def as_arrays(self):
        """Stack every per-step list into an array, keyed by field name."""
        out = {k: np.asarray(v) for k, v in vars(self).items() if isinstance(v, list)}
        out["time"] = out["step"] * self.dt
        out["centroid"] = out["pos"].mean(axis=1)
        out["n_leaders"] = out["is_leader"].sum(axis=1)
        out["n_gate_leaders"] = (out["gate_target"] >= 0).sum(axis=1)
        return out


class Simulation:
    def __init__(self, n_agents=25, params=None, goal=None, leader_type="free", seed=None, state=None,
                 gates=(), explore=True, check_course=True):
        if leader_type not in leadership.LEADER_TYPES:
            raise ValueError(f"leader_type must be one of {leadership.LEADER_TYPES}, got {leader_type!r}")
        self.params = params or ModelParams()
        self.goal: GoalModel = goal or RandomWalkGoal("levy")
        self.leader_type = leader_type
        self.explore = explore
        self.gates = tuple(gates)
        if check_course:
            gates_mod.check_course(self.gates, self.params.gate_sense_range, self.params.gate_lookahead)
        self.rng = np.random.default_rng(seed)
        self.state = state if state is not None else grid_swarm(n_agents, self.rng)
        self.gate_pass_time = np.full((len(self.gates), self.state.n), np.nan)
        self.gate_first_target_time = np.full(len(self.gates), np.nan)
        self.step_count = 0
        self.total_demotions = 0
        self._last_gate_target_time = -math.inf
        self._update_leadership_measures(self._geometry())
        self.total_promotions = self._update_gate_leaders()[0]

    def _geometry(self):
        return motion.neighbour_geometry(self.state.pos, self.state.heading, self.params.sense_range)

    def _update_leadership_measures(self, geometry):
        dist, bearing = geometry
        self.state.stability = leadership.stability(dist, self.state.heading, self.params)
        self.state.full_coverage = leadership.full_quadrant_coverage(dist, bearing)

    def _start_legs(self, idx):
        s = self.state
        s.leg_length[idx], s.goal_heading[idx] = self.goal.assign(s, idx, self.rng)
        s.leg_travelled[idx] = 0.0

    def _record_passages(self, prev_pos):
        t = self.step_count * self.params.dt
        for g, gate in enumerate(self.gates):
            first = gates_mod.crossed(prev_pos, self.state.pos, gate) & np.isnan(self.gate_pass_time[g])
            self.gate_pass_time[g, first] = t

    def _update_gate_leaders(self):
        """Returns (n_promoted, n_demoted)."""
        if not self.gates:
            return 0, 0
        s = self.state
        had_target = s.gate_target >= 0
        s.gate_target, s.posts_seen, target = gates_mod.gate_targets(s.pos, s.heading, self.gates, self.params)
        sees = s.gate_target >= 0
        lost = had_target & ~sees
        promoted = sees & ~s.is_leader
        s.is_leader[lost] = False
        s.is_leader[sees] = True
        to_target = target[sees] - s.pos[sees]
        s.goal_heading[sees] = np.arctan2(to_target[:, 1], to_target[:, 0])
        s.legs_done[sees | lost] = 0
        s.leg_travelled[sees | lost] = 0.0
        first = np.isnan(self.gate_first_target_time)
        first[first] = np.isin(np.flatnonzero(first), s.gate_target)
        self.gate_first_target_time[first] = self.step_count * self.params.dt
        return int(promoted.sum()), int(lost.sum())

    def step(self):
        """Advance one time step; returns (n_promoted, n_demoted)."""
        s, p = self.state, self.params

        f_x, f_y = motion.virtual_force(s.pos, s.heading, s.is_leader, s.goal_heading, p, self._geometry())
        u, w = motion.velocity_commands(f_x, f_y, p)
        prev_pos = s.pos
        s.pos, s.heading = motion.integrate(s.pos, s.heading, u, w, p.dt)
        if self.gates:
            s.pos = gates_mod.push_out_of_posts(s.pos, self.gates, p.post_radius)
        self.step_count += 1
        self._record_passages(prev_pos)
        self._update_leadership_measures(self._geometry())
        n_prom, n_dem = self._update_gate_leaders()

        # Pausing exploration near gates
        paused = self.exploration_paused()
        if paused:
            step_down = s.is_leader & (s.gate_target < 0)
            s.is_leader[step_down] = False
            s.legs_done[step_down] = 0
            s.leg_travelled[step_down] = 0.0
            n_dem += int(step_down.sum())

        # Explorers: leg bookkeeping and demotion
        explorer = s.is_leader & (s.gate_target < 0)
        s.leg_travelled[explorer] += u[explorer] * p.dt
        finished = explorer & (s.leg_travelled >= s.leg_length)
        s.legs_done[finished] += 1
        demote = finished & (s.legs_done >= p.legs_per_tenure)
        s.is_leader[demote] = False
        s.legs_done[demote] = 0
        s.leg_travelled[demote] = 0.0
        self._start_legs(np.flatnonzero(finished & ~demote))

        # Promotion to explorer
        if self.explore and not paused:
            candidates = ~s.is_leader & leadership.promotion_eligible(s.full_coverage, self.leader_type)
            prob = leadership.promotion_probability(s.stability, p)
            promote = candidates & (self.rng.random(s.n) < prob)
            s.is_leader[promote] = True
            s.legs_done[promote] = 0
            self._start_legs(np.flatnonzero(promote))
            n_prom += int(promote.sum())

        n_dem += int(demote.sum())
        self.total_promotions += n_prom
        self.total_demotions += n_dem
        return n_prom, n_dem

    def exploration_paused(self):
        """Whether exploration is paused for gates now (see ModelParams)."""
        if not self.params.pause_exploration_at_gates:
            return False
        t = self.step_count * self.params.dt
        if (self.state.gate_target >= 0).any():
            self._last_gate_target_time = t
        return t - self._last_gate_target_time <= self.params.exploration_hold_off

    def next_gate_in_view(self):
        """For each gate but the last: whether some agent aimed at the next
        gate before the last agent through this one had passed (None if
        nobody passed this gate)."""
        out = []
        for times, t_next in zip(self.gate_pass_time, self.gate_first_target_time[1:]):
            out.append(None if np.isnan(times).all() else bool(t_next <= np.nanmax(times)))
        return out

    def run(self, n_steps, record_every=1, callback=None):
        """Run n_steps steps, recording every record_every-th state.

        callback(sim), if given, is called after every recorded step (e.g. for
        live drawing) and may return True to stop early.
        """
        history = History(dt=self.params.dt, gates=self.gates, post_radius=self.params.post_radius)
        history.record(self.step_count, self.state, 0, 0)
        prom = dem = 0
        for _ in range(n_steps):
            dp, dd = self.step()
            prom, dem = prom + dp, dem + dd
            if self.step_count % record_every == 0:
                history.record(self.step_count, self.state, prom, dem)
                prom = dem = 0
                if callback is not None and callback(self):
                    break
        history.gate_pass_time = self.gate_pass_time.copy()
        history.gate_first_target_time = self.gate_first_target_time.copy()
        return history
