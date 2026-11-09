"""Swarm simulation: motion model + leadership emergence + goal directions.

One time step:
  1. compute the virtual force and speed commands from the current state;
  2. integrate positions and headings;
  3. on the new configuration, update each agent's stability and quadrant
     coverage;
  4. leaders accumulate travelled distance; a leader that completes a leg
     either gets a new leg from the goal model or, after legs_per_tenure
     legs, is demoted to follower;
  5. eligible followers are promoted with probability P(stability) and get
     their first leg.
"""
from dataclasses import dataclass, field
import math

import numpy as np

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
                   full_coverage=np.zeros(n, dtype=bool))


def grid_swarm(n_agents, rng, spacing=0.8, jitter=0.25):
    """Agents on a jittered square lattice with uniformly random headings
    (the reference's initial condition)."""
    side = math.ceil(math.sqrt(n_agents))
    xx, yy = np.meshgrid(np.arange(side) * spacing, np.arange(side) * spacing)
    pos = np.column_stack([xx.ravel(), yy.ravel()])[:n_agents]
    pos = pos + rng.uniform(-jitter, jitter, pos.shape) * spacing
    heading = rng.uniform(-np.pi, np.pi, n_agents)
    return SwarmState.from_poses(pos, heading)


@dataclass
class History:
    dt: float
    step: list = field(default_factory=list)
    pos: list = field(default_factory=list)
    heading: list = field(default_factory=list)
    is_leader: list = field(default_factory=list)
    goal_heading: list = field(default_factory=list)
    stability: list = field(default_factory=list)
    promotions: list = field(default_factory=list)   # promotions during each recorded interval
    demotions: list = field(default_factory=list)

    def record(self, step, state, promotions, demotions):
        self.step.append(step)
        self.pos.append(state.pos.copy())
        self.heading.append(state.heading.copy())
        self.is_leader.append(state.is_leader.copy())
        self.goal_heading.append(state.goal_heading.copy())
        self.stability.append(state.stability.copy())
        self.promotions.append(promotions)
        self.demotions.append(demotions)

    def as_arrays(self):
        """Stack every per-step list into an array, keyed by field name."""
        out = {k: np.asarray(v) for k, v in vars(self).items() if isinstance(v, list)}
        out["time"] = out["step"] * self.dt
        out["centroid"] = out["pos"].mean(axis=1)
        out["n_leaders"] = out["is_leader"].sum(axis=1)
        return out


class Simulation:
    def __init__(self, n_agents=25, params=None, goal=None, leader_type="free", seed=None, state=None):
        if leader_type not in leadership.LEADER_TYPES:
            raise ValueError(f"leader_type must be one of {leadership.LEADER_TYPES}, got {leader_type!r}")
        self.params = params or ModelParams()
        self.goal: GoalModel = goal or RandomWalkGoal("levy")
        self.leader_type = leader_type
        self.rng = np.random.default_rng(seed)
        self.state = state if state is not None else grid_swarm(n_agents, self.rng)
        self.step_count = 0
        self.total_promotions = 0
        self.total_demotions = 0
        self._update_leadership_measures(self._geometry())

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

    def step(self):
        """Advance one time step; returns (n_promoted, n_demoted)."""
        s, p = self.state, self.params

        f_x, f_y = motion.virtual_force(s.pos, s.heading, s.is_leader, s.goal_heading, p, self._geometry())
        u, w = motion.velocity_commands(f_x, f_y, p)
        s.pos, s.heading = motion.integrate(s.pos, s.heading, u, w, p.dt)
        self._update_leadership_measures(self._geometry())

        # Leg bookkeeping and demotion
        s.leg_travelled[s.is_leader] += u[s.is_leader] * p.dt
        finished = s.is_leader & (s.leg_travelled >= s.leg_length)
        s.legs_done[finished] += 1
        demote = finished & (s.legs_done >= p.legs_per_tenure)
        s.is_leader[demote] = False
        s.legs_done[demote] = 0
        s.leg_travelled[demote] = 0.0
        self._start_legs(np.flatnonzero(finished & ~demote))

        # Promotion
        candidates = ~s.is_leader & leadership.promotion_eligible(s.full_coverage, self.leader_type)
        prob = leadership.promotion_probability(s.stability, p)
        promote = candidates & (self.rng.random(s.n) < prob)
        s.is_leader[promote] = True
        s.legs_done[promote] = 0
        self._start_legs(np.flatnonzero(promote))

        self.step_count += 1
        n_prom, n_dem = int(promote.sum()), int(demote.sum())
        self.total_promotions += n_prom
        self.total_demotions += n_dem
        return n_prom, n_dem

    def run(self, n_steps, record_every=1, callback=None):
        """Run n_steps steps, recording every record_every-th state.

        callback(sim), if given, is called after every recorded step (e.g. for
        live drawing) and may return True to stop early.
        """
        history = History(dt=self.params.dt)
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
        return history
