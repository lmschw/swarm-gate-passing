"""A tunable, privileged gate teacher.

The emergent_leaders rule with lone_post_aim="centre", generalised by three
aiming parameters (Aim):

- lateral: the aim point is shifted along the gate line by -lateral times the
  agent's own offset from the gate centre. 0 aims at the centre; > 0 makes
  agents off to one side aim past the centre, towards the other side;
- through: the aim point lies this far past the gate line (along the passing
  direction); negative values put it before the line;
- next_margin: an agent with targets at two gates switches to the later gate
  once that target is less than next_margin farther than the current one
  (0: the nearer target wins, as in the rule).

Which agents aim (sensing within gate_sense_range, the sensed post or the
centre in front) is the rule's. The motion side is ModelParams: Aim.goal_gain
and Aim.beta replace its goal_gain and beta (see Aim.model_params).

AimSimulation drives the swarm with this teacher and no cameras, for
evolve.py; VisionSimulation(aim=...) uses it as the teacher for training.
"""
from dataclasses import asdict, dataclass, fields, replace
import json

import numpy as np

from emergent_leaders import ModelParams, Simulation
from emergent_leaders.gates import _in_front


@dataclass(frozen=True)
class Aim:
    lateral: float = 0.0
    through: float = 0.0
    next_margin: float = 0.0
    goal_gain: float = 4.0
    beta: float = 0.5

    def model_params(self, params=None):
        return replace(params or ModelParams(sigma_leader=0.7), goal_gain=self.goal_gain, beta=self.beta,
                       lone_post_aim="centre")

    def save(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(**{k: v for k, v in json.load(f).items() if k in {x.name for x in fields(cls)}})


def aim_targets(pos, heading, gates, params, aim, visible=None):
    """As emergent_leaders.gates.gate_targets, with the aim point and the
    switch to the next gate set by aim."""
    n = len(heading)
    gate_idx, posts_seen, target = np.full(n, -1), np.zeros(n, dtype=int), np.full((n, 2), np.nan)
    for g, gate in enumerate(gates):
        posts = gate.posts
        sensed = np.linalg.norm(pos[:, None, :] - posts[None, :, :], axis=2) <= params.gate_sense_range
        if visible is not None:
            sensed &= visible[:, g]
        rule_aim = np.where(sensed[:, [0]], posts[0], posts[1])
        rule_aim[sensed.all(axis=1)] = gate.centre
        aims = sensed.any(axis=1) & _in_front(pos, heading, rule_aim, params.gate_front_half_angle)
        centre = np.asarray(gate.centre, dtype=float)
        lateral = (pos - centre) @ gate.tangent
        point = centre + np.outer(-aim.lateral * lateral, gate.tangent) + aim.through * gate.normal
        aims &= ~(np.linalg.norm(point - pos, axis=1) >= np.linalg.norm(target - pos, axis=1) + aim.next_margin)
        gate_idx[aims] = g
        posts_seen[aims] = sensed[aims].sum(axis=1)
        target[aims] = point[aims]
    return gate_idx, posts_seen, target


class AimSimulation(Simulation):
    def __init__(self, *args, aim=Aim(), **kwargs):
        self.aim = aim
        super().__init__(*args, **kwargs)

    def perceive_gates(self):
        s = self.state
        return aim_targets(s.pos, s.heading, self.gates, self.params, self.aim)
