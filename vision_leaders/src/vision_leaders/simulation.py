"""The emergent_leaders swarm with camera-based gate perception.

Every step, each agent renders its camera image (camera.py). The teacher's
gate perception is computed alongside, and the network's if one is given:

- teacher="full" (default): emergent_leaders' gate rule with its perfect
  sensing within gate_sense_range, whether or not the camera shows the posts.
  The network has to infer from the image what the rule senses directly,
  e.g. that a lone post close beside the agent means it is passing between
  the posts (the rule also senses the post behind and stops aiming). With
  params.lone_post_aim = "centre", an agent sensing one post aims at the
  gate's centre instead of at the post. With aim (an aiming.Aim), the tuned
  teacher of aiming.py replaces the rule (pass aim.model_params() as params);
- teacher="camera": the same rule, but a post is sensed only when the camera
  also shows at least visible_threshold of a pixel of it. This teacher fails
  at gates: an agent passing between the posts sees only the one in front
  and turns into it;
- the network (if given): GoalNet on the image. An agent leads when the
  predicted probability q of a gate target reaches lead_threshold, and aims
  along the predicted bearing.

How the network's output drives the goal term G gamma_i g_i of the force:

- goal_weighting="threshold" (default): gamma_i is 1 for leaders, 0 for
  followers, as in emergent_leaders;
- goal_weighting="confidence": gamma_i = q_i for every agent, and every
  agent's goal direction is its predicted bearing, so an agent pulls towards
  what it sees in proportion to how sure the network is that a gate target
  is there. Leaders (q >= lead_threshold) still use the leader spacing and
  are counted as leaders; explorers keep weight 1 and their own legs.

The motion model, leader bookkeeping and explorers are emergent_leaders'.
Without a network, the teacher drives the swarm. With one, the teacher's
labels are still kept (teacher_lead, teacher_bearing) for training on the
states the network visits. The network's outputs are kept as net_prob and
net_bearing.

A network leader's gate_target is bookkeeping only: the first gate the agent
has not passed yet. Its posts_seen is 2 (the network doesn't say how many
posts it saw).

The gate options of emergent_leaders' ModelParams:
- gate_sticky: the teacher keeps its own previous targets (teacher_target).
  A network with memory gets its previous output as input (prev_input, see
  network.py): the teacher's previous labels while the teacher drives, its
  own while it drives;
- gate_loose_radius: teacher_near labels the agents near their target
  gate's centre. Network leaders stop aligning when the network's near
  output reaches 0.5, so the network must have one;
- post_avoid_sigma: motion only, as in emergent_leaders.

With decide_every = k > 1, the network (or the teacher, if it drives)
perceives only every k steps: images, labels and outputs are those of the
last decision, and leaders keep its goal direction in the world frame until
the next. Rendering only then makes the simulation several times faster.
"""
import numpy as np

from emergent_leaders import ModelParams, Simulation
from emergent_leaders.gates import gate_targets

from .aiming import aim_targets
from .camera import CameraParams, render

TEACHERS = ("full", "camera")
GOAL_WEIGHTINGS = ("threshold", "confidence")


def _wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


class VisionSimulation(Simulation):
    def __init__(self, *args, camera=CameraParams(), net=None, teacher="full", visible_threshold=0.5,
                 lead_threshold=0.5, goal_weighting="threshold", aim=None, decide_every=1, **kwargs):
        if teacher not in TEACHERS:
            raise ValueError(f"teacher must be one of {TEACHERS}, got {teacher!r}")
        if goal_weighting not in GOAL_WEIGHTINGS:
            raise ValueError(f"goal_weighting must be one of {GOAL_WEIGHTINGS}, got {goal_weighting!r}")
        if goal_weighting == "confidence" and net is None:
            raise ValueError("goal_weighting='confidence' needs a network")
        params = kwargs.get("params") or ModelParams()
        if net is not None and params.gate_loose_radius and not getattr(net, "near", False):
            raise ValueError("gate_loose_radius needs a network with a near output")
        if aim is not None and params.gate_sticky:
            raise ValueError("the tuned teacher (aim) has no sticky targets")
        self.goal_weighting = goal_weighting
        self.decide_every = decide_every
        self._held = None   # (gate_target, posts_seen, goal direction) of the last decision
        self.aim = aim
        self.teacher_target = None
        self._last = None   # (lead, bearing, heading) of the last output, for prev_input
        self.camera = camera
        self.net = net
        self.teacher = teacher
        self.visible_threshold = visible_threshold
        self.lead_threshold = lead_threshold
        super().__init__(*args, **kwargs)

    def perceive_gates(self):
        if self._held is not None and self.step_count % self.decide_every:
            return self.held_targets()
        target, posts, aim = self.decide()
        to_aim = aim - self.state.pos
        self._held = (target, posts, np.arctan2(to_aim[:, 1], to_aim[:, 0]))
        return target, posts, aim

    def held_targets(self):
        """The last decision's targets, with each leader's goal direction kept in the world frame."""
        target, posts, direction = self._held
        s = self.state
        aim = np.where((target >= 0)[:, None], s.pos + np.column_stack([np.cos(direction), np.sin(direction)]), np.nan)
        if self.net is not None:   # bookkeeping as in decide(): the first gate not passed yet
            not_passed = np.isnan(self.gate_pass_time)
            next_gate = np.where(not_passed.any(axis=0), not_passed.argmax(axis=0), len(self.gates) - 1)
            target = np.where(target >= 0, next_gate, -1)
        return target, posts, aim

    def decide(self):
        """Render, label and (with a network) predict; returns gate_targets' (target, posts_seen, aim point)."""
        s, p = self.state, self.params
        if self._last is None:
            self.prev_input = np.zeros((s.n, 3), dtype=np.float32)
        else:
            lead, bearing, heading = self._last
            turned = bearing - _wrap(s.heading - heading)
            self.prev_input = np.column_stack([lead, lead * np.sin(turned), lead * np.cos(turned)]).astype(np.float32)
        self.images, self.post_visibility = render(s.pos, s.heading, self.gates, p.post_radius, self.camera)
        visible = self.post_visibility >= self.visible_threshold if self.teacher == "camera" else None
        if self.aim is None:
            teacher = gate_targets(s.pos, s.heading, self.gates, p, visible=visible, previous=self.teacher_target,
                                   passed=np.isfinite(self.gate_pass_time))
        else:
            teacher = aim_targets(s.pos, s.heading, self.gates, p, self.aim, visible=visible)
        self.teacher_target = teacher[0]
        to_target = teacher[2] - s.pos
        self.teacher_lead = teacher[0] >= 0
        self.teacher_bearing = np.where(self.teacher_lead,
                                        _wrap(np.arctan2(to_target[:, 1], to_target[:, 0]) - s.heading), 0.0)
        self.teacher_near = np.zeros(s.n, dtype=bool)
        if p.gate_loose_radius and self.gates:
            centres = np.array([g.centre for g in self.gates])
            d = np.linalg.norm(s.pos - centres[np.maximum(teacher[0], 0)], axis=1)
            self.teacher_near = self.teacher_lead & (d < p.gate_loose_radius)
        if self.net is None:
            self._last = (self.teacher_lead.astype(float), self.teacher_bearing, s.heading.copy())
            return teacher

        prob, bearing, near = self.net.predict(self.images, self.prev_input if getattr(self.net, "memory", False)
                                               else None)
        self.net_prob, self.net_bearing, self.net_near = prob, bearing, near
        lead = prob >= self.lead_threshold
        self._last = (lead.astype(float), np.where(lead, bearing, 0.0), s.heading.copy())
        direction = s.heading + bearing
        self.net_goal_heading = direction
        target = np.where(lead[:, None], s.pos + np.column_stack([np.cos(direction), np.sin(direction)]), np.nan)
        not_passed = np.isnan(self.gate_pass_time)
        next_gate = np.where(not_passed.any(axis=0), not_passed.argmax(axis=0), len(self.gates) - 1)
        return np.where(lead, next_gate, -1), np.where(lead, 2, 0), target

    def near_gate(self):
        if self.net is None or not self.params.gate_loose_radius:
            return super().near_gate()
        return (self.state.gate_target >= 0) & (self.net_near >= 0.5)

    def goal_terms(self):
        weight, goal_heading = super().goal_terms()
        if self.goal_weighting == "threshold":
            return weight, goal_heading
        s = self.state
        perceiving = ~(s.is_leader & (s.gate_target < 0))   # everyone but explorers
        return (np.where(perceiving, self.net_prob, weight),
                np.where(perceiving, self.net_goal_heading, goal_heading))
