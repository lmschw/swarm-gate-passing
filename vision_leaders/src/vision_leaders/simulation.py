"""The emergent_leaders swarm with camera-based gate perception.

Every step, each agent renders its camera image (camera.py). The teacher's
gate perception is computed alongside, and the network's if one is given:

- teacher="full" (default): emergent_leaders' gate rule with its perfect
  sensing within gate_sense_range, whether or not the camera shows the posts.
  The network has to infer from the image what the rule senses directly,
  e.g. that a lone post close beside the agent means it is passing between
  the posts (the rule also senses the post behind and stops aiming);
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
"""
import numpy as np

from emergent_leaders import Simulation
from emergent_leaders.gates import gate_targets

from .camera import CameraParams, render

TEACHERS = ("full", "camera")
GOAL_WEIGHTINGS = ("threshold", "confidence")


def _wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


class VisionSimulation(Simulation):
    def __init__(self, *args, camera=CameraParams(), net=None, teacher="full", visible_threshold=0.5,
                 lead_threshold=0.5, goal_weighting="threshold", **kwargs):
        if teacher not in TEACHERS:
            raise ValueError(f"teacher must be one of {TEACHERS}, got {teacher!r}")
        if goal_weighting not in GOAL_WEIGHTINGS:
            raise ValueError(f"goal_weighting must be one of {GOAL_WEIGHTINGS}, got {goal_weighting!r}")
        if goal_weighting == "confidence" and net is None:
            raise ValueError("goal_weighting='confidence' needs a network")
        self.goal_weighting = goal_weighting
        self.camera = camera
        self.net = net
        self.teacher = teacher
        self.visible_threshold = visible_threshold
        self.lead_threshold = lead_threshold
        super().__init__(*args, **kwargs)

    def perceive_gates(self):
        s = self.state
        self.images, self.post_visibility = render(s.pos, s.heading, self.gates, self.params.post_radius, self.camera)
        visible = self.post_visibility >= self.visible_threshold if self.teacher == "camera" else None
        teacher = gate_targets(s.pos, s.heading, self.gates, self.params, visible=visible)
        to_target = teacher[2] - s.pos
        self.teacher_lead = teacher[0] >= 0
        self.teacher_bearing = np.where(self.teacher_lead,
                                        _wrap(np.arctan2(to_target[:, 1], to_target[:, 0]) - s.heading), 0.0)
        if self.net is None:
            return teacher

        prob, bearing = self.net.predict(self.images)
        self.net_prob, self.net_bearing = prob, bearing
        lead = prob >= self.lead_threshold
        direction = s.heading + bearing
        self.net_goal_heading = direction
        target = np.where(lead[:, None], s.pos + np.column_stack([np.cos(direction), np.sin(direction)]), np.nan)
        not_passed = np.isnan(self.gate_pass_time)
        next_gate = np.where(not_passed.any(axis=0), not_passed.argmax(axis=0), len(self.gates) - 1)
        return np.where(lead, next_gate, -1), np.where(lead, 2, 0), target

    def goal_terms(self):
        weight, goal_heading = super().goal_terms()
        if self.goal_weighting == "threshold":
            return weight, goal_heading
        s = self.state
        perceiving = ~(s.is_leader & (s.gate_target < 0))   # everyone but explorers
        return (np.where(perceiving, self.net_prob, weight),
                np.where(perceiving, self.net_goal_heading, goal_heading))
