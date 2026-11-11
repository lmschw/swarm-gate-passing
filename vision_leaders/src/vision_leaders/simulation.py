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
  predicted probability of a gate target reaches lead_threshold, and aims
  along the predicted bearing.

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


def _wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


class VisionSimulation(Simulation):
    def __init__(self, *args, camera=CameraParams(), net=None, teacher="full", visible_threshold=0.5,
                 lead_threshold=0.5, **kwargs):
        if teacher not in TEACHERS:
            raise ValueError(f"teacher must be one of {TEACHERS}, got {teacher!r}")
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
        target = np.where(lead[:, None], s.pos + np.column_stack([np.cos(direction), np.sin(direction)]), np.nan)
        not_passed = np.isnan(self.gate_pass_time)
        next_gate = np.where(not_passed.any(axis=0), not_passed.argmax(axis=0), len(self.gates) - 1)
        return np.where(lead, next_gate, -1), np.where(lead, 2, 0), target
