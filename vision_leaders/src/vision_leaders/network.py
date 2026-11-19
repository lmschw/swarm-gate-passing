"""A small CNN from one camera image to (gate in view?, goal bearing).

The bearing is relative to the agent's heading, positive to the left, and is
predicted as (sin, cos) so it has no wrap-around. A fixed channel with each
column's bearing is appended to the image, so the network can read bearings
off positions in the image.

Optional, for the gate options of emergent_leaders:
- memory: the agent's previous output is a second input, (lead 0/1, sin,
  cos of its previous bearing, turned by how far the agent has turned since;
  zeros if it didn't lead), so the network can keep a gate target the image
  alone doesn't show (gate_sticky). Only the bearing sees it, through a head
  of its own: a lead output that saw its previous lead would learn to copy it
  (the teacher's lead almost never changes from one step to the next) and
  never start leading by itself;
- near: a fourth output, the logit of being near the target gate's centre
  (gate_loose_radius), where agents stop aligning.
"""
import math

import numpy as np
import torch
from torch import nn

from .camera import CameraParams


class GoalNet(nn.Module):
    def __init__(self, cam=CameraParams(), memory=False, near=False, channels=(16, 32, 32), hidden=64):
        """channels: of the three convolutions; hidden: units of the hidden layer."""
        super().__init__()
        self.cam = cam
        self.memory, self.near = memory, near
        self.config = dict(memory=memory, near=near, channels=tuple(channels), hidden=hidden)
        c1, c2, c3 = channels
        step = cam.fov / cam.width
        columns = cam.fov / 2 - step * (np.arange(cam.width) + 0.5)
        self.register_buffer("column_bearing",
                             torch.tensor(columns, dtype=torch.float32).expand(1, 1, cam.height, cam.width).clone())
        self.features = nn.Sequential(
            nn.Conv2d(4, c1, 3, padding=1), nn.ReLU(),
            nn.Conv2d(c1, c2, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(c2, c3, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        n_features = c3 * math.ceil(cam.height / 4) * math.ceil(cam.width / 4)
        if memory:
            self.head = nn.Sequential(nn.Linear(n_features, hidden), nn.ReLU(), nn.Linear(hidden, 1 + near))
            self.bearing_head = nn.Sequential(nn.Linear(n_features + 3, hidden), nn.ReLU(), nn.Linear(hidden, 2))
        else:
            self.head = nn.Sequential(nn.Linear(n_features, hidden), nn.ReLU(), nn.Linear(hidden, 3 + near))

    def forward(self, images, prev=None):
        """images: (b, 3, height, width) uint8; prev: (b, 3) float, with memory.
        Returns (lead logit, sin, cos, near logit or None), each (b,)."""
        x = images.float() / 255.0
        x = torch.cat([x, self.column_bearing.expand(len(x), -1, -1, -1)], dim=1)
        x = self.features(x)
        out = self.head(x)
        if self.memory:
            bearing = self.bearing_head(torch.cat([x, prev], dim=1))
            return out[:, 0], bearing[:, 0], bearing[:, 1], (out[:, 1] if self.near else None)
        return out[:, 0], out[:, 1], out[:, 2], (out[:, 3] if self.near else None)

    @torch.no_grad()
    def predict(self, images, prev=None):
        """images: (n, 3, height, width) uint8 array; prev: (n, 3) array, with
        memory. Returns (lead probability, bearing [rad], near probability or
        None) arrays."""
        logit, s, c, near = self(torch.from_numpy(images), None if prev is None else torch.from_numpy(prev).float())
        return (torch.sigmoid(logit).numpy(), torch.atan2(s, c).numpy(),
                None if near is None else torch.sigmoid(near).numpy())

    def save(self, path):
        torch.save({"cam": self.cam, **self.config, "state": self.state_dict()}, path)

    @classmethod
    def load(cls, path):
        data = torch.load(path, weights_only=False)
        net = cls(data["cam"], data.get("memory", False), data.get("near", False), data.get("channels", (16, 32, 32)),
                  data.get("hidden", 64))
        net.load_state_dict(data["state"])
        return net.eval()
