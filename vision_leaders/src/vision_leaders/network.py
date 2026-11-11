"""A small CNN from one camera image to (gate in view?, goal bearing).

The bearing is relative to the agent's heading, positive to the left, and is
predicted as (sin, cos) so it has no wrap-around. A fixed channel with each
column's bearing is appended to the image, so the network can read bearings
off positions in the image.
"""
import math

import numpy as np
import torch
from torch import nn

from .camera import CameraParams


class GoalNet(nn.Module):
    def __init__(self, cam=CameraParams()):
        super().__init__()
        self.cam = cam
        step = cam.fov / cam.width
        columns = cam.fov / 2 - step * (np.arange(cam.width) + 0.5)
        self.register_buffer("column_bearing",
                             torch.tensor(columns, dtype=torch.float32).expand(1, 1, cam.height, cam.width).clone())
        self.features = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        n_features = 32 * math.ceil(cam.height / 4) * math.ceil(cam.width / 4)
        self.head = nn.Sequential(nn.Linear(n_features, 64), nn.ReLU(), nn.Linear(64, 3))

    def forward(self, images):
        """images: (b, 3, height, width) uint8. Returns (lead logit, sin, cos), each (b,)."""
        x = images.float() / 255.0
        x = torch.cat([x, self.column_bearing.expand(len(x), -1, -1, -1)], dim=1)
        out = self.head(self.features(x))
        return out[:, 0], out[:, 1], out[:, 2]

    @torch.no_grad()
    def predict(self, images):
        """images: (n, 3, height, width) uint8 array. Returns (lead probability, bearing [rad]) arrays."""
        logit, s, c = self(torch.from_numpy(images))
        return torch.sigmoid(logit).numpy(), torch.atan2(s, c).numpy()

    def save(self, path):
        torch.save({"cam": self.cam, "state": self.state_dict()}, path)

    @classmethod
    def load(cls, path):
        data = torch.load(path, weights_only=False)
        net = cls(data["cam"])
        net.load_state_dict(data["state"])
        return net.eval()
