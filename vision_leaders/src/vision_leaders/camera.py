"""Low-resolution forward cameras, one per agent.

The world is flat. Gate posts are tall cylinders (post_radius from the motion
model's ModelParams, post_height here), agents are short cylinders
(agent_radius, agent_height); the agents of the motion model are points, so
the cylinders only matter to the cameras. Each camera sits mount_height above
the ground at its agent's position and looks along its heading.

Projection: columns are equal steps in bearing over the field of view (left
edge = +fov/2), rows use the same angular step, and heights project by
pinhole perspective: an object at horizontal distance d spans the rows
horizon_row - f (top - mount_height) / d to horizon_row + f mount_height / d,
with f = width / fov pixels per radian.

Pixels are area-sampled: an object covers the fraction of a pixel its angular
extent overlaps, so a far post fades out instead of flickering in and out.
Objects are composited front to back; what a nearer object covers hides
what lies behind it. Uncovered pixels show sky above the horizon and ground
below. Images are 8-bit RGB.
"""
from dataclasses import dataclass
import math

import numpy as np

SKY = np.array([0.55, 0.70, 0.90])
GROUND = np.array([0.35, 0.32, 0.28])
POST = np.array([1.00, 0.45, 0.10])
AGENT = np.array([0.85, 0.85, 0.85])


@dataclass(frozen=True)
class CameraParams:
    width: int = 72                 # pixels
    height: int = 16
    fov: float = math.radians(180)  # horizontal field of view
    horizon_row: float = 10.0       # rows from the top edge to the horizon
    mount_height: float = 0.2       # camera above the ground [m]
    post_height: float = 1.0
    agent_radius: float = 0.1
    agent_height: float = 0.25

    @property
    def focal(self):
        """Pixels per radian."""
        return self.width / self.fov


def _scene(pos, gates, post_radius, cam):
    """Objects every camera can see: (centres, radii, heights, colours, post index or -1)."""
    posts = np.array([p for gate in gates for p in gate.posts]).reshape(-1, 2)
    n, m = len(pos), len(posts)
    centres = np.concatenate([pos, posts])
    radii = np.concatenate([np.full(n, cam.agent_radius), np.full(m, post_radius)])
    heights = np.concatenate([np.full(n, cam.agent_height), np.full(m, cam.post_height)])
    colours = np.concatenate([np.tile(AGENT, (n, 1)), np.tile(POST, (m, 1))])
    post_idx = np.concatenate([np.full(n, -1), np.arange(m)])
    return centres, radii, heights, colours, post_idx


def render(pos, heading, gates, post_radius, cam=CameraParams()):
    """Every agent's camera image and how visible each post is to it.

    Returns (images, post_visibility): images is (n, 3, height, width) uint8;
    post_visibility is (n, n_gates, 2), the most any one pixel shows of each
    post (0: hidden or out of view, 1: fills a pixel unoccluded).
    """
    n = len(heading)
    centres, radii, heights, colours, post_idx = _scene(pos, gates, post_radius, cam)
    m = len(centres)

    rel = centres[None, :, :] - pos[:, None, :]                       # (n, m, 2)
    dist = np.linalg.norm(rel, axis=2)
    dist[np.arange(n), np.arange(n)] = np.inf                         # no agent sees itself
    bearing = (np.arctan2(rel[..., 1], rel[..., 0]) - heading[:, None] + np.pi) % (2 * np.pi) - np.pi
    half = np.arcsin(np.clip(radii[None, :] / np.maximum(dist, 1e-9), 0.0, 1.0))

    # horizontal coverage of each column: column j spans bearings [left_j - step, left_j]
    step = cam.fov / cam.width
    left = cam.fov / 2 - step * np.arange(cam.width)
    h_cov = np.clip(np.minimum((bearing + half)[..., None], left)
                    - np.maximum((bearing - half)[..., None], left - step), 0.0, None) / step   # (n, m, W)

    # vertical coverage of each row: row r spans [r, r + 1] from the top edge
    d = np.where(np.isfinite(dist), dist, 1.0)
    top = cam.horizon_row - cam.focal * (heights[None, :] - cam.mount_height) / d
    bottom = cam.horizon_row + cam.focal * cam.mount_height / d
    rows = np.arange(cam.height)
    v_cov = np.clip(np.minimum(bottom[..., None], rows + 1) - np.maximum(top[..., None], rows), 0.0, 1.0)  # (n, m, H)

    img = np.zeros((n, 3, cam.height, cam.width))
    transmit = np.ones((n, cam.height, cam.width))
    visibility = np.zeros((n, len(gates) * 2))
    order = np.argsort(dist, axis=1)
    agents = np.arange(n)
    for rank in range(m):
        k = order[:, rank]
        alpha = v_cov[agents, k][:, :, None] * h_cov[agents, k][:, None, :]
        alpha[~np.isfinite(dist[agents, k])] = 0.0
        shown = transmit * alpha
        img += shown[:, None] * colours[k][:, :, None, None]
        is_post = post_idx[k] >= 0
        if is_post.any():
            seen = shown[is_post].reshape(is_post.sum(), -1).max(axis=1)
            visibility[agents[is_post], post_idx[k][is_post]] = seen
        transmit *= 1.0 - alpha

    sky = np.clip(cam.horizon_row - rows, 0.0, 1.0)                   # fraction of each row above the horizon
    background = sky[:, None] * SKY + (1 - sky)[:, None] * GROUND   # (H, 3)
    img += transmit[:, None] * background.T[None, :, :, None]
    images = np.round(255 * np.clip(img, 0.0, 1.0)).astype(np.uint8)
    return images, visibility.reshape(n, len(gates), 2)
