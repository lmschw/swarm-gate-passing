"""Visualisation: live view, MP4 replay and a static run summary.

The swarm view shows followers in grey and leaders in blue, each with a
heading arrow; leaders also carry a faint arrow along their goal heading. The
centroid's path is traced in orange and a strip below the arena plots the
number of leaders over time.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, to_rgba

from .leadership import promotion_probability

SURFACE = "#1a1a19"
INK = "#f0efec"
INK_MUTED = "#8a8983"
FOLLOWER = "#c3c2b7"
LEADER = "#3987e5"
TRAIL = "#d95926"

HEADING_LEN = 0.45
GOAL_LEN = 1.2


class SwarmView:
    """Matplotlib figure that redraws a swarm frame by updating its artists in place."""

    def __init__(self, n_agents, t_max, n_max=None, follow=True, view_radius=8.0, extent=None,
                 figsize=(7, 8.4)):
        self.follow, self.view_radius, self.extent = follow, view_radius, extent
        self.fig, (self.ax, self.ax_n) = plt.subplots(
            2, 1, figsize=figsize, gridspec_kw={"height_ratios": [5, 1]}, facecolor=SURFACE)
        for ax in (self.ax, self.ax_n):
            ax.set_facecolor(SURFACE)
            ax.tick_params(colors=INK_MUTED, labelsize=8)
            for spine in ax.spines.values():
                spine.set_color(INK_MUTED)
                spine.set_linewidth(0.6)
        self.ax.set_aspect("equal")
        self.ax.set_xlabel("x [m]", color=INK_MUTED, fontsize=9)
        self.ax.set_ylabel("y [m]", color=INK_MUTED, fontsize=9)

        zeros = np.zeros(n_agents)
        self.trail, = self.ax.plot([], [], color=TRAIL, lw=1.5, alpha=0.9, zorder=1)
        self.goals = self.ax.quiver(zeros, zeros, zeros, zeros, color=LEADER, alpha=0.35,
                                    angles="xy", scale_units="xy", scale=1, width=0.004, zorder=2)
        self.arrows = self.ax.quiver(zeros, zeros, zeros, zeros, angles="xy", scale_units="xy",
                                     scale=1, width=0.005, zorder=3)
        self.dots = self.ax.scatter(zeros, zeros, s=28, zorder=4, linewidths=0)
        self.label = self.ax.text(0.02, 0.98, "", transform=self.ax.transAxes, va="top",
                                  color=INK, fontsize=9, family="monospace")
        handles = [plt.Line2D([], [], ls="", marker="o", color=FOLLOWER, label="follower"),
                   plt.Line2D([], [], ls="", marker="o", color=LEADER, label="leader"),
                   plt.Line2D([], [], color=TRAIL, label="centroid path")]
        legend = self.ax.legend(handles=handles, loc="upper right", fontsize=8, frameon=False)
        for text in legend.get_texts():
            text.set_color(INK)

        self.n_line, = self.ax_n.plot([], [], color=LEADER, lw=1.5)
        self.ax_n.set_xlim(0, t_max)
        self.ax_n.set_ylim(0, n_max or n_agents)
        self.ax_n.set_xlabel("time [s]", color=INK_MUTED, fontsize=9)
        self.ax_n.set_ylabel("leaders", color=INK_MUTED, fontsize=9)
        self.fig.tight_layout()
        self._t, self._n, self._trail = [], [], []

    def update(self, t, pos, heading, is_leader, goal_heading):
        centroid = pos.mean(axis=0)
        self._trail.append(centroid)
        self._t.append(t)
        self._n.append(int(is_leader.sum()))

        colors = np.where(is_leader[:, None], to_rgba(LEADER), to_rgba(FOLLOWER))
        self.dots.set_offsets(pos)
        self.dots.set_facecolors(colors)
        self.arrows.set_offsets(pos)
        self.arrows.set_UVC(HEADING_LEN * np.cos(heading), HEADING_LEN * np.sin(heading))
        self.arrows.set_color(colors)
        g = GOAL_LEN * is_leader
        self.goals.set_offsets(pos)
        self.goals.set_UVC(g * np.cos(goal_heading), g * np.sin(goal_heading))
        trail = np.asarray(self._trail)
        self.trail.set_data(trail[:, 0], trail[:, 1])
        self.n_line.set_data(self._t, self._n)
        self.label.set_text(f"t = {t:7.1f} s\nleaders {self._n[-1]:2d} / {len(heading)}")

        if self.follow:
            r = self.view_radius
            self.ax.set_xlim(centroid[0] - r, centroid[0] + r)
            self.ax.set_ylim(centroid[1] - r, centroid[1] + r)
        elif self.extent is not None:
            self.ax.set_xlim(*self.extent[:2])
            self.ax.set_ylim(*self.extent[2:])

    def frame_bgr(self):
        self.fig.canvas.draw()
        rgb = np.asarray(self.fig.canvas.buffer_rgba())[..., :3]
        return np.ascontiguousarray(rgb[..., ::-1])


def _extent(pos, margin=2.0):
    lo, hi = pos.reshape(-1, 2).min(axis=0) - margin, pos.reshape(-1, 2).max(axis=0) + margin
    centre, half = (lo + hi) / 2, (hi - lo).max() / 2
    return (centre[0] - half, centre[0] + half, centre[1] - half, centre[1] + half)


def render_video(history, output_path, fps=30, frame_stride=1, follow=True, view_radius=8.0):
    """Replay a recorded History as an MP4 (one frame per frame_stride records)."""
    import cv2

    h = history.as_arrays()
    n_agents = h["pos"].shape[1]
    view = SwarmView(n_agents, t_max=h["time"][-1], n_max=max(1, h["n_leaders"].max()),
                     follow=follow, view_radius=view_radius,
                     extent=None if follow else _extent(h["pos"]))
    writer = None
    try:
        for k in range(0, len(h["time"]), frame_stride):
            view.update(h["time"][k], h["pos"][k], h["heading"][k], h["is_leader"][k], h["goal_heading"][k])
            frame = view.frame_bgr()
            if writer is None:
                writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                         (frame.shape[1], frame.shape[0]))
            writer.write(frame)
    finally:
        if writer is not None:
            writer.release()
        plt.close(view.fig)


def live_callback(sim, n_steps, record_every, follow=True, view_radius=8.0):
    """Callback for Simulation.run that draws every recorded step in a window.
    Closing the window stops the run."""
    plt.ion()
    view = SwarmView(sim.state.n, t_max=n_steps * sim.params.dt, follow=follow, view_radius=view_radius,
                     extent=None if follow else (-50, 50, -50, 50))

    def callback(sim):
        if not plt.fignum_exists(view.fig.number):
            return True
        s = sim.state
        view.update(sim.step_count * sim.params.dt, s.pos, s.heading, s.is_leader, s.goal_heading)
        plt.pause(1e-3)
        return False

    return callback


def plot_summary(history, params, output_path=None):
    """Three panels: centroid path, number of leaders, mean stability vs time."""
    h = history.as_arrays()
    t, c = h["time"], h["centroid"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    ink, muted, grid = "#0b0b0b", "#52514e", "#e4e3df"

    ax = axes[0]
    blues = ListedColormap(plt.get_cmap("Blues")(np.linspace(0.3, 1.0, 256)))
    points = ax.scatter(c[:, 0], c[:, 1], c=t, cmap=blues, s=3, linewidths=0)
    ax.plot(*c[0], "o", color=ink, ms=6)
    ax.annotate("start", c[0], textcoords="offset points", xytext=(6, 6), fontsize=8, color=muted)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title("Centroid path", loc="left", fontsize=11, color=ink)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    fig.colorbar(points, ax=ax, label="time [s]", shrink=0.8)

    ax = axes[1]
    ax.plot(t, h["n_leaders"], color="#2a78d6", lw=1.2)
    ax.set_title(f"Leaders (of {h['pos'].shape[1]} agents)", loc="left", fontsize=11, color=ink)
    ax.set_xlabel("time [s]")
    ax.set_ylim(bottom=0)

    ax = axes[2]
    mean_s = h["stability"].mean(axis=1)
    ax.plot(t, mean_s, color="#2a78d6", lw=1.2)
    # stability at which promotion reaches half its maximum probability
    s_half = params.b_prom - np.log(1.0 / params.p_prom_max) / params.k_prom
    ax.axhline(s_half, color=muted, lw=1, ls="--")
    ax.annotate(f"half-max promotion, P = {promotion_probability(s_half, params):.3f}/step", (t[0], s_half),
                textcoords="offset points", xytext=(4, 4), fontsize=8, color=muted)
    ax.set_title("Mean stability", loc="left", fontsize=11, color=ink)
    ax.set_xlabel("time [s]")

    for ax in axes:
        ax.grid(color=grid, lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=130)
        plt.close(fig)
    return fig
