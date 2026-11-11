"""Visualisation: live view, MP4 replay and a static run summary.

The swarm view shows followers in grey, random-walk leaders in blue and gate
leaders in green (hollow while they sense only one post and aim at it), each
with a heading arrow; leaders also carry a faint arrow along their goal
heading. Gates are drawn as two posts, at their true radius, with a dashed
opening.
The centroid's path is traced in orange and a strip below the arena plots the
number of leaders over time.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import ListedColormap, to_rgba

from .leadership import promotion_probability

SURFACE = "#1a1a19"
INK = "#f0efec"
INK_MUTED = "#8a8983"
FOLLOWER = "#c3c2b7"
LEADER = "#3987e5"
GATE_LEADER = "#199e70"
TRAIL = "#d95926"

HEADING_LEN = 0.45
GOAL_LEN = 1.2


class SwarmView:
    """Matplotlib figure that redraws a swarm frame by updating its artists in place."""

    def __init__(self, n_agents, t_max, n_max=None, follow=True, view_radius=8.0, extent=None,
                 gates=(), post_radius=0.15, figsize=(7, 8.4)):
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

        for gate in gates:
            posts = gate.posts
            self.ax.plot(posts[:, 0], posts[:, 1], color=INK_MUTED, lw=1, ls="--", zorder=0)
            for post in posts:
                self.ax.add_patch(Circle(post, post_radius, color=INK, zorder=5))

        zeros = np.zeros(n_agents)
        self.trail, = self.ax.plot([], [], color=TRAIL, lw=1.5, alpha=0.9, zorder=1)
        self.goals = self.ax.quiver(zeros, zeros, zeros, zeros, alpha=0.35,
                                    angles="xy", scale_units="xy", scale=1, width=0.004, zorder=2)
        self.arrows = self.ax.quiver(zeros, zeros, zeros, zeros, angles="xy", scale_units="xy",
                                     scale=1, width=0.005, zorder=3)
        self.dots = self.ax.scatter(zeros, zeros, s=28, zorder=4, linewidths=0)
        self.label = self.ax.text(0.02, 0.98, "", transform=self.ax.transAxes, va="top",
                                  color=INK, fontsize=9, family="monospace")
        handles = [plt.Line2D([], [], ls="", marker="o", color=FOLLOWER, label="follower"),
                   plt.Line2D([], [], ls="", marker="o", color=LEADER,
                              label="leader (random walk)" if gates else "leader")]
        if gates:
            handles += [plt.Line2D([], [], ls="", marker="o", color=GATE_LEADER, label="leader (gate centre)"),
                        plt.Line2D([], [], ls="", marker="o", mfc="none", mew=1.2, color=GATE_LEADER,
                                   label="leader (one post)"),
                        plt.Line2D([], [], ls="", marker="o", color=INK, label="gate post")]
        handles.append(plt.Line2D([], [], color=TRAIL, label="centroid path"))
        legend = self.ax.legend(handles=handles, loc="upper right", fontsize=8, frameon=False)
        for text in legend.get_texts():
            text.set_color(INK)

        self.n_line, = self.ax_n.plot([], [], color=LEADER, lw=1.5, label="all leaders")
        self.n_gate_line = None
        if gates:
            self.n_gate_line, = self.ax_n.plot([], [], color=GATE_LEADER, lw=1.5, label="gate leaders")
            strip_legend = self.ax_n.legend(loc="upper right", fontsize=7, frameon=False)
            for text in strip_legend.get_texts():
                text.set_color(INK)
        self.ax_n.set_xlim(0, t_max)
        self.ax_n.set_ylim(0, 1.1 * (n_max or n_agents))
        self.ax_n.set_xlabel("time [s]", color=INK_MUTED, fontsize=9)
        self.ax_n.set_ylabel("leaders", color=INK_MUTED, fontsize=9)
        # fixed margins: tight_layout would only see the initial tick labels
        self.fig.subplots_adjust(left=0.1, right=0.97, top=0.98, bottom=0.07, hspace=0.22)
        self._t, self._n, self._n_gate, self._trail = [], [], [], []

    def update(self, t, pos, heading, is_leader, goal_heading, gate_target=None, posts_seen=None, n_passed=None):
        """gate_target: per-agent gate index (-1 if none); posts_seen: posts of
        that gate each agent senses; n_passed: agents that have passed each
        gate so far."""
        centroid = pos.mean(axis=0)
        gate_leader = gate_target >= 0 if gate_target is not None else np.zeros_like(is_leader)
        self._trail.append(centroid)
        self._t.append(t)
        self._n.append(int(is_leader.sum()))
        self._n_gate.append(int(gate_leader.sum()))

        colors = np.where(is_leader[:, None], to_rgba(LEADER), to_rgba(FOLLOWER))
        colors[gate_leader] = to_rgba(GATE_LEADER)
        self.dots.set_offsets(pos)
        one_post = posts_seen == 1 if posts_seen is not None else np.zeros_like(is_leader)
        faces = colors.copy()
        faces[one_post] = to_rgba(SURFACE)
        self.dots.set_facecolors(faces)
        self.dots.set_edgecolors(colors)
        self.dots.set_linewidths(np.where(one_post, 1.2, 0.0))
        self.arrows.set_offsets(pos)
        self.arrows.set_UVC(HEADING_LEN * np.cos(heading), HEADING_LEN * np.sin(heading))
        self.arrows.set_color(colors)
        g = GOAL_LEN * is_leader
        self.goals.set_offsets(pos)
        self.goals.set_UVC(g * np.cos(goal_heading), g * np.sin(goal_heading))
        self.goals.set_color(colors)
        trail = np.asarray(self._trail)
        self.trail.set_data(trail[:, 0], trail[:, 1])
        self.n_line.set_data(self._t, self._n)
        text = f"t = {t:7.1f} s\nleaders {self._n[-1]:2d} / {len(heading)}"
        if self.n_gate_line is not None:
            self.n_gate_line.set_data(self._t, self._n_gate)
            text += f" ({self._n_gate[-1]} gate)"
        if n_passed is not None:
            text += "".join(f"\ngate {k + 1}: {m:2d} / {len(heading)} through" for k, m in enumerate(n_passed))
        self.label.set_text(text)

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


def _n_passed(gate_pass_time, t):
    if gate_pass_time is None or len(gate_pass_time) == 0:
        return None
    return list((np.nan_to_num(gate_pass_time, nan=np.inf) <= t).sum(axis=1))


def render_video(history, output_path, fps=30, frame_stride=1, follow=True, view_radius=8.0):
    """Replay a recorded History as an MP4 (one frame per frame_stride records)."""
    import cv2

    h = history.as_arrays()
    n_agents = h["pos"].shape[1]
    view = SwarmView(n_agents, t_max=h["time"][-1], n_max=max(1, h["n_leaders"].max()),
                     follow=follow, view_radius=view_radius, gates=history.gates,
                     post_radius=history.post_radius, extent=None if follow else _extent(h["pos"]))
    writer = None
    try:
        for k in range(0, len(h["time"]), frame_stride):
            t = h["time"][k]
            view.update(t, h["pos"][k], h["heading"][k], h["is_leader"][k], h["goal_heading"][k],
                        h["gate_target"][k], h["posts_seen"][k], _n_passed(history.gate_pass_time, t))
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
                     gates=sim.gates, post_radius=sim.params.post_radius,
                     extent=None if follow else (-50, 50, -50, 50))

    def callback(sim):
        if not plt.fignum_exists(view.fig.number):
            return True
        s = sim.state
        t = sim.step_count * sim.params.dt
        view.update(t, s.pos, s.heading, s.is_leader, s.goal_heading, s.gate_target, s.posts_seen,
                    _n_passed(sim.gate_pass_time, t))
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
    for gate in history.gates:
        posts = gate.posts
        ax.plot(posts[:, 0], posts[:, 1], color=muted, lw=1, ls="--")
        ax.plot(posts[:, 0], posts[:, 1], ls="", marker="s", ms=5, color=ink)
    ax.plot(*c[0], "o", color=ink, ms=6)
    ax.annotate("start", c[0], textcoords="offset points", xytext=(6, 6), fontsize=8, color=muted)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title("Centroid path", loc="left", fontsize=11, color=ink)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    fig.colorbar(points, ax=ax, label="time [s]", shrink=0.8)

    ax = axes[1]
    ax.plot(t, h["n_leaders"], color="#2a78d6", lw=1.2, label="all leaders")
    if history.gates:
        ax.plot(t, h["n_gate_leaders"], color="#1baf7a", lw=1.2, label="gate leaders")
        ax.legend(fontsize=8, frameon=False, loc="upper right")
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
