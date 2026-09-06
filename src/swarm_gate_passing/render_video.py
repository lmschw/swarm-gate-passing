"""Render a recorded episode trajectory as an MP4 video.

Replays telemetry already produced by simulate_hebbian_episode(...,
record_trajectory=True, record_battery=True) frame-by-frame with matplotlib,
writing each frame via cv2.VideoWriter -- the same matplotlib-to-cv2-frame
pipeline energy_efficient_flocking's original render_hebbian_episode_video used
(see wind_physics._open_video_writer), but decoupled from the physics loop
itself: rendering never slows down or is coupled to simulation/training.
"""
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import config


def render_trajectory_video(result, output_path, gradient_sensor=None, gates=None, finish_x=None,
                             fps=10.0, figsize=(8, 8), max_battery=100.0, title=""):
    """result: an EpisodeResult from simulate_hebbian_episode(...,
    record_trajectory=True, record_battery=True) -- both flags are required,
    since this replays exactly what was recorded rather than re-simulating."""
    telemetry = result.telemetry
    if telemetry is None or telemetry.get("positions") is None or telemetry.get("headings") is None:
        raise ValueError("result.telemetry must include 'positions' and 'headings' -- "
                          "call simulate_hebbian_episode(..., record_trajectory=True)")
    positions = telemetry["positions"]   # (n_steps, n_agents, 2), arena frame
    headings = telemetry["headings"]     # (n_steps, n_agents)
    battery = telemetry.get("battery")   # (n_steps, n_agents) or None
    n_steps, n_agents, _ = positions.shape

    x_candidates = [positions[:, :, 0].min(), positions[:, :, 0].max()]
    if finish_x is not None:
        x_candidates.append(finish_x)
    if gates:
        x_candidates.extend(g.x_arena for g in gates)
    x_min, x_max = min(x_candidates) - 1.0, max(x_candidates) + 1.0
    y_min, y_max = config.Y_RANGE[0], config.Y_RANGE[1]

    size = config.VIDEO_SIZE
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    fig, ax = plt.subplots(figsize=figsize)

    map_extent = [config.X_RANGE[0], config.X_RANGE[1], config.Y_RANGE[0], config.Y_RANGE[1]]

    for t in range(n_steps):
        ax.clear()
        if gradient_sensor is not None:
            ax.imshow(gradient_sensor.map, cmap="gray", origin="upper", extent=map_extent,
                      alpha=0.5, aspect="auto")
        if gates:
            for gate in gates:
                if gate.blocking:
                    # solid wall spans the rest of the arena's y-range -- a real barrier
                    ax.plot([gate.x_arena, gate.x_arena], [y_min, gate.y_lo_arena], color="red", linewidth=3)
                    ax.plot([gate.x_arena, gate.x_arena], [gate.y_hi_arena, y_max], color="red", linewidth=3)
                else:
                    # "just a pole on either side" -- two short landmark posts, NOT a wall;
                    # nothing stops an agent from crossing anywhere else along this x.
                    post_half_len = 0.15
                    for post_y in (gate.y_lo_arena, gate.y_hi_arena):
                        ax.plot([gate.x_arena, gate.x_arena], [post_y - post_half_len, post_y + post_half_len],
                                color="red", linewidth=5, solid_capstyle="round")
        if finish_x is not None:
            ax.axvline(finish_x, color="orange", linestyle="--", linewidth=1.5)

        for i in range(n_agents):
            x, y = positions[t, i]
            theta = headings[t, i]
            if battery is not None:
                norm_b = float(np.clip(battery[t, i] / max_battery, 0.0, 1.0))
                color = (1.0 - norm_b, norm_b, 0.0)
            else:
                color = "steelblue"
            ax.add_patch(plt.Circle((x, y), config.ROBOT_RAD, facecolor=color, edgecolor="k", zorder=2))
            arrow_len = config.VIDEO_ARROW_LEN
            ax.quiver(x, y, -arrow_len * np.sin(theta), arrow_len * np.cos(theta),
                      angles="xy", scale_units="xy", scale=1, color="k",
                      width=config.VIDEO_QUIVER_WIDTH, zorder=3)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal")
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_title(f"{title}  t={t * config.DT:.1f}s" if title else f"t={t * config.DT:.1f}s")

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        frame = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
        frame = cv2.resize(frame, size)
        writer.write(frame)

    writer.release()
    plt.close(fig)
