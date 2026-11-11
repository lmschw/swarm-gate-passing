"""MP4 replay: the emergent_leaders swarm view plus one agent's camera.

The tracked agent is ringed in the arena. Its camera image is shown below
the leader strip, with the rule's goal bearing as a dashed white line and the
network's (if any) as a solid one, in the gate-leader colour while the
network leads.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from emergent_leaders.visualize import GATE_LEADER, INK, INK_MUTED, TRAIL, SwarmView, _n_passed


def camera_frame(sim, agent):
    """What render_video needs from sim about one agent at this step."""
    frame = {"image": sim.images[agent].copy(), "teacher_lead": bool(sim.teacher_lead[agent]),
             "teacher_bearing": float(sim.teacher_bearing[agent])}
    if sim.net is not None:
        frame.update(net_prob=float(sim.net_prob[agent]), net_bearing=float(sim.net_bearing[agent]))
    return frame


def render_video(history, frames, camera, agent, output_path, fps=30, view_radius=8.0):
    """history: a recorded History; frames: camera_frame(sim, agent) for each record."""
    import cv2

    h = history.as_arrays()
    n_agents = h["pos"].shape[1]
    fig_w, fig_h = 7.0, 10.4
    view = SwarmView(n_agents, t_max=h["time"][-1], n_max=max(1, h["n_leaders"].max()), view_radius=view_radius,
                     gates=history.gates, post_radius=history.post_radius, figsize=(fig_w, fig_h))
    ring = Circle((0, 0), 0.35, fill=False, ec=TRAIL, lw=1.5, zorder=6)
    view.ax.add_patch(ring)
    # camera row below the leader strip, full width
    left, width = 0.1, 0.87
    cam_h = width * fig_w * camera.height / camera.width / fig_h
    view.fig.subplots_adjust(bottom=0.07 + cam_h + 0.05)
    ax_cam = view.fig.add_axes([left, 0.03, width, cam_h])
    ax_cam.set_xticks([])
    ax_cam.set_yticks([])
    for spine in ax_cam.spines.values():
        spine.set_color(INK_MUTED)
    picture = ax_cam.imshow(frames[0]["image"].transpose(1, 2, 0), interpolation="nearest")
    step = camera.fov / camera.width

    def column(bearing):
        return (camera.fov / 2 - bearing) / step - 0.5

    rule_line = ax_cam.axvline(0, color=INK, ls="--", lw=1)
    net_line = ax_cam.axvline(0, color=GATE_LEADER, lw=1.5)
    caption = ax_cam.text(0.0, 1.04, "", transform=ax_cam.transAxes, color=INK, fontsize=8,
                          family="monospace", va="bottom")
    writer = None
    try:
        for k, frame in enumerate(frames):
            t = h["time"][k]
            view.update(t, h["pos"][k], h["heading"][k], h["is_leader"][k], h["goal_heading"][k],
                        h["gate_target"][k], h["posts_seen"][k], _n_passed(history.gate_pass_time, t))
            ring.center = h["pos"][k, agent]
            picture.set_data(frame["image"].transpose(1, 2, 0))
            rule_line.set_visible(frame["teacher_lead"])
            rule_line.set_xdata([column(frame["teacher_bearing"])] * 2)
            text = f"agent {agent} camera   rule: " + (
                f"{np.degrees(frame['teacher_bearing']):+4.0f} deg" if frame["teacher_lead"] else "no gate ")
            if "net_prob" in frame:
                leads = frame["net_prob"] >= 0.5
                net_line.set_visible(leads)
                net_line.set_xdata([column(frame["net_bearing"])] * 2)
                text += f"   network: p = {frame['net_prob']:.2f}" + (
                    f", {np.degrees(frame['net_bearing']):+4.0f} deg" if leads else "")
            else:
                net_line.set_visible(False)
            caption.set_text(text)
            image = view.frame_bgr()
            if writer is None:
                writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                         (image.shape[1], image.shape[0]))
            writer.write(image)
    finally:
        if writer is not None:
            writer.release()
        plt.close(view.fig)
