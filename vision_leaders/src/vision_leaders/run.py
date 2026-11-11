"""Run the swarm with camera-based gate perception and record it.

    python -m vision_leaders.run --model models/goalnet.pt --gates 3 --seed 0 --video slalom.mp4
    python -m vision_leaders.run --model models/goalnet.pt --gates 5 --course winding --seed 5 --duration 440
    python -m vision_leaders.run --gates 3 --seed 0 --video rule.mp4     # no model: the rule drives

Courses and starts are built like emergent_leaders.run's (same --seed, same
course). Explorers are off.
"""
import argparse

import matplotlib
import numpy as np

from emergent_leaders import ModelParams

from .courses import course, start
from .simulation import TEACHERS, VisionSimulation


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=None, help="trained GoalNet; without one the rule drives")
    parser.add_argument("--teacher", choices=TEACHERS, default="full",
                        help="rule used without a model, and shown beside the network")
    parser.add_argument("--duration", type=float, default=320.0, help="simulated time [s]")
    parser.add_argument("--start-heading", type=float, default=0.0, help="[deg], +/- 30 deg")
    parser.add_argument("--sigma-leader", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gates", type=int, default=3)
    parser.add_argument("--course", choices=["slalom", "winding"], default="slalom")
    parser.add_argument("--gate-spacing", type=float, default=None,
                        help="default: the largest slalom spacing for the sensing range")
    parser.add_argument("--max-turn", type=float, default=90.0, help="largest winding turn [deg]")
    parser.add_argument("--record-every", type=int, default=10)
    parser.add_argument("--video", default=None, help="MP4 with one agent's camera inset")
    parser.add_argument("--camera-agent", type=int, default=0, help="agent whose camera the video shows")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args(argv)

    matplotlib.use("Agg")
    from .visualize import camera_frame, render_video

    net = None
    if args.model:
        import torch
        from .network import GoalNet
        torch.set_num_threads(1)   # 25 small images per step: more threads only add overhead
        net = GoalNet.load(args.model)
    params = ModelParams(sigma_leader=args.sigma_leader)
    gates = course(args.course, args.gates, args.seed, args.gate_spacing, args.max_turn, params)
    kwargs = {} if net is None else {"camera": net.cam}
    sim = VisionSimulation(params=params, state=start(args.seed, heading_deg=args.start_heading), gates=gates,
                           explore=False, seed=args.seed, net=net, teacher=args.teacher, check_course=False,
                           **kwargs)
    frames = [camera_frame(sim, args.camera_agent)] if args.video else None

    def record(sim):
        frames.append(camera_frame(sim, args.camera_agent))

    history = sim.run(round(args.duration / params.dt), record_every=args.record_every,
                      callback=record if args.video else None)
    print(f"{'network' if net else args.teacher + ' rule'} perception, {sim.step_count * params.dt:.0f} s")
    for g, times in enumerate(sim.gate_pass_time):
        print(f"gate {g + 1}: {int((~np.isnan(times)).sum())} / {sim.state.n} agents through")
    if args.video:
        render_video(history, frames, sim.camera, args.camera_agent, args.video, fps=args.fps)
        print(f"video -> {args.video}")


if __name__ == "__main__":
    main()
