"""Run a leader-emergence simulation and visualise it.

    python -m emergent_leaders.run --walk levy --n-agents 25 --duration 500 --live
    python -m emergent_leaders.run --walk correlated --video run.mp4 --summary run.png
    python -m emergent_leaders.run --goal-heading 0 --summary fixed.png
"""
import argparse
import math
import pickle

import matplotlib


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-agents", type=int, default=25)
    parser.add_argument("--duration", type=float, default=500.0, help="simulated time [s]")
    parser.add_argument("--walk", choices=["brown", "levy", "correlated"], default="levy",
                        help="random-walk goal model for leaders")
    parser.add_argument("--goal-heading", type=float, default=None,
                        help="use one fixed global goal heading [deg] instead of a random walk")
    parser.add_argument("--leader-type", choices=["free", "inner", "outer"], default="free")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--record-every", type=int, default=10, help="record every k-th step")
    parser.add_argument("--live", action="store_true", help="draw the run in a window")
    parser.add_argument("--video", help="write an MP4 replay to this path")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--summary", help="write a summary figure to this path")
    parser.add_argument("--save", help="pickle the recorded history to this path")
    parser.add_argument("--fixed-view", action="store_true", help="fixed camera instead of following the centroid")
    args = parser.parse_args(argv)

    if not args.live:
        matplotlib.use("Agg")
    from .goals import FixedGoal, RandomWalkGoal
    from .simulation import Simulation
    from . import visualize

    goal = FixedGoal(math.radians(args.goal_heading)) if args.goal_heading is not None else RandomWalkGoal(args.walk)
    sim = Simulation(args.n_agents, goal=goal, leader_type=args.leader_type, seed=args.seed)
    n_steps = round(args.duration / sim.params.dt)
    callback = (visualize.live_callback(sim, n_steps, args.record_every, follow=not args.fixed_view)
                if args.live else None)
    history = sim.run(n_steps, record_every=args.record_every, callback=callback)

    print(f"{sim.step_count} steps ({sim.step_count * sim.params.dt:.0f} s), "
          f"{sim.total_promotions} promotions, {sim.total_demotions} demotions, "
          f"{int(sim.state.is_leader.sum())} leaders at the end")
    if args.save:
        with open(args.save, "wb") as f:
            pickle.dump(history, f)
    if args.summary:
        visualize.plot_summary(history, sim.params, args.summary)
        print(f"summary -> {args.summary}")
    if args.video:
        visualize.render_video(history, args.video, fps=args.fps, follow=not args.fixed_view)
        print(f"video -> {args.video}")


if __name__ == "__main__":
    main()
