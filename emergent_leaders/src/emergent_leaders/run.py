"""Run a leader-emergence simulation and visualise it.

    python -m emergent_leaders.run --walk levy --n-agents 25 --duration 500 --live
    python -m emergent_leaders.run --walk correlated --video run.mp4 --summary run.png
    python -m emergent_leaders.run --goal-heading 0 --summary fixed.png
    python -m emergent_leaders.run --gates 3 --start-heading 0 --no-explore --sigma-leader 0.7 --video gates.mp4
    python -m emergent_leaders.run --gates 5 --course winding --gate-spacing 12 --no-course-check --start-heading 0
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
    parser.add_argument("--no-explore", action="store_true",
                        help="no stability-driven random-walk leaders (only gate leaders)")
    parser.add_argument("--pause-exploration", type=float, default=None, metavar="HOLD_OFF",
                        help="pause random-walk leaders while any agent has a gate target and for "
                             "HOLD_OFF seconds after")
    parser.add_argument("--local-alignment", action="store_true",
                        help="align with neighbours within sense range instead of the whole swarm")
    parser.add_argument("--sigma-leader", type=float, default=None, help="leader spacing parameter")
    parser.add_argument("--start-heading", type=float, default=None,
                        help="start headings around this value [deg] (+/- 30 deg) instead of uniformly random")
    parser.add_argument("--seed", type=int, default=None)

    gates = parser.add_argument_group("gates (course starting along +x)")
    gates.add_argument("--gates", type=int, default=0, help="number of gates")
    gates.add_argument("--course", choices=["slalom", "winding"], default="slalom",
                       help="slalom: alternating offsets along +x; winding: random turns (drawn with --seed)")
    gates.add_argument("--max-turn", type=float, default=90.0, help="largest turn between winding gates [deg]")
    gates.add_argument("--gate-spacing", type=float, default=None,
                       help="distance between gate centres [m] (default: the largest the gate range allows)")
    gates.add_argument("--gate-offset", type=float, default=1.0, help="alternating lateral offset of the slalom [m]")
    gates.add_argument("--gate-width", type=float, default=4.0, help="distance between the posts [m]")
    gates.add_argument("--gate-range", type=float, default=None, help="gate post sensing range [m]")
    gates.add_argument("--gate-lookahead", type=float, default=None,
                       help="how far past a gate the swarm's front must see the next gate from [m]")
    gates.add_argument("--no-course-check", action="store_true",
                       help="allow courses that break the course rules (e.g. gates out of view of each other)")

    out = parser.add_argument_group("output")
    out.add_argument("--record-every", type=int, default=10, help="record every k-th step")
    out.add_argument("--live", action="store_true", help="draw the run in a window")
    out.add_argument("--video", help="write an MP4 replay to this path")
    out.add_argument("--fps", type=int, default=30)
    out.add_argument("--view-radius", type=float, default=10.0, help="half-width of the following camera [m]")
    out.add_argument("--summary", help="write a summary figure to this path")
    out.add_argument("--save", help="pickle the recorded history to this path")
    out.add_argument("--fixed-view", action="store_true", help="fixed camera instead of following the centroid")
    args = parser.parse_args(argv)

    if not args.live:
        matplotlib.use("Agg")
    import numpy as np
    from .gates import max_spacing, random_turns, slalom_course, winding_course
    from .goals import FixedGoal, RandomWalkGoal
    from .params import ModelParams
    from .simulation import Simulation, grid_swarm
    from . import visualize

    overrides = {"local_alignment": args.local_alignment}
    if args.sigma_leader is not None:
        overrides["sigma_leader"] = args.sigma_leader
    if args.gate_range is not None:
        overrides["gate_sense_range"] = args.gate_range
    if args.pause_exploration is not None:
        overrides.update(pause_exploration_at_gates=True, exploration_hold_off=args.pause_exploration)
    if args.gate_lookahead is not None:
        overrides["gate_lookahead"] = args.gate_lookahead
    params = ModelParams(**overrides)
    goal = FixedGoal(math.radians(args.goal_heading)) if args.goal_heading is not None else RandomWalkGoal(args.walk)
    if args.course == "slalom":
        def build(spacing):
            return slalom_course(max(args.gates, 2), spacing=spacing, offset=args.gate_offset, width=args.gate_width)
    else:
        turns = random_turns(max(args.gates, 2), math.radians(args.max_turn), np.random.default_rng(args.seed))

        def build(spacing):
            return winding_course(turns, spacing, width=args.gate_width)
    spacing = args.gate_spacing
    if spacing is None:
        try:
            spacing = max_spacing(build, params.gate_sense_range, params.gate_lookahead)
        except ValueError:
            parser.error("no spacing satisfies the course rules for this course; "
                         "give --gate-spacing and --no-course-check")
    course = build(spacing)[:args.gates]
    rng = np.random.default_rng(args.seed)
    start = grid_swarm(args.n_agents, rng,
                       heading=None if args.start_heading is None else math.radians(args.start_heading))
    sim = Simulation(params=params, goal=goal, leader_type=args.leader_type, seed=args.seed, state=start,
                     gates=course, explore=not args.no_explore, check_course=not args.no_course_check)
    n_steps = round(args.duration / sim.params.dt)
    callback = (visualize.live_callback(sim, n_steps, args.record_every, follow=not args.fixed_view,
                                        view_radius=args.view_radius)
                if args.live else None)
    history = sim.run(n_steps, record_every=args.record_every, callback=callback)

    print(f"{sim.step_count} steps ({sim.step_count * sim.params.dt:.0f} s), "
          f"{sim.total_promotions} promotions, {sim.total_demotions} demotions, "
          f"{int(sim.state.is_leader.sum())} leaders at the end")
    if course:
        print(f"gates {spacing:.1f} m apart, gate range {params.gate_sense_range:.1f} m")
    in_view = sim.next_gate_in_view() + [None]
    for g, times in enumerate(sim.gate_pass_time):
        through = ~np.isnan(times)
        when = f", t = {np.nanmin(times):.0f}-{np.nanmax(times):.0f} s" if through.any() else ""
        nxt = {True: ", next gate in view while passing", False: ", next gate NOT in view while passing",
               None: ""}[in_view[g]]
        print(f"gate {g + 1}: {through.sum()} / {sim.state.n} agents through{when}{nxt}")
    if args.save:
        with open(args.save, "wb") as f:
            pickle.dump(history, f)
    if args.summary:
        visualize.plot_summary(history, sim.params, args.summary)
        print(f"summary -> {args.summary}")
    if args.video:
        visualize.render_video(history, args.video, fps=args.fps, follow=not args.fixed_view,
                               view_radius=args.view_radius)
        print(f"video -> {args.video}")


if __name__ == "__main__":
    main()
