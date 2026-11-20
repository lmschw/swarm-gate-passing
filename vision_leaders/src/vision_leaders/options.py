"""Command-line options for emergent_leaders' gate parameters, shared by
train.py, evaluate.py and run.py."""
from dataclasses import replace


def add_gate_options(parser):
    group = parser.add_argument_group("gate options (emergent_leaders.ModelParams)")
    group.add_argument("--sense-range", type=float, default=None, help="gate_sense_range [m]")
    group.add_argument("--sticky-targets", action="store_true",
                       help="gate_sticky: keep a gate target until crossing that gate's line")
    group.add_argument("--loose-radius", type=float, default=0.0,
                       help="gate_loose_radius: agents this close to their target gate's centre stop aligning [m]")
    group.add_argument("--free-leaders", action="store_true",
                       help="gate_leader_free: gate leaders neither attract to nor align with neighbours")
    group.add_argument("--post-avoid", type=float, default=0.0, metavar="SIGMA",
                       help="post_avoid_sigma: posts repel agents within sqrt(2) SIGMA of their edge [m]")
    group.add_argument("--post-avoid-weight", type=float, default=3.0, help="post_avoid_weight, times alpha")


def gate_overrides(args):
    """The ModelParams fields the options of add_gate_options set, as a dict."""
    out = dict(gate_sticky=args.sticky_targets, gate_loose_radius=args.loose_radius,
               gate_leader_free=args.free_leaders,
               post_avoid_sigma=args.post_avoid, post_avoid_weight=args.post_avoid_weight)
    if args.sense_range is not None:
        out["gate_sense_range"] = args.sense_range
    return out


def gate_options(params, args):
    """params with the options of add_gate_options applied."""
    return replace(params, **gate_overrides(args))
