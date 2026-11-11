"""Evaluation courses, built the way emergent_leaders.run builds them."""
import math

import numpy as np

from emergent_leaders import ModelParams, grid_swarm, random_turns, slalom_course, winding_course
from emergent_leaders.gates import max_slalom_spacing


def course(kind, n_gates, seed, spacing=None, max_turn=90.0, params=ModelParams()):
    """slalom: alternating 1 m offsets; winding: random turns drawn from seed.
    spacing defaults to the largest slalom spacing for the sensing range."""
    if spacing is None:
        spacing = max_slalom_spacing(params.gate_sense_range, params.gate_lookahead)
    if kind == "slalom":
        return slalom_course(max(n_gates, 2), spacing=spacing)[:n_gates]
    turns = random_turns(max(n_gates, 2), math.radians(max_turn), np.random.default_rng(seed))
    return winding_course(turns, spacing)[:n_gates]


def start(seed, n_agents=25, heading_deg=0.0):
    """The swarm on a jittered grid, facing heading_deg +/- 30 deg (random headings if None)."""
    rng = np.random.default_rng(seed)
    return grid_swarm(n_agents, rng, heading=None if heading_deg is None else math.radians(heading_deg))
