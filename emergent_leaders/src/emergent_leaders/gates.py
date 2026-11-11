"""Gates: pairs of posts the swarm should pass between.

A gate has two posts `width` apart, centred on `centre`, and a passing
direction `heading` (the normal of the line between the posts). Posts are
solid discs of radius post_radius: agents (points) that would end a step
inside one are pushed back onto its edge.

Sensing: an agent senses a post perfectly when it lies within
gate_sense_range. What it aims for depends on how many posts of a gate it
senses:

- both: the gate's centre, if the centre lies in front of it (within
  gate_front_half_angle of its heading);
- one: that post, if the post lies in front of it. Closing in on the post
  brings the other one into range, and the aim shifts to the centre.

Agents ignore targets behind them, so they do not turn back towards a gate
they have just passed. With both posts sensed and the centre behind, the agent
is through (or beside) the gate and aims for nothing. An agent with targets at
two gates aims for the nearer target: the gate it is about to pass, and, once
that gate's centre is behind it, the next one.

Courses (check_course) must let the swarm see the next gate while it passes a
gate: a post of gate k+1 must lie within gate_sense_range of the point
gate_lookahead straight past gate k's centre, where the front of a passing
swarm is. Posts of non-consecutive gates must be more than
2 * gate_sense_range apart, so that no agent senses a gate further along the
course than the next one. gate_sense_range must also exceed width / 2, or no
agent could sense both posts of a gate.
"""
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class Gate:
    centre: tuple
    heading: float   # passing direction [rad]
    width: float

    @property
    def normal(self):
        return np.array([math.cos(self.heading), math.sin(self.heading)])

    @property
    def tangent(self):
        return np.array([-math.sin(self.heading), math.cos(self.heading)])

    @property
    def posts(self):
        c = np.asarray(self.centre, dtype=float)
        half = 0.5 * self.width * self.tangent
        return np.array([c - half, c + half])


def check_course(gates, gate_sense_range, lookahead):
    for a, gate in enumerate(gates):
        if gate.width / 2 >= gate_sense_range:
            raise ValueError(f"gate {a} is {gate.width:.2f} m wide; gate_sense_range = {gate_sense_range:.2f} m "
                             f"must exceed half of that or no agent can sense both posts")
    for a, (gate, nxt) in enumerate(zip(gates, gates[1:])):
        front = np.asarray(gate.centre) + lookahead * gate.normal
        d = np.linalg.norm(nxt.posts - front, axis=1).min()
        if d > gate_sense_range:
            raise ValueError(f"the nearer post of gate {a + 1} is {d:.2f} m from the point {lookahead:.2f} m past "
                             f"gate {a}'s centre; it must be within gate_sense_range = {gate_sense_range:.2f} m so that "
                             f"the swarm sees the next gate while passing")
    for a, gate in enumerate(gates):
        for b in range(a + 2, len(gates)):
            d = np.linalg.norm(gate.posts[:, None, :] - gates[b].posts[None, :, :], axis=2).min()
            if d <= 2 * gate_sense_range:
                raise ValueError(f"posts of gates {a} and {b} are {d:.2f} m apart; non-consecutive gates must be "
                                 f"more than 2 * gate_sense_range = {2 * gate_sense_range:.2f} m apart")


def max_spacing(build, gate_sense_range, lookahead, step=0.5):
    """Largest spacing (a multiple of step) for which build(spacing), a list
    of gates, passes check_course."""
    # no post of the next gate can be within range beyond this
    width = max(g.width for g in build(1.0))
    spacing = step * math.ceil((gate_sense_range + lookahead + width) / step)
    while spacing > 0:
        try:
            check_course(build(spacing), gate_sense_range, lookahead)
            return spacing
        except ValueError:
            spacing -= step
    raise ValueError("no spacing lets the swarm see the next gate")


def max_slalom_spacing(gate_sense_range, lookahead, n_gates=3, step=0.5, **course):
    """Largest spacing (a multiple of step) for which slalom_course(n_gates)
    passes check_course."""
    return max_spacing(lambda s: slalom_course(n_gates, spacing=s, **course), gate_sense_range, lookahead, step)


def _in_front(pos, heading, target, half_angle):
    to_target = target - pos
    bearing = np.arctan2(to_target[:, 1], to_target[:, 0]) - heading
    return np.abs((bearing + np.pi) % (2.0 * np.pi) - np.pi) <= half_angle


def gate_targets(pos, heading, gates, params):
    """What each agent aims for: (gate index or -1, posts sensed of that gate
    (0 if none), target point (nan if none))."""
    n = len(heading)
    gate_idx, posts_seen, target = np.full(n, -1), np.zeros(n, dtype=int), np.full((n, 2), np.nan)
    for g, gate in enumerate(gates):
        posts = gate.posts
        sensed = np.linalg.norm(pos[:, None, :] - posts[None, :, :], axis=2) <= params.gate_sense_range
        aim = np.where(sensed[:, [0]], posts[0], posts[1])
        aim[sensed.all(axis=1)] = gate.centre
        aims = sensed.any(axis=1) & _in_front(pos, heading, aim, params.gate_front_half_angle)
        # with targets at two gates, keep the nearer one
        aims &= ~(np.linalg.norm(aim - pos, axis=1) >= np.linalg.norm(target - pos, axis=1))
        gate_idx[aims] = g
        posts_seen[aims] = sensed[aims].sum(axis=1)
        target[aims] = aim[aims]
    return gate_idx, posts_seen, target


def push_out_of_posts(pos, gates, radius):
    """Move agents inside a post back onto its edge."""
    pos = pos.copy()
    for gate in gates:
        for post in gate.posts:
            d = pos - post
            r = np.linalg.norm(d, axis=1)
            inside = r < radius
            # an agent exactly on the post's centre leaves backwards along the passing direction
            out = np.where(r[inside, None] > 0, d[inside] / np.where(r[inside] > 0, r[inside], 1)[:, None],
                           -gate.normal)
            pos[inside] = post + radius * out
    return pos


def crossed(prev_pos, pos, gate):
    """Agents that passed between the posts in the passing direction this step."""
    c = np.asarray(gate.centre, dtype=float)
    before = (prev_pos - c) @ gate.normal
    after = (pos - c) @ gate.normal
    lateral = (pos - c) @ gate.tangent
    return (before < 0) & (after >= 0) & (np.abs(lateral) <= gate.width / 2)


def slalom_course(n_gates=3, start=(8.0, 1.6), spacing=7.0, offset=1.0, width=4.0):
    """Gates `spacing` apart along +x, alternately `offset` to either side of
    the start line. The first gate faces +x and every later one faces away
    from its predecessor. The default spacing is the largest the default
    gate_sense_range and gate_lookahead allow."""
    centres = [np.asarray(start, dtype=float)]
    for k in range(1, n_gates):
        y = start[1] + (offset if k % 2 else -offset)
        dy = y - centres[-1][1]
        dx = math.sqrt(max(spacing**2 - dy**2, 0.0))
        centres.append(np.array([centres[-1][0] + dx, y]))
    gates = []
    for k, c in enumerate(centres):
        d = c - centres[k - 1] if k > 0 else np.array([1.0, 0.0])
        gates.append(Gate(centre=(float(c[0]), float(c[1])), heading=math.atan2(d[1], d[0]), width=width))
    return gates


def winding_course(turns, spacing, start=(8.0, 1.6), width=4.0):
    """Gates `spacing` apart. The first faces +x; gate k+1 lies at turns[k]
    [rad] from gate k's passing direction and faces away from gate k, so the
    swarm turns by turns[k] between them."""
    centres, headings = [np.asarray(start, dtype=float)], [0.0]
    for turn in turns:
        headings.append(headings[-1] + turn)
        centres.append(centres[-1] + spacing * np.array([math.cos(headings[-1]), math.sin(headings[-1])]))
    return [Gate(centre=(float(c[0]), float(c[1])), heading=h, width=width) for c, h in zip(centres, headings)]


def random_turns(n_gates, max_turn, rng, min_separation=1.5):
    """Turns for an n_gates winding_course, uniform in [-max_turn, max_turn],
    redrawn until no two non-consecutive gate centres are closer than
    min_separation spacings, so the course does not fold back on itself.
    Gates two apart are at most 2 spacings apart (straight on), so
    min_separation must be below 2."""
    if n_gates >= 3 and min_separation >= 2.0:
        raise ValueError(f"min_separation must be below 2 spacings, got {min_separation}")
    while True:
        turns = rng.uniform(-max_turn, max_turn, n_gates - 1)
        c = np.array([g.centre for g in winding_course(turns, 1.0)])
        d = np.linalg.norm(c[:, None] - c[None], axis=2)
        if (d[np.triu_indices(n_gates, 2)] >= min_separation).all():
            return turns
