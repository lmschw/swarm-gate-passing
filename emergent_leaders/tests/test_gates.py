import inspect
import math

import numpy as np
import pytest

from emergent_leaders import (FixedGoal, Gate, ModelParams, Simulation, SwarmState, grid_swarm, random_turns,
                              slalom_course, winding_course)
from emergent_leaders import gates

P = ModelParams()
GATE = Gate(centre=(10.0, 0.0), heading=0.0, width=4.0)


def test_posts_are_width_apart_across_the_passing_direction():
    posts = Gate(centre=(1.0, 2.0), heading=math.pi / 2, width=4.0).posts
    np.testing.assert_allclose(posts, [[3.0, 2.0], [-1.0, 2.0]], atol=1e-12)


def test_next_gate_must_be_in_range_of_the_lookahead_beyond_a_gate():
    r, ahead = P.gate_sense_range, P.gate_lookahead
    # next gate shifted sideways by half its width: one post straight ahead
    gates.check_course([GATE, Gate((10.0 + ahead + r, 2.0), 0.0, 4.0)], r, ahead)
    with pytest.raises(ValueError):
        gates.check_course([GATE, Gate((10.0 + ahead + r + 0.1, 2.0), 0.0, 4.0)], r, ahead)
    # centred: both posts are 2 m to the side of the point, so the gate must be closer
    with pytest.raises(ValueError):
        gates.check_course([GATE, Gate((10.0 + ahead + r, 0.0), 0.0, 4.0)], r, ahead)
    gates.check_course([GATE, Gate((10.0 + ahead + math.sqrt(r**2 - 4.0), 0.0), 0.0, 4.0)], r, ahead)


@pytest.mark.parametrize("r", [3.0, 4.0, 8.0])
def test_largest_slalom_spacing_passes_the_check_and_the_next_does_not(r):
    ahead = P.gate_lookahead
    spacing = gates.max_slalom_spacing(r, ahead, n_gates=5)
    gates.check_course(slalom_course(5, spacing=spacing), r, ahead)
    with pytest.raises(ValueError):
        gates.check_course(slalom_course(5, spacing=spacing + 0.5), r, ahead)


def test_default_slalom_spacing_is_the_largest_the_default_range_allows():
    default = inspect.signature(slalom_course).parameters["spacing"].default
    assert default == gates.max_slalom_spacing(P.gate_sense_range, P.gate_lookahead)


def test_non_consecutive_gates_within_twice_the_sensing_range_are_rejected():
    # a course that bends back: the third gate's posts are 6.3 m from the first gate's.
    # With a 7 m range each next gate is in view, but the third is also in range of the first.
    course = winding_course([math.pi / 2, math.pi / 2], 6.0)
    with pytest.raises(ValueError, match="non-consecutive"):
        gates.check_course(course, 7.0, P.gate_lookahead)
    gates.check_course(course[:2], 7.0, P.gate_lookahead)


def test_winding_course_turns_by_the_given_angles():
    turns = [math.radians(60), math.radians(-30)]
    course = winding_course(turns, 7.0)
    assert [g.heading for g in course] == pytest.approx([0.0, math.radians(60), math.radians(30)])
    centres = np.array([g.centre for g in course])
    np.testing.assert_allclose(np.linalg.norm(np.diff(centres, axis=0), axis=1), 7.0)


def test_random_turns_keep_the_course_from_folding_back():
    rng = np.random.default_rng(0)
    for _ in range(20):
        turns = random_turns(6, math.pi / 2, rng)
        assert np.all(np.abs(turns) <= math.pi / 2)
        c = np.array([g.centre for g in winding_course(turns, 1.0)])
        d = np.linalg.norm(c[:, None] - c[None], axis=2)
        assert d[np.triu_indices(6, 2)].min() >= 1.5


def test_random_turns_reject_a_separation_only_a_straight_course_meets():
    with pytest.raises(ValueError):
        random_turns(3, math.pi / 2, np.random.default_rng(0), min_separation=2.0)


def test_gates_wider_than_twice_the_sensing_range_are_rejected():
    with pytest.raises(ValueError):
        gates.check_course([Gate((0.0, 0.0), 0.0, 2 * P.gate_sense_range)], P.gate_sense_range, 0.0)


def test_agent_sensing_two_gates_aims_at_the_nearer_target():
    nxt = Gate((12.5, 0.0), 0.0, 4.0)   # both gates sensed from both positions
    before, after = np.array([[9.5, 0.0]]), np.array([[10.5, 0.0]])
    assert gates.gate_targets(before, np.zeros(1), [GATE, nxt], P)[0][0] == 0
    assert gates.gate_targets(after, np.zeros(1), [GATE, nxt], P)[0][0] == 1


R = P.gate_sense_range
@pytest.mark.parametrize("pos, heading, gate, posts_seen, target", [
    ((7.0, 0.0), 0.0, 0, 2, (10.0, 0.0)),                  # both posts in range: the centre
    ((7.0, 0.0), math.radians(80), 0, 2, (10.0, 0.0)),     # centre still in the front half
    ((7.0, 0.0), math.pi, -1, 0, None),                     # centre behind
    ((10.0 - R, 2.0), 0.0, 0, 1, (10.0, 2.0)),             # only the nearer post in range: that post
    ((10.0 - R, 2.0), math.pi / 2, 0, 1, (10.0, 2.0)),     # post at the edge of the front half
    ((10.0 - R, 2.0), math.pi, -1, 0, None),                # post behind
    ((10.0, 2.0 + R + 0.1), 0.0, -1, 0, None),             # no post in range
    ((10.5, 2.0 + R - 0.1), 0.0, -1, 0, None),             # passed beside the gate, post behind
])
def test_agents_aim_at_the_centre_or_the_only_sensed_post_if_in_front(pos, heading, gate, posts_seen, target):
    g, n, aim = gates.gate_targets(np.array([pos]), np.array([heading]), [GATE], P)
    assert (g[0], n[0]) == (gate, posts_seen)
    if target is None:
        assert np.isnan(aim[0]).all()
    else:
        np.testing.assert_allclose(aim[0], target)


def test_posts_marked_invisible_are_not_sensed():
    pos, heading = np.array([[7.0, 0.0]]), np.zeros(1)
    one_post = np.array([[[True, False]]])
    g, n, aim = gates.gate_targets(pos, heading, [GATE], P, visible=one_post)
    assert (g[0], n[0]) == (0, 1)
    np.testing.assert_allclose(aim[0], GATE.posts[0])
    assert gates.gate_targets(pos, heading, [GATE], P, visible=np.zeros((1, 1, 2), bool))[0][0] == -1


def test_agents_are_pushed_out_of_posts():
    post = GATE.posts[1]
    pos = np.array([post + [0.05, 0.0], post, post + [1.0, 0.0]])
    out = gates.push_out_of_posts(pos, [GATE], 0.15)
    np.testing.assert_allclose(out, [post + [0.15, 0.0], post + [-0.15, 0.0], post + [1.0, 0.0]])


def test_crossing_counts_only_forward_passages_between_the_posts():
    prev = np.array([[9.9, 0.0], [9.9, 3.0], [10.1, 0.0]])
    now = np.array([[10.1, 0.0], [10.1, 3.0], [9.9, 0.0]])
    np.testing.assert_array_equal(gates.crossed(prev, now, GATE), [True, False, False])


def _single_agent(pos, heading, **kw):
    state = SwarmState.from_poses(np.array([pos]), np.array([heading]))
    return Simulation(state=state, gates=[GATE], explore=False, **kw)


def test_agent_seeing_a_gate_leads_towards_its_centre_until_it_is_behind():
    sim = _single_agent((7.0, 0.5), 0.0)
    assert sim.state.is_leader[0] and sim.state.gate_target[0] == 0 and sim.state.posts_seen[0] == 2
    assert sim.state.goal_heading[0] == pytest.approx(math.atan2(-0.5, 3.0))
    for _ in range(2000):
        sim.step()
        if sim.state.pos[0, 0] > 10.5:
            break
    assert not np.isnan(sim.gate_pass_time[0, 0])
    assert not sim.state.is_leader[0] and sim.state.gate_target[0] == -1
    assert sim.total_promotions == sim.total_demotions == 1


def test_agent_sensing_one_post_heads_for_it_then_for_the_centre():
    sim = _single_agent((10.0 - R + 0.1, 2.5), 0.0)
    assert sim.state.posts_seen[0] == 1
    assert sim.state.goal_heading[0] == pytest.approx(math.atan2(-0.5, R - 0.1))
    for _ in range(2000):
        sim.step()
        if sim.state.posts_seen[0] == 2:
            break
    assert sim.state.goal_heading[0] == pytest.approx(math.atan2(-sim.state.pos[0, 1], 10.0 - sim.state.pos[0, 0]))


def test_agent_driving_at_a_post_does_not_pass_through_it():
    # sense range too short to see the other post: it keeps aiming at this one
    params = ModelParams(gate_sense_range=2.5, post_radius=0.15)
    sim = _single_agent((8.0, 2.0), 0.0, params=params)
    for _ in range(3000):
        sim.step()
        assert np.linalg.norm(sim.state.pos[0] - GATE.posts[1]) >= 0.15 - 1e-9
    assert sim.state.pos[0, 0] < 10.0


def test_without_exploration_or_gates_nobody_leads():
    sim = Simulation(9, explore=False, seed=0)
    h = sim.run(1000, record_every=50).as_arrays()
    assert h["n_leaders"].max() == 0


def test_swarm_facing_a_gate_passes_through_it():
    params = ModelParams(sigma_leader=0.7)
    rng = np.random.default_rng(0)
    sim = Simulation(params=params, state=grid_swarm(25, rng, heading=0.0), gates=slalom_course(1),
                     explore=False, seed=0)
    sim.run(3000, record_every=100)
    assert np.mean(~np.isnan(sim.gate_pass_time[0])) >= 0.9


def test_paused_exploration_has_no_explorers_near_gates_and_resumes_after_the_hold_off():
    params = ModelParams(sigma_leader=0.7, pause_exploration_at_gates=True, exploration_hold_off=20.0)
    far_gate = Gate(centre=(20.0, 1.6), heading=0.0, width=4.0)   # explorers emerge before it is in view
    sim = Simulation(params=params, state=grid_swarm(25, np.random.default_rng(0), heading=0.0),
                     gates=[far_gate], goal=FixedGoal(0.0), seed=0)
    last_target, explorers_before, explorers_after = -np.inf, 0, 0
    for _ in range(12000):
        sim.step()
        s, t = sim.state, sim.step_count * params.dt
        if (s.gate_target >= 0).any():
            last_target = t
        explorers = int((s.is_leader & (s.gate_target < 0)).sum())
        if t - last_target <= 20.0:
            assert explorers == 0
        elif np.isinf(last_target):
            explorers_before += explorers
        else:
            explorers_after += explorers
    assert explorers_before > 0 and explorers_after > 0
    assert sim.total_promotions - sim.total_demotions == int(sim.state.is_leader.sum())


def test_exploration_is_not_paused_by_default():
    sim = Simulation(state=grid_swarm(25, np.random.default_rng(0), heading=0.0), gates=slalom_course(1), seed=0)
    explorers_near_gate = 0
    for _ in range(3000):
        sim.step()
        s = sim.state
        if (s.gate_target >= 0).any():
            explorers_near_gate += int((s.is_leader & (s.gate_target < 0)).sum())
    assert explorers_near_gate > 0
