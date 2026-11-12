import math

import numpy as np
import pytest

from emergent_leaders import FixedGoal, ModelParams, RandomWalkGoal, Simulation, SwarmState
from emergent_leaders import leadership, motion

P = ModelParams()


def _lattice(n_side, spacing, heading=0.0):
    xx, yy = np.meshgrid(np.arange(n_side) * spacing, np.arange(n_side) * spacing)
    pos = np.column_stack([xx.ravel(), yy.ravel()])
    return pos, np.full(len(pos), heading)


# --- motion ---------------------------------------------------------------

@pytest.mark.parametrize("d, sign", [(P.desired_distance, 0), (1.5, 1), (0.8, -1)])
def test_proximal_force_vanishes_at_desired_distance_and_attracts_beyond(d, sign):
    pos = np.array([[0.0, 0.0], [d, 0.0]])
    heading = np.zeros(2)
    dist, bearing = motion.neighbour_geometry(pos, heading, P.sense_range)
    p_x, p_y = motion.proximal_control(dist, bearing, np.full(2, P.sigma), P.epsilon)
    assert np.sign(round(p_x[0], 10)) == sign   # agent 0 pulled towards (+) or pushed away from (-) agent 1
    assert p_y[0] == pytest.approx(0.0, abs=1e-12)


def test_neighbours_outside_sensing_range_exert_no_force():
    pos = np.array([[0.0, 0.0], [P.sense_range + 0.01, 0.0]])
    dist, bearing = motion.neighbour_geometry(pos, np.zeros(2), P.sense_range)
    p_x, p_y = motion.proximal_control(dist, bearing, np.full(2, P.sigma), P.epsilon)
    assert np.all(p_x == 0) and np.all(p_y == 0)


def test_bearing_is_in_body_frame():
    pos = np.array([[0.0, 0.0], [0.0, 1.0]])
    _, bearing = motion.neighbour_geometry(pos, np.array([np.pi / 2, 0.0]), P.sense_range)
    assert bearing[0, 1] == pytest.approx(0.0)       # straight ahead of agent 0
    assert bearing[1, 0] == pytest.approx(-np.pi / 2)  # to the right of agent 1


def test_heading_alignment_is_unnormalised_sum_in_body_frame():
    heading = np.zeros(4)
    h_x, h_y = motion.heading_alignment(heading)
    np.testing.assert_allclose(h_x, 4.0)
    np.testing.assert_allclose(h_y, 0.0, atol=1e-12)
    h_x, h_y = motion.heading_alignment(np.array([0.0, 0.0, 0.0, np.pi / 2]))
    assert h_y[3] < 0 < h_y[0]   # the odd one out turns right, the others turn left towards it


@pytest.mark.parametrize("local, turns", [(True, False), (False, True)])
def test_local_alignment_ignores_agents_beyond_sense_range(local, turns):
    pos = np.array([[0.0, 0.0], [10.0, 0.0]])
    heading = np.array([0.0, np.pi / 2])
    params = ModelParams(local_alignment=local)
    _, f_y = motion.virtual_force(pos, heading, np.zeros(2, dtype=bool), np.zeros(2), params)
    assert (f_y[0] > 1e-9) == turns   # agent 0 turns towards agent 1's heading only when aligning globally


def test_local_alignment_includes_sensed_neighbours():
    pos = np.array([[0.0, 0.0], [1.0, 0.0]])
    heading = np.array([0.0, np.pi / 2])
    _, f_y = motion.virtual_force(pos, heading, np.zeros(2, dtype=bool), np.zeros(2), P)
    assert f_y[0] > 0 > f_y[1]


def test_goal_term_only_drives_leaders():
    pos = np.array([[0.0, 0.0], [10.0, 0.0]])  # out of range: only alignment + goal act
    heading = np.zeros(2)
    goal = np.full(2, np.pi / 2)
    f_x, f_y = motion.virtual_force(pos, heading, np.array([True, False]), goal, P)
    assert f_y[0] == pytest.approx(P.goal_gain)
    assert f_y[1] == pytest.approx(0.0, abs=1e-12)


def test_goal_weight_scales_the_goal_term_for_any_agent():
    pos = np.array([[0.0, 0.0], [10.0, 0.0]])
    goal = np.full(2, np.pi / 2)
    f_x, f_y = motion.virtual_force(pos, np.zeros(2), np.array([True, False]), goal, P,
                                    goal_weight=np.array([0.25, 0.5]))
    np.testing.assert_allclose(f_y, [0.25 * P.goal_gain, 0.5 * P.goal_gain])


def test_velocity_commands_are_clipped():
    u, w = motion.velocity_commands(np.array([-100.0, 0.0, 100.0]), np.array([-1e6, 0.0, 1e6]), P)
    np.testing.assert_allclose(u, [0.0, P.u_c, P.u_max])
    np.testing.assert_allclose(w, [-P.w_max, 0.0, P.w_max])


def test_integrate_moves_along_heading_and_wraps():
    pos, heading = motion.integrate(np.zeros((1, 2)), np.array([np.pi - 0.01]), np.array([1.0]),
                                    np.array([1.0]), 0.05)
    assert pos[0, 0] < 0 and pos[0, 1] > 0
    assert -np.pi <= heading[0] < 0


# --- leadership -----------------------------------------------------------

def test_stability_is_one_for_aligned_agents_at_desired_distance():
    pos = np.array([[0.0, 0.0], [P.desired_distance, 0.0]])
    dist, _ = motion.neighbour_geometry(pos, np.zeros(2), P.sense_range)
    np.testing.assert_allclose(leadership.stability(dist, np.zeros(2), P), 1.0)


def test_isolated_agent_has_only_alignment_stability_and_is_never_promoted():
    dist, _ = motion.neighbour_geometry(np.zeros((1, 2)), np.zeros(1), P.sense_range)
    s = leadership.stability(dist, np.zeros(1), P)
    assert s[0] == pytest.approx(P.w_head)
    assert leadership.promotion_probability(s, P)[0] < 1e-20


def test_promotion_probability_is_monotone_and_bounded():
    s = np.linspace(0, 1, 101)
    prob = leadership.promotion_probability(s, P)
    assert np.all(np.diff(prob) >= 0)
    assert prob.max() <= P.p_prom_max
    assert leadership.promotion_probability(P.b_prom, P) == pytest.approx(P.p_prom_max / (1 + P.p_prom_max))


def test_quadrant_coverage_separates_inner_and_outer_agents():
    pos, heading = _lattice(3, 1.0)
    dist, bearing = motion.neighbour_geometry(pos, heading, P.sense_range)
    covered = leadership.full_quadrant_coverage(dist, bearing)
    assert covered[4]                       # centre of the 3x3 lattice
    assert not covered[[0, 2, 6, 8]].any()  # corners
    eligible = leadership.promotion_eligible(covered, "outer")
    assert not eligible[4] and eligible[0]


# --- goals ----------------------------------------------------------------

@pytest.mark.parametrize("walk", ["brown", "levy", "correlated"])
def test_random_walk_legs_respect_length_bounds(walk):
    rng = np.random.default_rng(0)
    goal = RandomWalkGoal(walk)
    lengths, headings = goal.assign(None, np.arange(2000), rng)
    assert lengths.min() >= goal.min_length and lengths.max() <= goal.max_length
    assert np.all(np.abs(headings) <= np.pi)


def test_correlated_walk_headings_concentrate_around_zero():
    rng = np.random.default_rng(0)
    mean_cos = {w: np.cos(RandomWalkGoal(w).sample_headings(5000, rng)).mean() for w in ("levy", "correlated")}
    assert abs(mean_cos["levy"]) < 0.05
    assert mean_cos["correlated"] == pytest.approx(0.7, abs=0.05)   # E[cos] of wrapped Cauchy = rho


# --- simulation -----------------------------------------------------------

def test_simulation_is_reproducible_with_seed():
    a = Simulation(9, seed=7).run(400).as_arrays()
    b = Simulation(9, seed=7).run(400).as_arrays()
    np.testing.assert_array_equal(a["pos"], b["pos"])
    np.testing.assert_array_equal(a["is_leader"], b["is_leader"])


def test_leaders_emerge_and_swarm_stays_cohesive():
    sim = Simulation(25, seed=1)
    h = sim.run(4000, record_every=10).as_arrays()
    assert sim.total_promotions > 0
    assert h["n_leaders"][-1] == sim.total_promotions - sim.total_demotions
    spread = np.linalg.norm(h["pos"] - h["centroid"][:, None], axis=2).max(axis=1)
    assert spread.max() < 4.0


def test_leader_is_demoted_after_its_tenure_of_legs():
    params = ModelParams(p_prom_max=1e-12)   # no spontaneous promotions
    pos, heading = _lattice(1, 1.0)
    sim = Simulation(params=params, goal=FixedGoal(0.0, leg_length=0.5), state=SwarmState.from_poses(pos, heading))
    sim.state.is_leader[0] = True
    sim._start_legs(np.array([0]))
    legs_seen, steps = 0, 0
    while sim.state.is_leader[0]:
        before = sim.state.legs_done[0]
        sim.step()
        legs_seen += sim.state.legs_done[0] > before
        steps += 1
    assert legs_seen == params.legs_per_tenure - 1   # the last completion demotes instead of counting
    assert sim.total_demotions == 1
    travelled = steps * params.u_max * params.dt
    assert travelled == pytest.approx(params.legs_per_tenure * 0.5, abs=params.u_max * params.dt * 2)


def test_emergent_leaders_steer_the_swarm_towards_the_goal_heading():
    goal_heading = math.radians(60)
    sim = Simulation(25, goal=FixedGoal(goal_heading), seed=2)
    c = sim.run(10000, record_every=50).as_arrays()["centroid"]
    disp = c[-1] - c[0]
    assert np.linalg.norm(disp) > 20.0
    assert math.atan2(disp[1], disp[0]) == pytest.approx(goal_heading, abs=math.radians(15))
