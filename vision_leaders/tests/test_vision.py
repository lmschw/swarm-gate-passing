import math

import numpy as np
import pytest
import torch

from emergent_leaders import Gate, ModelParams, SwarmState
from vision_leaders.camera import AGENT, GROUND, POST, SKY, CameraParams, render
from vision_leaders.network import GoalNet
from vision_leaders.simulation import VisionSimulation

CAM = CameraParams()
GATE = Gate(centre=(10.0, 0.0), heading=0.0, width=4.0)


def _pixel(colour):
    return np.round(255 * colour).astype(np.uint8)


def _one_post(post, pos=(0.0, 0.0), heading=0.0, others=(), cam=CAM):
    """Render agent 0's view of a single post (a gate of width ~0 at post)."""
    gate = Gate(centre=post, heading=0.0, width=1e-9)
    pos = np.array([pos, *others], dtype=float)
    images, vis = render(pos, np.full(len(pos), heading), [gate], 0.15, cam)
    return images[0], vis[0, 0].max()


def test_an_agent_alone_sees_only_sky_and_ground():
    images, _ = render(np.zeros((1, 2)), np.zeros(1), [], 0.15, CAM)
    rows = images[0].transpose(1, 2, 0)
    assert (rows[0] == _pixel(SKY)).all() and (rows[-1] == _pixel(GROUND)).all()


@pytest.mark.parametrize("bearing, side", [(0.0, "centre"), (math.radians(40), "left"), (math.radians(-40), "right")])
def test_a_post_appears_at_its_bearing(bearing, side):
    img, vis = _one_post((3.0 * math.cos(bearing), 3.0 * math.sin(bearing)))
    orange = (img.transpose(1, 2, 0) == _pixel(POST)).all(axis=2).any(axis=0)
    columns = np.flatnonzero(orange)
    centre = (CAM.width - 1) / 2
    expected = centre - bearing / (CAM.fov / CAM.width)
    assert vis == pytest.approx(1.0)
    assert abs(columns.mean() - expected) <= 1.0, side


def test_posts_outside_the_field_of_view_are_invisible():
    assert _one_post((-3.0, 0.0))[1] == 0.0
    narrow = CameraParams(width=48, fov=math.radians(120))
    assert _one_post((0.0, 3.0), cam=narrow)[1] == 0.0      # 90 deg to the left
    assert _one_post((3.0, 0.0), cam=narrow)[1] > 0.5


def test_far_posts_fade():
    vis = [_one_post((d, 0.0))[1] for d in (4.0, 8.0, 16.0, 32.0)]
    assert vis[0] > 0.5   # less than 1 when the post straddles two columns
    assert np.all(np.diff(vis[1:]) < 0) and vis[-1] < 0.3


def test_a_nearer_agent_hides_the_lower_part_of_a_post_behind_it():
    img, vis = _one_post((5.0, 0.0), others=[(0.6, 0.0)])
    column = img[:, :, CAM.width // 2].T
    horizon = int(CAM.horizon_row)
    assert (column[horizon] == _pixel(AGENT)).all()
    agent_top = CAM.horizon_row - CAM.focal * (CAM.agent_height - CAM.mount_height) / 0.6
    above = column[:int(agent_top)].astype(int)
    assert (above[:, 0] - above[:, 2] > 100).any()   # orange: the post's top shows above the agent
    assert vis > 0.5


def test_teacher_aims_at_the_centre_of_a_gate_the_camera_shows():
    state = SwarmState.from_poses(np.array([[7.0, 0.5]]), np.zeros(1))
    sim = VisionSimulation(state=state, gates=[GATE], explore=False)
    assert sim.teacher_lead[0] and sim.state.posts_seen[0] == 2
    assert sim.teacher_bearing[0] == pytest.approx(math.atan2(-0.5, 3.0))


@pytest.mark.parametrize("teacher, leads", [("full", True), ("camera", False)])
def test_only_the_camera_teacher_ignores_posts_the_camera_does_not_show(teacher, leads):
    # close to the gate, both posts are outside a +/- 60 deg field of view
    narrow = CameraParams(width=48, fov=math.radians(120))
    state = SwarmState.from_poses(np.array([[9.5, 0.0]]), np.zeros(1))
    sim = VisionSimulation(state=state, gates=[GATE], explore=False, camera=narrow, teacher=teacher)
    assert sim.teacher_lead[0] == leads


def test_network_outputs_a_probability_and_a_bearing_per_image(tmp_path):
    net = GoalNet(CAM).eval()
    images = np.random.default_rng(0).integers(0, 256, (5, 3, CAM.height, CAM.width), dtype=np.uint8)
    prob, bearing, near = net.predict(images)
    assert near is None
    assert prob.shape == bearing.shape == (5,)
    assert ((prob >= 0) & (prob <= 1)).all() and (np.abs(bearing) <= math.pi).all()
    net.save(tmp_path / "net.pt")
    np.testing.assert_allclose(GoalNet.load(tmp_path / "net.pt").predict(images)[1], bearing, rtol=1e-6)


def test_network_with_memory_and_near_output_round_trips(tmp_path):
    net = GoalNet(CAM, memory=True, near=True).eval()
    images = np.random.default_rng(0).integers(0, 256, (4, 3, CAM.height, CAM.width), dtype=np.uint8)
    prev = np.array([[1.0, 0.5, 0.866], [0.0, 0.0, 0.0]] * 2, dtype=np.float32)
    prob, bearing, near = net.predict(images, prev)
    assert prob.shape == bearing.shape == near.shape == (4,)
    zeroed = net.predict(images, np.zeros_like(prev))
    assert not np.allclose(zeroed[1], bearing)   # the previous output steers the bearing...
    np.testing.assert_array_equal(zeroed[0], prob)   # ...but not whether to lead (no copying)
    np.testing.assert_array_equal(zeroed[2], near)
    net.save(tmp_path / "net.pt")
    loaded = GoalNet.load(tmp_path / "net.pt")
    assert loaded.memory and loaded.near
    np.testing.assert_allclose(loaded.predict(images, prev)[2], near, rtol=1e-6)


def test_a_larger_network_round_trips_with_its_size(tmp_path):
    net = GoalNet(CAM, near=True, channels=(32, 64, 64), hidden=128).eval()
    images = np.random.default_rng(0).integers(0, 256, (4, 3, CAM.height, CAM.width), dtype=np.uint8)
    net.save(tmp_path / "net.pt")
    loaded = GoalNet.load(tmp_path / "net.pt")
    assert loaded.config == net.config
    np.testing.assert_allclose(loaded.predict(images)[1], net.predict(images)[1], rtol=1e-6)


def test_training_data_round_trips(tmp_path):
    from vision_leaders.train import load_data, save_data
    rng = np.random.default_rng(0)
    data = (rng.integers(0, 256, (5, 3, CAM.height, CAM.width), dtype=np.uint8), rng.random((5, 3), dtype=np.float32),
            rng.random(5) < 0.5, rng.random(5, dtype=np.float32), rng.random(5) < 0.5)
    save_data(tmp_path / "data", data, next_seed=70)
    loaded, seed = load_data(tmp_path / "data")
    assert seed == 70
    for a, b in zip(data, loaded):
        np.testing.assert_array_equal(a, b)


def test_previous_output_input_is_the_teachers_last_label_turned_with_the_agent():
    params = ModelParams(gate_sticky=True, gate_loose_radius=3.5)
    state = SwarmState.from_poses(np.array([[7.0, 0.0], [-10.0, 0.0]]), np.zeros(2))
    sim = VisionSimulation(params=params, state=state, gates=[GATE], explore=False, check_course=False)
    lead, bearing = sim.teacher_lead.copy(), sim.teacher_bearing.copy()
    assert list(lead) == [True, False] and list(sim.teacher_near) == [True, False]
    sim.state.heading[:] = 0.2   # the agents turn before the next perception
    sim.perceive_gates()
    np.testing.assert_allclose(sim.prev_input[0], [1.0, math.sin(bearing[0] - 0.2), math.cos(bearing[0] - 0.2)],
                               rtol=1e-6)
    np.testing.assert_allclose(sim.prev_input[1], 0.0)


def test_loose_radius_needs_a_network_with_a_near_output():
    with pytest.raises(ValueError):
        VisionSimulation(params=ModelParams(gate_loose_radius=3.5), gates=[GATE], explore=False,
                         net=GoalNet(CAM).eval(), check_course=False)


class _AlwaysLeft(torch.nn.Module):
    """Stands in for GoalNet: always a gate, 30 deg to the left."""
    def predict(self, images, prev=None):
        return np.ones(len(images)), np.full(len(images), math.radians(30)), None


def test_network_leaders_aim_along_the_predicted_bearing_at_their_next_gate():
    state = SwarmState.from_poses(np.array([[0.0, 0.0], [1.0, 0.0]]), np.array([0.0, 1.0]))
    sim = VisionSimulation(state=state, gates=[GATE, Gate((20.0, 0.0), 0.0, 4.0)], explore=False,
                           net=_AlwaysLeft(), check_course=False)
    assert sim.state.is_leader.all() and (sim.state.gate_target == 0).all()
    np.testing.assert_allclose(sim.state.goal_heading, [math.radians(30), 1.0 + math.radians(30)])


class _UnsureLeft(torch.nn.Module):
    """Stands in for GoalNet: 30% sure of a gate, 30 deg to the left."""
    def predict(self, images, prev=None):
        return np.full(len(images), 0.3), np.full(len(images), math.radians(30)), None


@pytest.mark.parametrize("weighting, weight", [("threshold", 0.0), ("confidence", 0.3)])
def test_confidence_weighting_pulls_every_agent_by_the_network_confidence(weighting, weight):
    state = SwarmState.from_poses(np.array([[0.0, 0.0], [10.0, 0.0]]), np.array([0.0, 1.0]))
    sim = VisionSimulation(state=state, gates=[GATE], explore=False, net=_UnsureLeft(), goal_weighting=weighting,
                           check_course=False)
    assert not sim.state.is_leader.any()
    gamma, goal = sim.goal_terms()
    np.testing.assert_allclose(gamma, weight)
    if weighting == "confidence":
        np.testing.assert_allclose(goal, [math.radians(30), 1.0 + math.radians(30)])


def test_confidence_weighting_needs_a_network():
    with pytest.raises(ValueError):
        VisionSimulation(gates=[GATE], explore=False, goal_weighting="confidence")
