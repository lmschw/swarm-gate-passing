import math

import numpy as np
import pytest
import torch

from emergent_leaders import Gate
from vision_leaders.camera import CameraParams
from vision_leaders.network import GoalNet
from vision_leaders.rl import MISS_MARGIN, Policy, gae, missed_gates

GATES = [Gate(centre=(0.0, 0.0), heading=0.0, width=4.0), Gate(centre=(7.0, 0.0), heading=0.0, width=4.0)]


def test_crossing_a_gate_line_just_outside_the_posts_misses_it():
    prev = np.array([[-0.1, 0.0], [-0.1, 2.5], [-0.1, 2.0 + MISS_MARGIN + 1.0], [0.1, 2.5]])
    pos = prev + [0.2, 0.0]
    passed = np.zeros((2, 4), dtype=bool)
    passed[0, 0] = True   # agent 0 went through (as the simulation records it)
    missed = missed_gates(prev, pos, GATES, passed)
    # through the gate; beside it; far beside it (not this gate's business); already beyond the line
    assert list(missed[0]) == [False, True, False, False]
    assert not missed[1].any()


def test_passing_a_later_gate_misses_the_earlier_ones():
    pos = np.array([[8.0, 0.0], [8.0, 0.0]])
    passed = np.array([[True, False], [True, True]])
    missed = missed_gates(pos, pos, GATES, passed)
    assert list(missed[0]) == [False, True] and not missed[1].any()


def test_gae_with_no_discounting_is_the_return_to_go_minus_the_value():
    rewards = np.array([[1.0], [0.0], [2.0]], dtype=np.float32)
    values = np.array([[0.5], [1.0], [0.0]], dtype=np.float32)
    adv, ret = gae(values, rewards, gamma=1.0, lam=1.0)
    np.testing.assert_allclose(ret[:, 0], [3.0, 2.0, 2.0])
    np.testing.assert_allclose(adv[:, 0], [2.5, 1.0, 2.0])


def test_bearing_and_near_count_only_for_leaders():
    cam = CameraParams()
    policy = Policy(GoalNet(cam, near=True).eval(), bearing_std=0.1)
    images = torch.from_numpy(np.random.default_rng(0).integers(0, 256, (2, 3, cam.height, cam.width), dtype=np.uint8))
    lead = torch.tensor([False, True])
    with torch.no_grad():
        a, _ = policy.log_prob(images, lead, torch.tensor([0.0, 0.0]), torch.tensor([False, False]))
        b, _ = policy.log_prob(images, lead, torch.tensor([1.0, 1.0]), torch.tensor([True, True]))
    assert a[0] == pytest.approx(b[0].item()) and a[1] != pytest.approx(b[1].item())


def test_finer_rewards_pay_for_progress_centred_passes_and_time():
    from vision_leaders.rl import Reward, RewardTracker
    w = Reward(miss_penalty=1.0, team_weight=0.0, progress=1.0, centre=0.5, time=0.01)
    pos = np.array([[-3.5, 0.0], [-3.5, 1.0]])
    tracker = RewardTracker(GATES, pos, w)
    none = np.zeros((2, 2), dtype=bool)
    offset = np.full((2, 2), np.nan)
    # both move 3.5 m forwards: agent 0 straight at gate 0's centre, agent 1 1 m beside it
    r = tracker.interval(none, none, offset, pos + [3.0, 0.0])
    np.testing.assert_allclose(r, [3.0 / 7 - 0.01, (math.hypot(3.5, 1.0) - math.hypot(0.5, 1.0)) / 7 - 0.01])
    # both pass gate 0, agent 0 at its centre, agent 1 halfway to a post; progress is still towards gate 0
    passed = np.array([[True, True], [False, False]])
    offset[0] = [0.0, 0.5]
    r = tracker.interval(passed, none, offset, pos + [3.5, 0.0])
    np.testing.assert_allclose(r, [1 + 0.5 + 0.5 / 7 - 0.01,
                                   1 + 0.25 + (math.hypot(0.5, 1.0) - 1.0) / 7 - 0.01])
    # now aiming at gate 1, 7 m on; at the end gate 1 is neither passed nor missed
    np.testing.assert_allclose(tracker.target, [1, 1])
    np.testing.assert_allclose(tracker.final(), [-1.0, -1.0])
