import numpy as np
import pytest

from emergent_leaders import Gate, ModelParams, slalom_course
from emergent_leaders.gates import gate_targets

from vision_leaders.aiming import Aim, AimSimulation, aim_targets
from vision_leaders.courses import start

P = ModelParams(lone_post_aim="centre")
GATE = Gate(centre=(10.0, 0.0), heading=0.0, width=4.0)


def test_default_aim_is_the_centre_aiming_rule():
    rng = np.random.default_rng(0)
    pos = rng.uniform([0, -6], [30, 8], (500, 2))
    heading = rng.uniform(-np.pi, np.pi, 500)
    gates = slalom_course(3)
    for a, b in zip(aim_targets(pos, heading, gates, P, Aim()), gate_targets(pos, heading, gates, P)):
        np.testing.assert_array_equal(a, b)


def test_lateral_and_through_move_the_aim_point():
    pos = np.array([[8.0, 1.0]])
    _, _, target = aim_targets(pos, np.zeros(1), [GATE], P, Aim(lateral=0.5, through=1.0))
    np.testing.assert_allclose(target[0], [11.0, -0.5])


def test_next_margin_switches_to_the_later_gate_earlier():
    gates = [GATE, Gate(centre=(12.5, 0.0), heading=0.0, width=4.0)]
    pos = np.array([[9.5, 0.0]])   # 0.5 m before gate 0, 3 m before gate 1
    assert aim_targets(pos, np.zeros(1), gates, P, Aim())[0][0] == 0
    assert aim_targets(pos, np.zeros(1), gates, P, Aim(next_margin=2.0))[0][0] == 0
    assert aim_targets(pos, np.zeros(1), gates, P, Aim(next_margin=3.0))[0][0] == 1


def test_aim_round_trips_through_json(tmp_path):
    aim = Aim(lateral=0.3, through=-1.0, next_margin=2.0, goal_gain=6.0, beta=0.4)
    aim.save(tmp_path / "aim.json")
    assert Aim.load(tmp_path / "aim.json") == aim
    assert aim.model_params().goal_gain == 6.0


def test_aim_simulation_runs():
    aim = Aim(lateral=0.2)
    p = aim.model_params()
    sim = AimSimulation(params=p, state=start(0), gates=slalom_course(3, spacing=7.0), explore=False, seed=0,
                        check_course=False, aim=aim)
    for _ in range(600):
        sim.step()
    assert (sim.state.gate_target >= 0).any()


def test_vision_simulation_labels_with_the_tuned_teacher():
    from vision_leaders.simulation import VisionSimulation
    pos, heading = np.array([[8.0, 1.0]]), np.zeros(1)
    from emergent_leaders import SwarmState
    aim = Aim(lateral=0.5, through=1.0)
    sim = VisionSimulation(params=aim.model_params(), state=SwarmState.from_poses(pos, heading), gates=[GATE],
                           explore=False, check_course=False, aim=aim)
    assert sim.teacher_bearing[0] == pytest.approx(np.arctan2(-1.5, 3.0))
