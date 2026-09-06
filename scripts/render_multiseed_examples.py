"""Render representative videos from the multi-seed comparison: for the best
seed of each config (vision_gate vs vision_gate_curriculum), find one held-out
gate_seed that succeeds and one that fails, and render both -- so the numbers
in compare_multiseed.py can be checked against actual behavior, not just
trusted as reported.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from swarm_gate_passing import config
from swarm_gate_passing.hebbian_controller import unflatten_abcd
from swarm_gate_passing.simulation import simulate_hebbian_episode
from swarm_gate_passing.environment import random_gate
from swarm_gate_passing.render_video import render_trajectory_video

N_TEST_GATES = 15
OUT_DIR = "videos"

RUNS = [
    dict(name="multiseed_vision_gate_s123",
         genome="hebbian_results_multiseed/vision_gate/seed_123/hebbian_vision_gate_vision_best.npy",
         label="vision_gate seed=123 (47% post-hoc)"),
    dict(name="multiseed_curriculum_s123",
         genome="hebbian_results_multiseed/vision_gate_curriculum/seed_123/hebbian_vision_gate_curriculum_vision_best.npy",
         label="vision_gate_curriculum seed=123 (60% post-hoc, best overall)"),
]


def run_episode(rules, gate_seed):
    rng = np.random.default_rng(gate_seed)
    gate = random_gate(rng, config.GATE_OPENING_WIDTH_M, config.GATE_RANDOM_X_BOUNDS, config.GATE_RANDOM_Y_BOUNDS)
    result = simulate_hebbian_episode(
        rules, seed=42, n_agents=10, wind_enabled=False, sensor_mode="vision",
        gates=[gate], crossing_success=True, max_steps=config.GATE_MAX_STEPS,
        record_trajectory=True, record_battery=True)
    return result, [gate]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for run in RUNS:
        print(f"=== {run['name']} ===")
        genome = np.load(run["genome"])
        rules = unflatten_abcd(genome, n_inputs=config.n_inputs_for_sensor_mode("vision"))

        success_case = None
        fail_case = None
        for gate_seed in range(N_TEST_GATES):
            result, gates = run_episode(rules, gate_seed)
            print(f"  gate_seed={gate_seed}: success={bool(result.success)} "
                  f"excess_crossings={result.excess_crossings:.1f} cohesion={result.cohesion_dist:.2f}m")
            if result.success and success_case is None:
                success_case = (gate_seed, result, gates)
            if not result.success and fail_case is None:
                fail_case = (gate_seed, result, gates)
            if success_case and fail_case:
                break

        for tag, case in (("success", success_case), ("fail", fail_case)):
            if case is None:
                print(f"  no {tag} case found in first pass")
                continue
            gate_seed, result, gates = case
            out_path = os.path.join(OUT_DIR, f"{run['name']}_{tag}_gs{gate_seed}.mp4")
            render_trajectory_video(result, out_path, gates=gates,
                                     max_battery=config.HEBBIAN_MAX_BATTERY,
                                     title=f"{run['label']} [{tag}, gate_seed={gate_seed}]")
            print(f"  saved {out_path}")


if __name__ == "__main__":
    main()
