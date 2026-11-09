"""Render representative videos of the trained 2-gate-sequential genome against
held-out random gate placements, to see what's actually happening behind the
0/15 post-hoc success rate (compare_multiseed-style methodology, adapted for
2 sequential gates instead of 1).
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

OUT_DIR = "videos"
GENOME = "hebbian_results_2gate/vision_gate_2g_seed42/hebbian_vision_gate_vision_best.npy"
CASES = [1, 6]  # gate_seed=1: quiet (0 excess crossings), gate_seed=6: busy (23 excess crossings)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    genome = np.load(GENOME)
    rules = unflatten_abcd(genome, n_inputs=config.n_inputs_for_sensor_mode("vision"))

    for gate_seed in CASES:
        rng = np.random.default_rng(gate_seed)
        gates = [random_gate(rng, config.GATE_OPENING_WIDTH_M, config.GATE_RANDOM_X_BOUNDS, config.GATE_RANDOM_Y_BOUNDS)
                 for _ in range(2)]
        result = simulate_hebbian_episode(
            rules, seed=42, n_agents=10, wind_enabled=False, sensor_mode="vision",
            gates=gates, crossing_success=True, max_steps=config.GATE_MAX_STEPS,
            record_trajectory=True, record_battery=True)
        print(f"gate_seed={gate_seed}: success={bool(result.success)} "
              f"excess_crossings={result.excess_crossings:.1f} cohesion={result.cohesion_dist:.2f}m "
              f"n_steps={result.telemetry['positions'].shape[0]}")
        out_path = os.path.join(OUT_DIR, f"2gate_seed42_gs{gate_seed}.mp4")
        render_trajectory_video(result, out_path, gates=gates,
                                 max_battery=config.HEBBIAN_MAX_BATTERY,
                                 title=f"2-gate sequential [gate_seed={gate_seed}]")
        print(f"  saved {out_path}")


if __name__ == "__main__":
    main()
