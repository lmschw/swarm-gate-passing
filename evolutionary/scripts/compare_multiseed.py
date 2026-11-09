"""Post-hoc, honest comparison of multi-seed trained genomes: every genome is
independently re-tested against the SAME 15 held-out random gate placements
(not the noisy training-time fitness numbers), and results are aggregated per
configuration (mean/std across seeds) so genuine differences can be told apart
from run-to-run CMA-ES variance.
"""
import glob
import os

import numpy as np

from swarm_gate_passing import config
from swarm_gate_passing.hebbian_controller import unflatten_abcd
from swarm_gate_passing.simulation import simulate_hebbian_episode
from swarm_gate_passing.environment import random_gate

N_TEST_GATES = 15
CONFIGS = {
    "vision_gate": "hebbian_results_multiseed/vision_gate",
    "vision_gate_curriculum": "hebbian_results_multiseed/vision_gate_curriculum",
}


def evaluate_genome(genome_path, stage_name):
    genome = np.load(genome_path)
    rules = unflatten_abcd(genome, n_inputs=config.n_inputs_for_sensor_mode("vision"))
    successes = 0
    total_excess = 0.0
    total_cohesion = 0.0
    for gate_seed in range(N_TEST_GATES):
        rng = np.random.default_rng(gate_seed)
        gate = random_gate(rng, config.GATE_OPENING_WIDTH_M, config.GATE_RANDOM_X_BOUNDS, config.GATE_RANDOM_Y_BOUNDS)
        result = simulate_hebbian_episode(
            rules, seed=42, n_agents=10, wind_enabled=False, sensor_mode="vision",
            gates=[gate], crossing_success=True, max_steps=config.GATE_MAX_STEPS)
        successes += int(result.success)
        total_excess += result.excess_crossings
        total_cohesion += result.cohesion_dist
    return {
        "success_rate": successes / N_TEST_GATES,
        "successes": successes,
        "avg_excess_crossings": total_excess / N_TEST_GATES,
        "avg_cohesion": total_cohesion / N_TEST_GATES,
    }


def main():
    all_results = {}
    for label, base_dir in CONFIGS.items():
        seed_dirs = sorted(glob.glob(os.path.join(base_dir, "seed_*")))
        results = []
        print(f"\n=== {label} ===")
        for seed_dir in seed_dirs:
            seed = os.path.basename(seed_dir).replace("seed_", "")
            genome_path = os.path.join(seed_dir, f"hebbian_{label}_vision_best.npy")
            if not os.path.exists(genome_path):
                print(f"  seed {seed}: MISSING ({genome_path})")
                continue
            r = evaluate_genome(genome_path, label)
            results.append(r)
            print(f"  seed {seed}: {r['successes']}/{N_TEST_GATES} successes "
                  f"({r['success_rate']*100:.0f}%), avg_cohesion={r['avg_cohesion']:.2f}m, "
                  f"avg_excess_crossings={r['avg_excess_crossings']:.1f}")
        all_results[label] = results
        if results:
            rates = [r["success_rate"] for r in results]
            cohesions = [r["avg_cohesion"] for r in results]
            excess = [r["avg_excess_crossings"] for r in results]
            print(f"  --- {label} summary (n={len(results)} seeds) ---")
            print(f"  success_rate: mean={np.mean(rates)*100:.1f}%  std={np.std(rates)*100:.1f}%  "
                  f"min={min(rates)*100:.0f}%  max={max(rates)*100:.0f}%")
            print(f"  cohesion:     mean={np.mean(cohesions):.2f}m  std={np.std(cohesions):.2f}m")
            print(f"  excess_cross: mean={np.mean(excess):.1f}  std={np.std(excess):.1f}")

    print("\n=== FINAL COMPARISON ===")
    for label, results in all_results.items():
        if not results:
            continue
        rates = [r["success_rate"] for r in results]
        print(f"{label:28s}: mean success rate = {np.mean(rates)*100:5.1f}%  "
              f"(seeds: {[f'{r*100:.0f}%' for r in rates]})")


if __name__ == "__main__":
    main()
