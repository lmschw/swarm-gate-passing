"""One-off script: render a handful of representative gate_passing episodes as
videos, using a FIXED wavelength/gate placement (rather than the domain
randomization used during training) so the videos are directly comparable to
each other.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from swarm_gate_passing import config
from swarm_gate_passing.hebbian_controller import unflatten_abcd
from swarm_gate_passing.simulation import simulate_hebbian_episode
from swarm_gate_passing.environment import GradientSensor, evenly_spaced_gates, render_path_map
from swarm_gate_passing.render_video import render_trajectory_video

FIXED_FREQ = 4.0
FIXED_GATE_X = -3.5
SEED = 42
OUT_DIR = "videos"

RUNS = [
    dict(name="thymio_n10", genome="hebbian_results_gate/thymio_n10/hebbian_gate_passing_thymio_best.npy",
         sensor_mode="thymio", n_agents=10, n_gates=1,
         title="thymio sensor, n=10, 1 gate (best eff 206.2)"),
    dict(name="quadrant_n10", genome="hebbian_results_gate/quadrant_n10/hebbian_gate_passing_best.npy",
         sensor_mode="quadrant", n_agents=10, n_gates=1,
         title="quadrant (positions) sensor, n=10, 1 gate (best eff 224.0)"),
    dict(name="quadrant_n20", genome="hebbian_results_gate/sweep_n_agents_20/hebbian_gate_passing_best.npy",
         sensor_mode="quadrant", n_agents=20, n_gates=1,
         title="quadrant sensor, n=20, 1 gate (best eff 246.4)"),
    dict(name="quadrant_n10_5gates", genome="hebbian_results_gate/sweep_n_gates_5/hebbian_gate_passing_best.npy",
         sensor_mode="quadrant", n_agents=10, n_gates=5,
         title="quadrant sensor, n=10, 5 gates (best eff 139.5)"),
]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    world_w = config.X_RANGE[1] - config.X_RANGE[0]
    world_h = config.Y_RANGE[1] - config.Y_RANGE[0]

    grid = render_path_map("sine_curve", world_w, world_h,
                            meters_per_pixel=config.GRADIENT_MAP_METERS_PER_PIXEL,
                            path_width_m=config.GRADIENT_DEFAULT_PATH_WIDTH_M,
                            path_kwargs={"freq": FIXED_FREQ})
    sensor = GradientSensor(grid, world_w, world_h, noise_magnitude=config.GRADIENT_MAP_NOISE_MAGNITUDE)

    for run in RUNS:
        print(f"=== rendering {run['name']} ===")
        n_inputs = config.n_inputs_for_sensor_mode(run["sensor_mode"])
        genome = np.load(run["genome"])
        rules = unflatten_abcd(genome, n_inputs=n_inputs)

        gates = evenly_spaced_gates(sensor, run["n_gates"], FIXED_GATE_X, config.GATE_OPENING_WIDTH_M,
                                     config.X_RANGE, config.Y_RANGE)
        finish_x = FIXED_GATE_X - config.GATE_POST_GATE_DISTANCE_M

        result = simulate_hebbian_episode(
            rules, seed=SEED, n_agents=run["n_agents"], wind_enabled=False,
            sensor_mode=run["sensor_mode"], gradient_sensor=sensor, gates=gates, finish_x=finish_x,
            record_trajectory=True, record_battery=True)

        print(f"  dist={result.dist_travelled:.2f} success={bool(result.success)} "
              f"path_dev={result.path_deviation_m:.2f}m speed={result.mean_speed:.3f}m/s "
              f"post_gate_cohesion={result.post_gate_cohesion_dist} n_steps={result.telemetry['positions'].shape[0]}")

        out_path = os.path.join(OUT_DIR, f"{run['name']}.mp4")
        render_trajectory_video(result, out_path, gradient_sensor=sensor, gates=gates, finish_x=finish_x,
                                 max_battery=config.HEBBIAN_MAX_BATTERY, title=run["title"])
        print(f"  saved {out_path}")


if __name__ == "__main__":
    main()
