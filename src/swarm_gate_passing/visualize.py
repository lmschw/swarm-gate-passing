"""CLI: replay a trained (or random) genome for one episode and plot the swarm's
trajectory over its gradient map, as a static image (no video/cv2 dependency).

Usage: python -m swarm_gate_passing.visualize --genome hebbian_results/hebbian_follow_gradient_path_best.npy \\
           --gradient-map maps_gradient/path_sine_curve_freq2.png --output trajectory.png
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np

from . import config
from .hebbian_controller import unflatten_abcd
from .simulation import simulate_hebbian_episode
from .environment import GradientSensor


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--genome", default=None, metavar="PATH",
                         help="Trained genome .npy (default: a fresh random genome).")
    parser.add_argument("--gradient-map", default=None, metavar="PATH")
    parser.add_argument("--seed", type=int, default=config.HEBBIAN_DEFAULT_SEED)
    parser.add_argument("--n-agents", type=int, default=config.HEBBIAN_N_AGENTS)
    parser.add_argument("--wind-grid", type=int, default=None)
    parser.add_argument("--output", default="trajectory.png")
    args = parser.parse_args(argv)

    if args.genome is not None:
        genome = np.load(args.genome)
    else:
        rng = np.random.default_rng(args.seed)
        genome = rng.uniform(*config.HEBBIAN_ABCD_BOUNDS, config.HEBBIAN_N_ABCD)
    rules = unflatten_abcd(genome)

    gradient_sensor = None
    if args.gradient_map is not None:
        world_size_x = config.X_RANGE[1] - config.X_RANGE[0]
        world_size_y = config.Y_RANGE[1] - config.Y_RANGE[0]
        gradient_sensor = GradientSensor.from_png(args.gradient_map, world_size_x, world_size_y,
                                                   noise_magnitude=config.GRADIENT_MAP_NOISE_MAGNITUDE)

    result = simulate_hebbian_episode(
        rules, seed=args.seed, n_agents=args.n_agents, wind_enabled=True,
        nx=args.wind_grid, ny=args.wind_grid, gradient_sensor=gradient_sensor,
        record_trajectory=True)
    dist, batt, ct, wct, coh, prox, path_alignment, telemetry = result
    positions = telemetry["positions"]  # (n_steps, n_agents, 2), arena frame

    fig, ax = plt.subplots(figsize=(8, 8))
    if gradient_sensor is not None:
        extent = [config.X_RANGE[0], config.X_RANGE[1], config.Y_RANGE[0], config.Y_RANGE[1]]
        ax.imshow(gradient_sensor.map, cmap="gray", origin="upper", extent=extent, alpha=0.6)
    for i in range(positions.shape[1]):
        ax.plot(positions[:, i, 0], positions[:, i, 1], linewidth=1)
    ax.scatter(positions[-1, :, 0], positions[-1, :, 1], c="red", s=20, zorder=3, label="final position")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title(f"dist={dist:.2f}  battery={batt:.1f}  path_alignment={path_alignment:.1f}")
    ax.legend(loc="lower right")
    fig.savefig(args.output, dpi=150)
    print(f"Saved {args.output}  (dist_travelled={dist:.3f}, average_batt={batt:.2f}, "
          f"collision_time={ct:.2f}, wall_collision_time={wct:.2f}, path_alignment={path_alignment:.2f})")


if __name__ == "__main__":
    main()
