"""CLI: replay a trained (or random) genome for one episode and plot the swarm's
trajectory over its gradient map (and gate, if any), as a static image (no
video/cv2 dependency).

Usage: python -m swarm_gate_passing.visualize --genome hebbian_results/hebbian_gate_passing_best.npy \\
           --sensor-mode thymio --gate-x -3.5 --path-freq 4.0 --output trajectory.png
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np

from . import config
from .hebbian_controller import unflatten_abcd
from .simulation import simulate_hebbian_episode
from .environment import GradientSensor, Gate, render_path_map


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--genome", default=None, metavar="PATH",
                         help="Trained genome .npy (default: a fresh random genome).")
    parser.add_argument("--sensor-mode", default="quadrant", choices=list(config.SENSOR_MODES))
    parser.add_argument("--gradient-map", default=None, metavar="PATH",
                         help="Fixed PNG map. Mutually exclusive with --path-freq (which renders one).")
    parser.add_argument("--path-freq", type=float, default=None,
                         help="Render a sine-curve map at this frequency instead of loading a fixed PNG.")
    parser.add_argument("--gate-x", type=float, default=None,
                         help="Place a physical gate at this arena-frame x (requires a map).")
    parser.add_argument("--gate-opening-width", type=float, default=config.GATE_OPENING_WIDTH_M)
    parser.add_argument("--seed", type=int, default=config.HEBBIAN_DEFAULT_SEED)
    parser.add_argument("--n-agents", type=int, default=config.HEBBIAN_N_AGENTS)
    parser.add_argument("--wind-grid", type=int, default=None)
    parser.add_argument("--wind-enabled", action="store_true")
    parser.add_argument("--output", default="trajectory.png")
    args = parser.parse_args(argv)

    n_inputs = config.n_inputs_for_sensor_mode(args.sensor_mode)
    n_abcd = config.n_abcd_for(n_inputs)
    if args.genome is not None:
        genome = np.load(args.genome)
    else:
        rng = np.random.default_rng(args.seed)
        genome = rng.uniform(*config.HEBBIAN_ABCD_BOUNDS, n_abcd)
    rules = unflatten_abcd(genome, n_inputs=n_inputs)

    world_size_x = config.X_RANGE[1] - config.X_RANGE[0]
    world_size_y = config.Y_RANGE[1] - config.Y_RANGE[0]
    gradient_sensor = None
    if args.gradient_map is not None:
        gradient_sensor = GradientSensor.from_png(args.gradient_map, world_size_x, world_size_y,
                                                   noise_magnitude=config.GRADIENT_MAP_NOISE_MAGNITUDE)
    elif args.path_freq is not None:
        grid = render_path_map("sine_curve", world_size_x, world_size_y,
                                meters_per_pixel=config.GRADIENT_MAP_METERS_PER_PIXEL,
                                path_width_m=config.GRADIENT_DEFAULT_PATH_WIDTH_M,
                                path_kwargs={"freq": args.path_freq})
        gradient_sensor = GradientSensor(grid, world_size_x, world_size_y,
                                          noise_magnitude=config.GRADIENT_MAP_NOISE_MAGNITUDE)

    gate = None
    finish_x = args.gate_x
    if gradient_sensor is not None and args.gate_x is not None:
        gate = Gate.centered_on_path(gradient_sensor, args.gate_x, args.gate_opening_width,
                                      config.X_RANGE, config.Y_RANGE)

    result = simulate_hebbian_episode(
        rules, seed=args.seed, n_agents=args.n_agents, wind_enabled=args.wind_enabled,
        nx=args.wind_grid, ny=args.wind_grid, sensor_mode=args.sensor_mode,
        gradient_sensor=gradient_sensor, gate=gate, finish_x=finish_x,
        record_trajectory=True)
    positions = result.telemetry["positions"]  # (n_steps, n_agents, 2), arena frame

    fig, ax = plt.subplots(figsize=(8, 8))
    if gradient_sensor is not None:
        extent = [config.X_RANGE[0], config.X_RANGE[1], config.Y_RANGE[0], config.Y_RANGE[1]]
        ax.imshow(gradient_sensor.map, cmap="gray", origin="upper", extent=extent, alpha=0.6)
    if gate is not None:
        ax.plot([gate.x_arena, gate.x_arena], [config.Y_RANGE[0], gate.y_lo_arena], color="red", linewidth=3)
        ax.plot([gate.x_arena, gate.x_arena], [gate.y_hi_arena, config.Y_RANGE[1]], color="red", linewidth=3)
    elif finish_x is not None:
        ax.axvline(finish_x, color="red", linestyle="--", linewidth=1.5, label="finish line")
    for i in range(positions.shape[1]):
        ax.plot(positions[:, i, 0], positions[:, i, 1], linewidth=1)
    ax.scatter(positions[-1, :, 0], positions[-1, :, 1], c="blue", s=20, zorder=3, label="final position")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title(f"dist={result.dist_travelled:.2f}  path_dev={result.path_deviation_m:.2f}m  "
                 f"speed={result.mean_speed:.3f}m/s  success={bool(result.success)}")
    ax.legend(loc="lower right")
    fig.savefig(args.output, dpi=150)
    print(f"Saved {args.output}\n{result}")


if __name__ == "__main__":
    main()
