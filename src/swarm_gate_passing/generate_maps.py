"""CLI: generate the default set of gradient/path maps sized to this project's
flocking arena (config.X_RANGE x config.Y_RANGE).

Usage: python -m swarm_gate_passing.generate_maps [--output-dir maps_gradient]
"""
import argparse

from . import config
from .environment import generate_default_map_set


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="maps_gradient")
    parser.add_argument("--meters-per-pixel", type=float, default=config.GRADIENT_MAP_METERS_PER_PIXEL)
    parser.add_argument("--path-width-m", type=float, default=config.GRADIENT_DEFAULT_PATH_WIDTH_M)
    args = parser.parse_args(argv)

    world_width = config.X_RANGE[1] - config.X_RANGE[0]
    world_height = config.Y_RANGE[1] - config.Y_RANGE[0]
    paths = generate_default_map_set(args.output_dir, world_width_m=world_width, world_height_m=world_height,
                                      meters_per_pixel=args.meters_per_pixel, path_width_m=args.path_width_m)
    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
