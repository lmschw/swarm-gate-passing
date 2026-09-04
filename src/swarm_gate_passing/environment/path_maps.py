"""Path / gradient-map generation.

Ported from volcano_gradient's gym_pybullet_drones/examples/gradient/generate_gradients.py
(the `thicken_path` exponential-falloff model and its handful of pixel-space
centerline functions: sine_curve, parabola, exponential, zigzag, circular_arc),
stripped of that project's PyBullet/drone dependencies and its machine-specific
output-path detection, and parameterized in meters instead of hardcoded pixel
counts so maps can be sized to match whatever arena a simulation actually uses.

Convention note: this module renders the path centerline as BRIGHT (high
intensity) with intensity decaying outward -- the opposite of some of
volcano_gradient's own maps, which invert that (dark = path, per that project's
"0 = black = goal region" sensor convention). Documented here so this doesn't
get silently assumed to match that project's per-file pixel convention; see
environment/sensing.py's GradientSensor for how out-of-bounds reads are scored
under this module's bright-on-path convention.
"""
import os

import numpy as np
from PIL import Image


def sine_curve_path(x, width, height, freq=2.0):
    amplitude = height / 4.0
    frequency = freq * np.pi / width
    return amplitude * np.sin(frequency * x) + height / 2.0


def parabola_path(x, width, height):
    return 0.001 * (x - width / 2.0) ** 2 + height / 4.0


def exponential_path(x, width, height):
    nx = x / width
    return height * (1.0 - np.exp(-3.0 * nx))


def zigzag_path(x, width, height):
    period = width / 5.0
    return height / 2.0 + (height / 4.0) * np.abs((x % period) / period * 2.0 - 1.0)


def circular_arc_path(x, width, height):
    center_x, radius = width / 2.0, width / 2.0
    x_offset = x - center_x
    valid = np.abs(x_offset) <= radius
    result = np.full_like(x, height / 2.0, dtype=float)
    result[valid] = height / 2.0 - np.sqrt(radius ** 2 - x_offset[valid] ** 2) * (height / (2.0 * radius))
    return result


PATH_FUNCTIONS = {
    "sine_curve": sine_curve_path,
    "parabola": lambda x, width, height, **_: parabola_path(x, width, height),
    "exponential": lambda x, width, height, **_: exponential_path(x, width, height),
    "zigzag": lambda x, width, height, **_: zigzag_path(x, width, height),
    "circular_arc": lambda x, width, height, **_: circular_arc_path(x, width, height),
}


def get_thickness_for_width(target_width_meters, meters_per_pixel, threshold_fraction=0.6):
    """Inverts intensity(d) = exp(-k*d) to find the decay constant k that puts the
    given physical corridor width at `threshold_fraction` of peak intensity.
    1:1 port of generate_gradients.py's get_thickness_for_width (there hardcoded to
    a 0.04 m/pixel scale and threshold=150/255; generalized here to any scale)."""
    width_pixels = target_width_meters / meters_per_pixel
    d_pixels = width_pixels / 2.0
    return -np.log(1.0 - threshold_fraction) / d_pixels


def render_path_map(path_name, world_width_m, world_height_m, meters_per_pixel=0.05,
                     path_width_m=0.6, path_kwargs=None, bright_on_path=True):
    """Renders one path centerline as a uint8 (height_px, width_px) intensity map.

    path_name: one of PATH_FUNCTIONS' keys. path_width_m: physical corridor width
    (see get_thickness_for_width) at which intensity has fallen to 60% of peak.
    """
    if path_name not in PATH_FUNCTIONS:
        raise ValueError(f"unknown path '{path_name}', choose from {sorted(PATH_FUNCTIONS)}")
    path_kwargs = path_kwargs or {}

    width_px = max(2, int(round(world_width_m / meters_per_pixel)))
    height_px = max(2, int(round(world_height_m / meters_per_pixel)))

    x_cols = np.arange(width_px, dtype=float)
    centerline_px = PATH_FUNCTIONS[path_name](x_cols, width_px, height_px, **path_kwargs)

    y_rows = np.arange(height_px, dtype=float)[:, None]
    distance_px = np.abs(y_rows - centerline_px[None, :])

    k = get_thickness_for_width(path_width_m, meters_per_pixel)
    intensity = np.exp(-k * distance_px)  # 1.0 on centerline, decaying outward
    if not bright_on_path:
        intensity = 1.0 - intensity

    span = intensity.max() - intensity.min()
    normalized = (intensity - intensity.min()) / span if span > 1e-12 else intensity
    return (normalized * 255.0).astype(np.uint8)


def save_map(grid, path):
    Image.fromarray(grid, mode="L").save(path)


def load_map(path):
    return np.array(Image.open(path).convert("L"))


def generate_default_map_set(output_dir, world_width_m=10.0, world_height_m=10.0,
                              meters_per_pixel=0.05, path_width_m=0.6):
    """Writes one representative PNG per path shape into output_dir, sized to a
    given arena -- the combined-project analogue of volcano_gradient's
    maps_gradient/ example set. Returns {name: file_path}."""
    os.makedirs(output_dir, exist_ok=True)
    specs = {
        "sine_curve_freq2": ("sine_curve", {"freq": 2.0}),
        "sine_curve_freq4": ("sine_curve", {"freq": 4.0}),
        "zigzag": ("zigzag", {}),
        "parabola": ("parabola", {}),
        "exponential": ("exponential", {}),
        "circular_arc": ("circular_arc", {}),
    }
    paths = {}
    for name, (shape, kwargs) in specs.items():
        grid = render_path_map(shape, world_width_m, world_height_m, meters_per_pixel,
                                path_width_m=path_width_m, path_kwargs=kwargs)
        out_path = os.path.join(output_dir, f"path_{name}.png")
        save_map(grid, out_path)
        paths[name] = out_path
    return paths
