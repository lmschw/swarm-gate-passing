"""Light-intensity gradient sensing.

Ported from volcano_gradient's read_light_intensity (gym_pybullet_drones/examples/
flocking/2d_flocking_with_real_utils.py): a simulated light-intensity sensor that
reads a gradient-map image at an agent's position, with additive uniform noise.

Refactored out of that function's module-level globals (gradient_map, WORLD_SIZE_X,
WORLD_SIZE_Y) into a small reusable class, and simplified to this project's own
coordinate convention: (0, 0) is the map's bottom-left corner, covering
[0, world_size_x] x [0, world_size_y] in meters, +x right / +y up. The source
function's x<->y axis swap (an inherited quirk from the original dm_ds_v2.py
research code, per volcano_gradient's own GRADIENT_FOLLOWING_STRATEGY.md) is
deliberately NOT carried over here, since this is a fresh mapping rather than a
literal port -- callers (simulation.py) are responsible for translating whatever
arena coordinate system they use into this frame before calling read().
"""
import numpy as np

from .path_maps import load_map


class GradientSensor:
    def __init__(self, gradient_map, world_size_x, world_size_y, noise_magnitude=0.5):
        self.map = np.asarray(gradient_map, dtype=float)
        self.world_size_x = world_size_x
        self.world_size_y = world_size_y
        self.noise_magnitude = noise_magnitude
        self._rows, self._cols = self.map.shape
        # Precomputed once per map: the row (bright_on_path) / (dark_on_path) index
        # of each column's most path-like pixel -- i.e. the rendered path's actual
        # centerline, read directly off the map rather than re-derived from
        # whatever path-function parameters generated it, so it's exact for any
        # map (including hand-edited or non-formulaic ones). Used by centerline_y().
        self._centerline_row_bright = np.argmax(self.map, axis=0)
        self._centerline_row_dark = np.argmin(self.map, axis=0)

    @classmethod
    def from_png(cls, path, world_size_x, world_size_y, **kwargs):
        return cls(load_map(path), world_size_x, world_size_y, **kwargs)

    def centerline_y(self, x, bright_on_path=True):
        """Vectorized: for each world-frame x (scalar or array), returns the
        world-frame y (meters, bottom-up) of the path centerline at that x."""
        x = np.asarray(x, dtype=float)
        col = np.clip(np.round(x / self.world_size_x * self._cols).astype(int), 0, self._cols - 1)
        row = (self._centerline_row_bright if bright_on_path else self._centerline_row_dark)[col]
        return self.world_size_y * (1.0 - (row + 0.5) / self._rows)

    def read(self, x, y, add_noise=True):
        """x, y: scalar or array of world-frame coordinates (meters). Returns
        intensity in [0, 255] (plus additive noise if enabled), one value per
        position, matching volcano_gradient's g_noise_mag=0.5 uniform noise model.
        Positions outside the map read as 0 (darkest / off-path under this
        module's bright-on-path convention -- the "no useful signal" case)."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        col = np.floor(x / self.world_size_x * self._cols).astype(int)
        row = np.floor((self.world_size_y - y) / self.world_size_y * self._rows).astype(int)

        out_of_bounds = (row < 0) | (row >= self._rows) | (col < 0) | (col >= self._cols)
        row_c = np.clip(row, 0, self._rows - 1)
        col_c = np.clip(col, 0, self._cols - 1)

        values = np.where(out_of_bounds, 0.0, self.map[row_c, col_c])

        if add_noise:
            noise = np.random.uniform(-self.noise_magnitude, self.noise_magnitude, size=values.shape)
            values = values + noise

        return values
