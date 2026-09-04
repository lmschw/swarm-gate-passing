"""Physical gate barrier for the gate-passing curriculum.

A Gate is a wall placed across the arena at a given x position, with a gap
("opening") straddling the gradient-mapped path's actual centerline at that x
-- read directly off the map via GradientSensor.centerline_y(), so the opening
lines up with the path regardless of its shape or wavelength. simulation.py's
_move() stops any agent that tries to cross the gate's x outside the opening
(counted together with arena-wall hits), and the same x doubles as the episode's
finish line for the success bonus (see simulation.EpisodeResult.success).
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Gate:
    x_arena: float      # arena-frame x position of the barrier
    y_lo_arena: float   # opening's lower y bound (arena frame)
    y_hi_arena: float   # opening's upper y bound (arena frame)

    def blocks(self, y_arena):
        """True if a crossing agent at this y would be stopped by the barrier."""
        return not (self.y_lo_arena <= y_arena <= self.y_hi_arena)

    @classmethod
    def centered_on_path(cls, gradient_sensor, x_arena, opening_width_m, x_range, y_range):
        world_x = x_arena - x_range[0]
        center_y_world = float(gradient_sensor.centerline_y(world_x))
        center_y_arena = center_y_world + y_range[0]
        half = opening_width_m / 2.0
        return cls(x_arena=x_arena, y_lo_arena=center_y_arena - half, y_hi_arena=center_y_arena + half)


def evenly_spaced_gates(gradient_sensor, n_gates, finish_x, opening_width_m, x_range, y_range,
                         start_x=0.0):
    """n_gates Gates, each centered on the path, spaced evenly between start_x
    (the spawn area, by default arena x=0) and finish_x -- the last gate sits
    exactly AT finish_x, so n_gates=1 reproduces the original single-gate
    behavior exactly (list of one Gate at finish_x). Returned in travel order
    (descending x, since the swarm moves in -x): gates[-1] is always the last
    (and overall finish) gate.
    """
    if n_gates < 1:
        return []
    xs = [start_x + (finish_x - start_x) * (i / n_gates) for i in range(1, n_gates + 1)]
    return [Gate.centered_on_path(gradient_sensor, x, opening_width_m, x_range, y_range) for x in xs]
