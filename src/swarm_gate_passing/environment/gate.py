"""Gate barrier / gate landmarks for the gate-passing curricula.

A Gate is defined by its x position and an opening [y_lo, y_hi]. Two distinct
physical behaviors share this same definition:

- blocking=True (the original design, used by the gradient-guided curriculum):
  a solid wall spans the arena's full y-range at x_arena EXCEPT the opening --
  simulation.py's _move() physically stops any agent that tries to cross
  outside it (counted together with arena-wall hits).
- blocking=False ("just a pole on either side", used by the vision-based
  curriculum): no wall at all -- agents can freely cross the gate's x anywhere.
  Only the two opening-edge points exist physically (as landmarks agents can
  see, per sensor_model_vision.py), and a crossing only counts toward
  simulate_hebbian_episode's crossing_success bookkeeping if it happens within
  [y_lo, y_hi]. Added after a solid wall was observed splitting a swarm that
  spawned straddling it into two groups that struggled to reconnect at all
  before ever finding the (possibly distant) opening -- removing the wall lets
  agents solve "regroup" and "find the gap" somewhat independently, while
  still requiring them to actually pass between the two poles together.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Gate:
    x_arena: float           # arena-frame x position of the barrier/poles
    y_lo_arena: float        # opening's lower y bound (arena frame)
    y_hi_arena: float        # opening's upper y bound (arena frame)
    blocking: bool = True    # False = "just a pole on either side" (see module docstring)

    def blocks(self, y_arena):
        """True if a crossing agent at this y would be stopped by the barrier
        (always False when blocking=False -- nothing physically stops anyone)."""
        if not self.blocking:
            return False
        return not (self.y_lo_arena <= y_arena <= self.y_hi_arena)

    def within_opening(self, y_arena):
        """True if y_arena falls within [y_lo, y_hi] -- used to decide whether
        a crossing (physically blocked or not) counts as "through the gate"."""
        return self.y_lo_arena <= y_arena <= self.y_hi_arena

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


def random_gate(rng, opening_width_m, x_bounds, y_bounds, blocking=False):
    """A single gate "strewn" at a uniformly-random (x, y-center) within the
    given bounds -- independent of any path/map (contrast evenly_spaced_gates,
    which centers gates on a rendered gradient path). Used by the vision-based
    curriculum (no gradient map, agents must find the gate by limited-range,
    occlusion-aware sensing -- see sensor_model_vision.py). blocking=False by
    default: "just a pole on either side" -- see module docstring for why."""
    x = float(rng.uniform(*x_bounds))
    y_center = float(rng.uniform(*y_bounds))
    half = opening_width_m / 2.0
    return Gate(x_arena=x, y_lo_arena=y_center - half, y_hi_arena=y_center + half, blocking=blocking)
