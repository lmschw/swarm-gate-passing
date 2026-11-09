"""Goal directions for leaders.

A leader walks a sequence of straight "legs". Each leg has a length (path
length the leader must travel) and a goal heading psi, which drives the goal
term g_i of the motion model. A goal model assigns a new leg whenever an agent
is promoted or finishes its previous leg; this is the hook where gate-directed
goals will plug in.
"""
import numpy as np
from scipy.stats import levy_stable

# (stability alpha, skewness beta, wrapped-Cauchy rho)
WALKS = {
    "brown": (2.0, 0.0, 0.0),
    "levy": (1.0, 0.0, 0.0),
    "correlated": (2.0, 0.0, 0.7),
}


class GoalModel:
    def assign(self, state, idx, rng):
        """Return (leg_lengths, goal_headings) for the agents in idx."""
        raise NotImplementedError


class RandomWalkGoal(GoalModel):
    """The reference's random-walk goals.

    Leg lengths are drawn from a Levy-stable pdf discretised on 100 points
    between its 20th and 99th percentiles, then clipped to
    [min_length, max_length]. Headings are drawn in the global frame from a
    wrapped Cauchy distribution centred on 0 with concentration rho (rho = 0
    is uniform).
    """

    def __init__(self, walk="levy", min_length=4.0, max_length=12.0, n_points=100):
        if walk not in WALKS:
            raise ValueError(f"walk must be one of {tuple(WALKS)}, got {walk!r}")
        self.walk = walk
        a, b, self.rho = WALKS[walk]
        self.min_length, self.max_length = min_length, max_length
        self._lengths = np.linspace(levy_stable.ppf(0.2, a, b), levy_stable.ppf(0.99, a, b), n_points)
        pdf = levy_stable.pdf(self._lengths, a, b)
        self._probs = pdf / pdf.sum()

    def sample_lengths(self, n, rng):
        lengths = rng.choice(self._lengths, size=n, p=self._probs)
        return np.clip(lengths, self.min_length, self.max_length)

    def sample_headings(self, n, rng):
        c = 2.0 * self.rho / (1.0 + self.rho**2)
        v = np.cos(rng.uniform(0.0, 2.0 * np.pi, n))
        headings = np.arccos((v + c) / (1.0 + c * v))
        return np.where(rng.random(n) < 0.5, -headings, headings)

    def assign(self, state, idx, rng):
        n = len(idx)
        return self.sample_lengths(n, rng), self.sample_headings(n, rng)


class FixedGoal(GoalModel):
    """Every leg points along the same global heading."""

    def __init__(self, heading=0.0, leg_length=8.0):
        self.heading = heading
        self.leg_length = leg_length

    def assign(self, state, idx, rng):
        n = len(idx)
        return np.full(n, self.leg_length), np.full(n, self.heading)
