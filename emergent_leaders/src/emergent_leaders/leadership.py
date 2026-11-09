"""Leadership emergence: stability measure and stochastic promotion.

A follower i becomes a leader with per-step probability P(s_i), where s_i is
a local stability measure combining how well i's neighbours sit at the
desired distance and how aligned they are:

    s_i = w_dist * (1 - mean_j (d_ij - d_des)^2 / d_des^2)
        + w_head * | sum_{j in N_i + i} e^{i theta_j} | / (|N_i| + 1)

    P(s) = p_max / (1 + exp(-k (s - s_0))),   s_0 = b - ln(1 / p_max) / k

Agents with no sensed neighbour get a spacing term of 0, so they are never
promoted in practice. Promotion can be restricted by position in the swarm:
"inner" agents have a neighbour in each of their four body-frame quadrants,
"outer" agents do not.
"""
import numpy as np

LEADER_TYPES = ("free", "inner", "outer")


def stability(dist, heading, params):
    sensed = np.isfinite(dist)
    n_sensed = sensed.sum(axis=1)
    d_des = params.desired_distance

    sq_err = np.where(sensed, (dist - d_des) ** 2, 0.0).sum(axis=1)
    spacing = np.zeros(len(heading))
    has_nb = n_sensed > 0
    spacing[has_nb] = 1.0 - sq_err[has_nb] / n_sensed[has_nb] / d_des**2

    with_self = sensed | np.eye(len(heading), dtype=bool)
    alignment = np.abs(with_self @ np.exp(1j * heading)) / (n_sensed + 1)

    return params.w_dist * spacing + params.w_head * alignment


def full_quadrant_coverage(dist, bearing):
    """True where every body-frame quadrant holds at least one sensed neighbour."""
    quadrant = np.clip(np.floor((bearing + np.pi) / (np.pi / 2)).astype(int), 0, 3)
    sensed = np.isfinite(dist)
    covered = np.stack([(sensed & (quadrant == q)).any(axis=1) for q in range(4)], axis=1)
    return covered.all(axis=1)


def promotion_probability(s, params):
    s0 = params.b_prom - np.log(1.0 / params.p_prom_max) / params.k_prom
    return params.p_prom_max / (1.0 + np.exp(-params.k_prom * (s - s0)))


def promotion_eligible(full_coverage, leader_type):
    if leader_type == "free":
        return np.ones_like(full_coverage, dtype=bool)
    if leader_type == "inner":
        return full_coverage.copy()
    if leader_type == "outer":
        return ~full_coverage
    raise ValueError(f"leader_type must be one of {LEADER_TYPES}, got {leader_type!r}")
