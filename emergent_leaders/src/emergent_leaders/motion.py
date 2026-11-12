"""Motion model: virtual-force flocking on unicycle agents.

Each agent i has position x_i and heading theta_i. Every term of the virtual
force is expressed in agent i's body frame (x forward, y left):

    f_i = alpha * p_i + beta * h_i + goal_gain * gamma_i * g_i

    p_i   proximal control: sum over sensed neighbours j of
          F(d_ij) * [cos(phi_ij), sin(phi_ij)],
          F(d) = -epsilon * (2 sigma_i^4 / d^5 - sigma_i^2 / d^3)
          (attractive for d > sqrt(2) sigma_i, repulsive below)
    h_i   heading alignment: the (unnormalised) sum of the heading vectors of
          i and its sensed neighbours (or of all agents, if
          local_alignment is off), rotated into i's frame
    g_i   goal direction: unit vector towards the goal heading psi_i
    gamma_i  1 for leaders, 0 for followers (or any weight, if goal_weight is given)

The force is mapped to speed commands and integrated with forward Euler:

    u_i = clip(k1 * f_x + u_c, 0, u_max)
    w_i = clip(k2 * f_y, -w_max, w_max)
    x_i += u_i [cos theta_i, sin theta_i] dt,   theta_i += w_i dt
"""
import numpy as np


def wrap_to_pi(angle):
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def neighbour_geometry(pos, heading, sense_range):
    """Pairwise distances and body-frame bearings.

    dist[i, j] is inf for j == i and for j outside i's sensing range, so that
    every force term vanishes for unsensed pairs. bearing[i, j] is the angle
    of j as seen from i, relative to theta_i, wrapped to [-pi, pi).
    """
    diff = pos[None, :, :] - pos[:, None, :]          # diff[i, j] = x_j - x_i
    dist = np.hypot(diff[..., 0], diff[..., 1])
    dist[(dist > sense_range) | (dist == 0.0)] = np.inf
    bearing = wrap_to_pi(np.arctan2(diff[..., 1], diff[..., 0]) - heading[:, None])
    return dist, bearing


def proximal_control(dist, bearing, sigma, epsilon):
    """sigma: per-agent spacing parameter, shape (n,)."""
    s = sigma[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        magnitude = -epsilon * (2.0 * s**4 / dist**5 - s**2 / dist**3)
    magnitude[~np.isfinite(dist)] = 0.0
    return np.sum(magnitude * np.cos(bearing), axis=1), np.sum(magnitude * np.sin(bearing), axis=1)


def heading_alignment(heading, sensed=None):
    """sensed: (n, n) bool mask of the agents each agent aligns with (itself is
    always included); None aligns every agent with the whole swarm."""
    if sensed is None:
        total = np.sum(np.exp(1j * heading))
    else:
        total = (sensed | np.eye(len(heading), dtype=bool)) @ np.exp(1j * heading)
    local = total * np.exp(-1j * heading)
    return local.real, local.imag


def goal_attraction(heading, goal_heading):
    rel = goal_heading - heading
    return np.cos(rel), np.sin(rel)


def virtual_force(pos, heading, is_leader, goal_heading, params, geometry=None, goal_weight=None, sigma=None):
    """Returns (f_x, f_y) in each agent's body frame. goal_weight, if given,
    replaces gamma_i (leaders 1, followers 0) as the weight of the goal term;
    sigma, if given, replaces the spacing parameters (sigma_leader for
    leaders, sigma for followers)."""
    dist, bearing = geometry if geometry is not None else neighbour_geometry(pos, heading, params.sense_range)
    if sigma is None:
        sigma = np.where(is_leader, params.sigma_leader, params.sigma)
    p_x, p_y = proximal_control(dist, bearing, sigma, params.epsilon)
    h_x, h_y = heading_alignment(heading, np.isfinite(dist) if params.local_alignment else None)
    g_x, g_y = goal_attraction(heading, goal_heading)
    gamma = is_leader.astype(float) if goal_weight is None else goal_weight
    f_x = params.alpha * p_x + params.beta * h_x + params.goal_gain * gamma * g_x
    f_y = params.alpha * p_y + params.beta * h_y + params.goal_gain * gamma * g_y
    return f_x, f_y


def velocity_commands(f_x, f_y, params):
    u = np.clip(params.k1 * f_x + params.u_c, 0.0, params.u_max)
    w = np.clip(params.k2 * f_y, -params.w_max, params.w_max)
    return u, w


def integrate(pos, heading, u, w, dt):
    step = (u * dt)[:, None] * np.column_stack([np.cos(heading), np.sin(heading)])
    return pos + step, wrap_to_pi(heading + w * dt)
