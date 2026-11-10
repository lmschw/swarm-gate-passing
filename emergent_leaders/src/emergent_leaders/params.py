"""Model parameters.

Defaults reproduce the constants of the reference implementation
(emergent_leaders/mas_random_walk, random_walk/simulation.py). Lengths are in
metres, time in seconds, angles in radians.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ModelParams:
    # --- Proximal control (Lennard-Jones-like virtual potential) ---
    epsilon: float = 15.0          # potential strength
    sigma: float = 0.7             # follower spacing parameter, d_des = sqrt(2) * sigma
    sigma_leader: float = 0.6      # leaders keep a slightly tighter spacing
    sense_range: float = 2.0       # neighbours beyond this distance are not sensed

    # --- Virtual-force weights ---
    alpha: float = 2.0             # proximal control
    beta: float = 0.5              # heading alignment
    goal_gain: float = 4.0         # goal direction (applied to leaders only)
    # Align with neighbours within sense_range; False aligns with the whole
    # swarm, as the reference does.
    local_alignment: bool = True

    # --- Unicycle motion ---
    k1: float = 0.6                # linear gain:  u = k1 * f_x + u_c
    k2: float = 0.05               # angular gain: w = k2 * f_y
    u_c: float = 0.05              # bias forward speed
    u_max: float = 0.1             # u is clipped to [0, u_max]
    w_max: float = math.pi         # w is clipped to [-w_max, w_max]
    dt: float = 0.05

    # --- Stability measure ---
    w_dist: float = 0.7            # weight of the spacing term
    w_head: float = 0.3            # weight of the local-alignment term

    # --- Promotion probability (logistic in stability, per time step) ---
    k_prom: float = 100.0          # steepness
    p_prom_max: float = 0.05       # saturation probability
    b_prom: float = 0.91           # stability at which P reaches ~p_prom_max / (1 + p_prom_max)

    # --- Leader tenure ---
    # The reference demotes once an agent has drawn 5 goal samples; the 5th is
    # drawn as the 4th leg completes, so a leader effectively walks 4 legs.
    legs_per_tenure: int = 4

    @property
    def desired_distance(self):
        return self.sigma * math.sqrt(2.0)
