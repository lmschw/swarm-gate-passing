"""Hebbian-plasticity MLP controller.

Vendored from energy_efficient_flocking's ants26_replication/experiment/
hebbian_controller.py, itself a 1:1 port of the MATLAB reference's hebbianStep.m
forward pass + Hebbian weight update, and the per-agent weight-initialization loop
in simulation_free_global_mod_2.m. Unchanged here except for import paths --
adding the gradient-sensing input (config.HEBBIAN_N_INPUTS) doesn't touch this
file at all, since layer shapes are derived entirely from config.
"""
import numpy as np

from . import config

# Layer shapes, in flatten/unflatten order (matches hebbianStep.m's W1, W2, W3).
_LAYER_SHAPES = (
    ("1", (config.HEBBIAN_N_INPUTS, config.HEBBIAN_N_HIDDEN)),
    ("2", (config.HEBBIAN_N_HIDDEN, config.HEBBIAN_N_HIDDEN)),
    ("3", (config.HEBBIAN_N_HIDDEN, config.HEBBIAN_N_OUTPUTS)),
)
_LETTERS = ("A", "B", "C", "D")


def init_weights():
    """Fresh, randomly-initialized NN weights for one agent -- not evolved, re-drawn
    every episode. Uniform distribution in [-1, 1] for all three matrices."""
    r = config.HEBBIAN_WEIGHT_INIT_RANGE
    w1 = np.random.uniform(-r, r, (config.HEBBIAN_N_INPUTS, config.HEBBIAN_N_HIDDEN))
    w2 = np.random.uniform(-r, r, (config.HEBBIAN_N_HIDDEN, config.HEBBIAN_N_HIDDEN))
    w3 = np.random.uniform(-r, r, (config.HEBBIAN_N_HIDDEN, config.HEBBIAN_N_OUTPUTS))
    return w1, w2, w3


def unflatten_abcd(flat):
    """Maps an HEBBIAN_N_ABCD-length genome vector to a dict of 12 matrices -- A1,
    A2, A3, B1, B2, B3, C1, C2, C3, D1, D2, D3 -- matching hebbianStep.m's
    R.A1..R.D3 fields."""
    flat = np.asarray(flat, dtype=float)
    rules = {}
    idx = 0
    for letter in _LETTERS:
        for suffix, shape in _LAYER_SHAPES:
            size = shape[0] * shape[1]
            rules[letter + suffix] = flat[idx:idx + size].reshape(shape)
            idx += size
    assert idx == config.HEBBIAN_N_ABCD, f"expected to consume {config.HEBBIAN_N_ABCD}, got {idx}"
    return rules


def _normalize(w):
    """Scales down only if a weight exceeds 1 in magnitude, preventing unbounded
    growth under repeated Hebbian updates."""
    maxval = np.max(np.abs(w))
    return w / maxval if maxval > 1.0 else w


def hebbian_step(x_in, w1, w2, w3, rules):
    """One forward pass + Hebbian update for a single agent.

    x_in: (N_INPUTS,) sensory vector. w1/w2/w3: this agent's current NN weights.
    rules: dict from unflatten_abcd(), shared across every agent in the swarm.
    Returns (v, w, w1_new, w2_new, w3_new).
    """
    x_in = x_in.reshape(-1, 1)               # (N_INPUTS, 1) column vector
    h1 = np.maximum(0.0, x_in.T @ w1)         # (1, N_HIDDEN)
    h2 = np.maximum(0.0, h1 @ w2)             # (1, N_HIDDEN)
    out = np.tanh(h2 @ w3)                    # (1, N_OUTPUTS)

    v = out[0, 0] * config.HEBBIAN_LINEAR_VEL_MAX
    w = out[0, 1] * config.HEBBIAN_ANGULAR_VEL_MAX

    eta = config.HEBBIAN_LEARNING_RATE
    # Broadcasting (N_INPUTS,1)*(N_INPUTS,N_HIDDEN) and (1,N_HIDDEN)*(N_HIDDEN,N_HIDDEN)
    # below reproduces MATLAB's repmat(in,1,N_HIDDEN) / repmat(h1,N_HIDDEN,1) without
    # an explicit repeat.
    w1_new = _normalize(w1 + eta * (rules["A1"] * (x_in @ h1) + rules["B1"] * x_in
                                     + rules["C1"] * h1 + rules["D1"]))
    w2_new = _normalize(w2 + eta * (rules["A2"] * (h1.T @ h2) + rules["B2"] * h1.T
                                     + rules["C2"] * h2 + rules["D2"]))
    w3_new = _normalize(w3 + eta * (rules["A3"] * (h2.T @ out) + rules["B3"] * h2.T
                                     + rules["C3"] * out + rules["D3"]))
    return v, w, w1_new, w2_new, w3_new
