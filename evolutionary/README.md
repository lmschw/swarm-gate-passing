# swarm-gate-passing

Evolutionary (CMA-ES + Hebbian-ABCD) flocking controllers, trained to navigate
gradient-mapped paths. This project combines two prior research codebases:

- **Evolutionary training pipeline** (`src/swarm_gate_passing/{config,hebbian_controller,
  sensor_model,wind_physics,simulation,optimize,fitness_plot}.py`) — vendored from
  `energy_efficient_flocking`'s `ants26_replication/experiment/` (a Python replication
  of the ANTS 2026 paper *"Energy-Efficient Flocking in Self-Organized Robot Swarms"*,
  Mahdavi et al.; MATLAB reference in that project's `ants-2026-polimi-*/` directory).
  A per-robot MLP is updated online by an evolved generalized-Hebbian rule (the "ABCD"
  coefficients); CMA-ES evolves those coefficients, shared across every agent, over a
  staged curriculum of increasingly complex fitness objectives (distance → battery
  efficiency via wind-drafting → collision avoidance).

- **Paths, maps, and sensing** (`src/swarm_gate_passing/environment/`) — ported from
  `volcano_gradient`'s gradient-map path generator and light-intensity sensor
  (`gym_pybullet_drones/examples/gradient/generate_gradients.py` and the
  `read_light_intensity` sensor in `examples/flocking/2d_flocking_with_real_utils.py`),
  stripped of that project's PyBullet/drone-physics dependencies.

## What's new here (the combination)

The flocking controller gained an 11th sensor input: a light-intensity reading from a
gradient-map PNG (`environment.sensing.GradientSensor`), sampled at each agent's
position every simulation step. A fourth curriculum stage, `follow_gradient_path`,
rewards staying on the mapped path on top of the original energy/collision terms —
see `config.HEBBIAN_STAGE_FITNESS_WEIGHTS` and `simulation.stage_fitness`.

## Install

```sh
pip install -e ".[dev]"
```

## Usage

Generate a default set of path/gradient maps sized to the flocking arena:

```sh
python -m swarm_gate_passing.generate_maps --output-dir maps_gradient
```

Run the staged CMA-ES training curriculum (paper-scale settings are slow — see
`--popsize`/`--maxiter`/`--wind-grid`/`--battery` to shrink a run for local testing):

```sh
python -m swarm_gate_passing.optimize \
    --gradient-map maps_gradient/path_sine_curve_freq2.png \
    --output-dir hebbian_results
```

Replay a trained genome and plot its trajectory over the gradient map:

```sh
python -m swarm_gate_passing.visualize \
    --genome hebbian_results/hebbian_follow_gradient_path_best.npy \
    --gradient-map maps_gradient/path_sine_curve_freq2.png \
    --output trajectory.png
```

## Tests

```sh
pytest tests/
```
