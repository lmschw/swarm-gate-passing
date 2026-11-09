"""swarm-gate-passing: evolutionary (CMA-ES + Hebbian-ABCD) flocking controllers
trained to navigate gradient-mapped paths.

Combines two prior research codebases:
  - evolution: the staged CMA-ES training pipeline and wind/drag/battery-aware
    flocking simulation, vendored from energy_efficient_flocking.
  - environment: path/map generation and light-intensity sensing, ported from
    volcano_gradient.

See config.py for the full list of tunable constants and README.md for usage.
"""
