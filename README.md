# swarm-gate-passing

Approaches to getting a robot swarm through gates, kept as independent
Python projects:

| Directory | Approach |
|---|---|
| [`evolutionary/`](evolutionary/) | CMA-ES-evolved Hebbian (ABCD) flocking controllers, trained over a curriculum of path-following and gate-passing objectives. |
| [`emergent_leaders/`](emergent_leaders/) | Analytical model: virtual-force flocking where leaders emerge stochastically from local stability and steer the swarm along a goal direction. With gates, agents that sense a gate post lead the swarm towards it, on slalom and winding courses. |
| [`vision_leaders/`](vision_leaders/) | The `emergent_leaders` swarm with a low-resolution camera per agent and a CNN, trained by imitation and DAgger, that turns each image into "gate in view?" and a goal bearing. |

Each directory has its own `pyproject.toml`, tests and README. Install any of them
in editable mode (`vision_leaders` needs `emergent_leaders`):

```sh
pip install -e "evolutionary[dev]" -e "emergent_leaders[dev]" -e "vision_leaders[dev]"
```
