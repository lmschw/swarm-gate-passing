# swarm-gate-passing

Two approaches to getting a robot swarm through gates, kept as independent
Python projects:

| Directory | Approach |
|---|---|
| [`evolutionary/`](evolutionary/) | CMA-ES-evolved Hebbian (ABCD) flocking controllers, trained over a curriculum of path-following and gate-passing objectives. |
| [`emergent_leaders/`](emergent_leaders/) | Analytical model: virtual-force flocking where leaders emerge stochastically from local stability and steer the swarm along a goal direction. Includes gates the swarm passes through (first version), on slalom and winding courses. |

Each directory has its own `pyproject.toml`, tests and README. Install either or
both in editable mode:

```sh
pip install -e "evolutionary[dev]" -e "emergent_leaders[dev]"
```
