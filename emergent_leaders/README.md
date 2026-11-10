# emergent-leaders

Flocking with emergent, goal-directed leaders. This is the model behind the
`random_walk/simulation.py` code in the `mas_random_walk` repository,
reimplemented as small, tested modules. All default parameters match that code
(see `params.py`). Gate handling will plug in as a new goal model (see
[Goal directions](#goal-directions)).

## Model

N unicycle agents, each with position $\mathbf{x}_i$ and heading $\theta_i$.
Agent $i$ senses the neighbours $\mathcal{N}_i$ within `sense_range`, at distance
$d_{ij}$ and body-frame bearing $\phi_{ij}$.

### Motion (`motion.py`)

A virtual force, expressed in agent $i$'s body frame:

$$\mathbf{f}_i = \alpha\,\mathbf{p}_i + \beta\,\mathbf{h}_i + G\,\gamma_i\,\mathbf{g}_i$$

- **Proximal control:**
  $\mathbf{p}_i = \sum_{j\in\mathcal{N}_i} F(d_{ij})\,[\cos\phi_{ij}, \sin\phi_{ij}]$ with
  $F(d) = -\varepsilon\,(2\sigma_i^4/d^5 - \sigma_i^2/d^3)$. The equilibrium spacing is
  $d_\text{des} = \sqrt2\,\sigma_i$: neighbours attract beyond it and repel inside it.
  Leaders use a smaller $\sigma$ (0.6 instead of 0.7).
- **Heading alignment:** $\mathbf{h}_i$ is the unnormalised sum of the heading unit
  vectors of $i$ and its neighbours in $\mathcal{N}_i$, rotated into $i$'s frame.
  `local_alignment=False` (`--global-alignment` on the command line) sums over
  all agents instead, as the reference does.
- **Goal direction:** $\mathbf{g}_i = [\cos(\psi_i-\theta_i), \sin(\psi_i-\theta_i)]$, where
  $\psi_i$ is the leader's goal heading. $\gamma_i = 1$ for leaders and 0 for followers.

The force maps to speed commands, which are integrated with forward Euler:

$$u_i = \operatorname{clip}(k_1 f_x + u_c,\ 0,\ u_\max),\qquad
\omega_i = \operatorname{clip}(k_2 f_y,\ -\omega_\max,\ \omega_\max)$$

### Leadership emergence (`leadership.py`)

Each agent computes a local **stability** from how evenly its neighbours are
spaced and how well they are aligned:

$$s_i = w_d\Big(1 - \tfrac{1}{|\mathcal{N}_i|}\textstyle\sum_j \tfrac{(d_{ij}-d_\text{des})^2}{d_\text{des}^2}\Big)
      + w_h\,\frac{\big|\sum_{j\in\mathcal{N}_i\cup\{i\}} e^{\mathrm{i}\theta_j}\big|}{|\mathcal{N}_i|+1}$$

At every step, a follower is promoted to leader with probability

$$P(s) = \frac{p_\max}{1 + e^{-k(s - s_0)}},\qquad s_0 = b - \ln(1/p_\max)/k.$$

Promotion can be restricted with `leader_type`: `inner` agents have a
neighbour in each of their four body-frame quadrants, and `outer` agents do
not. `free` allows any agent.

A new leader gets a **leg** from the goal model: a goal heading $\psi_i$ and a
path length. When it has travelled that length, it gets the next leg. After
`legs_per_tenure` legs it is demoted back to follower.

### Goal directions (`goals.py`)

A `GoalModel` assigns `(leg_lengths, goal_headings)` whenever an agent is
promoted or finishes a leg.

- `RandomWalkGoal(walk)`: the reference's random walks. Leg lengths come from a
  discretised Lévy-stable pdf, clipped to [4, 12] m. Headings are global and
  drawn from a wrapped Cauchy distribution centred on 0.

  | walk | stability α | ρ |
  |---|---|---|
  | `brown` | 2 | 0 |
  | `levy` | 1 | 0 |
  | `correlated` | 2 | 0.7 |

- `FixedGoal(heading)`: every leg points the same way. This is the simplest
  goal-directed case and the stepping stone to gate goals.

## Usage

```sh
pip install -e ".[dev]"

python -m emergent_leaders.run --walk levy --duration 500 --live          # interactive window
python -m emergent_leaders.run --walk correlated --seed 0 \
    --video run.mp4 --summary run.png                                    # MP4 replay + summary figure
python -m emergent_leaders.run --goal-heading 45 --summary goal45.png    # fixed goal direction
python -m emergent_leaders.run --global-alignment --live                 # reference's global alignment
```

From Python:

```python
from emergent_leaders import Simulation, RandomWalkGoal
from emergent_leaders.visualize import plot_summary, render_video

sim = Simulation(n_agents=25, goal=RandomWalkGoal("levy"), leader_type="free", seed=0)
history = sim.run(n_steps=10_000, record_every=10)
plot_summary(history, sim.params, "summary.png")
render_video(history, "run.mp4")
```

## Differences from the reference code

By default, heading alignment only uses neighbours within `sense_range`. The
reference aligns with the whole swarm, which `local_alignment=False` restores.
With local alignment, the swarm often splits under the random-walk goals: in
about 4–5 of 10 Lévy or Brownian runs of 600 s, and in 1 of 10 correlated runs.
It stays together under a fixed goal heading.

With global alignment the dynamics match the reference. Over 1000 s with 25
agents, both implementations average about 12–13 leaders, keep a swarm radius
of about 3 m and reach a mean stability of about 0.81. A few other details
differ:

- `n_agents` is the total number of agents. The reference's argument was the
  side length of the square starting grid.
- Run length is given in steps (`run(n_steps)`) or in seconds (`--duration`).
- `legs_per_tenure = 4` replaces "demote once 5 goal samples were drawn". These
  give the same behaviour: in the reference, the 5th sample is drawn as the 4th
  leg ends.
- In the reference, a leader that was demoted on the same step it finished a
  leg still drew a new sample, so its next tenure started with an inflated
  count. That no longer happens.
- Quadrant coverage is computed on the post-move configuration, the same as
  stability. The reference used the pre-move one, a difference of one time step.

Two behaviours are kept as they are in the reference but are worth knowing:

- With `brown`, every leg is exactly 4 m, because the Gaussian's 99th percentile
  (≈3.3) falls below the 4 m minimum.
- The `correlated` walk's headings are centred on global 0, not on the previous
  heading, so the swarm drifts towards +x.
