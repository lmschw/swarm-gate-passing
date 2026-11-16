# vision-leaders

Gate perception from low-resolution cameras. Each agent of the
[`emergent_leaders`](../emergent_leaders/) swarm gets a forward camera, and a
small CNN maps its image to whether a gate is in view and the bearing to steer
along. The motion model, the leader bookkeeping and the gates are
`emergent_leaders`'; only the gate perception is replaced.

In `emergent_leaders` an agent becomes a gate leader when it senses a gate
post within 4 m in front of it, and aims at the gate centre (both posts
sensed) or at the lone post. Here the network decides both from the image.

## Camera (`camera.py`)

- 72 × 16 px, 8-bit RGB, 180° horizontal field of view; columns are equal
  steps in bearing (2.5° each), rows use the same angular step, heights
  project by perspective. The camera sits 0.2 m above the ground.
- Gate posts are orange cylinders, 1 m tall, of the motion model's post
  radius (0.15 m). Agents are grey cylinders, 0.25 m tall with radius 0.1 m
  (they are points in the motion model). Sky above the horizon, ground below.
- Pixels are area-sampled: an object fills the fraction of a pixel it covers,
  so a far post fades out instead of flickering. A post 4 m away is about
  2 px wide and 6 px tall.
- Objects are composited front to back, so nearer agents hide what is behind
  them. The posts stand above the agents, so their tops stay visible over the
  swarm. Rendering all 25 cameras takes about 6 ms per step.

![camera views](docs/camera_views.png)

## Teacher (`simulation.py`)

`VisionSimulation` renders every camera each step and computes the teacher's
gate perception alongside the network's:

- `teacher="full"` (default): the `emergent_leaders` rule with its perfect
  sensing within 4 m, whether or not the camera shows the posts. The network
  is trained on these labels.
- `teacher="camera"`: the same rule, but a post counts only if the camera
  shows at least half a pixel of it. This teacher fails at the gates: on the
  default slalom only 1 of 10 runs gets most of the swarm through all three
  gates (68% / 27% / 11% through), against 10 of 10 for the full rule. An
  agent passing between the posts sees only the post in front of it and turns
  into it. The full rule also senses the post behind, sees that the gate
  centre is behind, and stops aiming. With a 180° field of view this happens
  right at the gate; with 120° (the first design) it happens about 1 m before
  it, and the results were no better (1 of 10). Agents hiding the posts made
  no difference: the results were identical with invisible agents.

With the full teacher, the network has to learn from the image what the rule
senses directly, e.g. that a lone post close beside it means it is passing
through the gate.

With `ModelParams(lone_post_aim="centre")` the full teacher aims at the gate
centre also when it senses only one post (`train.py --lone-post-aim
centre`). Aiming at a lone post is what makes most agents miss a gate (see
the `emergent_leaders` README); the centre is privileged information for the
rule, but the camera often shows the far post beyond the rule's 4 m.

## Network (`network.py`)

`GoalNet`: three 3 × 3 convolutions (16, 32, 32 channels, the last two with
stride 2), a 64-unit hidden layer, and three outputs: the logit of "gate
target" and the bearing as (sin, cos), relative to the heading, positive to
the left. A fourth input channel holds each column's bearing. An agent leads
when the probability reaches 0.5 and then aims along the predicted bearing.

With `goal_weighting="confidence"` (`--goal-weighting confidence`), the
probability q itself weights the goal term instead: every agent pulls along
its predicted bearing with γ = q, in f = αp + βh + Gγg. Agents with q ≥ 0.5
still count as leaders (spacing, statistics, video).

## Training (`train.py`)

1. **Behaviour cloning:** the full rule drives the swarm on 40 random courses
   (three-gate slaloms with 0–1.5 m offsets and four-gate winding courses with
   turns up to 45°, at the largest spacing the course rules allow), 300 s
   each. Every 20 steps (1 s) each agent's image is stored with the rule's
   label; images without a gate target are subsampled to 30%.
2. **DAgger:** three rounds of 20 runs in which the network drives; the rule
   labels the images it sees, they are added to the data, and the network is
   retrained on all of it.

Training on 22 CPU cores took 42 min (`goalnet.pt`; 44 min for
`goalnet_centre.pt`).

## Results

`models/goalnet.pt` was trained as above. Offline, on a held-out 10% of all
images (most of them from DAgger rounds), it agrees with the rule on whether
to lead for 98.8% of images; on images where the rule leads, the bearing
error has a median of 1.8° and a 90th percentile of 5.3°. After behaviour
cloning alone, these were 98.3%, 2.5° and 9.7°.

Closed loop (`python -m vision_leaders.evaluate models/goalnet.pt`; explorers
off, `sigma_leader` 0.7, swarm facing the course):

| suite | perception | through each gate | runs with > 50% through all |
|---|---|---|---|
| default slalom, seeds 0–9, 320 s | rule | 92 / 90 / 81% | 10 / 10 |
| | network | 90 / 91 / 71% | 10 / 10 |
| winding, 7 m, seeds 0–19, 600 s | rule | 93 / 57 / 40 / 31 / 25% | 5 / 20 |
| | network | 91 / 61 / 40 / 28 / 24% | 4 / 20 |

The network drives about as well as the rule it imitates. On the winding
courses both lose the swarm at sharp turns (see the `emergent_leaders`
README), so the network isn't the limit there. The closed-loop effect of
DAgger was not measured separately: the behaviour-cloning model of this run
was overwritten (`train.py` now keeps each stage's model).

`models/goalnet_centre.pt` imitates the centre-aiming teacher (otherwise
trained the same way; log in `models/train_centre.log`). Offline: 98.8% lead
agreement, bearing error median 1.6°, 90th percentile 5.2° (2.0° and 7.1°
after behaviour cloning). Closed loop (`python -m vision_leaders.evaluate
models/goalnet.pt models/goalnet_centre_bc.pt models/goalnet_centre.pt
--rules post centre`):

| suite | perception | through each gate | runs with > 50% through all |
|---|---|---|---|
| default slalom | rule, aims at a lone post | 92 / 90 / 81% | 10 / 10 |
| | rule, aims at the centre | 98 / 96 / 91% | 10 / 10 |
| | `goalnet.pt` | 90 / 91 / 71% | 10 / 10 |
| | `goalnet_centre_bc.pt` (behaviour cloning only) | 98 / 96 / 90% | 10 / 10 |
| | `goalnet_centre.pt` | 98 / 96 / 91% | 10 / 10 |
| winding, 7 m | rule, aims at a lone post | 93 / 57 / 40 / 31 / 25% | 5 / 20 |
| | rule, aims at the centre | 98 / 68 / 41 / 30 / 33% | 6 / 20 |
| | `goalnet.pt` | 91 / 61 / 40 / 28 / 24% | 4 / 20 |
| | `goalnet_centre_bc.pt` | 98 / 40 / 31 / 10 / 19% | 1 / 20 |
| | `goalnet_centre.pt` | 95 / 57 / 46 / 19 / 15% | 3 / 20 |

On the slalom the new network matches its teacher and gets 8–20 points more
of the swarm through each gate than `goalnet.pt`. On the winding courses it
gets 95% through gate 1 but falls behind its teacher at the turns; DAgger
helps there (1 → 3 of 20 runs), but the gap to the teacher (6 of 20) remains.
With 20 courses and whole-swarm failures at turns, these counts are noisy.

### Tuned teacher (`aiming.py`, `evolve.py`)

Imitation can't beat its teacher, so the teacher was tuned on gate passage
first. `aiming.Aim` generalises the centre-aiming rule: agents off to one
side aim past the centre by `lateral` times their own offset from it, at a
point `through` metres past the gate line, and switch to the next gate once
its target is less than `next_margin` farther than the current one;
`goal_gain` and `beta` replace G and β. CMA-ES (`python -m
vision_leaders.evolve`, no cameras, about 3 min per generation on 22 cores)
ran 30 generations of 16 candidates, each generation on 12 fresh training
courses (slaloms with 0.5–1.5 m offsets, five-gate winding courses with turns
up to 90°, 7 m; seeds disjoint from the evaluation suites); fitness is the
mean fraction of the swarm through each gate. Log: `models/evolve.log`. The
search had not converged and ended with every parameter at a bound
(`models/aim_evolved.json`: lateral 1.5, through 3 m, next_margin 3 m,
G 12, β 0.1). On the same courses, its mean beat the centre rule in 14 of
the last 16 generations, by 7 points on average.

On the evaluation suites (teachers driving, no cameras; detached: agents
outside the largest group connected within 2 m at the end, mean / max):

| teacher | slalom: through each gate | runs | winding: through each gate | runs | winding detached |
|---|---|---|---|---|---|
| centre rule | 98 / 96 / 91% | 10 / 10 | 98 / 68 / 41 / 30 / 33% | 6 / 20 | 0 / 0 |
| tuned | 99 / 99 / 98% | 10 / 10 | 99 / 82 / 56 / 54 / 38% | 8 / 20 | 2.6 / 12 |
| tuned, β 0.5 | 100 / 100 / 97% | 10 / 10 | 100 / 87 / 61 / 55 / 45% | 9 / 20 | 1.4 / 10 |
| tuned aiming, G 4, β 0.5 | 98 / 96 / 93% | 10 / 10 | 99 / 73 / 52 / 49 / 38% | 7 / 20 | 0 / 0 |
| tuned, through 0 | 100 / 100 / 100% | 10 / 10 | 100 / 55 / 23 / 9 / 6% | 1 / 20 | 5.2 / 15 |
| tuned, lateral 0 | 98 / 98 / 95% | 10 / 10 | 97 / 72 / 52 / 46 / 28% | 6 / 20 | 1.7 / 9 |
| tuned, next_margin 0 | 99 / 99 / 98% | 10 / 10 | 99 / 79 / 55 / 55 / 34% | 7 / 20 | 3.2 / 11 |

The low β splits the swarm on winding courses (the fitness doesn't penalise
that), and β 0.5 does better there. With the motion model unchanged
(G 4, β 0.5), the tuned aiming alone does little on the slalom
(98 / 96 / 93%, against 98 / 96 / 91%) and gets 7 of 20 winding runs
(62% through on average, against 54%), with no agent detached; aiming past the gate line
(`through`) matters most once G is high. That teacher
(`models/aim_evolved_default_motion.json`) is the one the networks below
imitate.

A second search kept the motion model fixed (`--fix goal_gain=4 --fix
beta=0.5`), started from that teacher (`--init`), and widened the bounds that
the first search ended on (lateral up to 4, through and next_margin up to
6 m; the first search used 1.5, 3 and 3): 30 generations of 10 candidates on
16 courses each (`models/evolve_fixed.log`). It converged (step size 0.2 →
0.09) to lateral 3.8, through 2.2 m, next_margin 1.7 m
(`models/aim_evolved_fixed.json`), but is no better: on the evaluation
suites 99 / 96 / 92% (10 / 10) and 98 / 76 / 56 / 46 / 33% (6 / 20), and on
200 fresh training courses, paired with the first teacher, +0.5 ± 0.2
points on slaloms and −0.8 ± 1.2 on winding courses (centre rule: −1.6 and
−4.4). Within this aiming family, the motion model unchanged, the teacher
has plateaued at about 62% through on the winding suite.

| suite | perception | through each gate | runs with > 50% through all |
|---|---|---|---|
| default slalom | centre rule | 98 / 96 / 91% | 10 / 10 |
| | tuned teacher | 98 / 96 / 93% | 10 / 10 |
| | `goalnet_aim.pt` (training turns ≤ 45°) | 98 / 97 / 92% | 10 / 10 |
| | `goalnet_aim90_bc.pt` (turns ≤ 90°, behaviour cloning only) | 98 / 97 / 92% | 10 / 10 |
| | `goalnet_aim90.pt` (turns ≤ 90°) | 98 / 96 / 91% | 10 / 10 |
| winding, 7 m | centre rule | 98 / 68 / 41 / 30 / 33% | 6 / 20 |
| | tuned teacher | 99 / 73 / 52 / 49 / 38% | 7 / 20 |
| | `goalnet_aim.pt` | 98 / 53 / 40 / 19 / 11% | 2 / 20 |
| | `goalnet_aim90_bc.pt` | 98 / 57 / 41 / 28 / 24% | 2 / 20 |
| | `goalnet_aim90.pt` | 98 / 76 / 52 / 40 / 34% | 6 / 20 |

`goalnet_aim.pt` was trained like the others, on winding courses with turns
up to 45°; the winding suite has turns up to 90°. On the states it visits
there, it agrees with its teacher at gate 1 and after turns below 45° (misses
1.3–1.7% of the teacher's leads, bearing error median 1.0–1.3°), but not
after sharper turns (misses 5–7%, median 17–31°). `goalnet_aim90.pt` was
trained with `--max-turn 90 --duration 450` (winding training courses with
five gates 6–8 m apart, turns up to 90°; about an hour): offline 99.1% lead
agreement, bearing error median 2.5°, 90th percentile 9.0°. It is the first
network that keeps up with its teacher on the winding suite (60% through on
average, against 62% for the teacher, 54% for the centre rule and 49% for
`goalnet.pt`); DAgger matters here (2 → 6 of 20 runs). Part of the gain over
`goalnet_centre.pt` may come from the sharper training turns rather than the
tuned teacher; a centre-teacher network trained with `--max-turn 90` would
tell.

Confidence weighting (`python -m vision_leaders.evaluate models/goalnet.pt
--goal-weighting threshold confidence`) makes no clear difference:

| suite | goal weight | through each gate | runs with > 50% through all | agent-steps with 0.05 < q < 0.95 |
|---|---|---|---|---|
| default slalom | 1 if q ≥ 0.5 | 90 / 91 / 71% | 10 / 10 | 8.8% |
| | q | 91 / 91 / 76% | 10 / 10 | 8.0% |
| winding, 7 m | 1 if q ≥ 0.5 | 91 / 61 / 40 / 28 / 24% | 4 / 20 | 5.6% |
| | q | 91 / 58 / 39 / 23 / 15% | 3 / 20 | 6.3% |

The network is unsure for only 6–9% of agent-steps, so q mostly sits near 0
or 1 and both weightings act alike. It was trained on hard labels (would the
rule lead or not), so q estimates that probability; it says nothing about
how reliable the predicted bearing is.

A graded pull doesn't help even with the rule's perfect sensing. Gate
leaders were weighted by distance to the target (γ = 1 − d / 4 m), by a fade
over the outer half of the sensing range (full within 2 m of the nearest
post, 0 at 4 m), or by posts seen (0.5 for a lone post), with the goal gain
G either unchanged or raised so the mean pull matches the binary rule
(runs with > 50% through all gates, slalom / winding):

| goal weight of gate leaders | G = 4 | G raised to match the mean pull |
|---|---|---|
| 1 (binary, as now) | 10 / 10, 5 / 20 | – |
| 1 − d / 4 m (mean 0.41) | 4 / 10, 0 / 20 | 5 / 10, 2 / 20 (G 9.8) |
| fade over 2–4 m (mean 0.70) | 5 / 10, 0 / 20 | 4 / 10, 1 / 20 (G 5.8) |
| 0.5 for a lone post (mean 0.75) | 8 / 10, 2 / 20 | 9 / 10, 2 / 20 (G 5.3) |

A weaker binary pull keeps the slalom (G 1.6: 10 / 10; G 2.8: 9 / 10) but
loses winding runs (1 / 20 and 3 / 20). So down-weighting the first, far
sightings of a gate is what hurts: the swarm starts turning late, and a
stronger pull close to the gate doesn't make up for it. A confidence that
grows as the gate approaches (e.g. soft labels by distance) would do the
same.

Videos (agent 14's camera below the arena; network leaders are drawn as
gate-centre leaders, since the network doesn't say how many posts it sees):

- `videos/network_slalom.mp4`: default slalom, seed 0; 23 / 23 / 15 agents
  through.
- `videos/network_winding.mp4`: the winding course of
  `emergent_leaders/videos/gates_winding.mp4` (seed 5, 7 m); 24 / 20 / 22 /
  23 / 23 through, against 23 / 17 / 24 / 25 / 20 with the rule.

```sh
python -m vision_leaders.run --model models/goalnet.pt --gates 3 --seed 0 --duration 320 \
    --camera-agent 14 --video videos/network_slalom.mp4
python -m vision_leaders.run --model models/goalnet.pt --gates 5 --course winding --gate-spacing 7 \
    --seed 5 --duration 440 --camera-agent 14 --video videos/network_winding.mp4
```

## Usage

```sh
pip install -e ../emergent_leaders -e ".[dev]"

python -m vision_leaders.train --out models/goalnet.pt               # ~40 min on 22 cores
python -m vision_leaders.evaluate models/goalnet.pt                  # network vs rule
python -m vision_leaders.run --model models/goalnet.pt --gates 3 --seed 0 --video slalom.mp4
python -m vision_leaders.run --gates 3 --seed 0 --teacher camera     # camera-limited rule drives
pip install -e ".[evolve]"
python -m vision_leaders.evolve --out models/aim_evolved.json --courses 12   # ~90 min on 22 cores
python -m vision_leaders.evolve --out models/aim_evolved_fixed.json --popsize 10 --courses 16 \
    --fix goal_gain=4 --fix beta=0.5 --init models/aim_evolved_default_motion.json
python -m vision_leaders.train --out models/goalnet_aim.pt --aim models/aim_evolved_default_motion.json
```

`run.py` builds courses and starts like `emergent_leaders.run` (same seed,
same course). Its videos show the swarm and, below it, the camera of one
agent (`--camera-agent`, ringed in the arena) with the rule's bearing
(dashed) and the network's (solid).
