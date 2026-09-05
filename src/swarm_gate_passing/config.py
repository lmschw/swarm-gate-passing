"""Central configuration for swarm-gate-passing.

Combines two prior research codebases:
  - The evolutionary (CMA-ES + Hebbian-ABCD) flocking controller and its wind/drag/
    battery physics, vendored from energy_efficient_flocking's
    ants26_replication/experiment/ (itself a Python replication of the ANTS 2026
    paper "Energy-Efficient Flocking in Self-Organized Robot Swarms", Mahdavi et
    al., whose MATLAB reference lives in that project's ants-2026-polimi-*/ dir).
  - Gradient-map path sensing, ported from volcano_gradient's light-intensity
    sensor and path/map generator (see environment/).

Two further pieces layer on top of that combination:
  - A second sensor mode ("thymio"), a 7-channel raw IR proximity model ported
    from ants26_replication/thymio_ir_variant/sensor_model.py, as an alternative
    to the idealized quadrant distance/bearing sensor ("quadrant" mode, the
    original default -- see sensor_model.py vs sensor_model_thymio.py).
  - A physical gate barrier (environment/gate.py) placed across the arena with
    its opening centered on the path, and a 4-stage incremental curriculum
    (flock_cohesion -> flock_gradient -> flock_gate -> flock_gate_speed) that
    trains a swarm to pass through it one skill at a time -- see
    simulation.py's EpisodeResult/stage_fitness.
"""
import math

# --- Core simulation parameters (wind/drag/battery physics + spawning) ---
DT = 0.5                    # time-step [s]
ROBOT_RAD = 0.055            # robot radius [m]
WIND_RAD = 0.15              # robot's wind-occlusion radius [m]
X_RANGE = [-5.0, 5.0]        # simulation X bounds [m]
Y_RANGE = [-5.0, 5.0]        # simulation Y bounds [m]
V_WIND = 10.0                # freestream wind speed

# --- Agent spawning ---
SPAWN_SQUARE_SIZE = 3.0      # side length of the square agents are randomly spawned in [m]
SPAWN_MIDPOINT = [0.0, 0.0]  # center of the spawn square
SPAWN_MIN_DIST_SLACK = 0.1   # extra slack (on top of 2*ROBOT_RAD) enforced between spawned agents

# --- Collision / walls ---
COLLISION_MIN_DIST_SLACK = 0.01      # min_dist = COLLISION_MIN_DIST_SLACK + 2*ROBOT_RAD
WALL_MARGIN_FACTOR = 0.5             # wall_margin = ROBOT_RAD * WALL_MARGIN_FACTOR
WALL_COLLISION_WEIGHT = 3            # each wall hit counts as this many collisions

# --- Wind-tracking camera window (the x-range RayTraceCircularRobots is evaluated over) ---
WIND_TRACKING_WINDOW_WIDTH = 10.0    # total width of the tracking window [m]
WIND_TRACKING_MAX_SPAN = 9.8         # cap on the swarm's own x-extent within that window [m]

# --- Wind / wake ray-tracing (RayTraceCircularRobots) ---
UINF = 100.0                  # freestream ("full power") wind value
KAPPA = 10.0                  # drag force scale factor (paper Eq. 3: kappa=10)
WAKE_RECOVERY_RATE = 1.0      # fraction of wake gap recovered per grid step outside a robot's radius
WAKE_PERCENT_DROP = 0.25      # wind intensity drop on entering/switching a robot's wake
WAKE_MAX_WALL_SPAN = 0.7      # controls how sharply the wall effect kicks in (lower = more wall effect)
WAKE_MIN_POWER_X = 30.0       # floor applied to power immediately behind a robot
WAKE_MIN_POWER_Y = 10.0       # floor applied to power after the wall-effect pass
WAKE_ALPHA = 0.5              # decay rate of the first smoothing kernel
WAKE_BETA = 0.5               # decay rate of the second smoothing kernel
WAKE_X_SMOOTHING_1 = 100      # first-pass smoothing kernel size divisor (x)
WAKE_Y_SMOOTHING_1 = 50       # first-pass smoothing kernel size divisor (y)
WAKE_X_SMOOTHING_2 = 50       # second-pass smoothing kernel size divisor (x)
WAKE_Y_SMOOTHING_2 = 50       # second-pass smoothing kernel size divisor (y)
WAKE_THR_OK_DELTA = 1.0       # a cell counts as "free-stream" once within this much of UINF

# --- Drag force (dragforce) ---
DRAG_UPSTREAM_LOOKAHEAD_FACTOR = 1.1   # how far upstream (in wind_rad) to sample the wind grid
DRAG_AIR_DENSITY = 1.225                # kg/m^3
DRAG_COEFFICIENT_AREA = 0.0045          # effective drag coefficient * frontal area

# --- Battery drainage (batterydrainage) ---
BATTERY_WHEEL_POWER_DIVISOR = 4.0    # divisor applied to summed absolute wheel speeds
BATTERY_MIN_DRAIN = 0.10             # floor on per-step drain (idle power draw)
BATTERY_DRAIN_SCALE = 2.0            # matches MATLAB reference's literal `agents(:,4) - 2*batt_drain`

# --- Video / plotting output ---
HEBBIAN_VIDEO_PATH = "hebbian_alone.mp4"
VIDEO_FPS = 10.0
VIDEO_SIZE = (1200, 800)
VIDEO_FIGSIZE = (12, 8)
VIDEO_VIEWPORT_HALF_WIDTH = 5.0   # camera half-width/height around the swarm's center of mass [m]
VIDEO_ARROW_LEN = 0.3             # heading-arrow length in the rendered frame [m]
VIDEO_QUIVER_WIDTH = 0.004        # heading-arrow line width

HEBBIAN_DEFAULT_SEED = 42

# =====================================================================================
# --- Hebbian ABCD neural-network controller ---
# Each robot runs a small MLP (ReLU, ReLU, tanh) updated online by a Hebbian rule;
# the rule's coefficients (not the weights themselves) are what CMA-ES evolves,
# shared by every agent in a swarm. See hebbian_controller.py, sensor_model.py,
# sensor_model_thymio.py, simulation.py, optimize.py.
# =====================================================================================

HEBBIAN_N_AGENTS = 20             # default swarm size (overridden per run, e.g. --n-agents 10)
HEBBIAN_SENSING_RADIUS = 2.01     # R: neighbor detection radius [m] for "quadrant" sensor mode
HEBBIAN_LINEAR_VEL_MAX = 0.2      # m/s, tanh output #1 rescaled to [-this, this]
HEBBIAN_ANGULAR_VEL_MAX = math.pi / 5  # rad/s, tanh output #2 rescaled to [-this, this]

# --- Battery & wind grid ---
HEBBIAN_MAX_BATTERY = 100.0        # starting battery for all agents but one
HEBBIAN_MIN_BATTERY = 100.0        # starting battery for the single "weakest" agent
HEBBIAN_NX = 200                   # wind grid resolution; lower to cut simulation cost
HEBBIAN_NY = 200

# --- Neural controller architecture ---
HEBBIAN_N_HIDDEN = 10             # both hidden layers
HEBBIAN_N_OUTPUTS = 2             # (v, w)
HEBBIAN_LEARNING_RATE = 0.1       # mu in delta_w = mu*(A*ni*nj + B*ni + C*nj + D)
HEBBIAN_WEIGHT_INIT_RANGE = 1.0   # NN weights (not the ABCD genome) initialized uniformly in [-this, this]

# --- Sensor modes ---
# "quadrant": 4 quadrants x (distance, bearing) + battery + heading + gradient light = 11
#     inputs -- the idealized sensor used throughout ants26_replication's other
#     variants (encodes each neighbor's relative POSITION: distance and bearing).
# "thymio": 7 raw Thymio-II IR proximity readings (no neighbor identity/bearing,
#     wall and neighbor reflectance indistinguishable) + battery + heading +
#     gradient light = 10 inputs, ported from thymio_ir_variant/sensor_model.py.
# Both put battery/heading/light in the same last-3 positions (see
# BATTERY_ROW/HEADING_ROW/LIGHT_ROW below), so simulation.py can ablate/feed them
# generically regardless of which sensor module is in use.
SENSOR_MODES = ("quadrant", "thymio")


def n_inputs_for_sensor_mode(sensor_mode):
    if sensor_mode == "quadrant":
        return 11
    if sensor_mode == "thymio":
        return 10
    raise ValueError(f"unknown sensor_mode {sensor_mode!r}, choose from {SENSOR_MODES}")


def battery_row(n_inputs):
    return n_inputs - 3


def heading_row(n_inputs):
    return n_inputs - 2


def light_row(n_inputs):
    return n_inputs - 1


def n_abcd_for(n_inputs, n_hidden=None, n_outputs=None):
    """Genome length for a given input width: 4 ABCD coefficients per NN weight,
    W1: n_inputs x n_hidden, W2: n_hidden x n_hidden, W3: n_hidden x n_outputs."""
    n_hidden = n_hidden if n_hidden is not None else HEBBIAN_N_HIDDEN
    n_outputs = n_outputs if n_outputs is not None else HEBBIAN_N_OUTPUTS
    return 4 * (n_inputs * n_hidden + n_hidden * n_hidden + n_hidden * n_outputs)


# Default ("quadrant") sizing -- kept as module constants for convenience/back-compat
# (existing stages, tests, and the default CLI all target this sensor mode).
HEBBIAN_N_INPUTS = n_inputs_for_sensor_mode("quadrant")   # 11
HEBBIAN_N_ABCD = n_abcd_for(HEBBIAN_N_INPUTS)              # 920

HEBBIAN_ABCD_INIT_RANGE = 5.0     # ABCD-rules initial mean sampled uniformly from [-this, this]
HEBBIAN_ABCD_BOUNDS = [-5.0, 5.0]  # CMA-ES hard bounds

# --- Real Thymio II IR proximity sensor geometry (prox.horizontal), "thymio" mode ---
# Sourced from Webots' community-calibrated Thymio2.proto model, converting each of
# the 7 DistanceSensor mount positions into a bearing angle; ordering matches
# thymio_swarm_platform's prox.horizontal[0..6] (indices 0-4 front left-to-right,
# 5-6 rear left/right; +angle = robot's own left).
THYMIO_IR_ANGLES = (
    0.6737, 0.3386, 0.0, -0.3386, -0.6737,   # front: left, front-left, center, front-right, right
    2.3387, -2.3387,                          # rear: left, right
)
THYMIO_IR_HALF_APERTURE = math.radians(10.0)   # disclosed assumption, not a measured spec
THYMIO_IR_RANGE = 0.12            # [m], surface gap (not center-to-center); effective range ~0-12cm

# --- CMA-ES hyperparameters ---
HEBBIAN_CMAES_POPSIZE = 30        # lambda
HEBBIAN_CMAES_GEN_MAX = 100       # Ngen, termination condition, PER STAGE
HEBBIAN_CMAES_SIGMA0 = 0.3        # initial covariance/step-size
HEBBIAN_N_REPEATS = 3             # simulations per candidate (different seeds); fitness = median

HEBBIAN_BATCH_SEEDS = [42, 123, 777, 2026, 888, 99, 412, 555, 1010, 8432]

# --- Gradient-map path sensing (combined from volcano_gradient) ---
GRADIENT_MAP_METERS_PER_PIXEL = 0.05   # map resolution when generating maps sized to X_RANGE/Y_RANGE
GRADIENT_MAP_NOISE_MAGNITUDE = 0.5      # matches volcano_gradient's g_noise_mag uniform-noise sensor model
GRADIENT_DEFAULT_PATH_WIDTH_M = 0.6     # default corridor width used by environment.generate_default_map_set

# --- Gate-passing task ---
# Physical barrier placed at the track's finish line, with its opening centered on
# the path's actual rendered centerline (read directly off the map -- see
# environment/gate.py -- so it's exact for any path shape/wavelength, not just
# the sine-curve formula). GATE_WIND_ENABLED is off: energy/drafting is not part
# of this task's fitness (see below), and skipping the O(Nx) wind ray-trace is
# also the single biggest per-step cost lever, which matters a lot with domain
# randomization run at scale.
GATE_WIND_ENABLED = False
GATE_OPENING_WIDTH_M = 1.0
GATE_COLLISION_WEIGHT = 250.0     # same convention/scale as save_battery_avoid_all's collision_w
GATE_WALL_COL_MULT = 3.0          # gate-barrier hits are counted together with arena-wall hits

# --- Incremental curriculum weights ---
# Each stage below ADDS one new fitness component to the previous stage's, rather
# than training all of cohesion + gradient-following + gate-passing + speed at once
# from a random genome -- confirmed necessary in practice: a single-shot combined
# fitness (even after fixing a physics bug and rebalancing magnitudes) produced
# genomes that neither followed the gradient nor stayed together. cohesion_dist and
# path_deviation_m are both already in meters and empirically similar magnitude
# (roughly 1-4m), so both get a plain weight of 1.0 -- i.e. eff -= metric directly,
# no rescaling -- rather than guessing a ratio between them. distance_w/success_bonus
# are introduced only once "reaching the gate" becomes the goal (stage 3): distance
# credit is capped (dist_travelled_capped, GATE_DISTANCE_CAP_SLACK_M/GATE_MAX_STEPS
# below) so it acts purely as a smooth "getting closer" gradient rather than
# something a genome can farm by cruising indefinitely, and success_bonus is set an
# order of magnitude above the other terms' typical range so genuinely succeeding is
# unambiguously the best outcome. speed_w is introduced last, and kept small (a
# polish/tiebreaker, not something worth trading cohesion or accuracy for).
GATE_COHESION_WEIGHT = 1.0          # eff -= cohesion_dist / this (mean pairwise distance, meters)
GATE_PATH_DEVIATION_WEIGHT = 1.0    # eff -= mean_path_deviation_m / this (meters)
GATE_POST_GATE_COHESION_WEIGHT = 1.0  # eff -= post_gate_cohesion_dist / this (meters, regrouping phase only)
GATE_DISTANCE_WEIGHT = 1.0          # eff += dist_travelled_capped / this (meters, capped -- see below)
GATE_SPEED_WEIGHT = 0.05            # eff += mean_speed_mps / this (mean_speed ~0.06-0.2 m/s observed
                                     # -> contributes roughly 1-4, comparable to the other terms, not dominant)
# mean_speed alone doesn't discourage stop-start oscillation: a fast burst followed by a
# full stop can still average out fine -- confirmed in practice (a flock_gate_speed genome
# with a good mean_speed still visibly moved, stopped, moved, stopped in its video). This
# penalizes actual STOPPED time directly, agent-seconds (same convention as collision_time:
# a per-step count of agents below the threshold, summed over the episode and scaled by dt).
# Threshold is ~15-20% of the ~0.1-0.16 m/s cruising speeds actually observed, well below
# real movement but above near-zero noise. Weight: worst case (all 10 agents stopped for a
# full ~200s episode) is 2000 agent-seconds; a genome stopped ~20% of the time for half the
# swarm is more like 200 agent-seconds -- dividing by 100 keeps that typical-bad-case penalty
# (~2) in the same range as the other smooth terms, not dominant but a real, direct cost.
GATE_STOP_SPEED_THRESHOLD_MPS = 0.02
GATE_STOPPED_TIME_WEIGHT = 100.0
GATE_SUCCESS_BONUS = 30.0           # eff += this iff EVERY agent crossed the finish line -- roughly an
                                     # order of magnitude above the smooth terms' typical spread, so success
                                     # is unambiguously better than any amount of near-miss behavior

# The finish line ("success") sits this far past the LAST gate, not at it -- a gate can
# physically scatter the swarm as individuals squeeze through separately, so requiring
# them to keep moving together for a bit further is what actually tests (and rewards)
# regrouping, rather than crediting "everyone got through" the instant they're all
# individually clear of the barrier. Only relevant once a gate exists (stage 3+).
GATE_POST_GATE_DISTANCE_M = 1.5
# Distance-reward cap and episode-length cap: without these, a genome that never
# attempts the (hard, all-or-nothing) success bonus can still rack up large reward by
# simply running the full battery-implied ~1000 steps heading roughly forward --
# confirmed in practice (an early reweighted-but-uncapped genome reached
# dist_travelled=57.8m against a real target of ~4-6.5m, by cruising the entire
# episode). GATE_MAX_STEPS also cuts typical episode cost substantially: covering the
# ~7m worst-case target distance at the observed ~0.1-0.16 m/s takes on the order of
# 100-140 steps, so 400 (200 simulated seconds) is a generous multiple of that for
# real maneuvering, not a tight ceiling.
GATE_DISTANCE_CAP_SLACK_M = 1.0
GATE_MAX_STEPS = 400
# Domain randomization: each simulated episode samples one wavelength (all stages)
# and, once a gate exists, one gate placement -- so the evolved genome doesn't just
# memorize a single layout.
GATE_FREQ_CHOICES = (2.0, 4.0, 6.0)
GATE_FINISH_X_CHOICES = (-2.5, -3.5, -4.5)

# --- Staged curricula ---
# Two independent curricula: the original energy-efficiency one (Table 2's 3
# stages, plus the fixed-map gradient-following stage added when this project
# first combined in volcano_gradient's sensing) is a separate, unrelated task
# from the gate-passing curriculum this project's current work is about --
# GATE_STAGES is what optimize.py's --stages defaults to; the energy stages
# remain selectable (--stages walk_left ...) but are never run unless asked
# for explicitly.
#
# GATE_STAGES is itself a 4-step incremental curriculum, each stage's genome
# seeded from the previous one (run_stage/train_one_seed always chains within
# a single --stages invocation): flock_cohesion (move together) ->
# flock_gradient (+ follow the gradient) -> flock_gate (+ pass the gate,
# rewarding capped progress-to-goal + the success bonus + post-gate
# regrouping) -> flock_gate_speed (+ reward speed). Only flock_gate and
# flock_gate_speed physically enable a gate (GATE_ENABLED_STAGES) -- the
# earlier stages have nothing to pass yet. Curricula don't chain into each
# other -- each --stages run starts its own fresh genome unless --init-genome
# is given.
ENERGY_STAGES = ("walk_left", "save_battery_avoid_wall", "save_battery_avoid_all", "follow_gradient_path")
GATE_STAGES = ("flock_cohesion", "flock_gradient", "flock_gate", "flock_gate_speed")
GATE_ENABLED_STAGES = ("flock_gate", "flock_gate_speed")
HEBBIAN_STAGES = ENERGY_STAGES + GATE_STAGES
HEBBIAN_STAGE_WIND_ENABLED = {
    "walk_left": False,
    "save_battery_avoid_wall": True,
    "save_battery_avoid_all": True,
    "follow_gradient_path": True,
    "flock_cohesion": GATE_WIND_ENABLED,
    "flock_gradient": GATE_WIND_ENABLED,
    "flock_gate": GATE_WIND_ENABLED,
    "flock_gate_speed": GATE_WIND_ENABLED,
}
# Per-stage fitness weights, all optional (a missing/None key means that term is
# entirely absent, and distance_w explicitly set to 0.0 means "off" too, since it's
# a multiplier rather than a divisor -- see simulation.stage_fitness for the full
# formula). Every term beyond distance is a divisor against the named metric
# (battery/collision/proximity/cohesion/path_deviation/post_gate_cohesion: penalties,
# i.e. eff -= metric/weight; battery/path/speed: rewards, i.e. eff += metric/weight),
# except success_bonus (a flat additive bonus) and distance_w (a multiplier, default
# HEBBIAN_EFF_DISTANCE_WEIGHT when the key is absent -- always set explicitly here,
# including to 0.0, so no gate-curriculum stage accidentally inherits that default).
HEBBIAN_STAGE_FITNESS_WEIGHTS = {
    "walk_left": {
        "wall_col_mult": 3.0,
    },
    "save_battery_avoid_wall": {
        "battery_w": 5.0, "collision_w": 500.0, "wall_col_mult": 3.0,
    },
    "save_battery_avoid_all": {
        "battery_w": 5.0, "collision_w": 250.0, "wall_col_mult": 3.0,
        "include_inter_robot_collision": True,
    },
    "follow_gradient_path": {
        "battery_w": 5.0, "collision_w": 250.0, "wall_col_mult": 3.0,
        "include_inter_robot_collision": True, "path_w": 5.0,
    },
    # --- gate curriculum: each stage = the previous stage's dict + one new term ---
    "flock_cohesion": {
        "distance_w": 0.0,
        "collision_w": GATE_COLLISION_WEIGHT, "wall_col_mult": GATE_WALL_COL_MULT,
        "include_inter_robot_collision": True,
        "cohesion_w": GATE_COHESION_WEIGHT,
    },
    "flock_gradient": {
        "distance_w": 0.0,
        "collision_w": GATE_COLLISION_WEIGHT, "wall_col_mult": GATE_WALL_COL_MULT,
        "include_inter_robot_collision": True,
        "cohesion_w": GATE_COHESION_WEIGHT,
        "path_deviation_w": GATE_PATH_DEVIATION_WEIGHT,
    },
    "flock_gate": {
        "distance_w": GATE_DISTANCE_WEIGHT,
        "collision_w": GATE_COLLISION_WEIGHT, "wall_col_mult": GATE_WALL_COL_MULT,
        "include_inter_robot_collision": True,
        "cohesion_w": GATE_COHESION_WEIGHT,
        "path_deviation_w": GATE_PATH_DEVIATION_WEIGHT,
        "success_bonus": GATE_SUCCESS_BONUS,
        "post_gate_cohesion_w": GATE_POST_GATE_COHESION_WEIGHT,
    },
    "flock_gate_speed": {
        "distance_w": GATE_DISTANCE_WEIGHT,
        "collision_w": GATE_COLLISION_WEIGHT, "wall_col_mult": GATE_WALL_COL_MULT,
        "include_inter_robot_collision": True,
        "cohesion_w": GATE_COHESION_WEIGHT,
        "path_deviation_w": GATE_PATH_DEVIATION_WEIGHT,
        "success_bonus": GATE_SUCCESS_BONUS,
        "post_gate_cohesion_w": GATE_POST_GATE_COHESION_WEIGHT,
        "speed_w": GATE_SPEED_WEIGHT,
        "stopped_w": GATE_STOPPED_TIME_WEIGHT,
    },
}

# Explicit distance weight -- without one, CMA-ES can cheaply preserve battery/avoid
# risk by barely moving; this keeps distance-travelled (progress toward/through the
# gate) dominant. Only used by the energy-efficiency curriculum -- every gate-
# curriculum stage sets its own distance_w explicitly (including 0.0) above.
HEBBIAN_EFF_DISTANCE_WEIGHT = 16.0
