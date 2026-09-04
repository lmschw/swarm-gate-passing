"""Central configuration for swarm-gate-passing.

Combines two prior research codebases:
  - The evolutionary (CMA-ES + Hebbian-ABCD) flocking controller and its wind/drag/
    battery physics, vendored from energy_efficient_flocking's
    ants26_replication/experiment/ (itself a Python replication of the ANTS 2026
    paper "Energy-Efficient Flocking in Self-Organized Robot Swarms", Mahdavi et
    al., whose MATLAB reference lives in that project's ants-2026-polimi-*/ dir).
  - Gradient-map path sensing, ported from volcano_gradient's light-intensity
    sensor and path/map generator (see environment/).

The new piece connecting them is the 11th sensor input (HEBBIAN_N_INPUTS below)
and the "follow_gradient_path" curriculum stage -- see sensor_model.py and
simulation.py.
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
# simulation.py, optimize.py.
# =====================================================================================

# --- Robot & sensing ---
HEBBIAN_N_AGENTS = 20             # swarm size
HEBBIAN_SENSING_RADIUS = 2.01     # R: neighbor detection radius [m]; also the "no neighbor" default distance
HEBBIAN_LINEAR_VEL_MAX = 0.2      # m/s, tanh output #1 rescaled to [-this, this]
HEBBIAN_ANGULAR_VEL_MAX = math.pi / 5  # rad/s, tanh output #2 rescaled to [-this, this]

# --- Battery & wind grid ---
HEBBIAN_MAX_BATTERY = 100.0        # starting battery for all agents but one
HEBBIAN_MIN_BATTERY = 100.0        # starting battery for the single "weakest" agent
HEBBIAN_NX = 200                   # wind grid resolution; lower to cut simulation cost
HEBBIAN_NY = 200

# --- Neural controller architecture ---
# 4 quadrants x (distance, bearing) + battery + compass heading + gradient light
# reading (the piece added by combining in volcano_gradient's sensing -- see
# sensor_model.py and environment/sensing.py).
HEBBIAN_N_INPUTS = 11
HEBBIAN_N_HIDDEN = 10             # both hidden layers
HEBBIAN_N_OUTPUTS = 2             # (v, w)
HEBBIAN_LEARNING_RATE = 0.1       # mu in delta_w = mu*(A*ni*nj + B*ni + C*nj + D)
# Weight-matrix shapes, in flatten/unflatten order: W1: N_INPUTS x N_HIDDEN,
# W2: N_HIDDEN x N_HIDDEN, W3: N_HIDDEN x N_OUTPUTS. Randomly initialized each
# episode using a uniform distribution in [-1, 1] for all three (not evolved --
# only the ABCD Hebbian-rule coefficients are).
HEBBIAN_WEIGHT_INIT_RANGE = 1.0

# --- ABCD genotype ---
# 4 coefficients (A, B, C, D) per NN weight, shared across all agents in a swarm:
# 4 * (11*10 + 10*10 + 10*2) = 920 total parameters.
HEBBIAN_N_ABCD = 4 * (HEBBIAN_N_INPUTS * HEBBIAN_N_HIDDEN + HEBBIAN_N_HIDDEN * HEBBIAN_N_HIDDEN
                      + HEBBIAN_N_HIDDEN * HEBBIAN_N_OUTPUTS)
HEBBIAN_ABCD_INIT_RANGE = 5.0     # ABCD-rules initial mean sampled uniformly from [-this, this]
HEBBIAN_ABCD_BOUNDS = [-5.0, 5.0]  # CMA-ES hard bounds

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

# --- Staged curriculum ---
# Stage 1 has no wind and rewards distance only, to avoid evolving the trivial strategy of
# just riding the tailwind. Stage 2 turns on wind and adds battery + wall-collision terms.
# Stage 3 adds a general inter-robot collision penalty on top of stage 2. Stage 4
# (added by this project) keeps stage 3's terms and adds a reward for staying on a
# gradient-mapped path, exercising the sensing added in environment/. Each stage's
# CMA-ES run is seeded from the previous stage's best genome; stage 1 alone starts
# from a fresh uniform-random ABCD_init.
HEBBIAN_STAGES = ("walk_left", "save_battery_avoid_wall", "save_battery_avoid_all", "follow_gradient_path")
HEBBIAN_STAGE_WIND_ENABLED = {
    "walk_left": False,
    "save_battery_avoid_wall": True,
    "save_battery_avoid_all": True,
    "follow_gradient_path": True,
}
# Fitness weights per stage: eff = HEBBIAN_EFF_DISTANCE_WEIGHT*dist + batt/battery_w -
# (collision_time + wall_col_mult*wall_collision_time) / collision_w - cohesion_dist /
# cohesion_w - proximity_penalty / proximity_w + path_alignment / path_w. A weight of
# None means that term is entirely absent. path_alignment is scaled to [0, 100] (see
# simulation.py), so path_w's scale is calibrated against battery_w's (also a
# [0, 100]-scaled term), not against the raw [0, 255] map intensity.
HEBBIAN_STAGE_FITNESS_WEIGHTS = {
    #                             battery_w   collision_w   wall_col_mult   include_inter_robot_collision   cohesion_w   proximity_w   path_w
    "walk_left":                 (None,        None,         3.0,            False,                          None,        None,         None),
    "save_battery_avoid_wall":   (5.0,         500.0,        3.0,            False,                          None,        None,         None),
    "save_battery_avoid_all":    (5.0,         250.0,        3.0,            True,                           None,        None,         None),
    "follow_gradient_path":      (5.0,         250.0,        3.0,            True,                           None,        None,         5.0),
}

# Explicit distance weight -- without one, CMA-ES can cheaply preserve battery by
# barely moving; this keeps distance-travelled dominant over battery preservation.
HEBBIAN_EFF_DISTANCE_WEIGHT = 16.0
