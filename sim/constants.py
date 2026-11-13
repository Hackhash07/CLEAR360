"""
sim/constants.py — Physical constants, camera specs, semantic class map
========================================================================

All values are authoritative for the CLEAR360 simulation.  Every constant is
documented with its units and source / rationale.

COORDINATE CONVENTIONS (never deviate):
  World / vehicle frame  — Blender default axes:
      X = RIGHT of vehicle
      Y = FORWARD of vehicle (direction of travel)
      Z = UP
  Vehicle centre at world origin; front bumper at Y ≈ +2.15 m.

  Camera frame  — OpenCV convention:
      x = right in image
      y = down in image
      z = into scene (forward)

  Blender camera looks along its local −Z axis.
  Conversion Blender → OpenCV C2W:
      T_c2w_cv = M_blender @ diag(1, −1, −1, 1)
  W2C stored in cameras.yaml:
      T_cam_world = inv(T_c2w_cv)

  Depth: Blender Z-pass = camera-space Z-depth (NOT Euclidean distance).
         This equals OpenCV positive-Z forward depth.  Stored as float32 m.
"""

# ---------------------------------------------------------------------------
# VEHICLE DIMENSIONS  (Hyundai Creta 2022-approximate)
# ---------------------------------------------------------------------------

# Overall body extents in metres (half-extents define bounding box from origin)
VEHICLE_LENGTH_M   = 4.30     # Y: −2.15 to +2.15 m  (front/rear bumper)
VEHICLE_WIDTH_M    = 1.79     # X: −0.895 to +0.895 m (body, excluding mirrors)
VEHICLE_HEIGHT_M   = 1.635    # Z: 0.00 to 1.635 m

# Half-extents for convenience
VEHICLE_HALF_LEN   = VEHICLE_LENGTH_M / 2.0   # 2.15 m
VEHICLE_HALF_W     = VEHICLE_WIDTH_M  / 2.0   # 0.895 m

# Wheelbase and axle Y-positions
WHEELBASE_M        = 2.60     # distance between front and rear axle centres
FRONT_AXLE_Y       = +1.30    # front axle at Y=+1.30 m
REAR_AXLE_Y        = -1.30    # rear axle  at Y=−1.30 m

# Wheel geometry
WHEEL_RADIUS_M     = 0.335    # outer radius of tyre
WHEEL_WIDTH_M      = 0.215    # tyre section width
WHEEL_OFFSET_X     = 0.90     # centre of wheel hub: X = ±0.90 m from vehicle CL

# Roof sub-panel (raised relative to main body top)
ROOF_HEIGHT_START_Z = 0.80    # roof panel starts at Z=0.80 m
ROOF_HEIGHT_M       = 0.835   # roof panel height: 0.80 → 1.635 m
ROOF_LENGTH_M       = 2.80    # Y extent (shorter than body)
ROOF_WIDTH_M        = 1.70    # X extent (slightly narrower than body)

# Side mirrors
MIRROR_TIP_X        = 1.085   # outer tip X = ±1.085 m from vehicle CL
MIRROR_Z_LOW        = 1.10    # mirror lower edge Z
MIRROR_Z_HIGH       = 1.40    # mirror upper edge Z
MIRROR_HEIGHT_M     = MIRROR_Z_HIGH - MIRROR_Z_LOW    # 0.30 m
MIRROR_DEPTH_M      = 0.12    # Y depth of mirror housing
MIRROR_WIDTH_M      = 0.19    # X extent (from body side to tip)
MIRROR_Y_CENTRE     = 1.50    # Y position (near front doors)

# Semantic pass index for ALL vehicle parts
VEHICLE_PASS_INDEX  = 2

# ---------------------------------------------------------------------------
# CAMERA RIG — FOUR FIXED CAMERAS
# ---------------------------------------------------------------------------
# All cameras share the same intrinsics.
# Orientation is computed as:
#   direction_vec = look_dir + tilt_component
#   euler = direction_vec.to_track_quat('-Z', 'Y').to_euler()
# (Blender camera looks along its local −Z axis; 'Y' is the up vector in cam space)

CAMERA_RESOLUTION_W  = 1280          # pixels
CAMERA_RESOLUTION_H  = 720           # pixels
CAMERA_SENSOR_W_MM   = 36.0          # sensor width in millimetres (35 mm equiv)
CAMERA_FOCAL_MM      = 10.39         # focal length in mm → ≈ 120° HFOV

# Derived intrinsics (pixel units)
# fx = (focal_mm / sensor_w_mm) * image_width_px
#    = (10.39 / 36.0) * 1280 ≈ 369.4 px
CAMERA_FX = (CAMERA_FOCAL_MM / CAMERA_SENSOR_W_MM) * CAMERA_RESOLUTION_W  # ≈ 369.4
CAMERA_FY = CAMERA_FX                        # square pixels → fy = fx
CAMERA_CX = CAMERA_RESOLUTION_W  / 2.0       # 640.0 px  (principal point centred)
CAMERA_CY = CAMERA_RESOLUTION_H  / 2.0       # 360.0 px

# Per-camera extrinsics: (position_xyz, look_direction_xyz, tilt_degrees)
# look_direction_xyz is a unit vector in world space (before tilt).
# Tilt is applied as a rotation around the camera's local X-axis (downward = negative degrees).
# Positive tilt would look UP; negative tilt points camera slightly toward ground.
import math as _math

CAMERA_CONFIGS = {
    "front": {
        "position":  (0.0,  2.15, 0.90),
        "look_dir":  (0.0,  1.0,  0.0),   # +Y (forward)
        "tilt_deg":  -8.0,                  # tilt down 8°
        "pass_index": None,                 # cameras have no semantic index
    },
    "rear": {
        "position":  (0.0, -2.15, 0.90),
        "look_dir":  (0.0, -1.0,  0.0),   # −Y (backward)
        "tilt_deg":  -8.0,
        "pass_index": None,
    },
    "left": {
        "position":  (-1.09, 0.30, 1.25),
        "look_dir":  (-1.0,  0.0,  0.0),  # −X (left)
        "tilt_deg":  -6.0,
        "pass_index": None,
    },
    "right": {
        "position":  (+1.09, 0.30, 1.25),
        "look_dir":  (+1.0,  0.0,  0.0),  # +X (right)
        "tilt_deg":  -6.0,
        "pass_index": None,
    },
}

CAMERA_NAMES = list(CAMERA_CONFIGS.keys())   # ["front","rear","left","right"]

# ---------------------------------------------------------------------------
# RENDER SETTINGS
# ---------------------------------------------------------------------------
RENDER_SAMPLES        = 32       # Cycles samples per pixel
RENDER_FRAME_START    = 1        # first Blender frame
RENDER_FRAME_END      = 20       # last  Blender frame  (→ indices 0–19)
RENDER_FRAME_COUNT    = 20       # total frames per camera

# ---------------------------------------------------------------------------
# ENVIRONMENT DIMENSIONS
# (all measurements in metres; positions are object centres in world space)
# ---------------------------------------------------------------------------

# Road ground plane
ROAD_SIZE_M = 20.0    # square, centred at origin, Z=0

# Walls (Cubes scaled to their bounding box)
WALL_CONFIGS = {
    "wall_north": {
        "scale":    (20.0,  0.3, 2.5),
        "location": (0.0,   9.5, 1.25),    # Y=+9.5 m; centre Z=1.25 m
        "pass_index": 4,
        "color":    (0.75, 0.74, 0.72, 1.0),   # light concrete RGBA
    },
    "wall_south": {
        "scale":    (20.0,  0.3, 2.5),
        "location": (0.0,  -9.5, 1.25),
        "pass_index": 4,
        "color":    (0.75, 0.74, 0.72, 1.0),
    },
    "wall_east": {
        "scale":    (0.3, 20.0, 2.5),
        "location": (9.5,  0.0, 1.25),
        "pass_index": 4,
        "color":    (0.75, 0.74, 0.72, 1.0),
    },
    "wall_west": {
        "scale":    (0.3, 20.0, 2.5),
        "location": (-9.5, 0.0, 1.25),
        "pass_index": 4,
        "color":    (0.75, 0.74, 0.72, 1.0),
    },
}

# Pillar (Cylinder)
PILLAR_CONFIG = {
    "pillar_1": {
        "radius":   0.20,
        "depth":    2.50,
        "location": (-6.0, 4.0, 1.25),   # centre at mid-height
        "pass_index": 5,
        "color":    (0.60, 0.60, 0.60, 1.0),
    },
}

# Parked cars (Cubes)
PARKED_CAR_CONFIGS = {
    "parked_car_1": {
        "scale":    (1.75, 4.2, 1.5),    # (X width, Y length, Z height)
        "location": (-5.0, -5.0, 0.75),  # centre at half-height
        "pass_index": 3,
        "color":    (0.08, 0.14, 0.60, 1.0),   # blue
    },
    "parked_car_2": {
        "scale":    (1.75, 4.2, 1.5),
        "location": (+5.0, -5.0, 0.75),
        "pass_index": 3,
        "color":    (0.60, 0.08, 0.08, 1.0),   # red
    },
}

# Kerbs and speed breaker (Cubes)
ROAD_FURNITURE_CONFIGS = {
    "kerb_left": {
        "scale":    (0.2, 15.0, 0.15),
        "location": (-4.5, 0.0, 0.075),
        "pass_index": 7,
        "color":    (0.85, 0.85, 0.85, 1.0),
    },
    "kerb_right": {
        "scale":    (0.2, 15.0, 0.15),
        "location": (+4.5, 0.0, 0.075),
        "pass_index": 7,
        "color":    (0.85, 0.85, 0.85, 1.0),
    },
    "speed_breaker": {
        "scale":    (3.5, 0.4, 0.10),
        "location": (0.0, 5.5, 0.05),
        "pass_index": 7,
        "color":    (0.90, 0.75, 0.10, 1.0),   # yellow
    },
}

# Moving obstacle (Cube, 0.5 m sides)
OBSTACLE_CONFIG = {
    "obstacle_mover": {
        "scale":    (0.5, 0.5, 0.5),
        "pass_index": 6,
        "color":    (0.90, 0.30, 0.05, 1.0),   # orange
    },
}

# Keyframes: list of (frame_number, (x, y, z))
OBSTACLE_KEYFRAMES = [
    (1,  ( 3.0,  7.0, 0.25)),
    (10, (-3.0,  5.5, 0.25)),
    (20, (-5.0,  3.0, 0.25)),
]

# ---------------------------------------------------------------------------
# LIGHTING
# ---------------------------------------------------------------------------
SUN_ELEVATION_DEG = 55.0    # elevation above horizon in degrees
SUN_AZIMUTH_DEG   = 160.0   # azimuth angle in degrees (measured from +Y/North, clockwise)
SUN_STRENGTH      = 3.0     # Blender sun lamp energy (W/m²-equivalent)

# ---------------------------------------------------------------------------
# SEMANTIC CLASS MAP  (pass_index → class name)
# ---------------------------------------------------------------------------
SEMANTIC_CLASSES = {
    0: "background / sky",
    1: "road / ground",
    2: "ego vehicle",
    3: "other vehicle (parked)",
    4: "wall / structure",
    5: "pillar / post",
    6: "dynamic obstacle (pedestrian/mover)",
    7: "road furniture (kerb / speed-breaker)",
}

# ---------------------------------------------------------------------------
# OUTPUT SUB-DIRECTORY TEMPLATES
# (all relative to the --output root passed to export_frames.py)
# ---------------------------------------------------------------------------
OUT_RGB_DIR      = "data/sim_frames/{camera}"        # → frame_{n:04d}.png
OUT_DEPTH_DIR    = "data/gt_depth/{camera}"          # → frame_{n:04d}.exr  (Z channel)
OUT_POSITION_DIR = "data/gt_position/{camera}"       # → frame_{n:04d}.exr  (X Y Z channels)
OUT_SEMANTIC_DIR = "data/gt_semantic/{camera}"       # → frame_{n:04d}.png  (uint8 pass_index)
CAMERAS_YAML_PATH = "data/cameras.yaml"

# Frame filename templates (Blender File Output uses its own zero-based counter,
# but we name them 0000-based: frame_0000.png … frame_0019.png).
FRAME_NAME_TEMPLATE = "frame_{n:04d}"
