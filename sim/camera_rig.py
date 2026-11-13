"""
sim/camera_rig.py — Four-camera surround rig for CLEAR360
==========================================================

This module creates four Blender camera objects arranged around the ego vehicle
and exports their calibration data to cameras.yaml.

COORDINATE CONVENTIONS — CRITICAL:
  World frame (Blender default):
    X = RIGHT of vehicle
    Y = FORWARD (direction of travel)
    Z = UP
    Vehicle centre at world origin.

  Camera frame (OpenCV convention, stored in cameras.yaml):
    x = right in image
    y = down  in image
    z = forward (into scene)

  Blender camera looks along its LOCAL −Z axis (OpenCV: +Z forward).
  Conversion from Blender C2W to OpenCV C2W:
    T_c2w_cv = M_blender @ diag(1, -1, -1, 1)
      where M_blender = cam_obj.matrix_world  (Blender's camera-to-world 4×4)
      diag(1,-1,-1,1) flips the Y and Z axes of CAMERA space:
        • Blender cam Y (up-in-cam)   → OpenCV cam -Y (down-in-image is +y)
        • Blender cam Z (behind cam)  → OpenCV cam -Z → becomes forward (+z)
      This yields a camera-to-world matrix in OpenCV convention.

  World-to-camera (what we store as T_cam_world):
    T_cam_world = inv(T_c2w_cv)
    p_cam = T_cam_world @ p_world_hom

  Depth: Blender's Z render pass = camera-space Z-depth (not Euclidean).
         Equivalent to OpenCV positive-Z forward depth. Stored float32 metres.

CAMERAS:
  front  — position (0.0,  2.15, 0.90), looks +Y, tilted −8° down
  rear   — position (0.0, −2.15, 0.90), looks −Y, tilted −8° down
  left   — position (−1.09, 0.30, 1.25), looks −X, tilted −6° down
  right  — position (+1.09, 0.30, 1.25), looks +X, tilted −6° down

INTRINSICS (identical for all cameras):
  Resolution   : 1280 × 720 pixels
  Sensor width : 36 mm  (35 mm-equivalent)
  Focal length : 10.39 mm  → ≈ 120° HFOV
  fx = fy      : (10.39/36.0) × 1280 ≈ 369.4 px
  cx           : 640.0 px  (image centre)
  cy           : 360.0 px  (image centre)

Usage:
  from sim.camera_rig import create_cameras, get_camera_intrinsics, export_cameras_yaml
  cameras = create_cameras(bpy.context.scene)
  export_cameras_yaml(cameras, "/path/to/CAR360/data/cameras.yaml")
"""

import math
import numpy as np
import yaml

import bpy
import mathutils

from sim.constants import (
    CAMERA_CONFIGS,
    CAMERA_RESOLUTION_W,
    CAMERA_RESOLUTION_H,
    CAMERA_SENSOR_W_MM,
    CAMERA_FOCAL_MM,
    CAMERA_FX,
    CAMERA_FY,
    CAMERA_CX,
    CAMERA_CY,
)


# ---------------------------------------------------------------------------
# CAMERA CREATION
# ---------------------------------------------------------------------------

def create_cameras(scene: bpy.types.Scene) -> dict:
    """Create four Blender camera objects and configure their intrinsics.

    Camera orientation is computed as follows:
      1. Start with the nominal look direction (unit vector in world space).
      2. Apply the downward tilt by rotating the vector around the camera's
         local right axis (the axis perpendicular to the look direction and Z):
           right_vec = look_dir × Z_up     (cross product)
           tilt_quat = Quaternion(right_vec, tilt_rad)
           final_dir = tilt_quat @ look_dir
      3. Convert to Blender Euler angles via:
           euler = final_dir.to_track_quat('-Z', 'Y').to_euler()
         '-Z' because Blender camera looks along its local −Z axis.
         'Y'  is the camera's up vector in camera space.

    Parameters
    ----------
    scene : bpy.types.Scene — the active Blender scene

    Returns
    -------
    dict
        Mapping of camera_name (str) → bpy.types.Object (camera data-block attached)
    """
    cameras: dict[str, bpy.types.Object] = {}

    # Remove any cameras created by previous calls (e.g., the default "Camera")
    for obj in list(bpy.data.objects):
        if obj.type == "CAMERA":
            bpy.data.objects.remove(obj, do_unlink=True)

    for cam_name, cfg in CAMERA_CONFIGS.items():
        pos_x, pos_y, pos_z = cfg["position"]
        look_x, look_y, look_z = cfg["look_dir"]
        tilt_deg = cfg["tilt_deg"]

        # ------------------------------------------------------------------
        # Step 1: Build look direction vector in world space
        # ------------------------------------------------------------------
        look_vec = mathutils.Vector((look_x, look_y, look_z)).normalized()

        # ------------------------------------------------------------------
        # Step 2: Compute the camera's local right axis for tilt rotation
        # right_vec = look_vec × Z_up
        # This gives the axis about which we tilt down.
        # For cameras looking ±Y, right_vec = ±X.
        # For cameras looking ±X, right_vec = ∓Y (by cross product rule).
        # ------------------------------------------------------------------
        z_up = mathutils.Vector((0.0, 0.0, 1.0))
        right_vec = look_vec.cross(z_up)

        # Guard: if look_vec is parallel to Z (edge case), use X as fallback
        if right_vec.length < 1e-6:
            right_vec = mathutils.Vector((1.0, 0.0, 0.0))
        else:
            right_vec.normalize()

        # ------------------------------------------------------------------
        # Step 3: Apply tilt rotation around the right axis
        # tilt_deg < 0 means tilting downward (rotating look vector toward -Z)
        # Quaternion.rotation convention: (axis, angle_in_radians)
        # ------------------------------------------------------------------
        tilt_rad = math.radians(tilt_deg)
        tilt_quat = mathutils.Quaternion(right_vec, tilt_rad)
        final_dir = tilt_quat @ look_vec
        final_dir.normalize()

        # ------------------------------------------------------------------
        # Step 4: Convert direction to Blender camera Euler rotation
        # to_track_quat('-Z', 'Y') rotates so that the object's local −Z
        # aligns with final_dir, and local Y points up as much as possible.
        # ------------------------------------------------------------------
        cam_euler = final_dir.to_track_quat("-Z", "Y").to_euler()

        # ------------------------------------------------------------------
        # Create camera data-block and object
        # ------------------------------------------------------------------
        cam_data = bpy.data.cameras.new(name=cam_name)
        cam_obj  = bpy.data.objects.new(name=cam_name, object_data=cam_data)
        scene.collection.objects.link(cam_obj)

        cam_obj.location      = (pos_x, pos_y, pos_z)
        cam_obj.rotation_euler = cam_euler

        # ------------------------------------------------------------------
        # Set camera intrinsics
        # ------------------------------------------------------------------
        cam_data.type         = "PERSP"
        cam_data.lens         = CAMERA_FOCAL_MM      # focal length in mm
        cam_data.sensor_width = CAMERA_SENSOR_W_MM   # sensor width in mm
        cam_data.sensor_fit   = "HORIZONTAL"          # derive height from aspect

        # Blender uses sensor_width + focal_length to compute FOV.
        # We set resolution separately on the render settings in export_frames.py.

        cameras[cam_name] = cam_obj

    return cameras


# ---------------------------------------------------------------------------
# INTRINSICS QUERY
# ---------------------------------------------------------------------------

def get_camera_intrinsics(camera_name: str) -> dict:
    """Return the intrinsic parameters for the named camera.

    All four cameras share identical intrinsics in CLEAR360.

    Parameters
    ----------
    camera_name : str — one of 'front', 'rear', 'left', 'right'

    Returns
    -------
    dict with keys:
      width            : int   — image width in pixels (1280)
      height           : int   — image height in pixels (720)
      fx               : float — focal length along X in pixels (≈369.4)
      fy               : float — focal length along Y in pixels (≈369.4)
      cx               : float — principal point X in pixels (640.0)
      cy               : float — principal point Y in pixels (360.0)
      focal_length_mm  : float — focal length in mm (10.39)
      sensor_width_mm  : float — sensor width in mm (36.0)

    Notes
    -----
    Derivation of fx (pixels):
      fx = (focal_length_mm / sensor_width_mm) * image_width_px
         = (10.39 / 36.0) * 1280 ≈ 369.42 px
    HFOV:
      θ_h = 2 * arctan(image_width / (2 * fx)) ≈ 120°
    """
    if camera_name not in CAMERA_CONFIGS:
        raise ValueError(
            f"Unknown camera '{camera_name}'. Valid names: {list(CAMERA_CONFIGS.keys())}"
        )

    return {
        "width":           CAMERA_RESOLUTION_W,
        "height":          CAMERA_RESOLUTION_H,
        "fx":              CAMERA_FX,
        "fy":              CAMERA_FY,
        "cx":              CAMERA_CX,
        "cy":              CAMERA_CY,
        "focal_length_mm": CAMERA_FOCAL_MM,
        "sensor_width_mm": CAMERA_SENSOR_W_MM,
    }


# ---------------------------------------------------------------------------
# EXTRINSICS COMPUTATION
# ---------------------------------------------------------------------------

def get_T_cam_world(cam_obj: bpy.types.Object) -> np.ndarray:
    """Compute the 4×4 OpenCV world-to-camera matrix for a Blender camera.

    Formula:
      M          = cam_obj.matrix_world     # Blender C2W (Blender convention)
                                            # columns: [right, up, -forward, pos]
      R_flip     = diag(1, -1, -1, 1)      # flip camera Y and Z to OpenCV convention
                                            # Blender cam Y (up) → OpenCV −Y (down-in-image)
                                            # Blender cam Z (behind) → OpenCV −Z (forward)
      T_c2w_cv   = M @ R_flip              # OpenCV camera-to-world in world coords
      T_cam_world = inv(T_c2w_cv)          # OpenCV world-to-camera   ← stored in YAML

    Usage of T_cam_world:
      p_cam_hom = T_cam_world @ p_world_hom    # project world point to camera frame
      (x_c, y_c, z_c, 1) = T_cam_world @ (X_w, Y_w, Z_w, 1)
      u = fx * x_c / z_c + cx    (z_c > 0 for visible points)
      v = fy * y_c / z_c + cy

    Parameters
    ----------
    cam_obj : bpy.types.Object — a camera object whose matrix_world is current

    Returns
    -------
    np.ndarray, shape (4, 4), dtype float64
        T_cam_world: transforms a homogeneous world point to OpenCV camera space.
    """
    # Step 1: Read Blender camera-to-world matrix (column-major in Blender,
    # np.array converts to row-major standard numpy layout).
    # M[i][j] = element at row i, column j.
    M = np.array(cam_obj.matrix_world, dtype=np.float64)  # shape (4,4)

    # Step 2: Define the axis-flip matrix.
    # In Blender camera space: X=right, Y=up, Z=behind (looking along −Z).
    # In OpenCV  camera space: x=right, y=down, z=forward.
    # Mapping: Blender_x → OpenCV_x (unchanged), Blender_y → −OpenCV_y, Blender_z → −OpenCV_z
    # Applying on the RIGHT of M transforms camera-space axes:
    #   new_col_j = M @ R_flip[:,j]
    R_flip = np.diag([1.0, -1.0, -1.0, 1.0])   # (4,4)

    # Step 3: OpenCV camera-to-world matrix
    T_c2w_cv = M @ R_flip   # (4,4), columns now: [right, −up, forward_into_scene, pos]

    # Step 4: Invert to get world-to-camera (what we need to project world points)
    T_cam_world = np.linalg.inv(T_c2w_cv)   # (4,4), dtype float64

    return T_cam_world


# ---------------------------------------------------------------------------
# CAMERAS.YAML EXPORT
# ---------------------------------------------------------------------------

def export_cameras_yaml(cameras_dict: dict, output_path: str) -> None:
    """Write cameras.yaml calibration file for all four cameras.

    The YAML file documents:
      • Coordinate convention (world and camera frames)
      • Depth convention
      • Per-camera: world position, intrinsics, and T_cam_world (4×4 OpenCV W2C)

    YAML structure (see sim/README_sim.md for the full schema):
      coordinate_convention: "..."
      depth_convention: "..."
      cameras:
        front:
          position_world: [x, y, z]
          intrinsics:
            width, height, fx, fy, cx, cy, focal_length_mm, sensor_width_mm
          T_cam_world:
            - [r00, r01, r02, r03]
            - [r10, r11, r12, r13]
            - [r20, r21, r22, r23]
            - [r30, r31, r32, r33]
        rear: ...
        left: ...
        right: ...

    Parameters
    ----------
    cameras_dict : dict
        Mapping of camera_name → bpy.types.Object (as returned by create_cameras).
    output_path : str
        Absolute path to the output cameras.yaml file.
    """
    from pathlib import Path

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Build the YAML data structure
    doc: dict = {
        "coordinate_convention": (
            "world=X_right_Y_forward_Z_up; "
            "camera=OpenCV(x_right,y_down,z_forward)"
        ),
        "depth_convention": (
            "camera_z_depth_metres; NOT euclidean ray distance"
        ),
        "cameras": {},
    }

    for cam_name, cam_obj in cameras_dict.items():
        # World position of camera (translation column of Blender C2W matrix)
        pos_world = list(cam_obj.location)   # [x, y, z] in world frame

        # Intrinsics dict
        intrinsics = get_camera_intrinsics(cam_name)

        # 4×4 OpenCV world-to-camera matrix
        T_cam_world = get_T_cam_world(cam_obj)

        # Serialise T_cam_world as a list of 4 rows, each a list of 4 floats.
        # We round to 8 decimal places for readability while keeping precision.
        T_list = [
            [round(float(T_cam_world[r, c]), 8) for c in range(4)]
            for r in range(4)
        ]

        doc["cameras"][cam_name] = {
            "position_world": [round(float(v), 6) for v in pos_world],
            "intrinsics": {
                "width":           int(intrinsics["width"]),
                "height":          int(intrinsics["height"]),
                "fx":              round(float(intrinsics["fx"]), 4),
                "fy":              round(float(intrinsics["fy"]), 4),
                "cx":              round(float(intrinsics["cx"]), 4),
                "cy":              round(float(intrinsics["cy"]), 4),
                "focal_length_mm": round(float(intrinsics["focal_length_mm"]), 4),
                "sensor_width_mm": round(float(intrinsics["sensor_width_mm"]), 4),
            },
            "T_cam_world": T_list,
        }

    # Write YAML with explicit ordering (Python 3.7+ dicts are ordered)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.dump(
            doc,
            f,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )

    print(f"[CLEAR360] cameras.yaml written → {out_path}")
