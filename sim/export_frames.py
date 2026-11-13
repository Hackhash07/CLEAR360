"""
sim/export_frames.py — Main Blender headless frame export script for CLEAR360
=============================================================================

This script runs inside Blender (in background/headless mode) to procedurally
construct the CLEAR360 Indian parking environment, ego vehicle, 4-camera
surround rig, configure Cycles render settings, and export synthetic ground-truth
multi-modal dataset:
  - RGB images (1280x720 8-bit PNG)
  - Camera-space Z-depth maps (1280x720 32-bit float OpenEXR, channel 'Z')
  - World-space position maps (1280x720 32-bit float OpenEXR, channels 'X','Y','Z')
  - Semantic segmentation masks (1280x720 8-bit grayscale PNG, pixel = pass_index)
  - Extrinsic & intrinsic camera calibration (cameras.yaml)

Command-line invocation:
------------------------
    blender --background --python sim/export_frames.py -- \\
        --output /path/to/CAR360 --frames 20

Arguments:
----------
    --output PATH     Root directory for output data. Subdirectories data/sim_frames,
                      data/gt_depth, data/gt_position, data/gt_semantic, and
                      data/cameras.yaml will be created relative to this path.
                      Defaults to the current working directory.
    --frames N        Number of frames to render (default: 20, corresponding to
                      Blender frames 1..N and pipeline frame indices 0..N-1).
    --samples S       Cycles render samples per pixel (default: 32).
    --dry-run         Build scene and write cameras.yaml without running Cycles render.

Coordinate conventions (NEVER DEVIATE):
---------------------------------------
World / Vehicle frame (Blender default axes):
  - X = RIGHT of vehicle
  - Y = FORWARD of vehicle (direction of travel)
  - Z = UP
  - Ego vehicle centre placed at world origin (0, 0, 0); front bumper at Y ≈ +2.15 m.

Camera frame (OpenCV convention, stored in cameras.yaml):
  - x = right in image
  - y = down in image
  - z = forward into scene

Blender camera looks along its local -Z axis. Conversion to OpenCV camera frame:
  T_c2w_cv = M_blender @ diag(1, -1, -1, 1)    # flip Y and Z of camera space
  T_cam_world = inv(T_c2w_cv)                  # stored as T_cam_world in cameras.yaml

Depth convention:
  Blender Z pass = camera-space Z-depth (in metres, NOT Euclidean ray distance).
  In OpenCV camera frame, this equals positive-Z forward. Stored as float32.

Semantic class map (pass_index):
  0: background / sky
  1: road / ground
  2: ego vehicle
  3: other vehicle (parked)
  4: wall / perimeter structure
  5: pillar / column
  6: dynamic obstacle (pedestrian / mover)
  7: road furniture (kerb / speed-breaker)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Add project root to sys.path so sim.* modules can be imported inside Blender
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    import bpy
    import mathutils
except ImportError:
    print("[CLEAR360 ERROR] This script must be run inside Blender: blender --background --python sim/export_frames.py -- ...")
    sys.exit(1)

import numpy as np

from sim.constants import (
    CAMERA_CONFIGS,
    CAMERA_NAMES,
    CAMERA_RESOLUTION_H,
    CAMERA_RESOLUTION_W,
    OBSTACLE_KEYFRAMES,
    RENDER_SAMPLES,
    SEMANTIC_CLASSES,
)
from sim.camera_rig import (
    create_cameras,
    export_cameras_yaml,
    get_camera_intrinsics,
    get_T_cam_world,
)
from sim.environment import (
    animate_obstacle,
    create_environment,
    setup_lighting,
)
from sim.vehicle_model import (
    clear_scene,
    create_vehicle,
)


# ---------------------------------------------------------------------------
# ARGUMENT PARSING
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command line arguments passed after '--' in the Blender CLI call."""
    argv = sys.argv
    if "--" in argv:
        args_to_parse = argv[argv.index("--") + 1:]
    else:
        args_to_parse = []

    parser = argparse.ArgumentParser(
        description="CLEAR360 Headless Blender Synthetic Frame Exporter"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path.cwd(),
        help="Root output directory (default: current working directory).",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=20,
        help="Number of frames to render (default: 20).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=RENDER_SAMPLES,
        help=f"Cycles samples per pixel (default: {RENDER_SAMPLES}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build scene and write cameras.yaml, skip rendering.",
    )

    return parser.parse_args(args_to_parse)


# ---------------------------------------------------------------------------
# CYCLES CONFIGURATION
# ---------------------------------------------------------------------------

def configure_cycles(scene: bpy.types.Scene, samples: int = 32) -> None:
    """Configure Cycles render engine, resolution, samples, and device.

    Auto-detects GPU acceleration in order of platform preference:
    METAL (macOS Apple Silicon / AMD), CUDA / OPTIX (NVIDIA), HIP (AMD Linux/Win).
    Falls back gracefully to CPU if no GPU device is active.
    """
    scene.render.engine = "CYCLES"
    scene.cycles.samples = samples
    scene.render.resolution_x = CAMERA_RESOLUTION_W
    scene.render.resolution_y = CAMERA_RESOLUTION_H
    scene.render.resolution_percentage = 100

    # Auto-detect GPU compute devices
    try:
        prefs = bpy.context.preferences
        cycles_prefs = prefs.addons.get("cycles")
        if cycles_prefs is not None:
            cprefs = cycles_prefs.preferences
            # Check for Metal, OptiX, CUDA, HIP
            for compute_type in ["METAL", "OPTIX", "CUDA", "HIP", "ONEAPI"]:
                try:
                    cprefs.compute_device_type = compute_type
                    cprefs.get_devices()
                    gpu_found = False
                    for dev in cprefs.devices:
                        if dev.type == compute_type:
                            dev.use = True
                            gpu_found = True
                    if gpu_found:
                        scene.cycles.device = "GPU"
                        print(f"[CLEAR360] Cycles GPU acceleration enabled ({compute_type})")
                        return
                except Exception:
                    continue
    except Exception as err:
        print(f"[CLEAR360] Note on GPU device check: {err}")

    scene.cycles.device = "CPU"
    print("[CLEAR360] Cycles running on CPU")


# ---------------------------------------------------------------------------
# RENDER PASSES & COMPOSITOR SETUP
# ---------------------------------------------------------------------------

def enable_render_passes(view_layer: bpy.types.ViewLayer) -> None:
    """Enable Z-depth, World Position, and Object Index passes on the view layer."""
    view_layer.use_pass_z = True
    view_layer.use_pass_position = True
    view_layer.use_pass_object_index = True


def setup_compositor(scene: bpy.types.Scene) -> dict[str, bpy.types.Node]:
    """Set up the compositor node tree with File Output nodes for all passes.

    Nodes created:
      1. File Output RGB: PNG, 8-bit RGB
      2. File Output Depth: OpenEXR, 32-bit float Z-depth
      3. File Output Position: OpenEXR, 32-bit float XYZ
      4. File Output Semantic: PNG, 8-bit grayscale pass_index

    Returns
    -------
    dict[str, bpy.types.Node]
        References to the 4 File Output nodes so base_paths can be updated per camera.
    """
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()

    # Render Layers node
    rl_node = tree.nodes.new("CompositorNodeRLayers")
    rl_node.name = "Render_Layers"
    rl_node.location = (0, 0)

    # ------------------------------------------------------------------
    # 1. RGB Image Output (PNG 8-bit)
    # ------------------------------------------------------------------
    node_rgb = tree.nodes.new("CompositorNodeOutputFile")
    node_rgb.name = "FileOutput_RGB"
    node_rgb.label = "File Output RGB"
    node_rgb.format.file_format = "PNG"
    node_rgb.format.color_mode = "RGB"
    node_rgb.format.color_depth = "8"
    node_rgb.file_slots[0].path = "frame_"
    node_rgb.location = (450, 300)
    tree.links.new(rl_node.outputs["Image"], node_rgb.inputs[0])

    # ------------------------------------------------------------------
    # 2. Depth Output (OpenEXR 32-bit float, channel 'Z')
    # ------------------------------------------------------------------
    node_depth = tree.nodes.new("CompositorNodeOutputFile")
    node_depth.name = "FileOutput_Depth"
    node_depth.label = "File Output Depth (Z)"
    node_depth.format.file_format = "OPEN_EXR"
    node_depth.format.color_depth = "32"
    node_depth.format.color_mode = "BW"
    if hasattr(node_depth.format, "use_zbuffer"):
        node_depth.format.use_zbuffer = True
    node_depth.file_slots[0].path = "frame_"
    node_depth.location = (450, 100)
    tree.links.new(rl_node.outputs["Depth"], node_depth.inputs[0])

    # ------------------------------------------------------------------
    # 3. Position Output (OpenEXR 32-bit float RGB -> X,Y,Z world pos)
    # ------------------------------------------------------------------
    node_pos = tree.nodes.new("CompositorNodeOutputFile")
    node_pos.name = "FileOutput_Position"
    node_label = "File Output Position (XYZ)"
    node_pos.format.file_format = "OPEN_EXR"
    node_pos.format.color_depth = "32"
    node_pos.format.color_mode = "RGB"
    node_pos.file_slots[0].path = "frame_"
    node_pos.location = (450, -100)
    if "Position" in rl_node.outputs:
        tree.links.new(rl_node.outputs["Position"], node_pos.inputs[0])

    # ------------------------------------------------------------------
    # 4. Semantic / IndexOB Output (PNG 8-bit BW)
    # ------------------------------------------------------------------
    node_sem = tree.nodes.new("CompositorNodeOutputFile")
    node_sem.name = "FileOutput_Semantic"
    node_sem.label = "File Output Semantic (IndexOB)"
    node_sem.format.file_format = "PNG"
    node_sem.format.color_mode = "BW"
    node_sem.format.color_depth = "8"
    node_sem.file_slots[0].path = "frame_"
    node_sem.location = (450, -300)
    if "IndexOB" in rl_node.outputs:
        tree.links.new(rl_node.outputs["IndexOB"], node_sem.inputs[0])

    return {
        "rgb": node_rgb,
        "depth": node_depth,
        "position": node_pos,
        "semantic": node_sem,
    }


# ---------------------------------------------------------------------------
# POST-PROCESSING: ZERO-BASED INDEX REMAPPING & EXR CHANNEL ENFORCEMENT
# ---------------------------------------------------------------------------

def remap_camera_frame_indices(cam_dir: Path, num_frames: int, ext: str) -> None:
    """Remap 1-based Blender rendered frames (frame_0001..frame_0020) to 0-based
    CLEAR360 pipeline indices (frame_0000..frame_0019).

    Parameters
    ----------
    cam_dir    : Path directory containing the files for a camera
    num_frames : total count of frames rendered
    ext        : file extension ('.png' or '.exr')
    """
    for k in range(1, num_frames + 1):
        target_idx = k - 1
        src = cam_dir / f"frame_{k:04d}{ext}"
        dst = cam_dir / f"frame_{target_idx:04d}{ext}"
        if src.exists() and not dst.exists():
            src.rename(dst)


def ensure_exr_channel_compliance(depth_path: Path, pos_path: Path) -> None:
    """Ensure depth EXR has channel 'Z' and position EXR has channels 'X','Y','Z'.
    If written with default OpenEXR channels (e.g. 'Y' or 'R','G','B'), this
    converts or aliases them if the OpenEXR/Imath Python modules are available.
    """
    try:
        import OpenEXR
        import Imath

        # Verify / standardize depth EXR
        if depth_path.exists():
            exr_in = OpenEXR.InputFile(str(depth_path))
            header = exr_in.header()
            channels = header.get("channels", {})
            if "Z" not in channels:
                dw = header["dataWindow"]
                w = dw.max.x - dw.min.x + 1
                h = dw.max.y - dw.min.y + 1
                pt = Imath.PixelType(Imath.PixelType.FLOAT)
                # Read whatever channel exists ('Y' or 'R')
                avail_ch = list(channels.keys())
                if avail_ch:
                    src_ch = "Y" if "Y" in channels else ("R" if "R" in channels else avail_ch[0])
                    raw = exr_in.channel(src_ch, pt)
                    # Rewrite with channel 'Z'
                    new_header = OpenEXR.Header(w, h)
                    new_header["channels"] = {"Z": Imath.Channel(pt)}
                    exr_out = OpenEXR.OutputFile(str(depth_path), new_header)
                    exr_out.writePixels({"Z": raw})
                    exr_out.close()
            exr_in.close()

        # Verify / standardize position EXR
        if pos_path.exists():
            exr_in = OpenEXR.InputFile(str(pos_path))
            header = exr_in.header()
            channels = header.get("channels", {})
            if not all(c in channels for c in ("X", "Y", "Z")):
                dw = header["dataWindow"]
                w = dw.max.x - dw.min.x + 1
                h = dw.max.y - dw.min.y + 1
                pt = Imath.PixelType(Imath.PixelType.FLOAT)
                if all(c in channels for c in ("R", "G", "B")):
                    raw_r = exr_in.channel("R", pt)
                    raw_g = exr_in.channel("G", pt)
                    raw_b = exr_in.channel("B", pt)
                    new_header = OpenEXR.Header(w, h)
                    new_header["channels"] = {
                        "X": Imath.Channel(pt),
                        "Y": Imath.Channel(pt),
                        "Z": Imath.Channel(pt),
                    }
                    exr_out = OpenEXR.OutputFile(str(pos_path), new_header)
                    exr_out.writePixels({"X": raw_r, "Y": raw_g, "Z": raw_b})
                    exr_out.close()
            exr_in.close()
    except Exception:
        # If OpenEXR is not installed in Blender's python, skip post-rewrite
        pass


# ---------------------------------------------------------------------------
# MAIN EXPORT WORKFLOW
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    output_root = args.output.resolve()
    num_frames = args.frames
    samples = args.samples
    dry_run = args.dry_run

    print("\n" + "=" * 70)
    print("CLEAR360 — HEADLESS BLENDER SYNTHETIC DATA EXPORTER")
    print("=" * 70)
    print(f"Output Root    : {output_root}")
    print(f"Frame Count    : {num_frames} (Blender frames 1..{num_frames} -> indices 0..{num_frames-1})")
    print(f"Cycles Samples : {samples}")
    print(f"Dry Run        : {dry_run}")
    print("=" * 70 + "\n")

    # Step 1: Create output directories
    sim_frames_root = output_root / "data" / "sim_frames"
    gt_depth_root   = output_root / "data" / "gt_depth"
    gt_pos_root     = output_root / "data" / "gt_position"
    gt_sem_root     = output_root / "data" / "gt_semantic"

    for cam in CAMERA_NAMES:
        (sim_frames_root / cam).mkdir(parents=True, exist_ok=True)
        (gt_depth_root   / cam).mkdir(parents=True, exist_ok=True)
        (gt_pos_root     / cam).mkdir(parents=True, exist_ok=True)
        (gt_sem_root     / cam).mkdir(parents=True, exist_ok=True)

    # Step 2: Build 3D procedural scene
    print("[1/5] Clearing default scene objects...")
    clear_scene()

    print("[2/5] Constructing ego vehicle (Hyundai-Creta-approximate, pass_index=2)...")
    vehicle_parts = create_vehicle()
    print(f"      Created vehicle parts: {list(vehicle_parts.keys())}")

    print("[3/5] Constructing Indian parking environment & obstacles...")
    env_objects = create_environment()
    print(f"      Created environment objects: {list(env_objects.keys())}")

    print("[4/5] Setting up sun lighting & shadows...")
    sun_lamp = setup_lighting()

    scene = bpy.context.scene
    print("[5/5] Creating 4-camera surround rig (front, rear, left, right)...")
    cameras = create_cameras(scene)

    # Configure timeline
    scene.frame_start = 1
    scene.frame_end   = num_frames

    # Export camera calibrations to YAML (both config/cameras.yaml and data/cameras.yaml)
    cameras_yaml_target = output_root / "config" / "cameras.yaml"
    data_yaml_target = output_root / "data" / "cameras.yaml"
    print(f"\nExporting camera calibrations to {cameras_yaml_target} and {data_yaml_target}...")
    export_cameras_yaml(cameras, str(cameras_yaml_target))
    export_cameras_yaml(cameras, str(data_yaml_target))

    if dry_run:
        print("\n[DRY RUN COMPLETE] Scene built and cameras.yaml written. Skipping render.")
        return

    # Configure Cycles & Compositor
    print("\nConfiguring Cycles renderer and multi-pass compositor...")
    configure_cycles(scene, samples=samples)
    view_layer = scene.view_layers[0]
    enable_render_passes(view_layer)
    file_nodes = setup_compositor(scene)

    # Render each camera
    print("\nStarting multi-camera rendering sequence...")
    for cam_name in CAMERA_NAMES:
        cam_obj = cameras[cam_name]
        print(f"\n>>> Rendering camera: [{cam_name.upper()}] (Blender frames 1..{num_frames})")

        # Set active camera
        scene.camera = cam_obj

        # Point File Output nodes to camera-specific subdirectories
        rgb_dir   = sim_frames_root / cam_name
        depth_dir = gt_depth_root   / cam_name
        pos_dir   = gt_pos_root     / cam_name
        sem_dir   = gt_sem_root     / cam_name

        file_nodes["rgb"].base_path      = str(rgb_dir)   + os.sep
        file_nodes["depth"].base_path    = str(depth_dir) + os.sep
        file_nodes["position"].base_path = str(pos_dir)   + os.sep
        file_nodes["semantic"].base_path = str(sem_dir)   + os.sep

        # Render all frames for this camera
        bpy.ops.render.render(animation=True, write_still=True)

        # Remap output frames from 1-based (frame_0001..0020) to 0-based (frame_0000..0019)
        remap_camera_frame_indices(rgb_dir, num_frames, ".png")
        remap_camera_frame_indices(depth_dir, num_frames, ".exr")
        remap_camera_frame_indices(pos_dir, num_frames, ".exr")
        remap_camera_frame_indices(sem_dir, num_frames, ".png")

        # Channel compliance check
        for f_idx in range(num_frames):
            ensure_exr_channel_compliance(
                depth_dir / f"frame_{f_idx:04d}.exr",
                pos_dir / f"frame_{f_idx:04d}.exr",
            )

        print(f"    Completed camera [{cam_name}]: {num_frames} frames saved.")

    # Re-verify cameras.yaml export in both locations
    export_cameras_yaml(cameras, str(cameras_yaml_target))
    export_cameras_yaml(cameras, str(data_yaml_target))

    # Print summary
    print("\n" + "=" * 70)
    print("CLEAR360 EXPORT COMPLETE")
    print("=" * 70)
    print(f"Total cameras : {len(CAMERA_NAMES)} ({', '.join(CAMERA_NAMES)})")
    print(f"Total frames  : {num_frames} per camera ({len(CAMERA_NAMES) * num_frames} total FrameSets)")
    print(f"RGB images    : {sim_frames_root}")
    print(f"GT Depth      : {gt_depth_root}")
    print(f"GT Position   : {gt_pos_root}")
    print(f"GT Semantic   : {gt_sem_root}")
    print(f"Calibration   : {cameras_yaml_target}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
