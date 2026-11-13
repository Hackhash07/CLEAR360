"""
sim/export_photoreal.py — Photorealistic Frame Exporter for CLEAR360 (M4.2)
===========================================================================

Executes headless Blender Cycles rendering for Frame 0000 across all 4 cameras
(front, rear, left, right) using the photorealistic PBR scene:
  - 1280x720 8-bit RGB PNG
  - 1280x720 32-bit float Z-depth EXR (channel 'Z')
  - 1280x720 32-bit float World Position EXR (channels 'X', 'Y', 'Z')
  - 1280x720 8-bit Semantic Mask PNG (pixel = pass_index)
  - Saves master scene to sim/blender_scene.blend

Invocation:
-----------
    blender --background --python sim/export_photoreal.py -- --frame 0 --samples 32
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import sys

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

user_site = Path.home() / ".local" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
if user_site.exists() and str(user_site) not in sys.path:
    sys.path.insert(0, str(user_site))

try:
    import bpy
    import mathutils
    import OpenEXR
    import Imath
    import numpy as np
    import cv2
except ImportError as e:
    print(f"[ERROR] Missing required library: {e}")
    sys.exit(1)

from sim.constants import (
    CAMERA_NAMES,
    CAMERA_RESOLUTION_H,
    CAMERA_RESOLUTION_W,
)
from sim.camera_rig import (
    create_cameras,
    export_cameras_yaml,
)
from sim.vehicle_model import clear_scene
from sim.photoreal_scene import (
    build_photoreal_environment,
    build_photoreal_ego_vehicle,
    setup_photoreal_lighting_and_sky,
)


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    args_to_parse = argv[argv.index("--") + 1:] if "--" in argv else []

    parser = argparse.ArgumentParser(description="CLEAR360 Photoreal Cycles Exporter")
    parser.add_argument("--frame", type=int, default=0, help="Frame index to render (default: 0)")
    parser.add_argument("--samples", type=int, default=32, help="Cycles samples per pixel (default: 32)")
    parser.add_argument("--save-blend", action="store_true", default=True, help="Save master .blend file")
    return parser.parse_args(args_to_parse)


def configure_cycles_renderer(scene: bpy.types.Scene, samples: int = 32, device: str = "CPU") -> None:
    scene.render.engine = "CYCLES"
    scene.cycles.samples = samples
    scene.render.resolution_x = CAMERA_RESOLUTION_W
    scene.render.resolution_y = CAMERA_RESOLUTION_H
    scene.render.resolution_percentage = 100

    # CPU rendering in Blender on Apple Silicon is multi-threaded and runs in seconds
    scene.cycles.device = "CPU"
    print(f"[CLEAR360] Cycles rendering active on {scene.cycles.device} ({samples} samples/px)")


def setup_compositor_nodes(scene: bpy.types.Scene, temp_dir: Path) -> tuple[bpy.types.Node, bpy.types.Node, bpy.types.Node]:
    """Configure Compositor File Output nodes for Z-depth, Position, and Semantics."""
    # Enable view layer passes
    vl = scene.view_layers[0]
    vl.use_pass_z = True
    vl.use_pass_position = True
    vl.use_pass_object_index = True
    vl.update_render_passes()

    # Create compositor node group
    ng = bpy.data.node_groups.new("CompositorTree", "CompositorNodeTree")
    scene.compositing_node_group = ng
    rl = ng.nodes.new("CompositorNodeRLayers")
    rl.location = (0, 0)

    # 1. Depth Output Node
    out_depth = ng.nodes.new("CompositorNodeOutputFile")
    out_depth.location = (450, 200)
    out_depth.directory = str(temp_dir) + os.sep
    out_depth.file_name = "raw_depth_"
    out_depth.file_output_items.clear()
    i_d = out_depth.file_output_items.new("FLOAT", "Z")
    i_d.override_node_format = True
    i_d.format.file_format = "OPEN_EXR"
    i_d.format.color_depth = "32"
    ng.links.new(rl.outputs["Depth"], out_depth.inputs["Z"])

    # 2. Position Output Node
    out_pos = ng.nodes.new("CompositorNodeOutputFile")
    out_pos.location = (450, 0)
    out_pos.directory = str(temp_dir) + os.sep
    out_pos.file_name = "raw_pos_"
    out_pos.file_output_items.clear()
    i_p = out_pos.file_output_items.new("VECTOR", "XYZ")
    i_p.override_node_format = True
    i_p.format.file_format = "OPEN_EXR"
    i_p.format.color_depth = "32"
    ng.links.new(rl.outputs["Position"], out_pos.inputs["XYZ"])

    # 3. Semantic Output Node
    out_sem = ng.nodes.new("CompositorNodeOutputFile")
    out_sem.location = (450, -200)
    out_sem.directory = str(temp_dir) + os.sep
    out_sem.file_name = "raw_sem_"
    out_sem.file_output_items.clear()
    i_s = out_sem.file_output_items.new("FLOAT", "IndexOB")
    i_s.override_node_format = True
    i_s.format.file_format = "OPEN_EXR"
    i_s.format.color_depth = "32"
    ng.links.new(rl.outputs["Object Index"], out_sem.inputs["IndexOB"])

    return out_depth, out_pos, out_sem


def process_rendered_passes(
    cam_name: str,
    target_frame: int,
    temp_dir: Path,
    base_dirs: dict[str, Path],
) -> None:
    """Read temporary multi-pass EXRs, reformat to standard CLEAR360 schemas, and save to final paths."""
    w, h = CAMERA_RESOLUTION_W, CAMERA_RESOLUTION_H
    pt = Imath.PixelType(Imath.PixelType.FLOAT)

    # 1. Depth EXR
    temp_depth_file = temp_dir / "raw_depth_.exr"
    final_depth_file = base_dirs["depth"] / cam_name / f"frame_{target_frame:04d}.exr"
    if temp_depth_file.exists():
        exr_in = OpenEXR.InputFile(str(temp_depth_file))
        raw_z = exr_in.channel("Z.V", pt)
        exr_in.close()

        # Write clean single-channel 'Z' EXR
        new_header = OpenEXR.Header(w, h)
        new_header["channels"] = {"Z": Imath.Channel(pt)}
        exr_out = OpenEXR.OutputFile(str(final_depth_file), new_header)
        exr_out.writePixels({"Z": raw_z})
        exr_out.close()
        temp_depth_file.unlink()

    # 2. Position EXR
    temp_pos_file = temp_dir / "raw_pos_.exr"
    final_pos_file = base_dirs["position"] / cam_name / f"frame_{target_frame:04d}.exr"
    if temp_pos_file.exists():
        exr_in = OpenEXR.InputFile(str(temp_pos_file))
        raw_x = exr_in.channel("XYZ.X", pt)
        raw_y = exr_in.channel("XYZ.Y", pt)
        raw_z = exr_in.channel("XYZ.Z", pt)
        exr_in.close()

        # Write clean 3-channel ('X', 'Y', 'Z') EXR
        new_header = OpenEXR.Header(w, h)
        new_header["channels"] = {
            "X": Imath.Channel(pt),
            "Y": Imath.Channel(pt),
            "Z": Imath.Channel(pt),
        }
        exr_out = OpenEXR.OutputFile(str(final_pos_file), new_header)
        exr_out.writePixels({"X": raw_x, "Y": raw_y, "Z": raw_z})
        exr_out.close()
        temp_pos_file.unlink()

    # 3. Semantic Mask PNG
    temp_sem_file = temp_dir / "raw_sem_.exr"
    final_sem_file = base_dirs["semantic"] / cam_name / f"frame_{target_frame:04d}.png"
    if temp_sem_file.exists():
        exr_in = OpenEXR.InputFile(str(temp_sem_file))
        raw_idx = exr_in.channel("IndexOB.V", pt)
        exr_in.close()

        sem_arr = np.frombuffer(raw_idx, dtype=np.float32).reshape((h, w))
        sem_uint8 = np.clip(np.round(sem_arr), 0, 255).astype(np.uint8)
        cv2.imwrite(str(final_sem_file), sem_uint8)
        temp_sem_file.unlink()


def main() -> None:
    args = parse_args()
    target_frame = args.frame
    samples = args.samples
    save_blend = args.save_blend

    print("\n" + "=" * 70)
    print("CLEAR360 — PHOTOREALISTIC CYCLES RENDER (M4.2)")
    print("=" * 70)
    print(f"Target Frame   : {target_frame} (Blender frame {target_frame + 1})")
    print(f"Cycles Samples : {samples}")
    print(f"Cameras        : {', '.join(CAMERA_NAMES)}")
    print("=" * 70 + "\n")

    # Output directories dedicated to photorealistic data (preserving baseline)
    data_root = project_root / "data"
    base_dirs = {
        "rgb": data_root / "photoreal_frames",
        "depth": data_root / "photoreal_gt_depth",
        "position": data_root / "photoreal_gt_position",
        "semantic": data_root / "photoreal_gt_semantic",
    }
    for d in base_dirs.values():
        for cam in CAMERA_NAMES:
            (d / cam).mkdir(parents=True, exist_ok=True)

    temp_dir = data_root / "photoreal_temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # 1. Clear scene
    print("[1/5] Clearing default scene...")
    clear_scene()

    # 2. Build Ego Vehicle with PBR paint, glass, and wheels
    print("[2/5] Building photorealistic ego vehicle (Creta dimensions, metallic paint & clearcoat)...")
    ego_parts = build_photoreal_ego_vehicle()
    print(f"      Created parts: {list(ego_parts.keys())}")

    # 3. Build Environment with PBR asphalt, road markings, concrete, and obstacles
    print("[3/5] Building photorealistic parking environment (PBR asphalt, markings, concrete)...")
    env_objs = build_photoreal_environment()
    print(f"      Created environment objects: {list(env_objs.keys())}")

    # 4. Natural Lighting and Multi-scattering Sky
    print("[4/5] Setting up natural sun lighting & multi-scattering sky...")
    sun_lamp = setup_photoreal_lighting_and_sky()

    # 5. Create Camera Rig
    scene = bpy.context.scene
    print("[5/5] Creating 4-camera surround rig...")
    cameras = create_cameras(scene)

    # Save master scene .blend file
    if save_blend:
        blend_path = project_root / "sim" / "blender_scene.blend"
        print(f"\nSaving master scene file to: {blend_path}")
        bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))

    # Configure Cycles & Compositor
    print("\nConfiguring Cycles engine and multi-pass compositor...")
    configure_cycles_renderer(scene, samples=samples)
    setup_compositor_nodes(scene, temp_dir)

    # Render target frame (Blender frame = target_frame + 1)
    blender_frame = target_frame + 1
    scene.frame_set(blender_frame)

    print(f"\nRendering Frame {target_frame:04d} across all 4 cameras...")
    for cam_name in CAMERA_NAMES:
        cam_obj = cameras[cam_name]
        print(f"  >>> Rendering [{cam_name.upper()}] ...", end="", flush=True)

        scene.camera = cam_obj

        # Direct RGB PNG save path
        rgb_file = base_dirs["rgb"] / cam_name / f"frame_{target_frame:04d}.png"
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_mode = "RGB"
        scene.render.image_settings.color_depth = "8"
        scene.render.filepath = str(rgb_file)

        # Render
        bpy.ops.render.render(write_still=True)

        # Format and save depth EXR, position EXR, and semantic mask PNG
        process_rendered_passes(cam_name, target_frame, temp_dir, base_dirs)
        print(" DONE")

    # Clean up temp dir
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)

    print("\n" + "=" * 70)
    print("PHOTOREALISTIC RENDERING COMPLETE (Frame 0000)")
    print("=" * 70)
    for k, d in base_dirs.items():
        print(f"  {k:10s}: {d}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
