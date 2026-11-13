"""
sim/generate_calibration_dataset.py — Part C: Dedicated Calibration Dataset Generator
=====================================================================================

Creates a dedicated, independent calibration environment in Blender Cycles with
known geometric objects positioned at exact metric ground-truth distances:
  - 0.75 m, 1.0 m, 1.5 m, 2.0 m, 3.0 m, 5.0 m, 8.0 m
from each camera's optical centre.

Object classes include:
  - Textured calibration boxes / cubes (0.5m)
  - Textured calibration cylinders / columns (r=0.25m, h=1.5m)
  - Vehicle mockup obstacles (metallic car paint)
  - PBR asphalt ground plane

This dataset is used strictly to fit metric depth calibration models
(global scale, per-camera scale, affine), while our parking scene Frame 0000
remains the strictly held-out evaluation set.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import shutil
import sys

project_root = Path(__file__).resolve().parent.parent
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
    CAMERA_CONFIGS,
    CAMERA_NAMES,
    CAMERA_RESOLUTION_H,
    CAMERA_RESOLUTION_W,
)
from sim.camera_rig import create_cameras
from sim.vehicle_model import clear_scene
from sim.photoreal_scene import (
    create_asphalt_pbr_material,
    create_car_paint_pbr_material,
    create_concrete_pbr_material,
    setup_photoreal_lighting_and_sky,
    _make_cube,
    _apply_scale,
)


def build_calibration_scene() -> dict[str, bpy.types.Object]:
    """Build calibration scene with known geometric targets at 0.75m, 1m, 1.5m, 2m, 3m, 5m, 8m."""
    objs = {}

    # 1. PBR Asphalt Road Plane
    bpy.ops.mesh.primitive_plane_add(location=(0.0, 0.0, 0.0))
    road = bpy.context.active_object
    road.name = "road_plane"
    road.scale = (25.0, 25.0, 1.0)
    _apply_scale(road)
    mat_asphalt = create_asphalt_pbr_material("mat_calib_asphalt")
    road.data.materials.append(mat_asphalt)
    road.pass_index = 1
    objs["road_plane"] = road

    # Materials
    mat_box = create_car_paint_pbr_material("mat_calib_orange", (0.85, 0.35, 0.05, 1.0), metallic=0.3, roughness=0.35)
    mat_concrete = create_concrete_pbr_material("mat_calib_concrete", (0.55, 0.54, 0.52))
    mat_car_blue = create_car_paint_pbr_material("mat_calib_blue", (0.05, 0.15, 0.65, 1.0), metallic=0.85, roughness=0.2)
    mat_car_red = create_car_paint_pbr_material("mat_calib_red", (0.65, 0.05, 0.08, 1.0), metallic=0.85, roughness=0.2)

    # Place known targets in front of each camera:
    # -------------------------------------------------------------
    # FRONT CAMERA (Position: 0, 2.15, 0.90 | Look: +Y | Tilt: -8 deg)
    # Targets at distance 0.75m, 2.0m, 5.0m
    # Target at 0.75m: Y = 2.15 + 0.75 = 2.90m, box
    objs["front_target_075"] = _make_cube("front_box_075", (0.4, 0.4, 0.5), (0.0, 2.90, 0.25), mat_box, pass_index=6)
    # Target at 2.0m: Y = 2.15 + 2.0 = 4.15m, cylinder
    bpy.ops.mesh.primitive_cylinder_add(radius=0.25, depth=1.4, location=(-0.8, 4.15, 0.7))
    p_cyl = bpy.context.active_object
    p_cyl.name = "front_cyl_200"
    p_cyl.data.materials.append(mat_concrete)
    p_cyl.pass_index = 5
    objs["front_cyl_200"] = p_cyl
    # Target at 5.0m: Y = 2.15 + 5.0 = 7.15m, vehicle mockup
    objs["front_car_500"] = _make_cube("front_car_500", (1.7, 3.5, 1.4), (0.5, 7.15, 0.7), mat_car_blue, pass_index=3)

    # -------------------------------------------------------------
    # REAR CAMERA (Position: 0, -2.15, 0.90 | Look: -Y | Tilt: -8 deg)
    # Targets at distance 1.0m, 3.0m, 8.0m
    # Target at 1.0m: Y = -2.15 - 1.0 = -3.15m
    objs["rear_box_100"] = _make_cube("rear_box_100", (0.5, 0.5, 0.5), (0.0, -3.15, 0.25), mat_box, pass_index=6)
    # Target at 3.0m: Y = -2.15 - 3.0 = -5.15m
    bpy.ops.mesh.primitive_cylinder_add(radius=0.3, depth=1.6, location=(0.7, -5.15, 0.8))
    r_cyl = bpy.context.active_object
    r_cyl.name = "rear_cyl_300"
    r_cyl.data.materials.append(mat_concrete)
    r_cyl.pass_index = 5
    objs["rear_cyl_300"] = r_cyl
    # Target at 8.0m: Y = -2.15 - 8.0 = -10.15m, concrete barrier
    objs["rear_wall_800"] = _make_cube("rear_wall_800", (6.0, 0.4, 2.0), (0.0, -10.15, 1.0), mat_concrete, pass_index=4)

    # -------------------------------------------------------------
    # LEFT CAMERA (Position: -1.09, 0.30, 1.25 | Look: -X | Tilt: -6 deg)
    # Targets at distance 1.5m, 3.0m, 8.0m
    # Target at 1.5m: X = -1.09 - 1.5 = -2.59m
    objs["left_box_150"] = _make_cube("left_box_150", (0.5, 0.5, 0.6), (-2.59, 0.3, 0.3), mat_box, pass_index=6)
    # Target at 3.0m: X = -1.09 - 3.0 = -4.09m, vehicle mockup
    objs["left_car_300"] = _make_cube("left_car_300", (3.8, 1.7, 1.4), (-4.09, 0.5, 0.7), mat_car_red, pass_index=3)
    # Target at 8.0m: X = -1.09 - 8.0 = -9.09m, wall
    objs["left_wall_800"] = _make_cube("left_wall_800", (0.4, 8.0, 2.2), (-9.09, 0.0, 1.1), mat_concrete, pass_index=4)

    # -------------------------------------------------------------
    # RIGHT CAMERA (Position: +1.09, 0.30, 1.25 | Look: +X | Tilt: -6 deg)
    # Targets at distance 0.75m, 2.0m, 5.0m
    # Target at 0.75m: X = 1.09 + 0.75 = 1.84m
    objs["right_box_075"] = _make_cube("right_box_075", (0.4, 0.4, 0.4), (1.84, 0.3, 0.2), mat_box, pass_index=6)
    # Target at 2.0m: X = 1.09 + 2.0 = 3.09m, cylinder
    bpy.ops.mesh.primitive_cylinder_add(radius=0.25, depth=1.5, location=(3.09, -0.4, 0.75))
    rt_cyl = bpy.context.active_object
    rt_cyl.name = "right_cyl_200"
    rt_cyl.data.materials.append(mat_concrete)
    rt_cyl.pass_index = 5
    objs["right_cyl_200"] = rt_cyl
    # Target at 5.0m: X = 1.09 + 5.0 = 6.09m, vehicle
    objs["right_car_500"] = _make_cube("right_car_500", (3.8, 1.7, 1.4), (6.09, 0.2, 0.7), mat_car_blue, pass_index=3)

    return objs


def setup_compositor_nodes(scene: bpy.types.Scene, temp_dir: Path) -> tuple[bpy.types.Node, bpy.types.Node]:
    vl = scene.view_layers[0]
    vl.use_pass_z = True
    vl.use_pass_object_index = True
    vl.update_render_passes()

    ng = bpy.data.node_groups.new("CalibCompositorTree", "CompositorNodeTree")
    scene.compositing_node_group = ng
    rl = ng.nodes.new("CompositorNodeRLayers")

    # Depth Output Node
    out_depth = ng.nodes.new("CompositorNodeOutputFile")
    out_depth.directory = str(temp_dir) + os.sep
    out_depth.file_name = "raw_depth_"
    out_depth.file_output_items.clear()
    i_d = out_depth.file_output_items.new("FLOAT", "Z")
    i_d.override_node_format = True
    i_d.format.file_format = "OPEN_EXR"
    i_d.format.color_depth = "32"
    ng.links.new(rl.outputs["Depth"], out_depth.inputs["Z"])

    # Semantic Output Node
    out_sem = ng.nodes.new("CompositorNodeOutputFile")
    out_sem.directory = str(temp_dir) + os.sep
    out_sem.file_name = "raw_sem_"
    out_sem.file_output_items.clear()
    i_s = out_sem.file_output_items.new("FLOAT", "IndexOB")
    i_s.override_node_format = True
    i_s.format.file_format = "OPEN_EXR"
    i_s.format.color_depth = "32"
    ng.links.new(rl.outputs["Object Index"], out_sem.inputs["IndexOB"])

    return out_depth, out_sem


def process_rendered_passes(cam_name: str, temp_dir: Path, out_dirs: dict[str, Path]) -> None:
    w, h = CAMERA_RESOLUTION_W, CAMERA_RESOLUTION_H
    pt = Imath.PixelType(Imath.PixelType.FLOAT)

    temp_depth = temp_dir / "raw_depth_.exr"
    final_depth = out_dirs["depth"] / cam_name / "calib.exr"
    if temp_depth.exists():
        exr_in = OpenEXR.InputFile(str(temp_depth))
        raw_z = exr_in.channel("Z.V", pt)
        exr_in.close()

        new_h = OpenEXR.Header(w, h)
        new_h["channels"] = {"Z": Imath.Channel(pt)}
        exr_out = OpenEXR.OutputFile(str(final_depth), new_h)
        exr_out.writePixels({"Z": raw_z})
        exr_out.close()
        temp_depth.unlink()

    temp_sem = temp_dir / "raw_sem_.exr"
    final_sem = out_dirs["semantic"] / cam_name / "calib.png"
    if temp_sem.exists():
        exr_in = OpenEXR.InputFile(str(temp_sem))
        raw_idx = exr_in.channel("IndexOB.V", pt)
        exr_in.close()

        sem_arr = np.frombuffer(raw_idx, dtype=np.float32).reshape((h, w))
        sem_uint8 = np.clip(np.round(sem_arr), 0, 255).astype(np.uint8)
        cv2.imwrite(str(final_sem), sem_uint8)
        temp_sem.unlink()


def main() -> None:
    print("\n" + "=" * 70)
    print("CLEAR360 — DEDICATED CALIBRATION DATASET RENDER (PART C)")
    print("=" * 70)

    calib_root = project_root / "data" / "calibration_dataset"
    out_dirs = {
        "rgb": calib_root / "rgb",
        "depth": calib_root / "gt_depth",
        "semantic": calib_root / "gt_semantic",
    }
    for d in out_dirs.values():
        for cam in CAMERA_NAMES:
            (d / cam).mkdir(parents=True, exist_ok=True)

    temp_dir = calib_root / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    clear_scene()
    build_calibration_scene()
    setup_photoreal_lighting_and_sky()

    scene = bpy.context.scene
    cameras = create_cameras(scene)

    scene.render.engine = "CYCLES"
    scene.cycles.samples = 16
    scene.cycles.device = "CPU"
    scene.render.resolution_x = CAMERA_RESOLUTION_W
    scene.render.resolution_y = CAMERA_RESOLUTION_H
    scene.render.resolution_percentage = 100

    setup_compositor_nodes(scene, temp_dir)

    for cam_name in CAMERA_NAMES:
        cam_obj = cameras[cam_name]
        print(f"Rendering calibration view: [{cam_name.upper()}] ...", end="", flush=True)

        scene.camera = cam_obj
        rgb_file = out_dirs["rgb"] / cam_name / "calib.png"
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_mode = "RGB"
        scene.render.filepath = str(rgb_file)

        bpy.ops.render.render(write_still=True)
        process_rendered_passes(cam_name, temp_dir, out_dirs)
        print(" DONE")

    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)

    print("\nCalibration dataset successfully generated at:", calib_root)


if __name__ == "__main__":
    main()

