"""
sim/generate_m8_scenarios.py — 10 Independent Generalization Scenarios for CLEAR360 (M8)
========================================================================================

Generates 10 diverse, realistic Blender Cycles environments:
  1. scenario_01_narrow_street        : Narrow Indian street with close walls & speed breaker
  2. scenario_02_double_parked        : Slalom corridor with vehicles parked on both sides
  3. scenario_03_close_pillars_walls  : Basement parking with tight pillars near side mirrors
  4. scenario_04_tight_parking_gap    : Tight parking slot flanked closely by two large SUVs
  5. scenario_05_speed_breakers_kerbs : Multiple raised speed bumps and high roadside kerbs
  6. scenario_06_approaching_pedestrians : Crossing pedestrian and oncoming vehicle
  7. scenario_07_harsh_lighting_shadows : Low golden hour sun with deep cast shadows & high contrast
  8. scenario_08_partially_occluded   : Occluded hazards positioned behind pillars and parked cars
  9. scenario_09_t_junction_turn      : 90-degree blind corner turn with corner obstacle
 10. scenario_10_construction_zone    : Roadwork barriers, traffic drums, and gravel debris

For each scenario, exports:
  - RGB PNG (1280x720) for 4 cameras (front, rear, left, right)
  - Metric Z-depth EXR (1280x720 float32) for 4 cameras
  - Semantic Mask PNG (1280x720 uint8) for 4 cameras
  - Scenario metadata JSON (ground-truth obstacle bounding boxes, classes, centroids, clearances)
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

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
    print(f"[ERROR] Missing required library in Blender python environment: {e}")
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
    create_concrete_pbr_material,
    create_car_paint_pbr_material,
    create_speed_breaker_material,
    create_kerb_hazard_material,
    build_photoreal_ego_vehicle,
    _make_cube,
    _apply_scale,
)


# ---------------------------------------------------------------------------
# SCENARIO DEFINITIONS
# ---------------------------------------------------------------------------

SCENARIOS_REGISTRY = [
    {
        "id": "scenario_01_narrow_street",
        "name": "Narrow Indian-Style Urban Street",
        "description": "Corridor width ~4.2m with close roadside concrete walls, parked scooter mockup, and forward speed breaker.",
        "sun_elev": 58.0,
        "sun_azim": 140.0,
        "sun_strength": 3.5,
    },
    {
        "id": "scenario_02_double_parked",
        "name": "Vehicles Parked on Both Sides",
        "description": "Narrow driving lane with multiple parked cars on left and right, leaving tight lateral clearances.",
        "sun_elev": 45.0,
        "sun_azim": 190.0,
        "sun_strength": 3.0,
    },
    {
        "id": "scenario_03_close_pillars_walls",
        "name": "Basement Parking Close Pillars & Walls",
        "description": "Underground parking garage with concrete structural pillars positioned within 0.6m of side mirrors.",
        "sun_elev": 25.0,
        "sun_azim": 90.0,
        "sun_strength": 1.2,
    },
    {
        "id": "scenario_04_tight_parking_gap",
        "name": "Tight Parking Slot Between Vehicles",
        "description": "Ego vehicle parked in a tight slot flanked by an SUV on left (<0.65m mirror clearance) and sedan on right.",
        "sun_elev": 50.0,
        "sun_azim": 170.0,
        "sun_strength": 3.0,
    },
    {
        "id": "scenario_05_speed_breakers_kerbs",
        "name": "Multiple Speed Breakers and High Kerbs",
        "description": "Urban street with two raised yellow/black speed breakers and elevated stone kerbs on both sides.",
        "sun_elev": 60.0,
        "sun_azim": 120.0,
        "sun_strength": 3.8,
    },
    {
        "id": "scenario_06_approaching_pedestrians",
        "name": "Approaching Pedestrians & Oncoming Traffic",
        "description": "Crossing pedestrian ahead of front bumper, secondary pedestrian near rear, and oncoming vehicle mockup.",
        "sun_elev": 40.0,
        "sun_azim": 210.0,
        "sun_strength": 2.8,
    },
    {
        "id": "scenario_07_harsh_lighting_shadows",
        "name": "Low Sun Angle & Deep Cast Shadows",
        "description": "Golden hour sun at 18 degrees elevation casting severe shadows across half the vehicle and high-contrast occlusion edges.",
        "sun_elev": 18.0,
        "sun_azim": 240.0,
        "sun_strength": 4.5,
    },
    {
        "id": "scenario_08_partially_occluded",
        "name": "Partially Occluded Obstacles",
        "description": "Pedestrian and hazard objects partially hidden behind pillars and parked vehicles to test occlusion reasoning.",
        "sun_elev": 48.0,
        "sun_azim": 160.0,
        "sun_strength": 3.0,
    },
    {
        "id": "scenario_09_t_junction_turn",
        "name": "T-Junction 90-Degree Blind Corner Turn",
        "description": "Right-angle corner with roadside building wall, corner kerb, and blind hazard obstacle positioned at corner apex.",
        "sun_elev": 52.0,
        "sun_azim": 130.0,
        "sun_strength": 3.2,
    },
    {
        "id": "scenario_10_construction_zone",
        "name": "Urban Roadwork & Construction Zone",
        "description": "Diagonal traffic barrier line, orange traffic drums, and rubble/debris obstacle forcing lane narrowing.",
        "sun_elev": 45.0,
        "sun_azim": 180.0,
        "sun_strength": 3.0,
    },
]


def setup_lighting(sun_elev_deg: float, sun_azim_deg: float, sun_strength: float):
    """Configure sun lamp with soft shadows and sky background."""
    for obj in list(bpy.data.objects):
        if obj.type == "LIGHT":
            bpy.data.objects.remove(obj, do_unlink=True)

    sun_data = bpy.data.lights.new(name="Sun_PBR", type="SUN")
    sun_data.energy = sun_strength
    sun_data.color = (1.0, 0.97, 0.92)
    sun_data.angle = 0.015

    sun_obj = bpy.data.objects.new(name="Sun_PBR", object_data=sun_data)
    bpy.context.scene.collection.objects.link(sun_obj)

    elev_rad = math.radians(sun_elev_deg)
    azim_rad = math.radians(sun_azim_deg)
    rot_x = -(math.pi / 2.0 - elev_rad)
    rot_z = azim_rad
    sun_obj.rotation_euler = (rot_x, 0.0, rot_z)

    world = bpy.context.scene.world
    if not world:
        world = bpy.data.worlds.new("World_PBR")
        bpy.context.scene.world = world
    world.use_nodes = True
    wnodes = world.node_tree.nodes
    wlinks = world.node_tree.links
    wnodes.clear()
    node_out = wnodes.new("ShaderNodeOutputWorld")
    node_bg = wnodes.new("ShaderNodeBackground")
    node_bg.inputs["Strength"].default_value = 0.8
    node_bg.inputs["Color"].default_value = (0.75, 0.82, 0.95, 1.0)
    wlinks.new(node_bg.outputs["Background"], node_out.inputs["Surface"])


def build_scenario_environment(scenario_id: str) -> list[dict]:
    """
    Constructs the specific 3D environment for the given scenario ID.
    Returns a list of obstacle metadata dicts with 3D bounding boxes and properties.
    """
    obstacles_meta = []

    # Shared materials
    mat_asphalt = create_asphalt_pbr_material("mat_asphalt")
    mat_concrete = create_concrete_pbr_material("mat_concrete", (0.65, 0.64, 0.62))
    mat_pillar = create_concrete_pbr_material("mat_pillar", (0.55, 0.54, 0.52))
    mat_car_blue = create_car_paint_pbr_material("mat_car_blue", (0.05, 0.15, 0.65, 1.0), 0.85, 0.2)
    mat_car_red = create_car_paint_pbr_material("mat_car_red", (0.65, 0.05, 0.08, 1.0), 0.85, 0.2)
    mat_car_silver = create_car_paint_pbr_material("mat_car_silver", (0.75, 0.77, 0.80, 1.0), 0.90, 0.18)
    mat_car_white = create_car_paint_pbr_material("mat_car_white", (0.85, 0.85, 0.85, 1.0), 0.40, 0.25)
    mat_pedestrian = create_car_paint_pbr_material("mat_pedestrian", (0.95, 0.45, 0.10, 1.0), 0.1, 0.6)
    mat_hazard = create_car_paint_pbr_material("mat_hazard_orange", (0.95, 0.35, 0.05, 1.0), 0.2, 0.4)
    mat_speed_breaker = create_speed_breaker_material("mat_speed_breaker")
    mat_kerb = create_kerb_hazard_material("mat_kerb")

    # 1. Ground Road Plane
    bpy.ops.mesh.primitive_plane_add(location=(0.0, 0.0, 0.0))
    road = bpy.context.active_object
    road.name = "road_plane"
    road.scale = (25.0, 25.0, 1.0)
    _apply_scale(road)
    road.data.materials.append(mat_asphalt)
    road.pass_index = 1

    def add_box_obs(name, scale, loc, mat, pass_idx, sem_name, is_dynamic=False):
        obj = _make_cube(name, scale, loc, mat, pass_index=pass_idx)
        # Compute bounding box
        sx, sy, sz = scale
        lx, ly, lz = loc
        min_pt = [lx - sx / 2.0, ly - sy / 2.0, lz - sz / 2.0]
        max_pt = [lx + sx / 2.0, ly + sy / 2.0, lz + sz / 2.0]
        obs_dict = {
            "name": name,
            "semantic_class": pass_idx,
            "semantic_name": sem_name,
            "centroid": [float(lx), float(ly), float(lz)],
            "dimensions": [float(sx), float(sy), float(sz)],
            "min_point": [float(p) for p in min_pt],
            "max_point": [float(p) for p in max_pt],
            "volume_m3": float(sx * sy * sz),
            "is_dynamic": is_dynamic,
        }
        obstacles_meta.append(obs_dict)
        return obj

    def add_cylinder_obs(name, radius, depth, loc, mat, pass_idx, sem_name, is_dynamic: bool = False):
        bpy.ops.mesh.primitive_cylinder_add(radius=radius, depth=depth, location=loc)
        cyl = bpy.context.active_object
        cyl.name = name
        cyl.data.materials.append(mat)
        cyl.pass_index = pass_idx
        lx, ly, lz = loc
        min_pt = [lx - radius, ly - radius, lz - depth / 2.0]
        max_pt = [lx + radius, ly + radius, lz + depth / 2.0]
        obs_dict = {
            "name": name,
            "semantic_class": pass_idx,
            "semantic_name": sem_name,
            "centroid": [float(lx), float(ly), float(lz)],
            "dimensions": [float(2 * radius), float(2 * radius), float(depth)],
            "min_point": [float(p) for p in min_pt],
            "max_point": [float(p) for p in max_pt],
            "volume_m3": float(math.pi * (radius ** 2) * depth),
            "is_dynamic": is_dynamic,
        }
        obstacles_meta.append(obs_dict)
        return cyl

    # -------------------------------------------------------------------------
    # SCENARIO 1: NARROW STREET
    # -------------------------------------------------------------------------
    if scenario_id == "scenario_01_narrow_street":
        # Left wall at X = -2.10 m (close to left mirror tip at -1.05m)
        add_box_obs("left_wall", (0.3, 18.0, 2.5), (-2.10, 0.0, 1.25), mat_concrete, 4, "Wall")
        # Right wall at X = +2.30 m
        add_box_obs("right_wall", (0.3, 18.0, 2.5), (+2.30, 0.0, 1.25), mat_concrete, 4, "Wall")
        # Speed breaker ahead at Y = +4.5 m
        add_box_obs("speed_breaker", (3.8, 0.4, 0.10), (0.1, 4.5, 0.05), mat_speed_breaker, 7, "Kerb / Speed Breaker")
        # Scooter mockup on left at Y = +1.8 m
        add_box_obs("parked_scooter", (0.6, 1.4, 1.1), (-1.65, 1.8, 0.55), mat_hazard, 6, "Moving / Small Obstacle")

    # -------------------------------------------------------------------------
    # SCENARIO 2: DOUBLE PARKED
    # -------------------------------------------------------------------------
    elif scenario_id == "scenario_02_double_parked":
        # Blue hatchback front-left: X = -2.5, Y = +2.8
        add_box_obs("car_front_left", (1.75, 4.0, 1.45), (-2.5, 2.8, 0.725), mat_car_blue, 3, "Parked Vehicle")
        # Red sedan front-right: X = +2.4, Y = +5.2
        add_box_obs("car_front_right", (1.80, 4.4, 1.45), (+2.4, 5.2, 0.725), mat_car_red, 3, "Parked Vehicle")
        # Silver SUV rear-left: X = -2.6, Y = -3.8
        add_box_obs("car_rear_left", (1.85, 4.5, 1.60), (-2.6, -3.8, 0.80), mat_car_silver, 3, "Parked Vehicle")
        # White car rear-right: X = +2.5, Y = -2.5
        add_box_obs("car_rear_right", (1.75, 3.8, 1.40), (+2.5, -2.5, 0.70), mat_car_white, 3, "Parked Vehicle")

    # -------------------------------------------------------------------------
    # SCENARIO 3: CLOSE PILLARS AND WALLS (BASEMENT)
    # -------------------------------------------------------------------------
    elif scenario_id == "scenario_03_close_pillars_walls":
        # Concrete pillar left: X = -1.65, Y = +0.40 (only 0.60m from left mirror tip!)
        add_cylinder_obs("pillar_left", 0.30, 2.8, (-1.65, 0.40, 1.4), mat_pillar, 5, "Pillar")
        # Concrete pillar right: X = +1.75, Y = +0.40 (only 0.70m from right mirror tip!)
        add_cylinder_obs("pillar_right", 0.30, 2.8, (+1.75, 0.40, 1.4), mat_pillar, 5, "Pillar")
        # Pillar rear: X = -1.8, Y = -3.5
        add_cylinder_obs("pillar_rear", 0.30, 2.8, (-1.80, -3.5, 1.4), mat_pillar, 5, "Pillar")
        # Forward perimeter wall at Y = +6.0
        add_box_obs("forward_wall", (12.0, 0.4, 2.5), (0.0, 6.0, 1.25), mat_concrete, 4, "Wall")

    # -------------------------------------------------------------------------
    # SCENARIO 4: TIGHT PARKING GAP
    # -------------------------------------------------------------------------
    elif scenario_id == "scenario_04_tight_parking_gap":
        # Flanking SUV on left: X = -1.85, Y = 0.0 (lateral clearance < 0.65m to mirror)
        add_box_obs("flanking_suv_left", (1.80, 4.6, 1.65), (-1.85, 0.0, 0.825), mat_car_blue, 3, "Parked Vehicle")
        # Flanking sedan on right: X = +1.90, Y = 0.0
        add_box_obs("flanking_sedan_right", (1.75, 4.3, 1.45), (+1.90, 0.0, 0.725), mat_car_silver, 3, "Parked Vehicle")
        # Back wall behind at Y = -3.15 (1.0m from rear bumper)
        add_box_obs("rear_parking_wall", (6.0, 0.4, 2.0), (0.0, -3.15, 1.0), mat_concrete, 4, "Wall")

    # -------------------------------------------------------------------------
    # SCENARIO 5: SPEED BREAKERS AND KERBS
    # -------------------------------------------------------------------------
    elif scenario_id == "scenario_05_speed_breakers_kerbs":
        # Elevated roadside kerb left: X = -2.8
        add_box_obs("kerb_left", (0.25, 16.0, 0.18), (-2.8, 0.0, 0.09), mat_kerb, 7, "Kerb / Speed Breaker")
        # Elevated roadside kerb right: X = +2.8
        add_box_obs("kerb_right", (0.25, 16.0, 0.18), (+2.8, 0.0, 0.09), mat_kerb, 7, "Kerb / Speed Breaker")
        # First speed breaker bump: Y = +3.8
        add_box_obs("speed_breaker_1", (3.6, 0.40, 0.12), (0.0, 3.8, 0.06), mat_speed_breaker, 7, "Kerb / Speed Breaker")
        # Second speed breaker bump: Y = +7.5
        add_box_obs("speed_breaker_2", (3.6, 0.40, 0.12), (0.0, 7.5, 0.06), mat_speed_breaker, 7, "Kerb / Speed Breaker")
        # Roadside planter box: X = 2.2, Y = 4.2
        add_box_obs("planter_box", (0.7, 0.7, 0.8), (2.2, 4.2, 0.4), mat_concrete, 7, "Kerb / Speed Breaker")

    # -------------------------------------------------------------------------
    # SCENARIO 6: APPROACHING PEDESTRIANS & ONCOMING TRAFFIC
    # -------------------------------------------------------------------------
    elif scenario_id == "scenario_06_approaching_pedestrians":
        # Crossing pedestrian in front-right: X = +1.6, Y = +3.5 (clearance ~1.45m)
        add_cylinder_obs("pedestrian_crossing", 0.25, 1.75, (1.6, 3.5, 0.875), mat_pedestrian, 6, "Pedestrian / Cyclist", True)
        # Second pedestrian near left-rear: X = -1.8, Y = -2.8
        add_cylinder_obs("pedestrian_rear", 0.22, 1.65, (-1.8, -2.8, 0.825), mat_pedestrian, 6, "Pedestrian / Cyclist", True)
        # Oncoming vehicle mockup ahead: X = -1.5, Y = +7.5
        add_box_obs("oncoming_car", (1.7, 3.8, 1.4), (-1.5, 7.5, 0.7), mat_car_red, 3, "Parked Vehicle")

    # -------------------------------------------------------------------------
    # SCENARIO 7: HARSH LIGHTING & DEEP SHADOWS
    # -------------------------------------------------------------------------
    elif scenario_id == "scenario_07_harsh_lighting_shadows":
        # High west wall casting massive shadow: X = -3.5, Z = 3.5
        add_box_obs("shadow_wall_west", (0.4, 16.0, 3.5), (-3.5, 0.0, 1.75), mat_concrete, 4, "Wall")
        # Parked car on right in bright direct sun: X = +3.0, Y = +1.5
        add_box_obs("sunlit_car_right", (1.75, 4.2, 1.5), (+3.0, 1.5, 0.75), mat_car_blue, 3, "Parked Vehicle")
        # Half-shadowed obstacle ahead: X = -0.5, Y = +4.5
        add_box_obs("box_half_shadow", (0.6, 0.6, 0.7), (-0.5, 4.5, 0.35), mat_hazard, 6, "Moving / Small Obstacle")

    # -------------------------------------------------------------------------
    # SCENARIO 8: PARTIALLY OCCLUDED OBSTACLES
    # -------------------------------------------------------------------------
    elif scenario_id == "scenario_08_partially_occluded":
        # Foreground pillar: X = +1.8, Y = +3.5
        add_cylinder_obs("pillar_occluder", 0.35, 2.6, (1.8, 3.5, 1.3), mat_pillar, 5, "Pillar")
        # Partially occluded pedestrian directly behind the pillar: X = +2.0, Y = +4.9
        add_cylinder_obs("occluded_pedestrian", 0.25, 1.75, (2.0, 4.9, 0.875), mat_pedestrian, 6, "Pedestrian / Cyclist", True)
        # Foreground parked vehicle left: X = -3.0, Y = +2.0
        add_box_obs("car_occluder_left", (1.8, 4.2, 1.5), (-3.0, 2.0, 0.75), mat_car_silver, 3, "Parked Vehicle")
        # Small obstacle hidden behind rear bumper of the parked car: X = -3.2, Y = +4.4
        add_box_obs("hidden_hazard_box", (0.5, 0.5, 0.5), (-3.2, 4.4, 0.25), mat_hazard, 6, "Moving / Small Obstacle")

    # -------------------------------------------------------------------------
    # SCENARIO 9: T-JUNCTION 90-DEGREE BLIND CORNER TURN
    # -------------------------------------------------------------------------
    elif scenario_id == "scenario_09_t_junction_turn":
        # Building wall along right: X = +2.5 from Y = -8 to +2.5
        add_box_obs("corner_wall_side", (0.4, 11.0, 2.6), (+2.5, -3.0, 1.3), mat_concrete, 4, "Wall")
        # Building wall extending eastward at the corner: Y = +2.5, X from +2.5 to +8
        add_box_obs("corner_wall_turn", (6.0, 0.4, 2.6), (+5.5, 2.5, 1.3), mat_concrete, 4, "Wall")
        # Blind obstacle located right around the corner: X = +3.5, Y = +3.5
        add_box_obs("blind_corner_obstacle", (0.7, 0.7, 0.9), (3.5, 3.5, 0.45), mat_hazard, 6, "Moving / Small Obstacle")
        # Across-the-street wall at Y = +6.0
        add_box_obs("opposite_street_wall", (14.0, 0.4, 2.5), (0.0, 6.0, 1.25), mat_concrete, 4, "Wall")

    # -------------------------------------------------------------------------
    # SCENARIO 10: CONSTRUCTION ZONE
    # -------------------------------------------------------------------------
    elif scenario_id == "scenario_10_construction_zone":
        # Construction barrier line on left (diagonal):
        add_box_obs("barrier_1", (0.3, 2.2, 0.9), (-2.2, 0.5, 0.45), mat_hazard, 7, "Road Furniture")
        add_box_obs("barrier_2", (0.3, 2.2, 0.9), (-1.8, 2.8, 0.45), mat_hazard, 7, "Road Furniture")
        add_box_obs("barrier_3", (0.3, 2.2, 0.9), (-1.4, 5.1, 0.45), mat_hazard, 7, "Road Furniture")
        # Traffic drum / barrel right: X = +1.8, Y = +2.5
        add_cylinder_obs("traffic_drum_1", 0.30, 0.95, (1.8, 2.5, 0.475), mat_hazard, 7, "Road Furniture")
        # Traffic drum / barrel right: X = +2.0, Y = +5.0
        add_cylinder_obs("traffic_drum_2", 0.30, 0.95, (2.0, 5.0, 0.475), mat_hazard, 7, "Road Furniture")
        # Rubble / gravel pile in forward lane: X = +0.4, Y = +4.8
        add_box_obs("rubble_pile", (1.1, 0.9, 0.35), (0.4, 4.8, 0.175), mat_concrete, 6, "Moving / Small Obstacle")

    return obstacles_meta


# ---------------------------------------------------------------------------
# RENDERING & PASS COMPOSITING
# ---------------------------------------------------------------------------

def configure_compositor(scene: bpy.types.Scene, temp_dir: Path) -> tuple[bpy.types.Node, bpy.types.Node]:
    vl = scene.view_layers[0]
    vl.use_pass_z = True
    vl.use_pass_object_index = True
    vl.update_render_passes()

    ng = bpy.data.node_groups.new("M8CompositorTree", "CompositorNodeTree")
    scene.compositing_node_group = ng
    rl = ng.nodes.new("CompositorNodeRLayers")

    out_depth = ng.nodes.new("CompositorNodeOutputFile")
    out_depth.directory = str(temp_dir) + os.sep
    out_depth.file_name = "raw_depth_"
    out_depth.file_output_items.clear()
    item_d = out_depth.file_output_items.new("FLOAT", "Z")
    item_d.override_node_format = True
    item_d.format.file_format = "OPEN_EXR"
    item_d.format.color_depth = "32"
    ng.links.new(rl.outputs["Depth"], out_depth.inputs["Z"])

    out_sem = ng.nodes.new("CompositorNodeOutputFile")
    out_sem.directory = str(temp_dir) + os.sep
    out_sem.file_name = "raw_sem_"
    out_sem.file_output_items.clear()
    item_s = out_sem.file_output_items.new("FLOAT", "IndexOB")
    item_s.override_node_format = True
    item_s.format.file_format = "OPEN_EXR"
    item_s.format.color_depth = "32"
    ng.links.new(rl.outputs["Object Index"], out_sem.inputs["IndexOB"])

    return out_depth, out_sem


def process_passes(cam_name: str, temp_dir: Path, out_dirs: dict[str, Path]):
    w, h = CAMERA_RESOLUTION_W, CAMERA_RESOLUTION_H
    pt = Imath.PixelType(Imath.PixelType.FLOAT)

    # Convert Depth EXR
    temp_depth = temp_dir / "raw_depth_.exr"
    final_depth = out_dirs["gt_depth"] / f"{cam_name}.exr"
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

    # Convert Semantic Mask PNG
    temp_sem = temp_dir / "raw_sem_.exr"
    final_sem = out_dirs["gt_semantic"] / f"{cam_name}.png"
    if temp_sem.exists():
        exr_in = OpenEXR.InputFile(str(temp_sem))
        raw_idx = exr_in.channel("IndexOB.V", pt)
        exr_in.close()

        sem_arr = np.frombuffer(raw_idx, dtype=np.float32).reshape((h, w))
        sem_uint8 = np.clip(np.round(sem_arr), 0, 255).astype(np.uint8)
        cv2.imwrite(str(final_sem), sem_uint8)
        temp_sem.unlink()


def render_scenario(scenario_cfg: dict, out_base: Path, samples: int = 16):
    scenario_id = scenario_cfg["id"]
    print(f"\n[{scenario_id}] Building: {scenario_cfg['name']}...")

    clear_scene()
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = samples
    scene.cycles.device = "CPU"
    scene.render.resolution_x = CAMERA_RESOLUTION_W
    scene.render.resolution_y = CAMERA_RESOLUTION_H
    scene.render.resolution_percentage = 100

    # Build environment and ego vehicle
    obs_meta = build_scenario_environment(scenario_id)
    build_photoreal_ego_vehicle()
    setup_lighting(scenario_cfg["sun_elev"], scenario_cfg["sun_azim"], scenario_cfg["sun_strength"])
    cameras = create_cameras(scene)

    # Output directory structure
    scen_dir = out_base / scenario_id
    out_dirs = {
        "rgb": scen_dir / "rgb",
        "gt_depth": scen_dir / "gt_depth",
        "gt_semantic": scen_dir / "gt_semantic",
    }
    for d in out_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    temp_dir = scen_dir / "_temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    configure_compositor(scene, temp_dir)

    # Render each camera
    t_start = time.perf_counter()
    for cam_name in CAMERA_NAMES:
        cam_obj = cameras[cam_name]
        scene.camera = cam_obj
        rgb_out = out_dirs["rgb"] / f"{cam_name}.png"
        scene.render.filepath = str(rgb_out)

        # Silent render
        bpy.ops.render.render(write_still=True)
        process_passes(cam_name, temp_dir, out_dirs)

    # Clean temp dir
    shutil.rmtree(temp_dir, ignore_errors=True)
    t_elapsed = time.perf_counter() - t_start

    # Compute ground truth vehicle clearance for each obstacle
    # Hyundai Creta body envelope: [-0.895, 0.895] x [-2.15, 2.15] x [0.0, 1.635]
    # Mirrors: X in [-1.05, -0.895] and [+0.895, +1.05] for Y in [0.35, 0.45], Z in [0.90, 1.10]
    boxes = [
        (np.array([-0.895, -2.15, 0.0]), np.array([+0.895, +2.15, 1.635])),
        (np.array([-1.05, 0.35, 0.90]), np.array([-0.895, 0.45, 1.10])),
        (np.array([+0.895, 0.35, 0.90]), np.array([+1.05, 0.45, 1.10])),
    ]

    for obs in obs_meta:
        b_min = np.array(obs["min_point"])
        b_max = np.array(obs["max_point"])
        best_dist = float("inf")
        for veh_min, veh_max in boxes:
            p1 = np.zeros(3)
            p2 = np.zeros(3)
            for i in range(3):
                if veh_max[i] < b_min[i]:
                    p1[i] = veh_max[i]
                    p2[i] = b_min[i]
                elif b_max[i] < veh_min[i]:
                    p1[i] = veh_min[i]
                    p2[i] = b_max[i]
                else:
                    mid = 0.5 * (max(veh_min[i], b_min[i]) + min(veh_max[i], b_max[i]))
                    p1[i] = mid
                    p2[i] = mid
            dist = float(np.linalg.norm(p1 - p2))
            if dist < best_dist:
                best_dist = dist
        obs["gt_clearance_to_vehicle_m"] = round(best_dist, 4)

    # Sort obstacles by clearance
    obs_meta.sort(key=lambda o: o["gt_clearance_to_vehicle_m"])

    metadata = {
        "scenario_id": scenario_id,
        "name": scenario_cfg["name"],
        "description": scenario_cfg["description"],
        "lighting": {
            "sun_elevation_deg": scenario_cfg["sun_elev"],
            "sun_azimuth_deg": scenario_cfg["sun_azim"],
            "sun_strength": scenario_cfg["sun_strength"],
        },
        "render_time_s": round(t_elapsed, 2),
        "obstacles_count": len(obs_meta),
        "closest_obstacle_clearance_m": obs_meta[0]["gt_clearance_to_vehicle_m"] if obs_meta else None,
        "obstacles": obs_meta,
    }

    with open(scen_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[{scenario_id}] Completed in {t_elapsed:.2f}s | Obstacles: {len(obs_meta)} | Closest clearance: {metadata['closest_obstacle_clearance_m']} m")


def main():
    parser = argparse.ArgumentParser(description="CLEAR360 M8 10 Scenarios Generator")
    parser.add_argument("--samples", type=int, default=16, help="Cycles samples per pixel (default: 16)")
    parser.add_argument("--scenario", type=str, default="all", help="Specific scenario ID or 'all'")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])

    out_base = project_root / "data" / "m8_scenarios"
    out_base.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("CLEAR360 — GENERATING 10 GENERALIZATION TEST SCENARIOS (M8)")
    print("=" * 70)

    selected = SCENARIOS_REGISTRY
    if args.scenario != "all":
        selected = [s for s in SCENARIOS_REGISTRY if s["id"] == args.scenario]
        if not selected:
            print(f"[ERROR] Unknown scenario: {args.scenario}")
            sys.exit(1)

    t_total_start = time.perf_counter()
    for scen_cfg in selected:
        render_scenario(scen_cfg, out_base, samples=args.samples)

    t_total = time.perf_counter() - t_total_start
    print("\n" + "=" * 70)
    print(f"All {len(selected)} M8 scenarios successfully generated in {t_total:.1f}s!")
    print(f"Output directory: {out_base}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
