"""
sim/photoreal_scene.py — Photorealistic PBR Scene Builder for CLEAR360 (M4.2)
=============================================================================

This module builds a photorealistic parking environment for Blender Cycles:
  - PBR Asphalt ground with Voronoi aggregate & multi-scale noise bump
  - PBR Concrete for walls and pillar with realistic micro-texture & roughness
  - Realistic white road markings & yellow/black hazard striping on speed breaker
  - Kerbs with alternating black-and-white safety stripes
  - Realistic metallic automotive paint with glossy clearcoat for vehicles
  - Dark tinted automotive glass for windshields and windows
  - Matte rubber tyres with metallic wheel rims
  - Natural outdoor sun lighting with soft shadow penumbra (angular diameter)
  - Atmospheric sky lighting (Blender 5.x MULTIPLE_SCATTERING sky)
  - Exact preservation of coordinate frames (X=right, Y=forward, Z=up)
  - Exact preservation of semantic pass_index (0: sky, 1: road, 2: ego,
    3: parked cars, 4: walls, 5: pillar, 6: obstacle, 7: furniture)
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import bpy
import mathutils

from sim.constants import (
    CAMERA_CONFIGS,
    CAMERA_NAMES,
    CAMERA_RESOLUTION_H,
    CAMERA_RESOLUTION_W,
    FRONT_AXLE_Y,
    MIRROR_DEPTH_M,
    MIRROR_HEIGHT_M,
    MIRROR_TIP_X,
    MIRROR_WIDTH_M,
    MIRROR_Y_CENTRE,
    MIRROR_Z_LOW,
    OBSTACLE_CONFIG,
    OBSTACLE_KEYFRAMES,
    PARKED_CAR_CONFIGS,
    PILLAR_CONFIG,
    REAR_AXLE_Y,
    ROAD_FURNITURE_CONFIGS,
    ROAD_SIZE_M,
    ROOF_HEIGHT_M,
    ROOF_HEIGHT_START_Z,
    ROOF_LENGTH_M,
    ROOF_WIDTH_M,
    SUN_AZIMUTH_DEG,
    SUN_ELEVATION_DEG,
    SUN_STRENGTH,
    VEHICLE_HALF_LEN,
    VEHICLE_HALF_W,
    VEHICLE_HEIGHT_M,
    VEHICLE_PASS_INDEX,
    WALL_CONFIGS,
    WHEEL_OFFSET_X,
    WHEEL_RADIUS_M,
    WHEEL_WIDTH_M,
)
from sim.camera_rig import create_cameras, export_cameras_yaml


# ---------------------------------------------------------------------------
# PROCEDURAL PBR SHADER GENERATORS
# ---------------------------------------------------------------------------

def create_asphalt_pbr_material(name: str = "mat_pbr_asphalt") -> bpy.types.Material:
    """Create procedural PBR asphalt with aggregate stones, fine sand grit, and wear."""
    if name in bpy.data.materials:
        return bpy.data.materials[name]

    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    node_out = nodes.new("ShaderNodeOutputMaterial")
    node_out.location = (650, 0)

    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (320, 0)
    links.new(bsdf.outputs["BSDF"], node_out.inputs["Surface"])

    tex_coord = nodes.new("ShaderNodeTexCoord")
    tex_coord.location = (-900, 0)

    mapping = nodes.new("ShaderNodeMapping")
    mapping.location = (-700, 0)
    links.new(tex_coord.outputs["Object"], mapping.inputs["Vector"])

    # Fine grit noise
    noise_grit = nodes.new("ShaderNodeTexNoise")
    noise_grit.location = (-450, 250)
    noise_grit.inputs["Scale"].default_value = 45.0
    noise_grit.inputs["Detail"].default_value = 6.0
    noise_grit.inputs["Roughness"].default_value = 0.7
    links.new(mapping.outputs["Vector"], noise_grit.inputs["Vector"])

    # Aggregate voronoi
    voronoi_pebbles = nodes.new("ShaderNodeTexVoronoi")
    voronoi_pebbles.location = (-450, -100)
    voronoi_pebbles.inputs["Scale"].default_value = 35.0
    voronoi_pebbles.inputs["Randomness"].default_value = 0.95
    links.new(mapping.outputs["Vector"], voronoi_pebbles.inputs["Vector"])

    # Color Ramp for asphalt base color (realistic dark charcoal to weathered grey)
    ramp_color = nodes.new("ShaderNodeValToRGB")
    ramp_color.location = (-180, 200)
    ramp_color.color_ramp.elements[0].position = 0.25
    ramp_color.color_ramp.elements[0].color = (0.09, 0.09, 0.10, 1.0)
    ramp_color.color_ramp.elements[1].position = 0.75
    ramp_color.color_ramp.elements[1].color = (0.17, 0.17, 0.18, 1.0)
    links.new(noise_grit.outputs["Fac"], ramp_color.inputs["Fac"])
    links.new(ramp_color.outputs["Color"], bsdf.inputs["Base Color"])

    # Bump node for stone aggregate relief
    bump = nodes.new("ShaderNodeBump")
    bump.location = (50, -150)
    bump.inputs["Strength"].default_value = 0.35
    bump.inputs["Distance"].default_value = 0.012
    links.new(voronoi_pebbles.outputs["Distance"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    bsdf.inputs["Roughness"].default_value = 0.85
    bsdf.inputs["Specular IOR Level"].default_value = 0.35
    return mat


def create_concrete_pbr_material(name: str = "mat_pbr_concrete", base_tint: tuple = (0.50, 0.49, 0.47)) -> bpy.types.Material:
    """Create procedural concrete material with formwork grain, micro-bump, and roughness."""
    if name in bpy.data.materials:
        return bpy.data.materials[name]

    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    node_out = nodes.new("ShaderNodeOutputMaterial")
    node_out.location = (650, 0)

    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (320, 0)
    links.new(bsdf.outputs["BSDF"], node_out.inputs["Surface"])

    tex_coord = nodes.new("ShaderNodeTexCoord")
    tex_coord.location = (-900, 0)

    noise_form = nodes.new("ShaderNodeTexNoise")
    noise_form.location = (-450, 150)
    noise_form.inputs["Scale"].default_value = 16.0
    noise_form.inputs["Detail"].default_value = 4.0
    noise_form.inputs["Roughness"].default_value = 0.65
    links.new(tex_coord.outputs["Object"], noise_form.inputs["Vector"])

    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.location = (-180, 150)
    r, g, b = base_tint
    ramp.color_ramp.elements[0].position = 0.2
    ramp.color_ramp.elements[0].color = (r * 0.72, g * 0.72, b * 0.72, 1.0)
    ramp.color_ramp.elements[1].position = 0.8
    ramp.color_ramp.elements[1].color = (min(1.0, r * 1.15), min(1.0, g * 1.15), min(1.0, b * 1.15), 1.0)
    links.new(noise_form.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])

    bump = nodes.new("ShaderNodeBump")
    bump.location = (50, -150)
    bump.inputs["Strength"].default_value = 0.22
    bump.inputs["Distance"].default_value = 0.008
    links.new(noise_form.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    bsdf.inputs["Roughness"].default_value = 0.86
    bsdf.inputs["Specular IOR Level"].default_value = 0.3
    return mat


def create_road_marking_material(name: str = "mat_road_marking_white") -> bpy.types.Material:
    """Create realistic traffic paint for parking lot stripes."""
    if name in bpy.data.materials:
        return bpy.data.materials[name]

    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    node_out = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    links.new(bsdf.outputs["BSDF"], node_out.inputs["Surface"])

    # Realistic weathered traffic paint
    bsdf.inputs["Base Color"].default_value = (0.84, 0.84, 0.82, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.48
    bsdf.inputs["Specular IOR Level"].default_value = 0.45
    return mat


def create_speed_breaker_material(name: str = "mat_speed_breaker_hazard") -> bpy.types.Material:
    """Create alternating yellow and dark hazard stripe material for speed breakers."""
    if name in bpy.data.materials:
        return bpy.data.materials[name]

    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    node_out = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    links.new(bsdf.outputs["BSDF"], node_out.inputs["Surface"])

    tex_coord = nodes.new("ShaderNodeTexCoord")
    tex_coord.location = (-700, 0)

    # Wave texture along X for stripes
    wave = nodes.new("ShaderNodeTexWave")
    wave.location = (-400, 0)
    wave.wave_type = "BANDS"
    wave.bands_direction = "X"
    wave.inputs["Scale"].default_value = 4.0
    links.new(tex_coord.outputs["Object"], wave.inputs["Vector"])

    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.location = (-150, 0)
    ramp.color_ramp.interpolation = "CONSTANT"
    ramp.color_ramp.elements[0].position = 0.5
    ramp.color_ramp.elements[0].color = (0.05, 0.05, 0.05, 1.0)     # dark black/charcoal
    ramp.color_ramp.elements[1].position = 0.501
    ramp.color_ramp.elements[1].color = (0.92, 0.72, 0.04, 1.0)     # road traffic yellow
    links.new(wave.outputs["Color"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])

    bsdf.inputs["Roughness"].default_value = 0.65
    return mat


def create_kerb_hazard_material(name: str = "mat_kerb_hazard") -> bpy.types.Material:
    """Create alternating black and white / yellow safety blocks on kerbs."""
    if name in bpy.data.materials:
        return bpy.data.materials[name]

    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    node_out = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    links.new(bsdf.outputs["BSDF"], node_out.inputs["Surface"])

    tex_coord = nodes.new("ShaderNodeTexCoord")
    wave = nodes.new("ShaderNodeTexWave")
    wave.wave_type = "BANDS"
    wave.bands_direction = "Y"
    wave.inputs["Scale"].default_value = 1.5
    links.new(tex_coord.outputs["Object"], wave.inputs["Vector"])

    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.interpolation = "CONSTANT"
    ramp.color_ramp.elements[0].position = 0.5
    ramp.color_ramp.elements[0].color = (0.07, 0.07, 0.07, 1.0)
    ramp.color_ramp.elements[1].position = 0.501
    ramp.color_ramp.elements[1].color = (0.85, 0.85, 0.83, 1.0)
    links.new(wave.outputs["Color"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])

    bsdf.inputs["Roughness"].default_value = 0.78
    return mat


def create_car_paint_pbr_material(
    name: str,
    color_rgba: tuple,
    metallic: float = 0.85,
    roughness: float = 0.18,
    coat_weight: float = 1.0,
) -> bpy.types.Material:
    """Create realistic metallic automotive paint with mirror clearcoat reflections."""
    if name in bpy.data.materials:
        return bpy.data.materials[name]

    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    node_out = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    links.new(bsdf.outputs["BSDF"], node_out.inputs["Surface"])

    bsdf.inputs["Base Color"].default_value = color_rgba
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = roughness
    if "Coat Weight" in bsdf.inputs:
        bsdf.inputs["Coat Weight"].default_value = coat_weight
        bsdf.inputs["Coat Roughness"].default_value = 0.03
    return mat


def create_glass_pbr_material(
    name: str = "mat_car_glass",
    tint_rgba: tuple = (0.04, 0.05, 0.06, 1.0),
) -> bpy.types.Material:
    """Create dark tinted automotive glass with smooth specular highlights."""
    if name in bpy.data.materials:
        return bpy.data.materials[name]

    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    node_out = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    links.new(bsdf.outputs["BSDF"], node_out.inputs["Surface"])

    bsdf.inputs["Base Color"].default_value = tint_rgba
    bsdf.inputs["Roughness"].default_value = 0.04
    bsdf.inputs["Specular IOR Level"].default_value = 0.85
    bsdf.inputs["IOR"].default_value = 1.52
    if "Transmission Weight" in bsdf.inputs:
        bsdf.inputs["Transmission Weight"].default_value = 0.25
    return mat


def create_rubber_pbr_material(name: str = "mat_tyre_rubber") -> bpy.types.Material:
    """Create matte vulcanized tyre rubber material."""
    if name in bpy.data.materials:
        return bpy.data.materials[name]

    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    node_out = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    links.new(bsdf.outputs["BSDF"], node_out.inputs["Surface"])

    bsdf.inputs["Base Color"].default_value = (0.035, 0.035, 0.04, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.82
    bsdf.inputs["Metallic"].default_value = 0.0
    return mat


def create_rim_material(name: str = "mat_wheel_rim") -> bpy.types.Material:
    """Create machined alloy wheel rim material."""
    if name in bpy.data.materials:
        return bpy.data.materials[name]

    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    node_out = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    links.new(bsdf.outputs["BSDF"], node_out.inputs["Surface"])

    bsdf.inputs["Base Color"].default_value = (0.75, 0.76, 0.78, 1.0)
    bsdf.inputs["Metallic"].default_value = 0.92
    bsdf.inputs["Roughness"].default_value = 0.20
    return mat


def create_chrome_mirror_material(name: str = "mat_mirror_glass") -> bpy.types.Material:
    """Create specular mirror reflection material."""
    if name in bpy.data.materials:
        return bpy.data.materials[name]

    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    node_out = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    links.new(bsdf.outputs["BSDF"], node_out.inputs["Surface"])

    bsdf.inputs["Base Color"].default_value = (0.95, 0.95, 0.95, 1.0)
    bsdf.inputs["Metallic"].default_value = 1.0
    bsdf.inputs["Roughness"].default_value = 0.01
    return mat


# ---------------------------------------------------------------------------
# GEOMETRY HELPERS
# ---------------------------------------------------------------------------

def _apply_scale(obj: bpy.types.Object) -> None:
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)


def _make_cube(
    name: str,
    scale_xyz: tuple,
    location_xyz: tuple,
    material: bpy.types.Material,
    pass_index: int,
) -> bpy.types.Object:
    sx, sy, sz = scale_xyz[0] / 2.0, scale_xyz[1] / 2.0, scale_xyz[2] / 2.0
    bpy.ops.mesh.primitive_cube_add(location=location_xyz)
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = (sx, sy, sz)
    _apply_scale(obj)
    obj.data.materials.append(material)
    obj.pass_index = pass_index
    return obj


# ---------------------------------------------------------------------------
# PHOTOREALISTIC SCENE CONSTRUCTOR
# ---------------------------------------------------------------------------

def build_photoreal_environment() -> dict[str, bpy.types.Object]:
    """Build the Indian parking environment with photorealistic PBR materials."""
    env: dict[str, bpy.types.Object] = {}

    # 1. PBR Asphalt Road Plane
    bpy.ops.mesh.primitive_plane_add(location=(0.0, 0.0, 0.0))
    road = bpy.context.active_object
    road.name = "road_plane"
    road.scale = (ROAD_SIZE_M / 2.0, ROAD_SIZE_M / 2.0, 1.0)
    _apply_scale(road)
    mat_asphalt = create_asphalt_pbr_material("mat_pbr_asphalt")
    road.data.materials.append(mat_asphalt)
    road.pass_index = 1
    env["road_plane"] = road

    # 2. White Parking Bay Markings (thin geometry at Z=0.001 to prevent z-fighting)
    # Marking 1: ego bay left line
    # Marking 2: ego bay right line
    # Marking 3: parked car 1 bay
    # Marking 4: parked car 2 bay
    mat_line = create_road_marking_material()
    markings = [
        ("marking_bay_left", (0.12, 5.0, 0.002), (-1.25, 0.0, 0.001)),
        ("marking_bay_right", (0.12, 5.0, 0.002), (1.25, 0.0, 0.001)),
        ("marking_parked1_left", (0.12, 5.0, 0.002), (-6.0, -5.0, 0.001)),
        ("marking_parked1_right", (0.12, 5.0, 0.002), (-4.0, -5.0, 0.001)),
        ("marking_parked2_left", (0.12, 5.0, 0.002), (4.0, -5.0, 0.001)),
        ("marking_parked2_right", (0.12, 5.0, 0.002), (6.0, -5.0, 0.001)),
        ("marking_center_dash", (0.15, 12.0, 0.002), (0.0, 0.0, 0.001)),
    ]
    for mname, mscale, mloc in markings:
        mobj = _make_cube(mname, mscale, mloc, mat_line, pass_index=1)
        env[mname] = mobj

    # 3. Concrete Perimeter Walls
    mat_concrete = create_concrete_pbr_material("mat_pbr_concrete", (0.52, 0.51, 0.49))
    for wname, cfg in WALL_CONFIGS.items():
        obj = _make_cube(
            name=wname,
            scale_xyz=cfg["scale"],
            location_xyz=cfg["location"],
            material=mat_concrete,
            pass_index=cfg["pass_index"],
        )
        env[wname] = obj

    # 4. Pillar with Concrete texture
    mat_pillar = create_concrete_pbr_material("mat_pillar_concrete", (0.48, 0.47, 0.45))
    for pname, cfg in PILLAR_CONFIG.items():
        bpy.ops.mesh.primitive_cylinder_add(
            radius=cfg["radius"],
            depth=cfg["depth"],
            location=cfg["location"],
        )
        pobj = bpy.context.active_object
        pobj.name = pname
        pobj.data.materials.append(mat_pillar)
        pobj.pass_index = cfg["pass_index"]
        env[pname] = pobj

    # 5. Parked Cars with realistic automotive metallic paints
    mat_parked1 = create_car_paint_pbr_material("mat_paint_navy_blue", (0.03, 0.09, 0.35, 1.0), metallic=0.88, roughness=0.18)
    mat_parked2 = create_car_paint_pbr_material("mat_paint_ruby_red", (0.50, 0.03, 0.05, 1.0), metallic=0.88, roughness=0.18)
    parked_mats = {"parked_car_1": mat_parked1, "parked_car_2": mat_parked2}
    for cname, cfg in PARKED_CAR_CONFIGS.items():
        obj = _make_cube(
            name=cname,
            scale_xyz=cfg["scale"],
            location_xyz=cfg["location"],
            material=parked_mats.get(cname, mat_parked1),
            pass_index=cfg["pass_index"],
        )
        env[cname] = obj

    # 6. Road Furniture: Kerbs with safety block paint, Speed Breaker with hazard stripes
    mat_kerb = create_kerb_hazard_material()
    mat_sb = create_speed_breaker_material()
    for fname, cfg in ROAD_FURNITURE_CONFIGS.items():
        mat_f = mat_sb if "speed_breaker" in fname else mat_kerb
        obj = _make_cube(
            name=fname,
            scale_xyz=cfg["scale"],
            location_xyz=cfg["location"],
            material=mat_f,
            pass_index=cfg["pass_index"],
        )
        env[fname] = obj

    # 7. Moving Obstacle (keyframe animated)
    cfg_obs = OBSTACLE_CONFIG["obstacle_mover"]
    mat_obs = create_car_paint_pbr_material("mat_obstacle_paint", (0.85, 0.28, 0.04, 1.0), metallic=0.5, roughness=0.25)
    obs = _make_cube(
        name="obstacle_mover",
        scale_xyz=cfg_obs["scale"],
        location_xyz=OBSTACLE_KEYFRAMES[0][1],
        material=mat_obs,
        pass_index=cfg_obs["pass_index"],
    )
    env["obstacle_mover"] = obs

    # Keyframe animate obstacle
    for frame_num, pos in OBSTACLE_KEYFRAMES:
        obs.location = pos
        obs.keyframe_insert(data_path="location", frame=frame_num)

    return env


def build_photoreal_ego_vehicle() -> dict[str, bpy.types.Object]:
    """Build the ego vehicle with metallic paint, clearcoat, glass, and wheels."""
    parts: dict[str, bpy.types.Object] = {}

    mat_paint = create_car_paint_pbr_material(
        "mat_ego_paint_silver",
        color_rgba=(0.72, 0.74, 0.76, 1.0), # Polar Silver Metallic
        metallic=0.90,
        roughness=0.16,
        coat_weight=1.0,
    )
    mat_glass = create_glass_pbr_material("mat_ego_glass", tint_rgba=(0.04, 0.05, 0.06, 1.0))
    mat_rubber = create_rubber_pbr_material("mat_ego_rubber")
    mat_rim = create_rim_material("mat_ego_rim")
    mat_mirror = create_chrome_mirror_material("mat_ego_mirror")

    # 1. Main lower body (SUV bumper-to-bumper)
    body_loc_z = VEHICLE_HEIGHT_M / 4.0
    body_scale = (VEHICLE_HALF_W * 2.0, VEHICLE_HALF_LEN * 2.0, VEHICLE_HEIGHT_M / 2.0)
    body = _make_cube("body", body_scale, (0.0, 0.0, body_loc_z), mat_paint, VEHICLE_PASS_INDEX)
    parts["body"] = body

    # 2. Cabin / Roof panel
    roof_loc_z = ROOF_HEIGHT_START_Z + ROOF_HEIGHT_M / 2.0
    roof_loc_y = -0.2
    roof = _make_cube(
        "roof_panel",
        (ROOF_WIDTH_M, ROOF_LENGTH_M, ROOF_HEIGHT_M),
        (0.0, roof_loc_y, roof_loc_z),
        mat_paint,
        VEHICLE_PASS_INDEX,
    )
    parts["roof_panel"] = roof

    # 3. Windshield & Windows (dark tinted glass panes)
    windshield = _make_cube(
        "windshield_front",
        (ROOF_WIDTH_M * 0.92, 0.05, ROOF_HEIGHT_M * 0.75),
        (0.0, roof_loc_y + ROOF_LENGTH_M / 2.0 + 0.02, roof_loc_z),
        mat_glass,
        VEHICLE_PASS_INDEX,
    )
    parts["windshield_front"] = windshield

    rear_window = _make_cube(
        "windshield_rear",
        (ROOF_WIDTH_M * 0.92, 0.05, ROOF_HEIGHT_M * 0.75),
        (0.0, roof_loc_y - ROOF_LENGTH_M / 2.0 - 0.02, roof_loc_z),
        mat_glass,
        VEHICLE_PASS_INDEX,
    )
    parts["windshield_rear"] = rear_window

    # 4. Four Wheels with Rubber Tyres
    wheel_positions = {
        "wheel_fl": (-WHEEL_OFFSET_X, FRONT_AXLE_Y, WHEEL_RADIUS_M),
        "wheel_fr": (WHEEL_OFFSET_X, FRONT_AXLE_Y, WHEEL_RADIUS_M),
        "wheel_rl": (-WHEEL_OFFSET_X, REAR_AXLE_Y, WHEEL_RADIUS_M),
        "wheel_rr": (WHEEL_OFFSET_X, REAR_AXLE_Y, WHEEL_RADIUS_M),
    }
    for wname, pos in wheel_positions.items():
        bpy.ops.mesh.primitive_cylinder_add(
            radius=WHEEL_RADIUS_M,
            depth=WHEEL_WIDTH_M,
            location=pos,
        )
        wobj = bpy.context.active_object
        wobj.name = wname
        # Cylinder axis is Z, rotate by 90 deg around Y to align with wheel axle X
        wobj.rotation_euler = (0, math.pi / 2.0, 0)
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
        wobj.data.materials.append(mat_rubber)
        wobj.pass_index = VEHICLE_PASS_INDEX
        parts[wname] = wobj

    # 5. Side Mirrors
    mirror_z = (MIRROR_Z_LOW + (MIRROR_Z_LOW + MIRROR_HEIGHT_M)) / 2.0
    mirror_scale = (MIRROR_WIDTH_M, MIRROR_DEPTH_M, MIRROR_HEIGHT_M)
    parts["mirror_left"] = _make_cube(
        "mirror_left",
        mirror_scale,
        (-MIRROR_TIP_X, MIRROR_Y_CENTRE, mirror_z),
        mat_paint,
        VEHICLE_PASS_INDEX,
    )
    parts["mirror_right"] = _make_cube(
        "mirror_right",
        mirror_scale,
        (MIRROR_TIP_X, MIRROR_Y_CENTRE, mirror_z),
        mat_paint,
        VEHICLE_PASS_INDEX,
    )
    return parts


# ---------------------------------------------------------------------------
# NATURAL LIGHTING & SKY
# ---------------------------------------------------------------------------

def setup_photoreal_lighting_and_sky() -> bpy.types.Object:
    """Set up physically-based Sun light with soft penumbra and multi-scattering sky."""
    # 1. Clear existing lights
    for obj in list(bpy.data.objects):
        if obj.type == "LIGHT":
            bpy.data.objects.remove(obj, do_unlink=True)

    # 2. Sun lamp with realistic angular diameter (soft shadows)
    sun_data = bpy.data.lights.new(name="Sun_PBR", type="SUN")
    sun_data.energy = SUN_STRENGTH * 1.5
    sun_data.color = (1.0, 0.98, 0.93)
    # Angular diameter in radians (0.015 rad ~ 0.86 deg produces natural soft shadows)
    sun_data.angle = 0.015

    sun_obj = bpy.data.objects.new(name="Sun_PBR", object_data=sun_data)
    bpy.context.scene.collection.objects.link(sun_obj)

    # Calculate sun orientation:
    elev_rad = math.radians(SUN_ELEVATION_DEG)
    azim_rad = math.radians(SUN_AZIMUTH_DEG)
    rot_x = -(math.pi / 2.0 - elev_rad)
    rot_z = azim_rad
    sun_obj.rotation_euler = (rot_x, 0.0, rot_z)

    # 3. World Sky node (Multi-scattering outdoor atmosphere)
    world = bpy.context.scene.world
    if not world:
        world = bpy.data.worlds.new("World_PBR")
        bpy.context.scene.world = world

    world.use_nodes = True
    wnodes = world.node_tree.nodes
    wlinks = world.node_tree.links
    wnodes.clear()

    node_wout = wnodes.new("ShaderNodeOutputWorld")
    node_wout.location = (400, 0)

    node_bg = wnodes.new("ShaderNodeBackground")
    node_bg.location = (150, 0)
    node_bg.inputs["Strength"].default_value = 1.0
    wlinks.new(node_bg.outputs["Background"], node_wout.inputs["Surface"])

    # Physical sky texture
    sky_node = wnodes.new("ShaderNodeTexSky")
    sky_node.location = (-150, 0)
    try:
        sky_node.sky_type = "MULTIPLE_SCATTERING"
    except Exception:
        pass
    # Point sky sun direction matching sun lamp
    sun_dir_x = math.cos(elev_rad) * math.sin(azim_rad)
    sun_dir_y = -math.cos(elev_rad) * math.cos(azim_rad)
    sun_dir_z = math.sin(elev_rad)
    if hasattr(sky_node, "sun_direction"):
        sky_node.sun_direction = (sun_dir_x, sun_dir_y, sun_dir_z)
    wlinks.new(sky_node.outputs["Color"], node_bg.inputs["Color"])

    return sun_obj

