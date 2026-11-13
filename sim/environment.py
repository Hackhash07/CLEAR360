"""
sim/environment.py — Indian parking lot environment, lighting, obstacle animation
==================================================================================

This module creates the full parking environment around the ego vehicle,
configures the sun lamp, and keyframes the moving obstacle.

COORDINATE CONVENTION (World frame, Blender default axes):
  X = RIGHT of vehicle
  Y = FORWARD of vehicle
  Z = UP
  Ground plane at Z = 0.

ENVIRONMENT OBJECTS CREATED:
  Road   : road_plane        (20×20 m ground, pass_index=1)
  Walls  : wall_north/south/east/west  (concrete enclosure, pass_index=4)
  Pillar : pillar_1          (cylindrical column, pass_index=5)
  Parked : parked_car_1/2   (simplified parked vehicles, pass_index=3)
  Kerbs  : kerb_left/right  (raised kerb strips, pass_index=7)
  Misc   : speed_breaker    (yellow speed bump, pass_index=7)
  Moving : obstacle_mover   (animated 0.5 m cube, pass_index=6)

Usage:
  from sim.environment import create_environment, setup_lighting, animate_obstacle
  env_objects = create_environment()
  setup_lighting()
  animate_obstacle(env_objects["obstacle_mover"], OBSTACLE_KEYFRAMES)
"""

import bpy
import math

from sim.constants import (
    ROAD_SIZE_M,
    WALL_CONFIGS,
    PILLAR_CONFIG,
    PARKED_CAR_CONFIGS,
    ROAD_FURNITURE_CONFIGS,
    OBSTACLE_CONFIG,
    OBSTACLE_KEYFRAMES,
    SUN_ELEVATION_DEG,
    SUN_AZIMUTH_DEG,
    SUN_STRENGTH,
)


# ---------------------------------------------------------------------------
# HELPER UTILITIES
# ---------------------------------------------------------------------------

def _make_material(name: str, color_rgba: tuple) -> bpy.types.Material:
    """Create (or reuse) a simple Principled BSDF material.

    Parameters
    ----------
    name       : str   — material data-block name in bpy.data.materials
    color_rgba : tuple — (R, G, B, A) linear sRGB, each in [0, 1]
    """
    if name in bpy.data.materials:
        return bpy.data.materials[name]

    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()

    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = color_rgba
    bsdf.inputs["Roughness"].default_value  = 0.80   # rough for environment surfaces
    bsdf.inputs["Metallic"].default_value   = 0.0

    out = nodes.new("ShaderNodeOutputMaterial")
    mat.node_tree.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    return mat


def _apply_scale(obj: bpy.types.Object) -> None:
    """Apply current scale to mesh data, resetting obj.scale to (1,1,1)."""
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)


def _make_cube(
    name: str,
    scale_xyz: tuple,
    location_xyz: tuple,
    color_rgba: tuple,
    pass_index: int,
    collection=None,
) -> bpy.types.Object:
    """Create a named box (cube scaled to target dimensions) at the given location.

    Blender's primitive_cube_add produces a 2×2×2 m cube centred at *location*.
    Setting obj.scale = (Sx, Sy, Sz) makes the final dimensions 2*Sx × 2*Sy × 2*Sz.
    We call transform_apply so the mesh vertices encode the final size.

    Parameters
    ----------
    name       : str   — object + material data-block name
    scale_xyz  : tuple — (X_size, Y_size, Z_size) in metres (full extents, not halves)
    location_xyz: tuple— (X, Y, Z) object centre in world space
    color_rgba : tuple — (R, G, B, A)
    pass_index : int   — Cycles IndexOB pass index (semantic class)
    collection : bpy.types.Collection or None
    """
    # Half-extents: Blender cube scale = half final dimension
    sx, sy, sz = scale_xyz[0] / 2.0, scale_xyz[1] / 2.0, scale_xyz[2] / 2.0

    bpy.ops.mesh.primitive_cube_add(location=location_xyz)
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = (sx, sy, sz)
    _apply_scale(obj)

    mat = _make_material(f"mat_{name}", color_rgba)
    obj.data.materials.append(mat)
    obj.pass_index = pass_index

    if collection is not None:
        for col in list(obj.users_collection):
            col.objects.unlink(obj)
        collection.objects.link(obj)

    return obj


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------

def create_environment(collection=None) -> dict:
    """Build all static and dynamic environment objects for the parking scene.

    Parameters
    ----------
    collection : bpy.types.Collection or None
        Objects are linked here if supplied; otherwise left in default collection.

    Returns
    -------
    dict
        Mapping of object_name (str) → bpy.types.Object for all created objects.
    """
    env: dict[str, bpy.types.Object] = {}

    # ------------------------------------------------------------------
    # 1. ROAD PLANE (20×20 m flat ground at Z=0)
    # ------------------------------------------------------------------
    # The plane is horizontal; its centre is at Z=0.
    # We use a Plane primitive then scale it to 20×20.
    # primitive_plane_add creates a 2×2 m quad; scale to 10×10 → 20×20 m.
    bpy.ops.mesh.primitive_plane_add(location=(0.0, 0.0, 0.0))
    road = bpy.context.active_object
    road.name = "road_plane"
    road.scale = (ROAD_SIZE_M / 2.0, ROAD_SIZE_M / 2.0, 1.0)
    _apply_scale(road)

    mat_road = _make_material("mat_road_plane", (0.20, 0.20, 0.22, 1.0))  # dark asphalt
    road.data.materials.append(mat_road)
    road.pass_index = 1

    if collection is not None:
        for col in list(road.users_collection):
            col.objects.unlink(col)
        collection.objects.link(road)

    env["road_plane"] = road

    # ------------------------------------------------------------------
    # 2. WALLS — four concrete perimeter walls
    # ------------------------------------------------------------------
    # Wall locations and dimensions come from WALL_CONFIGS in constants.py.
    # Each entry has: scale (X,Y,Z), location (centre), pass_index, color.
    for wname, cfg in WALL_CONFIGS.items():
        obj = _make_cube(
            name=wname,
            scale_xyz=cfg["scale"],
            location_xyz=cfg["location"],
            color_rgba=cfg["color"],
            pass_index=cfg["pass_index"],
            collection=collection,
        )
        env[wname] = obj

    # ------------------------------------------------------------------
    # 3. PILLAR — single cylindrical column (pillar_1)
    # ------------------------------------------------------------------
    # Blender cylinder axis is Z by default.  We place it at the given centre
    # (height centre = location Z = 1.25 m for a 2.5 m tall pillar).
    for pname, cfg in PILLAR_CONFIG.items():
        bpy.ops.mesh.primitive_cylinder_add(
            radius=cfg["radius"],
            depth=cfg["depth"],
            location=cfg["location"],
        )
        pobj = bpy.context.active_object
        pobj.name = pname

        mat_p = _make_material(f"mat_{pname}", cfg["color"])
        pobj.data.materials.append(mat_p)
        pobj.pass_index = cfg["pass_index"]

        if collection is not None:
            for col in list(pobj.users_collection):
                col.objects.unlink(pobj)
            collection.objects.link(pobj)

        env[pname] = pobj

    # ------------------------------------------------------------------
    # 4. PARKED CARS — two simplified box vehicles
    # ------------------------------------------------------------------
    for cname, cfg in PARKED_CAR_CONFIGS.items():
        obj = _make_cube(
            name=cname,
            scale_xyz=cfg["scale"],      # (X width, Y length, Z height)
            location_xyz=cfg["location"],
            color_rgba=cfg["color"],
            pass_index=cfg["pass_index"],
            collection=collection,
        )
        env[cname] = obj

    # ------------------------------------------------------------------
    # 5. ROAD FURNITURE — kerbs and speed breaker
    # ------------------------------------------------------------------
    for fname, cfg in ROAD_FURNITURE_CONFIGS.items():
        obj = _make_cube(
            name=fname,
            scale_xyz=cfg["scale"],
            location_xyz=cfg["location"],
            color_rgba=cfg["color"],
            pass_index=cfg["pass_index"],
            collection=collection,
        )
        env[fname] = obj

    # ------------------------------------------------------------------
    # 6. MOVING OBSTACLE — animated 0.5 m cube
    # ------------------------------------------------------------------
    # Initial location will be set by animate_obstacle(); we place it at frame 1 pos.
    cfg = OBSTACLE_CONFIG["obstacle_mover"]
    initial_pos = OBSTACLE_KEYFRAMES[0][1]   # (3.0, 7.0, 0.25)

    bpy.ops.mesh.primitive_cube_add(location=initial_pos)
    obs = bpy.context.active_object
    obs.name = "obstacle_mover"

    # Scale: 0.5 m cube → scale = 0.25 (half-extent)
    obs.scale = (
        cfg["scale"][0] / 2.0,
        cfg["scale"][1] / 2.0,
        cfg["scale"][2] / 2.0,
    )
    _apply_scale(obs)

    mat_obs = _make_material("mat_obstacle_mover", cfg["color"])
    obs.data.materials.append(mat_obs)
    obs.pass_index = cfg["pass_index"]

    if collection is not None:
        for col in list(obs.users_collection):
            col.objects.unlink(obs)
        collection.objects.link(obs)

    env["obstacle_mover"] = obs

    # Insert keyframes for the obstacle using the module-level constants
    animate_obstacle(obs, OBSTACLE_KEYFRAMES)

    return env


def setup_lighting() -> bpy.types.Object:
    """Add a Sun lamp configured to simulate outdoor Indian daylight.

    Sun Parameters (from constants.py):
      SUN_ELEVATION_DEG = 55°   — moderate sun angle (mid-morning / afternoon)
      SUN_AZIMUTH_DEG   = 160°  — south-south-east in world space
      SUN_STRENGTH      = 3.0   — energy in arbitrary Blender units

    Rotation computation:
      Blender sun direction is encoded as the light object's rotation in Euler angles.
      We build the rotation from elevation and azimuth:
        elevation: rotation around world X-axis (tilting sun up from horizontal)
        azimuth  : rotation around world Z-axis (rotating sun around horizon)

      In Blender's convention:
        - A sun lamp with rotation (0,0,0) shines straight DOWN (−Z).
        - To point it at elevation θ above horizon:
            rotate X by −(90° − θ)   i.e.  rot_x = −(π/2 − θ_rad)
        - Then rotate around Z for azimuth.

    Returns
    -------
    bpy.types.Object
        The created sun lamp object.
    """
    # Remove any existing sun lamps to avoid duplicates
    for obj in bpy.data.objects:
        if obj.type == "LIGHT" and obj.name.startswith("Sun"):
            bpy.data.objects.remove(obj, do_unlink=True)

    bpy.ops.object.light_add(type="SUN", location=(0.0, 0.0, 10.0))
    sun = bpy.context.active_object
    sun.name = "Sun_CLEAR360"

    # Blender sun rotation:
    #   rot_x = -(90° - elevation)  → tilts sun from straight-down toward horizon
    #   rot_z = azimuth              → rotates sun around vertical axis
    elevation_rad = math.radians(SUN_ELEVATION_DEG)   # 55° → 0.9599 rad
    azimuth_rad   = math.radians(SUN_AZIMUTH_DEG)     # 160° → 2.7925 rad

    rot_x = -(math.pi / 2.0 - elevation_rad)  # -(90° - 55°) = -35° ≈ -0.6109 rad
    rot_z = azimuth_rad                         # 160° ≈ 2.7925 rad

    sun.rotation_euler = (rot_x, 0.0, rot_z)

    # Set lamp strength
    sun.data.energy = SUN_STRENGTH

    # Enable contact shadows for realism
    sun.data.use_contact_shadow = True

    return sun


def animate_obstacle(obj: bpy.types.Object, frame_positions: list) -> None:
    """Insert location keyframes for the moving obstacle object.

    Parameters
    ----------
    obj             : bpy.types.Object
        The Blender object to animate (typically 'obstacle_mover').
    frame_positions : list of (frame_number: int, location: tuple(x, y, z))
        Each entry defines the object's world-space location at that frame.
        Example (from OBSTACLE_KEYFRAMES in constants.py):
          [(1, (3.0, 7.0, 0.25)), (10, (-3.0, 5.5, 0.25)), (20, (-5.0, 3.0, 0.25))]

    The function inserts a 'location' keyframe at each specified Blender frame.
    Blender will linearly interpolate between keyframes by default (BEZIER handles
    will be set; for constant or linear interpolation use fcurve.keyframe_points[i].interpolation).
    """
    for frame_num, (lx, ly, lz) in frame_positions:
        # Set the scene to the target frame and assign location
        bpy.context.scene.frame_set(frame_num)
        obj.location = (lx, ly, lz)

        # Insert a keyframe for the location channel at the current frame
        # 'LOCATION' inserts x, y, z sub-channels
        obj.keyframe_insert(data_path="location", frame=frame_num)
