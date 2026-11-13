"""
sim/vehicle_model.py — Procedural Hyundai-Creta-approximate SUV model
======================================================================

This module creates a simplified but dimensionally-accurate 3D vehicle model
in Blender representing a Hyundai Creta 2022 (approximate).

COORDINATE CONVENTION:
  World frame: X=right, Y=forward (travel), Z=up. Vehicle centre at origin.
  Front bumper at Y≈+2.15 m; rear bumper at Y≈−2.15 m.

VEHICLE PARTS CREATED:
  body          — main box, full vehicle bounding box (excluding roof detail)
  roof_panel    — raised upper section, sits atop body
  wheel_fl      — front-left  tyre cylinder
  wheel_fr      — front-right tyre cylinder
  wheel_rl      — rear-left   tyre cylinder
  wheel_rr      — rear-right  tyre cylinder
  mirror_left   — left side mirror housing
  mirror_right  — right side mirror housing

All parts receive pass_index=2 (ego vehicle semantic class).

Usage:
  from sim.vehicle_model import clear_scene, create_vehicle
  clear_scene()
  parts = create_vehicle()          # returns dict keyed by part name
"""

import bpy
import math

from sim.constants import (
    VEHICLE_HALF_LEN, VEHICLE_HALF_W, VEHICLE_HEIGHT_M,
    WHEEL_RADIUS_M, WHEEL_WIDTH_M, WHEEL_OFFSET_X,
    FRONT_AXLE_Y, REAR_AXLE_Y,
    ROOF_HEIGHT_START_Z, ROOF_HEIGHT_M, ROOF_LENGTH_M, ROOF_WIDTH_M,
    MIRROR_TIP_X, MIRROR_HEIGHT_M, MIRROR_DEPTH_M, MIRROR_WIDTH_M,
    MIRROR_Y_CENTRE, MIRROR_Z_LOW,
    VEHICLE_PASS_INDEX,
)


# ---------------------------------------------------------------------------
# HELPER UTILITIES
# ---------------------------------------------------------------------------

def _make_material(name: str, color_rgba: tuple) -> bpy.types.Material:
    """Create (or reuse) a simple diffuse Principled BSDF material.

    Parameters
    ----------
    name       : str   — material data-block name
    color_rgba : tuple — (R, G, B, A) in [0,1]

    Returns
    -------
    bpy.types.Material
    """
    if name in bpy.data.materials:
        return bpy.data.materials[name]

    mat = bpy.data.materials.new(name)
    mat.use_nodes = True

    # Clear auto-created nodes and set up Principled BSDF
    nodes = mat.node_tree.nodes
    nodes.clear()

    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = color_rgba
    bsdf.inputs["Roughness"].default_value = 0.4
    bsdf.inputs["Metallic"].default_value  = 0.3

    out = nodes.new("ShaderNodeOutputMaterial")
    mat.node_tree.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    return mat


def _apply_scale(obj: bpy.types.Object) -> None:
    """Apply current obj.scale to mesh data, resetting scale to (1,1,1).

    Blender's transform_apply modifies the underlying mesh vertices so that
    the displayed dimensions match the nominal scale values.  This must be
    done before reading obj.dimensions reliably.
    """
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)


def _assign_pass_index(obj: bpy.types.Object, index: int) -> None:
    """Set the Object Properties > Visibility > Pass Index field.

    This index is used by the Cycles 'Object Index' render pass and the
    compositor IndexOB socket to generate semantic segmentation maps.
    """
    obj.pass_index = index


def _link_to_collection(obj: bpy.types.Object, collection) -> None:
    """Move obj from the default scene collection into *collection* (if given)."""
    if collection is None:
        return
    # Unlink from current collections
    for col in list(obj.users_collection):
        col.objects.unlink(obj)
    collection.objects.link(obj)


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------

def clear_scene() -> None:
    """Delete ALL mesh objects currently in the scene.

    Called once at the start of export_frames.py to ensure a clean slate
    before the procedural scene is built.  Cameras, lights, and empties are
    left intact.
    """
    bpy.ops.object.select_all(action="DESELECT")

    # Select every mesh object in every collection
    for obj in bpy.data.objects:
        if obj.type == "MESH":
            obj.select_set(True)

    bpy.ops.object.delete(use_global=False)

    # Purge orphaned mesh data-blocks
    for mesh in bpy.data.meshes:
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)


def create_vehicle(collection=None) -> dict:
    """Build a simplified Hyundai-Creta-approximate SUV in Blender.

    The vehicle is centred at the world origin (0, 0, 0) with:
      - Bottom surface at Z = 0  (sitting on the ground plane)
      - Front bumper at Y ≈ +2.15 m
      - Rear  bumper at Y ≈ −2.15 m

    All parts are assigned pass_index = VEHICLE_PASS_INDEX (2) so that the
    Cycles IndexOB pass paints every vehicle pixel with value 2.

    Parameters
    ----------
    collection : bpy.types.Collection or None
        If provided, all created objects are linked into this collection
        instead of the default scene collection.

    Returns
    -------
    dict
        Mapping of part_name (str) → bpy.types.Object for all created parts.
    """
    parts: dict[str, bpy.types.Object] = {}

    # ------------------------------------------------------------------
    # 1. BODY — main bounding box
    # ------------------------------------------------------------------
    # Blender primitive_cube_add creates a 2×2×2 m cube centred at the given
    # location.  We scale it to (width, length, height) half-extents, then
    # shift it up by half its height so the bottom sits at Z=0.
    #
    # Dimensions:
    #   X: 1.79 m (VEHICLE_WIDTH_M)   → half = 0.895 m
    #   Y: 4.30 m (VEHICLE_LENGTH_M)  → half = 2.15 m
    #   Z: 1.47 m (body, up to roof base = ROOF_HEIGHT_START_Z=0.80 m; we model
    #              the body as the full cab + slightly lower than roof apex)
    #   Body Z height: roof starts at 0.80 m above ground, so body top at 0.80 m.
    #   Actually we keep the body as a single box from Z=0 to Z=VEHICLE_HEIGHT_M
    #   for simplicity; the roof_panel is an extra overlay on top.
    body_z_height = VEHICLE_HEIGHT_M   # 1.635 m — body goes full height
    body_z_centre = body_z_height / 2.0  # 0.8175 m — cube centre above ground

    bpy.ops.mesh.primitive_cube_add(location=(0.0, 0.0, body_z_centre))
    body = bpy.context.active_object
    body.name = "body"

    # Scale to target dimensions.
    # primitive_cube_add makes a 2×2×2 m unit cube; scale factors are half-extents.
    # obj.scale = (Sx, Sy, Sz) means final size = 2*Sx × 2*Sy × 2*Sz.
    body.scale = (
        VEHICLE_HALF_W,          # X: 0.895 → body is 1.79 m wide
        VEHICLE_HALF_LEN,         # Y: 2.15  → body is 4.30 m long
        body_z_height / 2.0,     # Z: 0.8175 → body is 1.635 m tall
    )
    _apply_scale(body)

    # Add a Bevel modifier for slightly rounded edges (purely cosmetic)
    bevel = body.modifiers.new(name="Bevel", type="BEVEL")
    bevel.width  = 0.03    # 3 cm bevel
    bevel.segments = 2

    # Material: dark charcoal body colour
    mat_body = _make_material("mat_vehicle_body", (0.08, 0.08, 0.10, 1.0))
    body.data.materials.append(mat_body)

    _assign_pass_index(body, VEHICLE_PASS_INDEX)
    _link_to_collection(body, collection)
    parts["body"] = body

    # ------------------------------------------------------------------
    # 2. ROOF PANEL — raised upper section overlaying the body top
    # ------------------------------------------------------------------
    # Sits from Z=ROOF_HEIGHT_START_Z (0.80 m) up to Z=VEHICLE_HEIGHT_M (1.635 m).
    # Narrower and shorter than body for the typical SUV silhouette.
    #
    # roof Z centre = ROOF_HEIGHT_START_Z + ROOF_HEIGHT_M/2
    roof_z_centre = ROOF_HEIGHT_START_Z + ROOF_HEIGHT_M / 2.0  # 0.80 + 0.4175 = 1.2175

    bpy.ops.mesh.primitive_cube_add(location=(0.0, 0.0, roof_z_centre))
    roof = bpy.context.active_object
    roof.name = "roof_panel"

    roof.scale = (
        ROOF_WIDTH_M  / 2.0,   # X: 0.85 → roof is 1.70 m wide
        ROOF_LENGTH_M / 2.0,   # Y: 1.40 → roof is 2.80 m long
        ROOF_HEIGHT_M / 2.0,   # Z: 0.4175 → roof panel is 0.835 m tall
    )
    _apply_scale(roof)

    mat_roof = _make_material("mat_vehicle_roof", (0.07, 0.07, 0.09, 1.0))
    roof.data.materials.append(mat_roof)

    _assign_pass_index(roof, VEHICLE_PASS_INDEX)
    _link_to_collection(roof, collection)
    parts["roof_panel"] = roof

    # ------------------------------------------------------------------
    # 3. WHEELS — four cylinders (tyres)
    # ------------------------------------------------------------------
    # Cylinders are created upright (Z axis = cylinder axis) by default.
    # We rotate 90° around X to lay them on their side (rolling axis = Y).
    #
    # Wheel centre Z = WHEEL_RADIUS_M (tyre sits on ground, Z=0)
    # Axle positions: front Y=+1.30 m, rear Y=−1.30 m
    # Lateral positions: X = ±WHEEL_OFFSET_X (±0.90 m)
    #
    # primitive_cylinder_add(radius, depth, ...) creates a cylinder of the
    # given radius and depth along Z.  After 90° rotation about X, depth
    # becomes the tyre width in Y, but we want it in X (vehicle's lateral
    # direction).  Actually after the rotation the cylinder axis lies along X.
    # We therefore rotate 90° around Y instead, so the axis lies along X.
    # Wait — to have tyre width along X (vehicle lateral), we rotate 90° about Y:
    #   original axis: Z → after rot90 about Y → X ✓

    wheel_specs = [
        ("wheel_fl", (-WHEEL_OFFSET_X, FRONT_AXLE_Y)),   # front-left
        ("wheel_fr", (+WHEEL_OFFSET_X, FRONT_AXLE_Y)),   # front-right
        ("wheel_rl", (-WHEEL_OFFSET_X, REAR_AXLE_Y)),    # rear-left
        ("wheel_rr", (+WHEEL_OFFSET_X, REAR_AXLE_Y)),    # rear-right
    ]

    for wname, (wx, wy) in wheel_specs:
        # Place cylinder at wheel hub centre; Z = wheel radius (bottom touches Z=0)
        bpy.ops.mesh.primitive_cylinder_add(
            radius=WHEEL_RADIUS_M,
            depth=WHEEL_WIDTH_M,
            location=(wx, wy, WHEEL_RADIUS_M),
        )
        wobj = bpy.context.active_object
        wobj.name = wname

        # Rotate 90° around Y so the cylinder axis aligns with X (vehicle lateral)
        # This makes the flat circular faces face left/right, and the curved surface
        # becomes the tyre tread touching the ground.
        wobj.rotation_euler = (0.0, math.radians(90.0), 0.0)
        _apply_scale(wobj)

        mat_wheel = _make_material("mat_vehicle_wheel", (0.05, 0.05, 0.05, 1.0))
        wobj.data.materials.append(mat_wheel)

        _assign_pass_index(wobj, VEHICLE_PASS_INDEX)
        _link_to_collection(wobj, collection)
        parts[wname] = wobj

    # ------------------------------------------------------------------
    # 4. SIDE MIRRORS — two box housings
    # ------------------------------------------------------------------
    # Mirror specs (from constants):
    #   MIRROR_TIP_X   = ±1.085 m from vehicle CL (outer tip X position)
    #   MIRROR_WIDTH_M = 0.19 m  (X extent)
    #   MIRROR_HEIGHT_M= 0.30 m  (Z: 1.10 to 1.40)
    #   MIRROR_DEPTH_M = 0.12 m  (Y depth)
    #   MIRROR_Y_CENTRE= 1.50 m  (near front doors)
    #   MIRROR_Z_LOW   = 1.10 m
    #
    # Mirror centre X = MIRROR_TIP_X ∓ MIRROR_WIDTH_M/2
    #   left  mirror: tip at X=−1.085, centre at X = −1.085 + 0.095 = −0.990
    #   right mirror: tip at X=+1.085, centre at X = +1.085 − 0.095 = +0.990
    # Mirror centre Z = MIRROR_Z_LOW + MIRROR_HEIGHT_M/2 = 1.10 + 0.15 = 1.25

    mirror_z_centre = MIRROR_Z_LOW + MIRROR_HEIGHT_M / 2.0   # 1.25 m

    mirror_specs = [
        ("mirror_left",  -(MIRROR_TIP_X - MIRROR_WIDTH_M / 2.0)),  # X centre for left
        ("mirror_right", +(MIRROR_TIP_X - MIRROR_WIDTH_M / 2.0)),  # X centre for right
    ]

    for mname, mx in mirror_specs:
        bpy.ops.mesh.primitive_cube_add(
            location=(mx, MIRROR_Y_CENTRE, mirror_z_centre)
        )
        mobj = bpy.context.active_object
        mobj.name = mname

        mobj.scale = (
            MIRROR_WIDTH_M  / 2.0,   # X half-extent
            MIRROR_DEPTH_M  / 2.0,   # Y half-extent
            MIRROR_HEIGHT_M / 2.0,   # Z half-extent
        )
        _apply_scale(mobj)

        mat_mirror = _make_material("mat_vehicle_mirror", (0.06, 0.06, 0.08, 1.0))
        mobj.data.materials.append(mat_mirror)

        _assign_pass_index(mobj, VEHICLE_PASS_INDEX)
        _link_to_collection(mobj, collection)
        parts[mname] = mobj

    return parts
