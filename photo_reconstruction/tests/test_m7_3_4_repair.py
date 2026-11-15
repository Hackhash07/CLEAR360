"""
photo_reconstruction/tests/test_m7_3_4_repair.py — Automated Regression Tests for CLEAR360 M7.3.4
==================================================================================================

Validates coordinate transformations, normal orientation, triangular artifact filtering,
texture UV alignment, landmark validation, and diagnostic generation.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

from photo_reconstruction.camera_model import PhoneCameraModel
from photo_reconstruction.coord_transform import (
    Clear360CoordinateTransformer,
    R_COLMAP_TO_CLEAR360,
    T_COLMAP_TO_CLEAR360,
)
from photo_reconstruction.landmark_validator import FountainLandmarkValidator
from photo_reconstruction.m7_3_4_diagnostics import M734DiagnosticRenderer
from photo_reconstruction.normal_orientation import NormalOrientationEngine
from photo_reconstruction.texture_engine import VisibilityAwareTextureEngine


def test_coord_transform_so3_properties():
    """Verify canonical coordinate transformation is a certified pure rotation in SO(3)."""
    det = np.linalg.det(R_COLMAP_TO_CLEAR360)
    assert abs(det - 1.0) < 1e-6, f"det(R) must be +1.0, got {det}"

    ortho = R_COLMAP_TO_CLEAR360 @ R_COLMAP_TO_CLEAR360.T
    assert np.allclose(ortho, np.eye(3), atol=1e-6), "R must be orthogonal"

    # Test camera axis transformations
    # In camera frame: +X right, +Y down, +Z forward
    # Up vector is [0, -1, 0]
    cam_up_colmap = np.array([0.0, -1.0, 0.0])
    cam_fwd_colmap = np.array([0.0, 0.0, 1.0])
    cam_right_colmap = np.array([1.0, 0.0, 0.0])

    world_up = R_COLMAP_TO_CLEAR360 @ cam_up_colmap
    world_fwd = R_COLMAP_TO_CLEAR360 @ cam_fwd_colmap
    world_right = R_COLMAP_TO_CLEAR360 @ cam_right_colmap

    # Clear360 vehicle frame: +X Right, +Y Forward, +Z Up
    assert np.allclose(world_up, [0.0, 0.0, 1.0]), f"Camera UP must map to World +Z, got {world_up}"
    assert np.allclose(world_fwd, [0.0, 1.0, 0.0]), f"Camera FWD must map to World +Y, got {world_fwd}"
    assert np.allclose(world_right, [1.0, 0.0, 0.0]), f"Camera RIGHT must map to World +X, got {world_right}"


def test_normal_orientation_flips_inverted_faces():
    """Verify faces pointing away from the camera are reliably inverted to face outward."""
    engine = NormalOrientationEngine()

    # Camera at (0, 0, 0)
    cam_c = np.array([0.0, 0.0, 0.0])

    # Vertices on plane at Y = 5.0 (looking from camera in +Y)
    # v0 = (-1, 5, 0), v1 = (1, 5, 0), v2 = (0, 5, 1)
    vertices = np.array([
        [-1.0, 5.0, 0.0],
        [1.0, 5.0, 0.0],
        [0.0, 5.0, 1.0],
    ], dtype=np.float32)

    # Face 0: wound counter-clockwise: normal points in -Y (towards camera at Y=0)
    # Cross: (v1 - v0) x (v2 - v0) = (2, 0, 0) x (1, 0, 1) = (0, -2, 0) -> n = (0, -1, 0)
    face_outward = np.array([[0, 1, 2]], dtype=np.uint32)

    # Face 1: wound clockwise: normal points in +Y (away from camera, inverted)
    face_inward = np.array([[0, 2, 1]], dtype=np.uint32)

    faces_both = np.vstack([face_outward, face_inward])

    oriented_faces, normals, flipped = engine.orient_faces_towards_cameras(vertices, faces_both, cam_c)

    assert flipped == 1, "Exactly one inverted face should have been flipped"
    # Both faces must now have normal pointing towards camera (dot with (cam - centroid) > 0)
    v0 = vertices[oriented_faces[:, 0]]
    v1 = vertices[oriented_faces[:, 1]]
    v2 = vertices[oriented_faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    dots = np.sum(cross * (cam_c - (v0 + v1 + v2)/3.0), axis=1)
    assert np.all(dots > 0), "All faces must face camera after orientation"


def test_triangular_sliver_filter():
    """Verify extreme slivers and degenerate triangles are cleanly pruned."""
    engine = NormalOrientationEngine(max_aspect_ratio=20.0, min_face_area=1e-6)

    # Normal equilateral triangle: edge ~ 1.0, area ~ 0.43 -> aspect ratio ~ 1.15
    v_norm = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 0.866, 0.0],
    ], dtype=np.float32)

    # Degenerate zero-area collinear triangle
    v_degen = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
    ], dtype=np.float32)

    # Extreme sliver triangle: edge 1.0, height 0.01 -> aspect ratio ~ 50.0
    v_sliver = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 0.01, 0.0],
    ], dtype=np.float32)

    all_v = np.vstack([v_norm, v_degen, v_sliver])
    faces = np.array([
        [0, 1, 2],  # Normal
        [3, 4, 5],  # Degenerate
        [6, 7, 8],  # Sliver
    ], dtype=np.uint32)

    v_clean, f_clean, stats = engine.filter_triangular_artifacts(all_v, faces)

    assert stats["initial_faces"] == 3
    assert stats["retained_faces"] == 1, "Only the normal triangle should be retained"
    assert len(f_clean) == 1
    assert len(v_clean) == 3


def test_upright_uv_coordinates(tmp_path):
    """Verify texture engine generates upright UV coordinates matching VisPy / OpenGL."""
    engine = VisibilityAwareTextureEngine(atlas_size=1024)

    # 1 camera in Clear360 vehicle frame looking along +Y with +Z up
    pose = T_COLMAP_TO_CLEAR360.copy()
    cam = PhoneCameraModel(
        name="test.jpg", model_name="OPENCV",
        width=800, height=600, params=np.array([600.0, 600.0, 400.0, 300.0]),
        fx=600.0, fy=600.0, cx=400.0, cy=300.0,
        T_world_cam=pose, T_cam_world=np.linalg.inv(pose),
    )
    img = np.ones((600, 800, 3), dtype=np.uint8) * 128

    # 1 triangle facing camera at Y=5.0m
    # Top vertex has Z = +1.0m, bottom vertices have Z = -1.0m
    v = np.array([
        [-0.5, 5.0, -1.0],
        [0.5, 5.0, -1.0],
        [0.0, 5.0, 1.0],
    ], dtype=np.float32)
    # Face oriented counter-clockwise towards camera
    f = np.array([[0, 1, 2]], dtype=np.uint32)

    new_v, new_f, new_uv, atlas, rep = engine.texture_mesh(
        vertices=v, faces=f, camera_models={"test.jpg": cam},
        images={"test.jpg": img}, out_dir=tmp_path, base_name="test_mesh"
    )

    assert rep.textured_faces == 1
    assert len(new_uv) == 3
    # UV coordinates must be strictly within [0, 1]
    assert np.all(new_uv >= 0.0) and np.all(new_uv <= 1.0)
    # V coordinates must have V(top) > V(bottom) under standard Wavefront OBJ UV convention
    # Top vertex is index 2 (Z = 1.0), bottom vertices are indices 0, 1 (Z = -1.0)
    v_top = new_uv[2, 1]
    v_bottom = new_uv[0, 1]
    assert v_top > v_bottom, f"Top vertex V ({v_top}) must be greater than bottom vertex V ({v_bottom}) for upright OBJ standard"


def test_landmark_validator():
    """Verify fountain landmark spatial relations and reprojection validation."""
    validator = FountainLandmarkValidator()
    # Mock camera models
    cams = {}
    for i, name in enumerate(["fountain_0003.jpg", "fountain_0004.jpg"]):
        pose = np.eye(4)
        pose[:3, 3] = [i * 2.0, 0.0, 2.1]
        cams[name] = PhoneCameraModel(
            name=name, model_name="OPENCV",
            width=1600, height=1067, params=np.array([1435.0, 1435.0, 800.0, 533.5]),
            fx=1435.0, fy=1435.0, cx=800.0, cy=533.5,
            T_world_cam=pose, T_cam_world=np.linalg.inv(pose),
        )

    # Custom ground-truth 3D landmarks
    custom_lms = {
        "fountain_top_crest": np.array([0.0, 5.4, 3.75], dtype=np.float32),
        "central_sculpture_center": np.array([0.0, 5.35, 2.15], dtype=np.float32),
        "left_ornamental_border": np.array([-1.35, 5.30, 2.15], dtype=np.float32),
        "right_ornamental_border": np.array([1.35, 5.30, 2.15], dtype=np.float32),
        "top_basin_rim": np.array([0.0, 4.85, 1.15], dtype=np.float32),
        "ground_contact_region": np.array([0.0, 5.0, 0.05], dtype=np.float32),
    }

    report = validator.validate_scene_landmarks(cams, custom_landmarks_3d=custom_lms)
    assert report.total_landmarks == 6
    assert report.vertical_ordering_valid is True
    assert report.relief_projection_valid is True
    assert report.lateral_symmetry_valid is True
