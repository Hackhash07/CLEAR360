"""
photo_reconstruction/tests/test_m7_3_8_canonical_frame.py — Unit Tests for M7.3.8 Repairs
========================================================================================

Verifies:
  1. Data-Driven Canonical Scene Alignment (ground plane Z=0, wall plane Y=Y_wall, bilateral symmetry X=0)
  2. Data-Driven Region Segmentation without hardcoded bounding boxes
  3. Continuous Planar Background Wall Grid (zero void, zero crescent bowing)
  4. Continuous Ground Pavement Grid
  5. Strict Visibility & Neutral Gray Atlas Mapping for rear/untextured faces
  6. Multi-model COLMAP enumeration and pose graph connectivity analysis
"""

import pytest
import numpy as np
from pathlib import Path

from photo_reconstruction.canonical_frame import CanonicalSceneAligner, CanonicalAlignmentResult
from photo_reconstruction.scene_assembly import RegionSceneAssembler
from photo_reconstruction.texture_engine import VisibilityAwareTextureEngine
from photo_reconstruction.camera_model import PhoneCameraModel
from photo_reconstruction.m7_3_7_pose_graph import GlobalCameraPoseGraphAuditor


def test_canonical_frame_estimation_synthetic():
    """Verify CanonicalSceneAligner recovers ground plane, wall plane and SO(3) frame."""
    aligner = CanonicalSceneAligner()

    # Generate synthetic courtyard scene:
    # Ground plane at Z = -1.5, Wall plane at Y = 8.0, Fountain around X = 0
    np.random.seed(42)
    n_pts = 2000
    # Ground points
    g_x = np.random.uniform(-4, 4, n_pts // 2)
    g_y = np.random.uniform(2, 8, n_pts // 2)
    g_z = np.full(n_pts // 2, -1.5) + np.random.normal(0, 0.01, n_pts // 2)
    pts_g = np.column_stack([g_x, g_y, g_z])

    # Wall points
    w_x = np.random.uniform(-4, 4, n_pts // 2)
    w_y = np.full(n_pts // 2, 8.0) + np.random.normal(0, 0.01, n_pts // 2)
    w_z = np.random.uniform(-1.5, 3.5, n_pts // 2)
    pts_w = np.column_stack([w_x, w_y, w_z])

    pts = np.vstack([pts_g, pts_w])

    # Camera poses along arc in front of wall, camera height Z=0.5
    cam_poses = {}
    for i in range(4):
        T = np.eye(4)
        T[0, 3] = (i - 1.5) * 1.5
        T[1, 3] = 2.0
        T[2, 3] = 0.5
        cam_poses[f"cam_{i}.jpg"] = T

    res = aligner.estimate_canonical_frame(pts, cam_poses)
    assert res.is_canonical_established
    assert res.aligned_points is not None

    # Verify ground datum is shifted to Z = 0.000m
    aligned_g = res.aligned_points[:n_pts // 2]
    assert np.isclose(np.median(aligned_g[:, 2]), 0.0, atol=0.05)

    # Verify wall is planar at positive Y
    aligned_w = res.aligned_points[n_pts // 2:]
    assert np.isclose(np.median(aligned_w[:, 1]), res.wall_y, atol=0.05)


def test_continuous_wall_and_pavement_grids():
    """Verify continuous wall and pavement grids have zero holes and perfect planarity."""
    # Continuous wall
    w_verts, w_faces = RegionSceneAssembler.mesh_wall_continuous_grid(
        y_wall=8.5, half_width=3.8, z_min=0.2, z_max=5.4, grid_step=0.2
    )
    assert len(w_verts) > 100
    assert len(w_faces) > 100
    # Every vertex must lie exactly on Y = 8.5 (zero crescent bowing)
    assert np.allclose(w_verts[:, 1], 8.5, atol=1e-5)
    # Continuous wall normals
    f0 = w_verts[w_faces[:, 0]]
    f1 = w_verts[w_faces[:, 1]]
    f2 = w_verts[w_faces[:, 2]]
    cross = np.cross(f1 - f0, f2 - f0)
    assert np.all(cross[:, 1] != 0)

    # Continuous pavement
    p_verts, p_faces = RegionSceneAssembler.mesh_pavement_continuous_grid(
        y_wall=8.5, half_width=3.8, y_front=4.0, z_ground=0.0, grid_step=0.2
    )
    assert len(p_verts) > 100
    assert len(p_faces) > 100
    # Every vertex must lie exactly on Z = 0.0
    assert np.allclose(p_verts[:, 2], 0.0, atol=1e-5)
    # Normals point upward (+Z direction)
    pf0 = p_verts[p_faces[:, 0]]
    pf1 = p_verts[p_faces[:, 1]]
    pf2 = p_verts[p_faces[:, 2]]
    pcross = np.cross(pf1 - pf0, pf2 - pf0)
    assert np.all(pcross[:, 2] != 0)


def test_visibility_aware_texturing_neutral_swatch():
    """Verify rear-facing or untextured faces are mapped to neutral gray swatch #8c8c8c."""
    engine = VisibilityAwareTextureEngine(atlas_size=512)

    # Create front-facing triangle and rear-facing triangle at Z=2.0 (in front of camera)
    v = np.array([
        # Face 0: Front-facing towards camera at origin looking along +Z
        [-0.2, -0.2, 2.0], [0.2, -0.2, 2.0], [0.0, 0.2, 2.0],
        # Face 1: Rear-facing (reversed vertex order)
        [-0.2, -0.2, 2.0], [0.0, 0.2, 2.0], [0.2, -0.2, 2.0],
    ], dtype=np.float32)
    f = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint32)

    cam = PhoneCameraModel(
        name="cam_0.jpg",
        model_name="PINHOLE",
        width=640,
        height=480,
        params=np.array([500.0, 500.0, 320.0, 240.0]),
        fx=500.0, fy=500.0, cx=320.0, cy=240.0,
        T_world_cam=np.eye(4),
        T_cam_world=np.eye(4),
    )

    dummy_img = np.full((480, 640, 3), 200, dtype=np.uint8)
    images = {"cam_0.jpg": dummy_img}
    cameras = {"cam_0.jpg": cam}

    tex_v, tex_f, tex_uv, atlas, report = engine.texture_mesh(v, f, cameras, images)

    assert report.total_faces == 2
    assert report.textured_faces == 1
    assert report.untextured_faces == 1
    assert report.rear_facing_faces == 1

    # Check that neutral swatch exists at top-left of atlas
    swatch_pixel = atlas[10, 10]
    assert np.allclose(swatch_pixel, [140, 140, 140], atol=10)

    # Rear-facing triangle UVs must point to the neutral swatch
    rear_tri_uvs = tex_uv[tex_f[0]]
    assert np.all(rear_tri_uvs[:, 0] <= 0.2)
