"""
photo_reconstruction/tests/test_m7_3_3_assembly.py
===================================================

Unit tests for CLEAR360 M7.3.3:
  1. Camera overlap fusion & surface consensus.
  2. 6-region physical architectural segmentation.
  3. Component-tailored surface meshing (Delaunay 2.5D, Alpha Shape).
  4. Scene Graph hierarchy & common world coordinate frame.
  5. Reconstruction stage auditor.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

from photo_reconstruction.camera_model import PhoneCameraModel
from photo_reconstruction.overlap_fusion import CameraOverlapFusion
from photo_reconstruction.scene_assembly import RegionSceneAssembler, SceneRegion
from photo_reconstruction.stage_auditor import ReconstructionStageAuditor


def test_camera_overlap_fusion_consensus():
    """Verify spatial consensus fuses overlapping points and preserves distinct relief."""
    fusion = CameraOverlapFusion(coincidence_radius=0.04, normal_agreement_threshold=0.707)

    # Synthetic camera
    K = np.array([[1000.0, 0.0, 400.0], [0.0, 1000.0, 300.0], [0.0, 0.0, 1.0]])
    cam1 = PhoneCameraModel(
        name="cam1.jpg", model_name="PINHOLE", width=800, height=600,
        params=np.array([1000.0, 1000.0, 400.0, 300.0]),
        fx=1000.0, fy=1000.0, cx=400.0, cy=300.0,
        T_world_cam=np.eye(4),
    )
    cam2 = PhoneCameraModel(
        name="cam2.jpg", model_name="PINHOLE", width=800, height=600,
        params=np.array([1000.0, 1000.0, 400.0, 300.0]),
        fx=1000.0, fy=1000.0, cx=400.0, cy=300.0,
        T_world_cam=np.array([
            [1.0, 0.0, 0.0, 0.2],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]),
    )

    depth1 = np.ones((600, 800), dtype=np.float32) * 5.0
    depth2 = np.ones((600, 800), dtype=np.float32) * 5.0

    res = fusion.fuse_camera_clouds(
        camera_models={"cam1.jpg": cam1, "cam2.jpg": cam2},
        aligned_depth_maps={"cam1.jpg": depth1, "cam2.jpg": depth2},
        stride=20,
        voxel_size=0.02,
    )

    assert res.total_raw_points > 0
    assert res.fused_point_count > 0
    assert res.fused_point_count < res.total_raw_points
    assert res.mean_normal_agreement >= 0.85
    assert len(res.source_camera_indices) == res.fused_point_count


def test_6_region_segmentation():
    """Verify 6-region physical architectural segmentation partitioning."""
    # Synthetic point cloud covering ground, wall, sculpture, basin, frame, adjacent
    pts = np.array([
        [0.0, 5.0, 0.2],    # Pavement (Z < 0.55)
        [0.85, 7.8, 2.4],   # Sculpture (X in [0.4, 1.3], Y in [7.5, 8.3], Z in [1.95, 2.95])
        [0.0, 6.5, 1.2],    # Basin (Z in [0.55, 1.90], |X| <= 1.75, Y in [5.8, 7.85])
        [0.0, 8.0, 3.5],    # Fountain Frame (|X| <= 2.1, Y in [7.5, 8.85], Z in [1.2, 5.2])
        [0.0, 9.5, 3.0],    # Wall (Y >= 8.35, Z >= 0.55)
        [5.0, 7.0, 2.0],    # Adjacent (X > 2.1)
    ], dtype=np.float32)
    cols = np.ones((len(pts), 3), dtype=np.float32)
    norms = np.tile([0.0, -1.0, 0.0], (len(pts), 1)).astype(np.float32)

    regions = RegionSceneAssembler.segment_point_cloud(pts, cols, norms)

    assert len(regions["pavement"][0]) == 1
    assert len(regions["sculpture"][0]) == 1
    assert len(regions["basin"][0]) == 1
    assert len(regions["fountain_frame"][0]) == 1
    assert len(regions["wall"][0]) == 1
    assert len(regions["adjacent"][0]) == 1


def test_modular_meshing_pavement_planar():
    """Verify 2.5D Delaunay pavement meshing guarantees manifoldness and upward normal."""
    # Flat horizontal plane
    x = np.linspace(-1.0, 1.0, 10)
    y = np.linspace(0.0, 2.0, 10)
    xx, yy = np.meshgrid(x, y)
    zz = np.zeros_like(xx)
    pts = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()]).astype(np.float32)
    cols = np.ones((len(pts), 3), dtype=np.float32)
    norms = np.tile([0.0, 0.0, 1.0], (len(pts), 1)).astype(np.float32)

    verts, faces = RegionSceneAssembler.mesh_pavement_planar(pts, cols, norms, max_edge_len=0.5)

    assert len(verts) > 0
    assert len(faces) > 0
    # Check that face normals point upward (+Z)
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    assert np.all(cross[:, 2] >= 0.0)


def test_modular_meshing_wall_planar():
    """Verify 2.5D Delaunay vertical wall meshing enforces forward-facing normals."""
    x = np.linspace(-1.0, 1.0, 10)
    z = np.linspace(1.0, 3.0, 10)
    xx, zz = np.meshgrid(x, z)
    yy = np.ones_like(xx) * 8.5
    pts = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()]).astype(np.float32)
    cols = np.ones((len(pts), 3), dtype=np.float32)
    norms = np.tile([0.0, -1.0, 0.0], (len(pts), 1)).astype(np.float32)

    verts, faces = RegionSceneAssembler.mesh_wall_planar(pts, cols, norms, max_edge_len=0.5)

    assert len(verts) > 0
    assert len(faces) > 0
    # Check that face normals point forward (-Y)
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    assert np.all(cross[:, 1] <= 0.0)


def test_reconstruction_stage_auditor_so3():
    """Verify stage auditor enforces det(R) = +1.0 and orthonormality."""
    auditor = ReconstructionStageAuditor(session_dir=Path("/tmp"))
    R_valid = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    T_valid = np.eye(4)
    T_valid[:3, :3] = R_valid

    cam = PhoneCameraModel(
        name="test.jpg", model_name="PINHOLE", width=800, height=600,
        params=np.array([1000.0, 1000.0, 400.0, 300.0]),
        fx=1000.0, fy=1000.0, cx=400.0, cy=300.0,
        T_world_cam=T_valid,
    )

    audit = auditor.audit_camera_transforms({"test.jpg": cam})
    assert audit["test.jpg"]["is_valid_so3"] is True
    assert audit["test.jpg"]["det_R"] == 1.0

