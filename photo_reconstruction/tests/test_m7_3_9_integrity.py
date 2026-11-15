"""
photo_reconstruction/tests/test_m7_3_9_integrity.py — M7.3.9 Geometry & Texturing Integrity Tests
================================================================================================
"""

import pytest
import numpy as np
from pathlib import Path
import open3d as o3d

from photo_reconstruction.m7_3_6_fusion import M736GeometricConsensusEngine
from photo_reconstruction.camera_model import PhoneCameraModel
from photo_reconstruction.texture_engine import VisibilityAwareTextureEngine
from photo_reconstruction.scene_assembly import RegionSceneAssembler


def test_m7_3_9_confidence_aware_separation():
    """Verify M7.3.9 confidence-aware point separation stores verified and uncertain points."""
    engine = M736GeometricConsensusEngine(voxel_fusion_size=0.03)

    # Synthetic camera looking at planar surface at Z = 2.0
    cam1 = PhoneCameraModel(
        name="cam1.jpg",
        model_name="PINHOLE",
        width=100,
        height=100,
        params=np.array([100.0, 100.0, 50.0, 50.0]),
        fx=100.0,
        fy=100.0,
        cx=50.0,
        cy=50.0,
        T_world_cam=np.eye(4),
        T_cam_world=np.eye(4),
    )
    # Second camera slightly shifted laterally
    T_cam2 = np.eye(4)
    T_cam2[0, 3] = 0.2
    cam2 = PhoneCameraModel(
        name="cam2.jpg",
        model_name="PINHOLE",
        width=100,
        height=100,
        params=np.array([100.0, 100.0, 50.0, 50.0]),
        fx=100.0,
        fy=100.0,
        cx=50.0,
        cy=50.0,
        T_world_cam=T_cam2,
        T_cam_world=np.linalg.inv(T_cam2),
    )

    cams = {"cam1.jpg": cam1, "cam2.jpg": cam2}
    depth_maps = {
        "cam1.jpg": np.full((100, 100), 2.0, dtype=np.float32),
        "cam2.jpg": np.full((100, 100), 2.0, dtype=np.float32),
    }
    imgs = {
        "cam1.jpg": np.full((100, 100, 3), 128, dtype=np.uint8),
        "cam2.jpg": np.full((100, 100, 3), 128, dtype=np.uint8),
    }

    f_pts, f_cols, f_norms, f_w, rep = engine.evaluate_multi_view_consensus(
        camera_models=cams,
        depth_maps=depth_maps,
        images_rgb=imgs,
        stride=2,
    )
    assert len(f_pts) > 0
    assert hasattr(engine, "uncertain_pts")
    assert rep.thickness_reduction_pct >= 50.0


def test_m7_3_9_smooth_vertex_normals():
    """Verify Wavefront OBJ export generates authentic smooth vertex normals."""
    engine = VisibilityAwareTextureEngine(atlas_size=512)

    # Simple 2-triangle quad
    v = np.array([
        [-1.0, 0.0, 2.0],
        [1.0, 0.0, 2.0],
        [-1.0, 2.0, 2.0],
        [1.0, 2.0, 2.0],
    ], dtype=np.float32)
    f = np.array([
        [0, 1, 2],
        [1, 3, 2],
    ], dtype=np.uint32)

    cam = PhoneCameraModel(
        name="cam.jpg",
        model_name="PINHOLE",
        width=200,
        height=200,
        params=np.array([150.0, 150.0, 100.0, 100.0]),
        fx=150.0,
        fy=150.0,
        cx=100.0,
        cy=100.0,
        T_world_cam=np.eye(4),
        T_cam_world=np.eye(4),
    )

    tmp_dir = Path("/tmp/clear360_m739_test")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out_obj = tmp_dir / "test_mesh.obj"

    tex_v, tex_f, tex_uv, atlas, rep = engine.texture_mesh(
        vertices=v,
        faces=f,
        camera_models={"cam.jpg": cam},
        images={"cam.jpg": np.full((200, 200, 3), 150, dtype=np.uint8)},
        out_dir=tmp_dir,
        base_name="test_mesh",
    )

    assert out_obj.exists()
    has_vn = False
    with open(out_obj) as f_obj:
        for line in f_obj:
            if line.startswith("vn "):
                has_vn = True
                break
    assert has_vn, "Smooth vertex normals (vn) must be written to OBJ"


def test_m7_3_9_bounded_observed_support():
    """Verify pavement and wall meshes do not construct unobserved 22-meter grids."""
    # Synthetic cluster of 100 wall points spanning [-1.5, 1.5]
    np.random.seed(42)
    pts = np.column_stack([
        np.random.uniform(-1.5, 1.5, 100),
        np.full(100, 8.5),
        np.random.uniform(0.5, 3.0, 100),
    ]).astype(np.float32)

    v_w, f_w = RegionSceneAssembler.mesh_wall_planar(pts, pts, pts, max_edge_len=0.30)
    assert len(v_w) > 0
    assert len(f_w) > 0
    # Width must strictly match observed cluster (< 3.5m, not 22m)
    w_width = float(v_w[:, 0].max() - v_w[:, 0].min())
    assert w_width <= 3.5, f"Expected observed support width <= 3.5m, got {w_width}m"
