"""
photo_reconstruction/tests/test_m7_3_2_innovations.py — M7.3.2 Verification Test Suite
====================================================================================

Verifies the mathematical and implementation correctness of all Phase 2 innovations:
  1. Authoritative CameraModel subpixel forward/backward projection and distortion inversion.
  2. Robust SfM-Scale Depth Alignment with held-out validation.
  3. Multi-View Forward-Backward Geometric Consistency filtering.
  4. Multi-Radius Ball-Pivoting Algorithm (BPA) and region-aware hybrid meshing.
  5. Visibility-Aware Texturing Engine, UV parameterization, and texture atlas generation.
"""

from pathlib import Path
import tempfile
import numpy as np
import pytest

from photo_reconstruction.camera_model import PhoneCameraModel
from photo_reconstruction.depth_aligner import SfMDepthAligner
from photo_reconstruction.mvs_consistency import MVSGeometricConsistencyFilter
from photo_reconstruction.mesh_builder import MeshBuilder
from photo_reconstruction.texture_engine import VisibilityAwareTextureEngine


def test_camera_model_subpixel_roundtrip():
    """Verify that unproject(project(P)) reconstructs 3D coordinates with sub-millimetre precision."""
    # Test SIMPLE_RADIAL camera model with authentic distortion coefficient
    cam = PhoneCameraModel(
        name="test_cam.jpg",
        model_name="SIMPLE_RADIAL",
        width=1600,
        height=1067,
        params=np.array([1435.0, 800.0, 533.5, 0.001], dtype=np.float64),
        fx=1435.0,
        fy=1435.0,
        cx=800.0,
        cy=533.5,
        distortion=np.array([0.001], dtype=np.float64),
    )

    # Synthetic 3D points within camera field of view
    np.random.seed(42)
    z = np.random.uniform(2.0, 8.0, size=100)
    x = np.random.uniform(-0.4, 0.4, size=100) * z
    y = np.random.uniform(-0.3, 0.3, size=100) * z
    pts_cam = np.column_stack([x, y, z])

    # 1. Forward project to image pixels
    pix_2d, depths_z, valid = cam.project_cam_to_image(pts_cam, check_bounds=True)
    assert np.all(valid)
    assert np.allclose(depths_z, pts_cam[:, 2])

    # 2. Backward unproject from pixels + depth
    pts_recovered = cam.unproject_image_to_cam(pix_2d, depths_z)

    # 3. Verify numerical reconstruction error is sub-millimetre (< 0.0001 m)
    err_3d = np.linalg.norm(pts_cam - pts_recovered, axis=1)
    assert np.max(err_3d) < 1e-4, f"Max unprojection error too large: {np.max(err_3d):.6e} m"


def test_camera_model_rescale():
    """Verify camera rescaling properly adapts focal length, principal point, and image boundaries."""
    cam = PhoneCameraModel(
        name="test_cam.jpg",
        model_name="PINHOLE",
        width=1600,
        height=1200,
        params=np.array([1600.0, 1600.0, 800.0, 600.0], dtype=np.float64),
        fx=1600.0,
        fy=1600.0,
        cx=800.0,
        cy=600.0,
    )

    scaled_cam = cam.rescale(800, 600)
    assert scaled_cam.width == 800
    assert scaled_cam.height == 600
    assert np.isclose(scaled_cam.fx, 800.0)
    assert np.isclose(scaled_cam.fy, 800.0)
    assert np.isclose(scaled_cam.cx, 399.75, atol=0.5)
    assert np.isclose(scaled_cam.cy, 299.75, atol=0.5)


def test_sfm_depth_aligner_held_out():
    """Verify SfM depth aligner correctly identifies scale factor and reduces residuals on held-out test set."""
    H, W = 100, 100
    true_scale = 1.85  # Ground truth scale factor
    np.random.seed(42)

    # Synthetic camera and depth map
    cam = PhoneCameraModel(
        name="view_1.jpg",
        model_name="PINHOLE",
        width=W,
        height=H,
        params=np.array([100.0, 100.0, 50.0, 50.0], dtype=np.float64),
        fx=100.0,
        fy=100.0,
        cx=50.0,
        cy=50.0,
    )

    # Raw mono depth: D_mono = 1.0 to 4.0 m
    d_mono = np.random.uniform(1.0, 4.0, size=(H, W)).astype(np.float32)

    # Generate 50 sparse anchor observations with true SfM depth = d_mono * true_scale + small noise
    tracks = []
    for _ in range(50):
        u = float(np.random.randint(10, W - 10))
        v = float(np.random.randint(10, H - 10))
        mono_val = float(d_mono[int(v), int(u)])
        sfm_z = mono_val * true_scale + np.random.normal(0, 0.01)
        # Unproject to 3D world
        xyz = [float((u - 50.0) * sfm_z / 100.0), float((v - 50.0) * sfm_z / 100.0), float(sfm_z)]
        tracks.append({"xy": (u, v), "xyz": xyz, "error": 0.25})

    aligner = SfMDepthAligner(mode="per_view", validation_split_ratio=0.20)
    report = aligner.align_depth_maps(
        depth_maps={"view_1.jpg": d_mono},
        camera_models={"view_1.jpg": cam},
        sparse_tracks={"view_1.jpg": tracks},
    )

    assert "view_1.jpg" in report.view_stats
    v_stat = report.view_stats["view_1.jpg"]
    assert np.isclose(v_stat.median_scale, true_scale, atol=0.05)
    assert v_stat.residual_reduction_pct > 80.0
    assert report.aligned_depth_maps["view_1.jpg"].shape == (H, W)


def test_mvs_geometric_consistency_rejection():
    """Verify that MVS consistency filter retains consistent points and rejects depth discrepancies."""
    W, H = 200, 200
    cam1 = PhoneCameraModel(
        name="cam1.jpg",
        model_name="PINHOLE",
        width=W,
        height=H,
        params=np.array([200.0, 200.0, 100.0, 100.0], dtype=np.float64),
        fx=200.0,
        fy=200.0,
        cx=100.0,
        cy=100.0,
        T_world_cam=np.eye(4),
    )
    # Cam 2 translated by 0.2m along X
    T2 = np.eye(4)
    T2[0, 3] = 0.20
    cam2 = PhoneCameraModel(
        name="cam2.jpg",
        model_name="PINHOLE",
        width=W,
        height=H,
        params=np.array([200.0, 200.0, 100.0, 100.0], dtype=np.float64),
        fx=200.0,
        fy=200.0,
        cx=100.0,
        cy=100.0,
        T_world_cam=T2,
    )

    # Consistent planar target at Z = 2.0m
    depth1 = np.ones((H, W), dtype=np.float32) * 2.0
    depth2 = np.ones((H, W), dtype=np.float32) * 2.0

    # Add an artificial severe depth inconsistency in top-left region of depth1
    depth1[:40, :40] = 5.0  # Massive depth jump

    filter_mvs = MVSGeometricConsistencyFilter(rel_depth_threshold=0.05, min_consistent_views=1)
    res = filter_mvs.evaluate_consistency(
        camera_models={"cam1.jpg": cam1, "cam2.jpg": cam2},
        depth_maps={"cam1.jpg": depth1, "cam2.jpg": depth2},
        stride=10,
    )

    assert res.retained_count > 0
    # Inconsistent points must be rejected
    assert res.rejected_count > 0


def test_multi_radius_bpa_meshing():
    """Verify Multi-Radius BPA reconstructs mesh from clean oriented point cloud."""
    # Create synthetic cylindrical / curved patch
    theta = np.linspace(-np.pi / 4, np.pi / 4, 30)
    z = np.linspace(-0.5, 0.5, 30)
    theta_g, z_g = np.meshgrid(theta, z)
    r = 1.0
    x = r * np.sin(theta_g).ravel()
    y = z_g.ravel()
    z_pts = r * np.cos(theta_g).ravel()

    pts = np.column_stack([x, y, z_pts]).astype(np.float32)
    normals = np.column_stack([np.sin(theta_g).ravel(), np.zeros_like(x), np.cos(theta_g).ravel()]).astype(np.float32)
    cols = np.ones_like(pts) * 0.8

    builder = MeshBuilder(method="bpa", bpa_radii_multipliers=(1.2, 2.5, 5.0))
    v, f, c, n = builder.build_mesh(pts, cols, normals)

    assert len(v) > 0
    assert len(f) > 0
    assert f.shape[1] == 3


def test_visibility_aware_texture_engine():
    """Verify VisibilityAwareTextureEngine UV parameterization and atlas generation."""
    # Single triangle facing camera 1
    v = np.array([
        [-0.2, -0.2, 2.0],
        [0.2, -0.2, 2.0],
        [0.0, 0.2, 2.0],
    ], dtype=np.float32)
    f = np.array([[0, 2, 1]], dtype=np.uint32)

    cam1 = PhoneCameraModel(
        name="cam1.jpg",
        model_name="PINHOLE",
        width=200,
        height=200,
        params=np.array([200.0, 200.0, 100.0, 100.0], dtype=np.float64),
        fx=200.0,
        fy=200.0,
        cx=100.0,
        cy=100.0,
        T_world_cam=np.eye(4),
    )

    # Red image for cam1
    img1 = np.full((200, 200, 3), 200, dtype=np.uint8)
    img1[:, :, 0] = 255  # Red

    with tempfile.TemporaryDirectory() as tmp_dir:
        engine = VisibilityAwareTextureEngine(atlas_size=512)
        new_v, new_f, uvs, atlas, report = engine.texture_mesh(
            vertices=v,
            faces=f,
            camera_models={"cam1.jpg": cam1},
            images={"cam1.jpg": img1},
            out_dir=Path(tmp_dir),
            base_name="test_textured",
        )

        assert report.textured_faces == 1
        assert report.coverage_percentage == 100.0
        assert (Path(tmp_dir) / "test_textured.obj").exists()
        assert (Path(tmp_dir) / "test_textured.mtl").exists()
        assert (Path(tmp_dir) / "test_textured.png").exists()
        assert len(uvs) == 3
        assert np.all(uvs >= 0.0) and np.all(uvs <= 1.0)
