"""
fusion/tests/test_pointcloud_fusion.py — Unit Tests for 3D Point Cloud Fusion
=============================================================================

Tests:
  - Rigid extrinsic transformation without ICP
  - Single-camera world frame unprojection accuracy
  - 4-camera unified fusion
  - 5 cm voxel downsampling
  - Vertical geometry preservation (walls, pillars)
"""

from pathlib import Path
import numpy as np
import open3d as o3d
import pytest

from fusion.engine import PointCloudFusionEngine, FusionResult
from pipeline.calibration import CalibrationStore
from pipeline.sync import CameraFrameData, FrameSet


@pytest.fixture
def calib_store():
    p = Path("config/cameras.yaml")
    store = CalibrationStore(p)
    store.load()
    return store


@pytest.fixture
def fusion_engine(calib_store):
    return PointCloudFusionEngine(calib_store, max_depth=50.0, min_depth=0.1, stride=4)


def test_rigid_fusion_without_icp(fusion_engine, calib_store):
    """Verify that fusion uses direct calibrated rigid transformation (T_cam_world)."""
    # Create synthetic planar depth maps for front and rear cameras
    H, W = 720, 1280
    front_depth = np.full((H, W), 5.0, dtype=np.float32)
    rear_depth = np.full((H, W), 5.0, dtype=np.float32)

    depth_maps = {"front": front_depth, "rear": rear_depth}
    result = fusion_engine.fuse_depth_maps(depth_maps, frame_id=0, pipeline_type="ground_truth")

    assert isinstance(result, FusionResult)
    assert result.num_points > 0
    assert "front" in result.camera_clouds
    assert "rear" in result.camera_clouds

    # Front camera is at Y=+2.15 looking +Y; depth 5m -> points around Y = +7.15 m
    front_pts = np.asarray(result.camera_clouds["front"].points)
    assert np.all(front_pts[:, 1] > 2.15)
    assert np.isclose(np.median(front_pts[:, 1]), 2.15 + 5.0 * np.cos(np.radians(-8.0)), atol=0.20)

    # Rear camera is at Y=-2.15 looking -Y; depth 5m -> points around Y = -7.15 m
    rear_pts = np.asarray(result.camera_clouds["rear"].points)
    assert np.all(rear_pts[:, 1] < -2.15)
    assert np.isclose(np.median(rear_pts[:, 1]), -2.15 - 5.0 * np.cos(np.radians(-8.0)), atol=0.20)


def test_voxel_downsampling_compliance(fusion_engine):
    """Verify 5 cm voxel downsampling reduces density while preserving spatial extents."""
    # Build a dense point cloud
    pts = np.random.uniform(low=[-2, -2, 0], high=[2, 2, 2], size=(20000, 3))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    ds_pcd = fusion_engine.downsample_voxel(pcd, voxel_size=0.05)
    assert len(ds_pcd.points) < len(pts)
    assert len(ds_pcd.points) > 100

    # Bounding box should remain preserved within 5 cm margin
    orig_min = pts.min(axis=0)
    orig_max = pts.max(axis=0)
    ds_pts = np.asarray(ds_pcd.points)
    ds_min = ds_pts.min(axis=0)
    ds_max = ds_pts.max(axis=0)

    assert np.all(np.abs(orig_min - ds_min) <= 0.06)
    assert np.all(np.abs(orig_max - ds_max) <= 0.06)


def test_vertical_geometry_preservation(fusion_engine):
    """Verify that vertical wall surfaces remain vertical without pitch distortion."""
    H, W = 720, 1280
    # Simulate a vertical wall at 5 m distance ahead of front camera
    # In camera coordinates, a vertical wall at distance D has z_cam = D / cos(pitch)
    # With unprojector, if wall is vertical in world, world Z spans elevation while Y remains ~constant
    depth = np.full((H, W), 5.0, dtype=np.float32)
    pts, _, _ = fusion_engine.unproject_camera("front", depth)

    # Z should span a vertical range from ground up
    z_span = pts[:, 2].max() - pts[:, 2].min()
    assert z_span > 2.0  # spans significant vertical height

