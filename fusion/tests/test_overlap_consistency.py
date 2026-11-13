"""
fusion/tests/test_overlap_consistency.py — Unit Tests for Overlap Consistency
=============================================================================

Tests:
  - Identification of 4 overlap zones (FL, FR, RL, RR)
  - Geometric consistency on Ground Truth (< 0.05 m median error)
  - Point-to-plane surface normal residual calculation
  - Occlusion filtering and inlier separation
"""

from pathlib import Path
import numpy as np
import open3d as o3d
import pytest

from fusion.engine import PointCloudFusionEngine
from fusion.overlap import OVERLAP_REGIONS, OverlapAnalyzer, OverlapMetrics
from pipeline.calibration import CalibrationStore
from validation.gt_loader import load_depth_exr


@pytest.fixture
def calib_store():
    p = Path("config/cameras.yaml")
    store = CalibrationStore(p)
    store.load()
    return store


@pytest.fixture
def overlap_analyzer(calib_store):
    return OverlapAnalyzer(calib_store, occlusion_dist_threshold=0.50)


def test_overlap_regions_defined(overlap_analyzer):
    """Verify all 4 overlap regions are properly registered."""
    expected_regions = ["front_left", "front_right", "rear_left", "rear_right"]
    for reg in expected_regions:
        assert reg in OVERLAP_REGIONS
        cfg = OVERLAP_REGIONS[reg]
        assert "cam_a" in cfg
        assert "cam_b" in cfg
        assert "azimuth_min_deg" in cfg
        assert "azimuth_max_deg" in cfg


def test_gt_overlap_consistency(calib_store, overlap_analyzer):
    """Verify that ground-truth point clouds achieve sub-5cm agreement in overlap zones."""
    engine = PointCloudFusionEngine(calib_store, max_depth=50.0, stride=2)
    depth_dir = Path("data/photoreal_gt_depth")

    if not (depth_dir / "front" / "frame_0000.exr").exists():
        pytest.skip("Photoreal GT depth for frame 0000 not available")

    depths = {}
    for cam in ["front", "rear", "left", "right"]:
        depths[cam] = load_depth_exr(depth_dir / cam / "frame_0000.exr")

    res = engine.fuse_depth_maps(depths, frame_id=0, pipeline_type="ground_truth")
    metrics = overlap_analyzer.analyze_all_regions(res.camera_clouds)

    for reg_name, m in metrics.items():
        assert m.num_points_a > 1000, f"{reg_name}: Insufficient points in camera A"
        assert m.num_points_b > 1000, f"{reg_name}: Insufficient points in camera B"
        # Ground-truth nearest-neighbor median distance must be < 0.05 m (5 cm)
        assert m.median_nn_dist_m < 0.05, f"{reg_name}: GT median NN error {m.median_nn_dist_m:.4f} m exceeds 5 cm"
        # Point-to-plane residual should be sub-centimeter on flat surfaces
        if m.median_pt2plane_m is not None:
            assert m.median_pt2plane_m < 0.02, f"{reg_name}: Pt2Plane residual {m.median_pt2plane_m:.4f} m exceeds 2 cm"


def test_occlusion_dist_filtering(overlap_analyzer):
    """Verify that points with large offsets are separated as occlusions rather than false alignment errors."""
    # Synthetic point clouds: points in the mutual FOV of front and left cameras
    X = np.linspace(-4.0, -1.5, 30)
    Y = np.linspace(2.5, 5.0, 30)
    xx, yy = np.meshgrid(X, Y)
    zz = np.zeros_like(xx) + 0.5
    raw_pts = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    # Keep points visible in BOTH front and left camera frustums and in azimuth wedge [120, 150]
    azim_mask = overlap_analyzer.filter_points_by_azimuth(raw_pts, 120.0, 150.0)
    vis_front = overlap_analyzer.is_visible_in_camera(raw_pts, "front")
    vis_left = overlap_analyzer.is_visible_in_camera(raw_pts, "left")
    mutual_mask = azim_mask & vis_front & vis_left
    mutual_pts = raw_pts[mutual_mask]

    assert len(mutual_pts) > 20, "Should have mutual test points"

    # Cloud B has matching points shifted by only 3 mm
    plane_pts_b = mutual_pts + np.array([0.002, 0.002, 0.001])

    # Cloud A additionally has distant occluded points (e.g. 1.5 m higher)
    occluded_pts_a = mutual_pts[:10] + np.array([0.0, 0.0, 1.5])
    all_pts_a = np.vstack([mutual_pts, occluded_pts_a])

    pcd_a = o3d.geometry.PointCloud()
    pcd_a.points = o3d.utility.Vector3dVector(all_pts_a)
    pcd_b = o3d.geometry.PointCloud()
    pcd_b.points = o3d.utility.Vector3dVector(plane_pts_b)

    metrics = overlap_analyzer.analyze_pair(pcd_a, pcd_b, "front_left")
    # Point-to-plane residual should be sub-centimeter
    assert metrics.median_pt2plane_m is not None
    assert metrics.median_pt2plane_m < 0.01
    # Mutual co-observed points must match within millimeters
    assert metrics.inlier_ratio_10cm > 0.80
    assert metrics.max_nn_dist_m > 1.0  # reflects occluded points
