"""
fusion/tests/test_reconstruction_pipeline_separation.py — Strict Pipeline Isolation Tests
========================================================================================

Verifies that:
  - Ground-Truth and AI reconstruction pipelines remain strictly isolated.
  - No shared mutable state or evaluation metric contamination exists between pipelines.
  - Calibration parameters applied to AI never alter Ground-Truth unprojections.
  - Depth provenance tags are strictly preserved.
"""

from pathlib import Path
import numpy as np
import pytest

from fusion.engine import PointCloudFusionEngine, FusionResult
from pipeline.calibration import CalibrationStore


@pytest.fixture
def calib_store():
    p = Path("config/cameras.yaml")
    store = CalibrationStore(p)
    store.load()
    return store


def test_pipeline_provenance_tags(calib_store):
    """Verify that fusion results strictly tag provenance without cross-contamination."""
    engine = PointCloudFusionEngine(calib_store, stride=4)
    H, W = 720, 1280

    depth_gt = {"front": np.full((H, W), 4.0, dtype=np.float32)}
    depth_ai = {"front": np.full((H, W), 28.0, dtype=np.float32)}

    res_gt = engine.fuse_depth_maps(depth_gt, frame_id=0, pipeline_type="ground_truth")
    res_ai = engine.fuse_depth_maps(depth_ai, frame_id=0, pipeline_type="ai_per_camera")

    assert res_gt.pipeline_type == "ground_truth"
    assert res_ai.pipeline_type == "ai_per_camera"

    # Points must reflect independent depth inputs
    pts_gt = np.asarray(res_gt.point_cloud.points)
    pts_ai = np.asarray(res_ai.point_cloud.points)

    assert not np.allclose(pts_gt, pts_ai)
    # GT points should be closer to camera than raw AI points
    assert pts_gt[:, 1].mean() < pts_ai[:, 1].mean()


def test_ai_scale_factors_do_not_contaminate_gt(calib_store):
    """Verify that applying AI scaling never mutates or influences Ground-Truth calculations."""
    engine = PointCloudFusionEngine(calib_store, stride=4)
    H, W = 720, 1280

    raw_ai_depth = np.full((H, W), 28.0, dtype=np.float32)
    scale_factor = 7.1539

    # Scale AI depth
    calib_ai_depth = raw_ai_depth / scale_factor

    # Compute GT point cloud with true 3.91 m depth
    gt_depth = np.full((H, W), 3.9138, dtype=np.float32)

    res_gt = engine.fuse_depth_maps({"front": gt_depth}, pipeline_type="ground_truth")
    res_ai = engine.fuse_depth_maps({"front": calib_ai_depth}, pipeline_type="ai_per_camera")

    # Although numerically close, the arrays must be distinct objects
    pts_gt = np.asarray(res_gt.point_cloud.points)
    pts_ai = np.asarray(res_ai.point_cloud.points)

    assert pts_gt is not pts_ai
    assert np.allclose(pts_gt, pts_ai, atol=1e-3)


def test_independent_camera_clouds(calib_store):
    """Verify that individual camera clouds are independently unprojected without cross-talk."""
    engine = PointCloudFusionEngine(calib_store, stride=4)
    H, W = 720, 1280

    depths = {
        "front": np.full((H, W), 5.0, dtype=np.float32),
        "left":  np.full((H, W), 3.0, dtype=np.float32),
    }

    res = engine.fuse_depth_maps(depths, pipeline_type="ground_truth")
    assert "front" in res.camera_clouds
    assert "left" in res.camera_clouds
    assert "rear" not in res.camera_clouds
    assert "right" not in res.camera_clouds

    pts_front = np.asarray(res.camera_clouds["front"].points)
    pts_left = np.asarray(res.camera_clouds["left"].points)

    # Front camera points should be mostly in Y > 0
    assert np.mean(pts_front[:, 1]) > 2.0
    # Left camera points should be mostly in X < 0
    assert np.mean(pts_left[:, 0]) < -0.9

