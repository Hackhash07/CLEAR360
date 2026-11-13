"""
Unit tests for CLEAR360 validation utilities and depth accuracy metrics.
"""

from __future__ import annotations

import numpy as np
import pytest

from validation.gt_loader import depth_z_to_euclidean, verify_depth_vs_position
from validation.metrics import (
    abs_rel,
    compute_all_metrics,
    per_camera_report,
    rmse,
    rmse_log,
    threshold_accuracy,
)


def test_depth_z_to_euclidean_center() -> None:
    """At the principal point (cx, cy), Euclidean distance equals camera Z-depth."""
    H, W = 720, 1280
    cx, cy = 640.0, 360.0
    fx, fy = 369.4, 369.4

    depth_z = np.ones((H, W), dtype=np.float32) * 5.0
    d_e = depth_z_to_euclidean(depth_z, fx, fy, cx, cy)

    # At center pixel (360, 640), ratio should be exactly 1.0 -> d_e == 5.0
    assert np.isclose(d_e[360, 640], 5.0, atol=1e-4)

    # At corner pixels, Euclidean distance must be strictly greater than Z-depth
    assert d_e[0, 0] > 5.0
    assert d_e[719, 1279] > 5.0


def test_verify_depth_vs_position_synthetic_consistency() -> None:
    """A mathematically consistent depth map and position map should produce near-zero error."""
    H, W = 720, 1280
    fx, fy = 369.4, 369.4
    cx, cy = 640.0, 360.0
    intrinsics = {"fx": fx, "fy": fy, "cx": cx, "cy": cy}

    # Identity camera transform: camera at origin pointing along +Z, world aligned
    T_cam_world = np.eye(4, dtype=np.float64)

    # Constant depth plane at Z = 4.0 metres
    depth = np.full((H, W), 4.0, dtype=np.float32)

    # Build consistent world position map
    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)

    x_w = (uu - cx) * depth / fx
    y_w = (vv - cy) * depth / fy
    z_w = depth

    pos_map = np.stack([x_w, y_w, z_w], axis=-1).astype(np.float32)

    res = verify_depth_vs_position(
        depth=depth,
        position_map=pos_map,
        T_cam_world=T_cam_world,
        intrinsics_dict=intrinsics,
        sample_stride=32,
    )

    assert res["passed"] is True
    assert res["mean_error_m"] < 1e-4
    assert res["max_error_m"] < 1e-4


def test_metrics_perfect_prediction() -> None:
    """When prediction perfectly matches ground-truth, errors should be 0 and accuracies 1."""
    gt = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    pred = gt.copy()

    assert abs_rel(pred, gt) == pytest.approx(0.0)
    assert rmse(pred, gt) == pytest.approx(0.0)
    assert rmse_log(pred, gt) == pytest.approx(0.0)
    assert threshold_accuracy(pred, gt, threshold=1.25) == pytest.approx(1.0)

    m = compute_all_metrics(pred, gt)
    assert m["abs_rel"] == pytest.approx(0.0)
    assert m["rmse"] == pytest.approx(0.0)
    assert m["rmse_log"] == pytest.approx(0.0)
    assert m["delta_1"] == pytest.approx(1.0)
    assert m["delta_2"] == pytest.approx(1.0)
    assert m["delta_3"] == pytest.approx(1.0)


def test_metrics_known_discrepancy() -> None:
    """Test metric calculations against hand-computed values."""
    gt = np.array([2.0, 4.0], dtype=np.float32)
    pred = np.array([2.5, 3.0], dtype=np.float32)

    # AbsRel: (|2.5 - 2.0| / 2.0 + |3.0 - 4.0| / 4.0) / 2 = (0.25 + 0.25) / 2 = 0.25
    assert abs_rel(pred, gt) == pytest.approx(0.25)

    # RMSE: sqrt(((0.5)^2 + (-1.0)^2) / 2) = sqrt((0.25 + 1.0) / 2) = sqrt(0.625) ≈ 0.790569
    assert rmse(pred, gt) == pytest.approx(np.sqrt(0.625))

    # Delta ratios: max(2.5/2.0, 2.0/2.5) = 1.25 -> not strictly < 1.25
    # max(3.0/4.0, 4.0/3.0) = 1.333 -> not < 1.25
    # delta_1 should be 0.0
    assert threshold_accuracy(pred, gt, threshold=1.25) == pytest.approx(0.0)
    # Both ratios are < 1.25^2 = 1.5625 -> delta_2 should be 1.0
    assert threshold_accuracy(pred, gt, threshold=1.25**2) == pytest.approx(1.0)


def test_per_camera_report() -> None:
    """Verify per-camera report structuring and overall averaging."""
    gt_front = np.ones((10, 10), dtype=np.float32) * 2.0
    pred_front = np.ones((10, 10), dtype=np.float32) * 2.0

    gt_rear = np.ones((10, 10), dtype=np.float32) * 2.0
    pred_rear = np.ones((10, 10), dtype=np.float32) * 2.2  # 10% error

    gts = {"front": gt_front, "rear": gt_rear}
    preds = {"front": pred_front, "rear": pred_rear}

    rep = per_camera_report(preds, gts)
    assert "front" in rep
    assert "rear" in rep
    assert "overall" in rep

    assert rep["front"]["abs_rel"] == pytest.approx(0.0)
    assert rep["rear"]["abs_rel"] == pytest.approx(0.1, rel=1e-3)
    assert rep["overall"]["abs_rel"] == pytest.approx(0.05, rel=1e-3)
