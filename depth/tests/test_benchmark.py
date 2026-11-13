"""
depth/tests/test_benchmark.py
==============================
Unit tests for benchmark metrics: MAE, RMSE, AbsRel, Delta-1.25.
"""

import numpy as np
import pytest

from depth.benchmark import compute_depth_metrics, format_metrics_table


def test_compute_depth_metrics_perfect():
    """Identical prediction and GT should result in zero error and 100% threshold accuracy."""
    gt = np.linspace(2.0, 40.0, 1000, dtype=np.float32).reshape(20, 50)
    pred = gt.copy()

    m = compute_depth_metrics(pred, gt)
    assert m["valid_pixels"] == 1000
    assert m["mae"] == pytest.approx(0.0, abs=1e-6)
    assert m["rmse"] == pytest.approx(0.0, abs=1e-6)
    assert m["abs_rel"] == pytest.approx(0.0, abs=1e-6)
    assert m["delta_1"] == pytest.approx(1.0, abs=1e-6)
    assert m["delta_2"] == pytest.approx(1.0, abs=1e-6)


def test_compute_depth_metrics_known_discrepancy():
    """Test metrics with a constant 2.0 m offset on a constant 10.0 m GT depth."""
    gt = np.full((10, 10), 10.0, dtype=np.float32)
    pred = np.full((10, 10), 12.0, dtype=np.float32)  # +2.0m offset

    m = compute_depth_metrics(pred, gt)
    assert m["valid_pixels"] == 100
    assert m["mae"] == pytest.approx(2.0, abs=1e-5)
    assert m["rmse"] == pytest.approx(2.0, abs=1e-5)
    assert m["abs_rel"] == pytest.approx(0.20, abs=1e-5)  # 2.0 / 10.0
    # ratio = 12 / 10 = 1.2 < 1.25 -> 100% delta_1
    assert m["delta_1"] == pytest.approx(1.0, abs=1e-5)


def test_compute_depth_metrics_mask_filtering():
    """Ensure invalid pixels (negative, >50m, NaN) are correctly masked out."""
    gt = np.array([0.1, 5.0, 10.0, 60.0, 80.0], dtype=np.float32)
    pred = np.array([0.1, 5.0, 10.0, 60.0, 80.0], dtype=np.float32)

    # Only 5.0 and 10.0 fall within [0.5, 50.0]
    m = compute_depth_metrics(pred, gt)
    assert m["valid_pixels"] == 2
    assert m["mae"] == pytest.approx(0.0, abs=1e-6)


def test_format_metrics_table():
    """Verify table formatting generates a non-empty string with camera labels."""
    dummy_m = {
        "front": {
            "mae": 1.5, "rmse": 2.0, "abs_rel": 0.15, "delta_1": 0.85, "delta_2": 0.95, "valid_pixels": 50000
        }
    }
    table = format_metrics_table(dummy_m)
    assert "front" in table
    assert "MAE" in table
    assert "1.500" in table

