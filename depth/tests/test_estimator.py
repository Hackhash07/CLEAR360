"""
depth/tests/test_estimator.py
==============================
Unit tests for DepthEstimator using Depth Anything V2 Metric ViT-Small.
"""

from pathlib import Path
import numpy as np
import pytest

from depth.estimator import DepthEstimator

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@pytest.fixture(scope="module")
def estimator():
    # Use CPU for deterministic, portable unit testing
    return DepthEstimator(device="cpu")


def test_estimator_initialization(estimator):
    """Verify DepthEstimator initializes correctly with ViT-Small and max_depth=80m."""
    assert estimator.encoder == "vits"
    assert estimator.max_depth == 80.0
    assert estimator.input_size == 518
    assert estimator.device.type == "cpu"
    assert estimator.model is not None


def test_estimator_output_shape_and_range(estimator):
    """Verify inference preserves 1280x720 dimensions and outputs float32 metric depth."""
    test_img = np.full((720, 1280, 3), 128, dtype=np.uint8)
    depth_pred, dt = estimator.estimate_depth(test_img, is_rgb=True)

    assert depth_pred.shape == (720, 1280)
    assert depth_pred.dtype == np.float32
    assert not np.any(np.isnan(depth_pred))
    assert not np.any(np.isinf(depth_pred))
    assert np.all(depth_pred >= 0.0)
    assert np.all(depth_pred <= 80.0)
    assert dt > 0.0
    assert estimator.last_inference_time_s == dt

