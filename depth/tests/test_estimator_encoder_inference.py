from pathlib import Path

import pytest

from depth.estimator import DepthEstimator


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("depth_anything_v2_metric_hypersim_vits.pth", "vits"),
        ("depth_anything_v2_metric_hypersim_vitb.pth", "vitb"),
        ("depth_anything_v2_metric_hypersim_vitl.pth", "vitl"),
        ("custom_vitb_checkpoint.pth", "vitb"),
        ("unknown_checkpoint.pth", "vits"),
    ],
)
def test_encoder_is_inferred_from_checkpoint_filename(filename, expected):
    assert DepthEstimator._infer_encoder_from_checkpoint(Path(filename)) == expected
