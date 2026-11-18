import numpy as np

from photo_reconstruction.camera_model import PhoneCameraModel
from photo_reconstruction.diagnose_two_view_consensus import (
    _bilinear_sample,
    _evaluate_direction,
)


def _camera(name: str) -> PhoneCameraModel:
    return PhoneCameraModel(
        name=name,
        model_name="PINHOLE",
        width=12,
        height=12,
        params=np.array([10.0, 10.0, 5.5, 5.5]),
        fx=10.0,
        fy=10.0,
        cx=5.5,
        cy=5.5,
        T_world_cam=np.eye(4),
        T_cam_world=np.eye(4),
    )


def test_bilinear_sample_interpolates_linear_depth_plane():
    yy, xx = np.mgrid[0:8, 0:8]
    depth = (2.0 + 0.25 * xx + 0.5 * yy).astype(np.float32)

    sampled = _bilinear_sample(
        depth,
        np.array([1.5, 3.25]),
        np.array([2.5, 4.5]),
    )

    expected = np.array([
        2.0 + 0.25 * 1.5 + 0.5 * 2.5,
        2.0 + 0.25 * 3.25 + 0.5 * 4.5,
    ])
    np.testing.assert_allclose(sampled, expected, atol=1e-6)


def test_identical_two_view_depths_pass_pairwise_geometry():
    source = np.full((12, 12), 3.0, dtype=np.float32)
    target = np.full((12, 12), 3.0, dtype=np.float32)

    result = _evaluate_direction(
        "5.jpg", "8.jpg",
        _camera("5.jpg"), _camera("8.jpg"),
        source, target, stride=3,
    )

    assert result["candidate_samples"] == 16
    assert result["projected_samples"] == 16
    assert result["target_depth_valid_samples"] == 16
    assert result["verified_samples"] == 16
    assert result["verified_pct_of_candidates"] == 100.0


def test_inconsistent_target_depth_is_rejected():
    source = np.full((12, 12), 3.0, dtype=np.float32)
    target = np.full((12, 12), 4.0, dtype=np.float32)

    result = _evaluate_direction(
        "5.jpg", "8.jpg",
        _camera("5.jpg"), _camera("8.jpg"),
        source, target, stride=3,
    )

    assert result["candidate_samples"] == 16
    assert result["target_depth_valid_samples"] == 16
    assert result["depth_agreement_samples"] == 0
    assert result["verified_samples"] == 0
    assert result["inconsistent_samples"] == 16
