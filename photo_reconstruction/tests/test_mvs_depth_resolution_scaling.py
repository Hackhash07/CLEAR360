"""Tests for camera intrinsics being rescaled to cached depth-map resolution."""

import numpy as np

from photo_reconstruction.camera_model import PhoneCameraModel
from photo_reconstruction.m7_3_6_fusion import M736GeometricConsensusEngine


def test_mvs_camera_is_rescaled_to_depth_map_grid_without_mutating_source():
    camera = PhoneCameraModel(
        name="view.jpg",
        model_name="PINHOLE",
        width=64,
        height=48,
        params=np.array([50.0, 52.0, 31.5, 23.5]),
        fx=50.0,
        fy=52.0,
        cx=31.5,
        cy=23.5,
        T_world_cam=np.array([
            [1.0, 0.0, 0.0, 1.2],
            [0.0, 1.0, 0.0, -0.4],
            [0.0, 0.0, 1.0, 2.0],
            [0.0, 0.0, 0.0, 1.0],
        ]),
    )
    depth_map = np.ones((96, 128), dtype=np.float32)

    scaled = M736GeometricConsensusEngine._camera_models_at_depth_resolution(
        {"view.jpg": camera},
        {"view.jpg": depth_map},
    )["view.jpg"]

    assert (scaled.width, scaled.height) == (128, 96)
    assert np.isclose(scaled.fx, 100.0)
    assert np.isclose(scaled.fy, 104.0)
    assert np.isclose(scaled.cx, 63.5)
    assert np.isclose(scaled.cy, 47.5)
    assert np.allclose(scaled.camera_center, camera.camera_center)

    # The original COLMAP model remains in its original coordinate system.
    assert (camera.width, camera.height) == (64, 48)
    assert np.isclose(camera.fx, 50.0)
    assert np.isclose(camera.fy, 52.0)
