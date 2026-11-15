"""Tests for held-out affine monocular-depth calibration and view rejection."""

import numpy as np

from photo_reconstruction.camera_model import PhoneCameraModel
from photo_reconstruction.depth_aligner import SfMDepthAligner


def _calibration_fixture(depth_to_sfm):
    height, width = 128, 128
    camera = PhoneCameraModel(
        name="view.jpg",
        model_name="PINHOLE",
        width=width,
        height=height,
        params=np.array([100.0, 100.0, 64.0, 64.0]),
        fx=100.0,
        fy=100.0,
        cx=64.0,
        cy=64.0,
        T_world_cam=np.eye(4),
        T_cam_world=np.eye(4),
    )

    depth_map = np.ones((height, width), dtype=np.float32)
    tracks = []
    samples = np.linspace(0.6, 5.5, 180, dtype=np.float64)
    positions = [
        (8 + (i % 15) * 7, 8 + (i // 15) * 9)
        for i in range(len(samples))
    ]

    for mono_depth, (u, v) in zip(samples, positions):
        sfm_depth = float(depth_to_sfm(mono_depth))
        depth_map[v, u] = float(mono_depth)
        tracks.append({
            "xy": (float(u), float(v)),
            "xyz": (0.0, 0.0, sfm_depth),
            "error": 0.5,
        })

    return camera, depth_map, tracks


def test_affine_depth_alignment_uses_held_out_anchors_and_recovers_offset():
    camera, depth_map, tracks = _calibration_fixture(
        lambda d: 1.4 * d + 0.35
    )

    aligner = SfMDepthAligner(mode="per_view")
    report = aligner.align_depth_maps(
        depth_maps={"view.jpg": depth_map},
        camera_models={"view.jpg": camera},
        sparse_tracks={"view.jpg": tracks},
    )

    stats = report.view_stats["view.jpg"]
    assert stats.success, stats.status_message
    assert abs(stats.median_scale - 1.4) < 0.03
    assert abs(stats.depth_offset - 0.35) < 0.05
    assert stats.validation_median_relative_error < 0.01
    assert stats.validation_p90_relative_error < 0.02
    assert "view.jpg" in report.aligned_depth_maps

    aligned = report.aligned_depth_maps["view.jpg"]
    u0, v0 = tracks[0]["xy"]
    assert abs(float(aligned[int(v0), int(u0)]) - float(tracks[0]["xyz"][2])) < 0.05


def test_depth_alignment_rejects_a_view_with_non_affine_held_out_residuals():
    camera, depth_map, tracks = _calibration_fixture(
        lambda d: 0.4 + d * d
    )

    aligner = SfMDepthAligner(mode="per_view")
    report = aligner.align_depth_maps(
        depth_maps={"view.jpg": depth_map},
        camera_models={"view.jpg": camera},
        sparse_tracks={"view.jpg": tracks},
    )

    stats = report.view_stats["view.jpg"]
    assert not stats.success
    assert stats.validation_p90_relative_error > aligner.max_val_p90_rel
    assert "view.jpg" not in report.aligned_depth_maps
    assert "Rejected" in stats.status_message
