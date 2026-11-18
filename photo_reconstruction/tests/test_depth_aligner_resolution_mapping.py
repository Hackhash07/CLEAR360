"""Regression test for matching COLMAP feature pixels to full-resolution depth maps."""

import numpy as np

from photo_reconstruction.camera_model import PhoneCameraModel
from photo_reconstruction.depth_aligner import SfMDepthAligner


def test_depth_alignment_maps_feature_coordinates_to_depth_map_resolution():
    # Simulate COLMAP running feature extraction at 64x64 while cached metric
    # depth remains at the original 128x128 photo resolution.
    camera = PhoneCameraModel(
        name="scaled_view.jpg",
        model_name="PINHOLE",
        width=64,
        height=64,
        params=np.array([80.0, 80.0, 31.5, 31.5]),
        fx=80.0,
        fy=80.0,
        cx=31.5,
        cy=31.5,
        T_world_cam=np.eye(4),
        T_cam_world=np.eye(4),
    )
    depth_map = np.full((128, 128), 8.5, dtype=np.float32)
    tracks = []

    mono_depths = np.linspace(0.7, 5.8, 130, dtype=np.float64)
    for i, mono_depth in enumerate(mono_depths):
        u = 6 + (i % 13) * 4
        v = 6 + (i // 13) * 5
        # Pixel-centre-preserving 2x mapping, identical to the aligner.
        u_depth = int(round((u + 0.5) * 2.0 - 0.5))
        v_depth = int(round((v + 0.5) * 2.0 - 0.5))
        depth_map[v_depth, u_depth] = float(mono_depth)
        tracks.append({
            "xy": (float(u), float(v)),
            "xyz": (0.0, 0.0, float(1.35 * mono_depth + 0.28)),
            "error": 0.4,
        })

    report = SfMDepthAligner(mode="per_view").align_depth_maps(
        depth_maps={"scaled_view.jpg": depth_map},
        camera_models={"scaled_view.jpg": camera},
        sparse_tracks={"scaled_view.jpg": tracks},
    )

    stats = report.view_stats["scaled_view.jpg"]
    assert stats.success, stats.status_message
    assert abs(stats.median_scale - 1.35) < 0.04
    assert abs(stats.depth_offset - 0.28) < 0.05
    assert stats.validation_p90_relative_error < 0.02
