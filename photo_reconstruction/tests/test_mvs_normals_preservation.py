"""
Regression test: MVS consensus normals must survive dense-cloud post-processing.

For an enclosed indoor scene, orienting every normal toward one average camera
position is invalid. The MVS engine's source-view-aware normals must be retained.
"""

from pathlib import Path

import cv2
import numpy as np

from photo_reconstruction.dense_reconstruction import DenseReconstructor


def test_dense_reconstruction_preserves_mvs_consensus_normals(tmp_path, monkeypatch):
    image_dir = tmp_path / "images"
    image_dir.mkdir()

    image = np.full((16, 16, 3), 127, dtype=np.uint8)
    for name in ("1.jpg", "2.jpg"):
        assert cv2.imwrite(str(image_dir / name), image)

    # 1,000 regularly spaced points; each carries a normal facing -X.
    # The synthetic cameras sit on +X. A global-average-camera reorientation
    # would incorrectly flip these normals toward +X.
    axis = np.arange(10, dtype=np.float32) * 0.02
    gx, gy, gz = np.meshgrid(axis, axis, axis, indexing="ij")
    fused_points = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()]).astype(np.float32)
    fused_colors = np.full((len(fused_points), 3), 0.5, dtype=np.float32)
    fused_normals = np.tile(
        np.array([[-1.0, 0.0, 0.0]], dtype=np.float32),
        (len(fused_points), 1),
    )
    fused_weights = np.ones(len(fused_points), dtype=np.float32)

    class FakeConsensusEngine:
        def __init__(self, voxel_fusion_size):
            self.voxel_fusion_size = voxel_fusion_size
            self.uncertain_pts = np.zeros((0, 3), dtype=np.float32)
            self.uncertain_cols = np.zeros((0, 3), dtype=np.float32)

        def evaluate_multi_view_consensus(
            self, camera_models, depth_maps, images_rgb, stride
        ):
            return (
                fused_points,
                fused_colors,
                fused_normals,
                fused_weights,
                None,
            )

    monkeypatch.setattr(
        "photo_reconstruction.m7_3_6_fusion.M736GeometricConsensusEngine",
        FakeConsensusEngine,
    )

    pose_1 = np.eye(4, dtype=np.float64)
    pose_2 = np.eye(4, dtype=np.float64)
    pose_1[:3, 3] = [10.0, 0.0, 0.0]
    pose_2[:3, 3] = [10.0, 0.2, 0.0]

    depth_maps = {
        "1.jpg": np.full((16, 16), 2.0, dtype=np.float32),
        "2.jpg": np.full((16, 16), 2.0, dtype=np.float32),
    }

    reconstructor = DenseReconstructor(
        voxel_size=0.006,
        nb_neighbors=20,
        std_ratio=2.0,
    )
    points, colors, normals = reconstructor.reconstruct_dense(
        image_dir=image_dir,
        camera_poses={"1.jpg": pose_1, "2.jpg": pose_2},
        depth_maps=depth_maps,
        sparse_points=None,
        sparse_colors=None,
        use_mvs_verification=True,
        camera_models={"1.jpg": object(), "2.jpg": object()},
        session_dir=tmp_path / "session",
    )

    assert len(points) >= 500
    assert len(colors) == len(points)
    assert len(normals) == len(points)
    assert np.all(np.isfinite(normals))

    mean_direction = np.mean(normals, axis=0)
    mean_direction /= np.linalg.norm(mean_direction) + 1e-8
    assert float(np.dot(mean_direction, np.array([-1.0, 0.0, 0.0]))) > 0.99
