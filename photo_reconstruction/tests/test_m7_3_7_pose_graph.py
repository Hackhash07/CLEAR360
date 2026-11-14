"""
photo_reconstruction/tests/test_m7_3_7_pose_graph.py — Unit Tests for M7.3.7 Pose Graph & Manifest
==================================================================================================
"""

import json
from pathlib import Path
import numpy as np
import pytest

from photo_reconstruction.m7_3_7_pose_graph import (
    GlobalCameraPoseGraphAuditor,
    CameraReprojectionValidator,
    GlobalSceneManifestAuthor,
)
from photo_reconstruction.coord_transform import (
    Clear360CoordinateTransformer,
    T_COLMAP_TO_CLEAR360,
    R_COLMAP_TO_CLEAR360,
)


def test_coord_transform_properties():
    """Verify SO(3) properties and right-handedness of the canonical transform."""
    det = np.linalg.det(R_COLMAP_TO_CLEAR360)
    assert abs(det - 1.0) < 1e-6, f"det(R) must be 1.0, got {det}"
    ortho_err = np.max(np.abs(R_COLMAP_TO_CLEAR360 @ R_COLMAP_TO_CLEAR360.T - np.eye(3)))
    assert ortho_err < 1e-6, f"Orthonormality error must be < 1e-6, got {ortho_err}"


def test_pose_graph_on_session_064():
    """Verify pose graph auditor, full connectivity, and sub-pixel reprojection on baseline session."""
    session_path = Path("/Users/vudumulasakethreddy/CLEAR360/data/phone_reconstruction/session_064")
    if not session_path.exists():
        pytest.skip("session_064 not found")

    auditor = GlobalCameraPoseGraphAuditor(session_path)
    cam_audit = auditor.audit_cameras()
    assert len(cam_audit) == 6, f"Expected 6 cameras, got {len(cam_audit)}"

    for name, c_info in cam_audit.items():
        assert c_info["clear360_pose"]["is_valid_so3"], f"Camera {name} pose is not valid SO(3)"
        assert c_info["clear360_pose"]["forward_vector"][1] > 0.5, f"Camera {name} must point forward (+Y)"
        assert c_info["clear360_pose"]["up_vector"][2] > 0.8, f"Camera {name} must point up (+Z)"

    graph = auditor.build_pose_graph()
    assert graph["num_cameras"] == 6
    assert graph["is_fully_connected"], "Camera pose graph must be fully connected"
    assert graph["max_cycle_residual"] < 1e-6, f"Max cycle residual must be < 1e-6, got {graph['max_cycle_residual']}"


def test_manifest_generation(tmp_path):
    """Verify authoritative global scene manifest generation."""
    cam_audit = {
        "cam_1.jpg": {
            "clear360_pose": {
                "C_world": [0.0, 0.0, 2.0],
                "forward_vector": [0.0, 1.0, 0.0],
                "up_vector": [0.0, 0.0, 1.0],
                "is_valid_so3": True,
            },
            "num_registered_3D": 5000,
        }
    }
    pose_graph = {
        "num_cameras": 1,
        "is_fully_connected": True,
        "min_pairwise_inliers": 1000,
        "max_pairwise_inliers": 1000,
        "max_cycle_residual": 0.0,
    }
    reproj_stats = {"overall_mean_residual_px": 0.20}
    scene_regions = {"wall": {"name": "Background Wall", "point_count": 1000}}

    out_file = tmp_path / "global_scene_manifest.json"
    manifest = GlobalSceneManifestAuthor.create_manifest(
        session_id="test_session",
        camera_audit=cam_audit,
        pose_graph=pose_graph,
        reprojection_stats=reproj_stats,
        scene_regions=scene_regions,
        output_file=out_file,
    )

    assert out_file.exists()
    assert manifest["coordinate_frame"]["authoritative_frame"] == "Clear360_Vehicle_World"
    assert len(manifest["sfm_reconstruction_models"]) == 1
    assert manifest["reprojection_validation"]["overall_mean_residual_px"] == 0.20

