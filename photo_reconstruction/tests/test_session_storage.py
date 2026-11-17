"""Focused tests for safe cleanup of bulky photogrammetry intermediates."""

from pathlib import Path

import pytest

from photo_reconstruction.session_manager import SessionManager


def test_cleanup_removes_only_allowlisted_intermediates_and_preserves_outputs(tmp_path: Path):
    manager = SessionManager(tmp_path)
    session = manager.create_session("session_001")
    older_session = manager.create_session("session_002")

    for folder in (
        "openmvs",
        "source_uploads",
        "original_images",
        "exif_metadata",
        "camera_poses",
        "sparse_cloud",
        "dense_cloud",
    ):
        path = session / folder
        path.mkdir(exist_ok=True)
        (path / "large-intermediate.bin").write_bytes(b"x" * 1024)

    (session / "mesh" / "scene_mesh.obj").write_text("final OBJ")
    (session / "mesh" / "scene_mesh.mtl").write_text("final MTL")
    (session / "mesh" / "scene_textured.glb").write_bytes(b"final GLB")
    (session / "mesh" / "scene_textured_material_00_map_Kd.jpg").write_bytes(b"atlas")
    (session / "report.json").write_text("{}")
    (session / "openmvs_pipeline.log").write_text("stage timings")
    (older_session / "keep.txt").write_text("older session must not be touched")

    result = manager.cleanup_session_intermediates("session_001")

    assert result["enabled"] is True
    assert result["megabytes_removed"] > 0
    assert result["final_mesh_and_reports_retained"] is True
    assert not (session / "openmvs").exists()
    assert not (session / "original_images").exists()
    assert not (session / "source_uploads").exists()
    assert (session / "mesh" / "scene_mesh.obj").is_file()
    assert (session / "mesh" / "scene_textured.glb").is_file()
    assert (session / "mesh" / "scene_textured_material_00_map_Kd.jpg").is_file()
    assert (session / "report.json").is_file()
    assert (session / "openmvs_pipeline.log").is_file()
    assert (older_session / "keep.txt").read_text() == "older session must not be touched"
    assert manager.list_sessions() == ["session_001", "session_002"]


def test_cleanup_rejects_non_session_paths(tmp_path: Path):
    manager = SessionManager(tmp_path)
    outside = tmp_path / "important"
    outside.mkdir()
    (outside / "keep.txt").write_text("safe")

    with pytest.raises(ValueError):
        manager.cleanup_session_intermediates("../important")

    assert (outside / "keep.txt").is_file()
