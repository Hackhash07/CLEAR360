from pathlib import Path

from scripts.rebuild_openmvs_dense import rebuild_dense_session


def test_dense_rebuild_refuses_when_required_colmap_workspace_is_missing(tmp_path: Path):
    session = tmp_path / "session_022"
    (session / "openmvs").mkdir(parents=True)
    (session / "original_images").mkdir()
    (session / "mesh").mkdir()
    (session / "original_images" / "a.jpg").write_bytes(b"x")
    with __import__("pytest").raises(FileNotFoundError, match="undistorted COLMAP workspace"):
        rebuild_dense_session(session)


def test_dense_rebuild_refuses_when_backup_name_already_exists(tmp_path: Path):
    import pytest

    session = tmp_path / "session_022"
    work = session / "openmvs"
    (work / "dense" / "images").mkdir(parents=True)
    (work / "sparse_text").mkdir()
    (session / "original_images").mkdir()
    (session / "mesh").mkdir()
    (session / "mesh_before_dense_quality_rebuild").mkdir()
    for index in range(6):
        (session / "original_images" / f"fountain_{index:04d}.jpg").write_bytes(b"x")
    for name in ("cameras.txt", "images.txt"):
        (work / "sparse_text" / name).write_text("# empty\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Backup already exists"):
        rebuild_dense_session(session)



def test_mesh_rebuild_preserves_existing_mesh_backup_guard(tmp_path):
    # Existing backup must make the rebuild stop before any OpenMVS binary runs.
    from scripts.rebuild_openmvs_mesh import rebuild_session
    import pytest

    session = tmp_path / "session_022"
    work = session / "openmvs"
    (work / "sparse_text").mkdir(parents=True)
    (work / "scene_fuse_dense.mvs").write_bytes(b"scene")
    (work / "sparse_text" / "images.txt").write_text(
        "1 1 0 0 0 0 0 0 1 fountain.jpg\n", encoding="utf-8"
    )
    (session / "original_images").mkdir()
    (session / "mesh").mkdir()
    (session / "mesh_before_quality_rebuild_v2").mkdir()
    with pytest.raises(FileExistsError, match="Backup already exists"):
        rebuild_session(session)
