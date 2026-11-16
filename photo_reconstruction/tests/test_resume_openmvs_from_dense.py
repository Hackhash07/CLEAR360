from pathlib import Path

import pytest

from scripts.resume_openmvs_from_dense import resume_from_dense


def _valid_session(tmp_path: Path) -> Path:
    session = tmp_path / "session_027"
    work = session / "openmvs"
    (work / "sparse_text").mkdir(parents=True)
    (work / "scene_fuse_dense.mvs").write_bytes(b"mvs")
    (work / "scene_fuse_dense.ply").write_bytes(b"ply")
    (work / "sparse_text" / "images.txt").write_text(
        "1 1 0 0 0 0 0 0 1 img1.jpg\n"
        "2 1 0 0 0 0 0 0 1 img2.jpg\n"
        "3 1 0 0 0 0 0 0 1 img3.jpg\n",
        encoding="utf-8",
    )
    (session / "original_images").mkdir()
    (session / "original_images" / "img1.jpg").write_bytes(b"x")
    (session / "mesh").mkdir()
    return session


def test_resume_requires_saved_dense_scene_pair(tmp_path: Path):
    session = tmp_path / "session_027"
    (session / "openmvs").mkdir(parents=True)
    (session / "openmvs" / "scene_fuse_dense.mvs").write_bytes(b"mvs")
    with pytest.raises(FileNotFoundError, match="complete saved dense pair"):
        resume_from_dense(session)


def test_resume_refuses_existing_backup_before_running_tools(tmp_path: Path):
    session = _valid_session(tmp_path)
    (session / "mesh_before_dense_resume").mkdir()
    with pytest.raises(FileExistsError, match="Backup already exists"):
        resume_from_dense(session)


def test_resume_refuses_nonempty_stage_before_running_tools(tmp_path: Path):
    session = _valid_session(tmp_path)
    stage = session / "openmvs" / "mesh_from_existing_dense"
    stage.mkdir()
    (stage / "partial.obj").write_text("partial", encoding="utf-8")
    with pytest.raises(FileExistsError, match="staging directory"):
        resume_from_dense(session)
