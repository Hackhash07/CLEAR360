from pathlib import Path

import pytest

from scripts.resume_openmvs_from_dense import resume_from_dense


def _valid_session(tmp_path: Path) -> Path:
    session = tmp_path / "session_027"
    work = session / "openmvs"
    (work / "sparse_text").mkdir(parents=True)
    (work / "scene_fuse_dense.mvs").write_bytes(b"mvs")
    (work / "scene_fuse_dense.ply").write_bytes(b"ply")
    (work / "dense" / "images").mkdir(parents=True)
    for image_name in ("img1.jpg", "img2.jpg", "img3.jpg"):
        (work / "dense" / "images" / image_name).write_bytes(b"undistorted")
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



def test_resume_runs_every_openmvs_stage_from_session_working_directory(tmp_path, monkeypatch):
    import numpy as np
    import scripts.resume_openmvs_from_dense as resume_module

    session = _valid_session(tmp_path)
    work = session / "openmvs"
    calls = []

    monkeypatch.setattr(resume_module, "_binary", lambda name: f"/fake/{name}")
    monkeypatch.setattr(resume_module, "obj_median_center", lambda _path: np.zeros(3))
    monkeypatch.setattr(resume_module, "align_obj_file", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(resume_module, "align_glb_file", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        resume_module, "align_open3d_geometry_file",
        lambda *_args, **_kwargs: True,
    )

    def fake_run(command, cwd, log_file, stage_name):
        calls.append((stage_name, Path(cwd)))
        output = Path(command[command.index("-o") + 1])
        output.parent.mkdir(parents=True, exist_ok=True)
        if stage_name == "Reconstruct mesh from saved dense scene":
            output.write_bytes(b"mvs")
            output.with_suffix(".ply").write_bytes(b"ply" * 1000)
        elif stage_name == "Refine mesh at native image resolution":
            output.write_bytes(b"mvs")
            output.with_suffix(".ply").write_bytes(b"ply" * 1000)
        elif stage_name == "Export textured GLB":
            output.write_bytes(b"glb" * 1000)
        elif stage_name == "Export textured OBJ and MTL":
            output.write_text("mtllib scene_textured.mtl\nv 0 0 0\n", encoding="utf-8")
            (output.parent / "scene_textured.mtl").write_text(
                "newmtl material_00\nmap_Kd texture.jpg\n", encoding="utf-8"
            )
            (output.parent / "texture.jpg").write_bytes(b"jpeg")
        else:
            raise AssertionError(f"Unexpected OpenMVS stage: {stage_name}")

    monkeypatch.setattr(resume_module, "_run", fake_run)

    report = resume_module.resume_from_dense(session)

    assert len(calls) == 4
    assert all(cwd == work for _, cwd in calls)
    assert (session / "mesh").is_dir()
    assert (session / "mesh_before_dense_resume_v2").is_dir()
    assert report["method"].startswith("resume from existing dense scene")



def test_resume_can_install_when_failed_session_has_no_mesh_folder(tmp_path, monkeypatch):
    import scripts.resume_openmvs_from_dense as resume_module

    session = _valid_session(tmp_path)
    (session / "mesh").rmdir()

    class ReachedOpenMVS(Exception):
        pass

    monkeypatch.setattr(resume_module, "_binary", lambda name: f"/fake/{name}")
    monkeypatch.setattr(
        resume_module,
        "_run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ReachedOpenMVS()),
    )
    with pytest.raises(ReachedOpenMVS):
        resume_module.resume_from_dense(session)
    assert not (session / "mesh_before_dense_resume_v2").exists()
