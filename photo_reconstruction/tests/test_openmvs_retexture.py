"""Safety and command-contract tests for texture-only OpenMVS reprocessing."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.retexture_openmvs_session import (
    FIX_SHA,
    TextureRetextureError,
    build_command,
    build_info_for,
    diffuse_maps,
    inspect_session,
    resolve_output_dir,
    run_texture_only,
    sha256_file,
)


def make_session(root: Path) -> Path:
    session = root / "session_049"
    work = session / "openmvs"
    images = work / "dense" / "images"
    images.mkdir(parents=True)
    (images / "fountain_0001.jpg").write_bytes(b"photo1")
    (images / "fountain_0002.jpg").write_bytes(b"photo2")
    (work / "scene_mesh.mvs").write_bytes(b"mvs scene")
    (work / "scene_mesh.ply").write_bytes(b"ply mesh")
    (session / "openmvs_pipeline.log").write_text(
        'Pipeline configuration: {"openmvs":{"texture_threads":3}}\n', encoding="utf-8"
    )
    (session / "mesh").mkdir()
    (session / "mesh" / "scene_mesh.ply").write_bytes(b"final baseline mesh")
    return session


def make_fixed_executable(root: Path) -> Path:
    exe = root / "TextureMesh"
    exe.write_bytes(b"fixed binary")
    exe.chmod(0o755)
    exe.with_name("TextureMesh.build.json").write_text(json.dumps({
        "source_commit": FIX_SHA, "binary_sha256": sha256_file(exe)
    }), encoding="utf-8")
    return exe


def test_inspect_session_finds_saved_mesh_images_and_original_thread_count(tmp_path: Path):
    inputs = inspect_session(make_session(tmp_path))
    assert inputs.scene_file.name == "scene_mesh.mvs"
    assert inputs.mesh_ply.name == "scene_mesh.ply"
    assert len(inputs.image_files) == 2
    assert inputs.texture_threads == 3


def test_inspect_session_stops_if_pretexture_scene_missing(tmp_path: Path):
    session = make_session(tmp_path)
    (session / "openmvs" / "scene_mesh.mvs").unlink()
    with pytest.raises(TextureRetextureError, match="incomplete"):
        inspect_session(session)


def test_output_folder_must_be_fresh_texture_ab_child(tmp_path: Path):
    session = make_session(tmp_path)
    assert resolve_output_dir(session, None) == session / "texture_ab_fixed"
    with pytest.raises(TextureRetextureError, match="texture_ab_"):
        resolve_output_dir(session, "mesh")
    with pytest.raises(TextureRetextureError, match="direct child"):
        resolve_output_dir(session, session / "openmvs" / "overwrite")
    (session / "texture_ab_existing").mkdir()
    with pytest.raises(TextureRetextureError, match="already exists"):
        resolve_output_dir(session, "texture_ab_existing")


def test_command_preserves_session_049_texture_flags(tmp_path: Path):
    inputs = inspect_session(make_session(tmp_path))
    exe = tmp_path / "TextureMesh"
    output = inputs.session_dir / "texture_ab_fixed"
    command = build_command(exe, inputs, output)
    assert command[:5] == [
        str(exe), "-i", str(inputs.scene_file), "-o", str(output / "scene_textured.obj")
    ]
    expected = {
        "--max-threads": "3", "--resolution-level": "0", "--min-resolution": "640",
        "--max-texture-size": "2048", "--virtual-face-images": "1",
        "--cost-smoothness-ratio": "0.7", "--close-holes": "0",
        "--sharpness-weight": "0.5", "--local-seam-leveling": "1",
        "--global-seam-leveling": "0",
    }
    for option, value in expected.items():
        assert command[command.index(option) + 1] == value
    assert "--ignore-mask-label=-1" in command


def test_mtl_diffuse_maps_resolve_paths_with_spaces(tmp_path: Path):
    atlas = tmp_path / "atlas one.jpg"
    atlas.write_bytes(b"atlas")
    mtl = tmp_path / "scene_textured.mtl"
    mtl.write_text('newmtl material_0\nmap_Kd "atlas one.jpg"\n', encoding="utf-8")
    assert diffuse_maps(mtl) == [atlas.resolve()]


def test_pinned_build_info_rejects_changed_binary_or_wrong_commit(tmp_path: Path):
    exe = make_fixed_executable(tmp_path)
    assert build_info_for(exe)["source_commit"] == FIX_SHA
    exe.with_name("TextureMesh.build.json").write_text(json.dumps({
        "source_commit": "wrong", "binary_sha256": sha256_file(exe)
    }), encoding="utf-8")
    with pytest.raises(TextureRetextureError, match="pinned upstream fix"):
        build_info_for(exe)


def test_texture_only_run_hashes_inputs_and_writes_separate_result(tmp_path: Path, monkeypatch):
    inputs = inspect_session(make_session(tmp_path))
    exe = make_fixed_executable(tmp_path)
    output = inputs.session_dir / "texture_ab_fixed"
    baseline_hashes = {
        p: sha256_file(p)
        for p in (inputs.scene_file, inputs.mesh_ply, *inputs.image_files, *inputs.baseline_mesh_files)
    }

    class Completed:
        returncode = 0
        stdout = "TextureMesh completed"

    def fake_run(command, **kwargs):
        (output / "scene_textured.obj").write_text("v 0 0 0\nf 1 1 1\n", encoding="utf-8")
        (output / "scene_textured.mtl").write_text(
            "newmtl material_0\nmap_Kd atlas.jpg\n", encoding="utf-8"
        )
        (output / "atlas.jpg").write_bytes(b"new atlas")
        return Completed()

    monkeypatch.setattr("scripts.retexture_openmvs_session.subprocess.run", fake_run)
    manifest = run_texture_only(inputs, exe, output)
    assert manifest["status"] == "success"
    assert manifest["geometry_stages_run"] == []
    assert manifest["input_files_unchanged"] is True
    assert manifest["baseline_final_mesh_unchanged"] is True
    assert (output / "texture_reprocess_manifest.json").is_file()
    assert all(sha256_file(p) == digest for p, digest in baseline_hashes.items())


def test_texture_only_run_refuses_existing_output(tmp_path: Path):
    inputs = inspect_session(make_session(tmp_path))
    exe = make_fixed_executable(tmp_path)
    output = inputs.session_dir / "texture_ab_fixed"
    output.mkdir()
    with pytest.raises(TextureRetextureError, match="already exists"):
        run_texture_only(inputs, exe, output)


@pytest.mark.parametrize("unexpected_folder", ["images", "mesh"])
def test_texture_only_detects_new_files_in_protected_directories(
    tmp_path: Path, monkeypatch, unexpected_folder: str
):
    inputs = inspect_session(make_session(tmp_path))
    exe = make_fixed_executable(tmp_path)
    output = inputs.session_dir / "texture_ab_fixed"

    class Completed:
        returncode = 0
        stdout = "TextureMesh completed"

    def fake_run(command, **kwargs):
        (output / "scene_textured.obj").write_text("v 0 0 0\nf 1 1 1\n", encoding="utf-8")
        (output / "scene_textured.mtl").write_text(
            "newmtl material_0\nmap_Kd atlas.jpg\n", encoding="utf-8"
        )
        (output / "atlas.jpg").write_bytes(b"new atlas")
        protected_dir = (
            inputs.image_dir if unexpected_folder == "images"
            else inputs.session_dir / "mesh"
        )
        (protected_dir / "unexpected-added-file.bin").write_bytes(b"unexpected")
        return Completed()

    monkeypatch.setattr("scripts.retexture_openmvs_session.subprocess.run", fake_run)
    with pytest.raises(TextureRetextureError, match="Input hashes changed"):
        run_texture_only(inputs, exe, output)

    manifest = json.loads((output / "texture_reprocess_manifest.json").read_text())
    assert manifest["input_files_unchanged"] is False
    assert manifest["status"] == "failed"


def test_texture_only_rejects_empty_diffuse_map(tmp_path: Path, monkeypatch):
    inputs = inspect_session(make_session(tmp_path))
    exe = make_fixed_executable(tmp_path)
    output = inputs.session_dir / "texture_ab_fixed"

    class Completed:
        returncode = 0
        stdout = "TextureMesh completed"

    def fake_run(command, **kwargs):
        (output / "scene_textured.obj").write_text("v 0 0 0\nf 1 1 1\n", encoding="utf-8")
        (output / "scene_textured.mtl").write_text(
            "newmtl material_0\nmap_Kd atlas.jpg\n", encoding="utf-8"
        )
        (output / "atlas.jpg").write_bytes(b"")

        return Completed()

    monkeypatch.setattr("scripts.retexture_openmvs_session.subprocess.run", fake_run)
    with pytest.raises(TextureRetextureError, match="Incomplete or empty"):
        run_texture_only(inputs, exe, output)

    manifest = json.loads((output / "texture_reprocess_manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert str(output / "atlas.jpg") in manifest["empty_output_files"]
