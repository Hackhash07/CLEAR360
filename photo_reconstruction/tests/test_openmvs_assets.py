"""Tests for copying OpenMVS OBJ material texture assets."""

from pathlib import Path
import shlex

import pytest

from photo_reconstruction.openmvs_assets import copy_mtl_diffuse_maps


def test_copy_mtl_diffuse_maps_copies_every_jpeg_and_rewrites_paths(tmp_path: Path):
    source_dir = tmp_path / "openmvs"
    mesh_dir = tmp_path / "mesh"
    source_dir.mkdir()
    (source_dir / "texture one.jpg").write_bytes(b"jpeg-one")
    (source_dir / "texture-two.jpg").write_bytes(b"jpeg-two")
    source_mtl = source_dir / "scene_textured.mtl"
    source_mtl.write_text(
        "newmtl material_00\n"
        'map_Kd "texture one.jpg"\n'
        "newmtl material_01\n"
        "map_Kd texture-two.jpg\n",
        encoding="utf-8",
    )

    output_mtl = copy_mtl_diffuse_maps(
        source_mtl, mesh_dir, output_name="scene_mesh.mtl"
    )

    assert output_mtl == mesh_dir / "scene_mesh.mtl"
    assert (mesh_dir / "texture one.jpg").read_bytes() == b"jpeg-one"
    assert (mesh_dir / "texture-two.jpg").read_bytes() == b"jpeg-two"
    text = output_mtl.read_text(encoding="utf-8")
    map_lines = [line for line in text.splitlines() if line.lstrip().lower().startswith("map_kd ")]
    assert shlex.split(map_lines[0]) == ["map_Kd", "texture one.jpg"]
    assert shlex.split(map_lines[1]) == ["map_Kd", "texture-two.jpg"]


def test_copy_mtl_diffuse_maps_preserves_map_options(tmp_path: Path):
    source_dir = tmp_path / "openmvs"
    mesh_dir = tmp_path / "mesh"
    source_dir.mkdir()
    (source_dir / "atlas.jpg").write_bytes(b"jpeg")
    source_mtl = source_dir / "scene.mtl"
    source_mtl.write_text(
        "newmtl material_00\nmap_Kd -s 1 1 1 atlas.jpg\n",
        encoding="utf-8",
    )

    output_mtl = copy_mtl_diffuse_maps(source_mtl, mesh_dir)

    assert "map_Kd -s 1 1 1 atlas.jpg" in output_mtl.read_text(encoding="utf-8")
    assert (mesh_dir / "atlas.jpg").is_file()


def test_copy_mtl_diffuse_maps_fails_for_missing_texture(tmp_path: Path):
    source_dir = tmp_path / "openmvs"
    source_dir.mkdir()
    source_mtl = source_dir / "scene.mtl"
    source_mtl.write_text(
        "newmtl material_00\nmap_Kd missing.jpg\n",
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="MTL diffuse map does not exist"):
        copy_mtl_diffuse_maps(source_mtl, tmp_path / "mesh")
