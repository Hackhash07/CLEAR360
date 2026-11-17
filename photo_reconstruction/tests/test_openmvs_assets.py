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



def test_export_obj_materials_to_glb_preserves_embedded_diffuse_texture(tmp_path: Path):
    import numpy as np
    from PIL import Image

    from photo_reconstruction.openmvs_assets import export_obj_materials_to_glb

    mesh_dir = tmp_path / "mesh"
    mesh_dir.mkdir()
    Image.fromarray(
        np.full((16, 16, 3), [180, 90, 40], dtype=np.uint8)
    ).save(mesh_dir / "diffuse.png")

    obj = mesh_dir / "scene_mesh.obj"
    obj.write_text(
        "mtllib scene_mesh.mtl\n"
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 1 1 0\n"
        "v 0 1 0\n"
        "vt 0 0\n"
        "vt 1 0\n"
        "vt 1 1\n"
        "vt 0 1\n"
        "usemtl material_00\n"
        "f 1/1 2/2 3/3\n"
        "f 1/1 3/3 4/4\n",
        encoding="utf-8",
    )
    (mesh_dir / "scene_mesh.mtl").write_text(
        "newmtl material_00\n"
        "Ka 1 1 1\n"
        "Kd 1 1 1\n"
        "Ks 0 0 0\n"
        "map_Kd diffuse.png\n",
        encoding="utf-8",
    )

    glb = mesh_dir / "scene_textured.glb"
    report = export_obj_materials_to_glb(obj, glb)

    assert glb.is_file()
    assert glb.stat().st_size > 0
    assert report["obj_vertices"] == 4
    assert report["obj_faces"] == 2
    assert report["glb_vertices"] > 0
    assert report["glb_faces"] == 2
    assert report["glb_textured_geometry_count"] == 1


def test_export_obj_materials_to_glb_rejects_untextured_obj(tmp_path: Path):
    from photo_reconstruction.openmvs_assets import export_obj_materials_to_glb

    obj = tmp_path / "untextured.obj"
    obj.write_text(
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 0 1 0\n"
        "f 1 2 3\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="no readable diffuse image"):
        export_obj_materials_to_glb(obj, tmp_path / "out.glb")
