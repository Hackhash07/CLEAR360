from pathlib import Path

import numpy as np
from PIL import Image

from viewer.obj_texture_loader import load_textured_obj


def test_loader_expands_independent_obj_vertex_and_uv_indices(tmp_path: Path):
    obj = tmp_path / "scene_mesh.obj"
    mtl = tmp_path / "scene_mesh.mtl"
    Image.new("RGB", (8, 8), (200, 40, 20)).save(tmp_path / "scene_mesh.png")
    mtl.write_text(
        "newmtl fountain\n"
        "Ka 1 1 1\n"
        "Kd 1 1 1\n"
        "map_Kd scene_mesh.png\n",
        encoding="utf-8",
    )
    # Four geometric vertices, but six UV coordinates. Vertex 1 is assigned
    # two different UV indices to model a texture seam across two triangles.
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
        "vt 0.2 0.3\n"
        "vt 0.8 0.7\n"
        "vn 0 0 1\n"
        "usemtl fountain\n"
        "f 1/1/1 2/2/1 3/3/1\n"
        "f 1/5/1 3/6/1 4/4/1\n",
        encoding="utf-8",
    )

    vertices, faces, normals, uvs, texture, atlas = load_textured_obj(obj)

    assert vertices.shape == (6, 3)
    assert faces.shape == (2, 3)
    assert normals is not None and normals.shape == (6, 3)
    assert uvs.shape == (6, 2)
    assert texture.shape == (8, 8, 3)
    assert tuple(texture[0, 0]) == (200, 40, 20)
    assert atlas.name == "scene_mesh.png"
    # Same geometric coordinate is duplicated because the texture seam differs.
    same_position = np.where(np.all(np.isclose(vertices, [0, 0, 0]), axis=1))[0]
    assert len(same_position) == 2
    assert not np.allclose(uvs[same_position[0]], uvs[same_position[1]])


def test_loader_combines_multiple_material_atlases(tmp_path: Path):
    obj = tmp_path / "multi.obj"
    mtl = tmp_path / "multi.mtl"
    Image.new("RGB", (4, 4), (255, 0, 0)).save(tmp_path / "red.png")
    Image.new("RGB", (4, 4), (0, 0, 255)).save(tmp_path / "blue.png")
    mtl.write_text(
        "newmtl red\nmap_Kd red.png\n"
        "newmtl blue\nmap_Kd blue.png\n",
        encoding="utf-8",
    )
    obj.write_text(
        "mtllib multi.mtl\n"
        "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n"
        "vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\n"
        "usemtl red\nf 1/1 2/2 3/3\n"
        "usemtl blue\nf 1/1 3/3 4/4\n",
        encoding="utf-8",
    )

    vertices, faces, normals, uvs, texture, atlas = load_textured_obj(obj)

    assert faces.shape == (2, 3)
    assert texture.shape == (4, 8, 3)
    assert atlas.name in {"red.png", "blue.png"}
    assert uvs.min() >= 0.0
    assert uvs.max() <= 1.0


def _write_textured_fixture(tmp_path: Path, obj_body: str) -> Path:
    Image.new("RGB", (8, 8), (200, 40, 20)).save(tmp_path / "atlas.png")
    (tmp_path / "scene.mtl").write_text(
        "newmtl surface\nmap_Kd atlas.png\n",
        encoding="utf-8",
    )
    obj = tmp_path / "scene.obj"
    obj.write_text(
        "mtllib scene.mtl\n" + obj_body,
        encoding="utf-8",
    )
    return obj


def test_loader_keeps_faces_without_uvs_as_separate_geometry(tmp_path: Path):
    obj = _write_textured_fixture(
        tmp_path,
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 1 1 0\n"
        "v 0 1 0\n"
        "vt 0 0\n"
        "vt 1 0\n"
        "vt 1 1\n"
        "vt 0 1\n"
        "vn 0 0 1\n"
        "usemtl surface\n"
        "f 1/1/1 2//1 3/3/1\n"
        "f 1/1/1 3/3/1 4/4/1\n",
    )

    loaded = load_textured_obj(obj)

    assert loaded.textured_triangle_count == 1
    assert loaded.untextured_triangle_count == 1
    assert loaded.total_triangle_count == 2
    assert loaded.texture_coverage == 0.5
    assert loaded.faces.shape == (1, 3)
    assert loaded.uvs.shape == (3, 2)
    assert loaded.untextured_faces.shape == (1, 3)
    assert loaded.untextured_vertices.shape == (3, 3)
    # Legacy six-value unpacking remains supported.
    vertices, faces, normals, uvs, texture, atlas = loaded
    assert vertices is loaded.vertices
    assert faces is loaded.faces
    assert atlas.name == "atlas.png"


def test_loader_accepts_v_and_v_double_slash_vn_corners(tmp_path: Path):
    obj = _write_textured_fixture(
        tmp_path,
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 1 1 0\n"
        "v 0 1 0\n"
        "v 0 0 1\n"
        "vt 0 0\n"
        "vt 1 0\n"
        "vt 1 1\n"
        "vn 0 0 1\n"
        "f 1//1 2//1 3//1\n"
        "f 1 3 4\n"
        "f 1/1 4/2 5/3\n",
    )

    loaded = load_textured_obj(obj)

    assert loaded.textured_triangle_count == 1
    assert loaded.untextured_triangle_count == 2
    assert loaded.faces.shape == (1, 3)
    assert loaded.untextured_faces.shape == (2, 3)


def test_loader_resolves_negative_vertex_and_uv_indices(tmp_path: Path):
    obj = _write_textured_fixture(
        tmp_path,
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 0 1 0\n"
        "vt 0 0\n"
        "vt 1 0\n"
        "vt 0 1\n"
        "f -3/-3 -2/-2 -1/-1\n",
    )

    loaded = load_textured_obj(obj)

    assert loaded.faces.shape == (1, 3)
    assert np.allclose(loaded.vertices, [[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    assert loaded.textured_triangle_count == 1
    assert loaded.untextured_triangle_count == 0
