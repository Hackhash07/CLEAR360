from pathlib import Path

import numpy as np
import pytest
import trimesh
from PIL import Image

from viewer.glb_texture_loader import load_textured_glb


def _triangle_mesh(color: tuple[int, int, int] | None = None):
    vertices = np.asarray(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64
    )
    faces = np.asarray([[0, 1, 2]], dtype=np.int64)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if color is not None:
        image = Image.new("RGB", (8, 8), color)
        material = trimesh.visual.material.PBRMaterial(
            baseColorTexture=image, metallicFactor=0.0, roughnessFactor=1.0
        )
        mesh.visual = trimesh.visual.TextureVisuals(
            uv=np.asarray([[0, 0], [1, 0], [0, 1]], dtype=np.float64),
            material=material,
        )
    return mesh


def _write_glb(path: Path, meshes: list[tuple[str, trimesh.Trimesh]]) -> None:
    scene = trimesh.Scene()
    for name, mesh in meshes:
        scene.add_geometry(mesh, node_name=name, geom_name=name)
    exported = scene.export(file_obj=str(path), file_type="glb")
    if isinstance(exported, bytes):
        path.write_bytes(exported)


def test_glb_loader_preserves_embedded_base_color_texture(tmp_path: Path):
    glb = tmp_path / "textured.glb"
    _write_glb(glb, [("body", _triangle_mesh((190, 65, 32)))])

    loaded = load_textured_glb(glb)

    assert loaded.texture is not None
    assert loaded.texture.shape[2] == 3
    assert loaded.textured_geometry_count == 1
    assert loaded.untextured_geometry_count == 0
    assert loaded.vertices.shape == (3, 3)
    assert loaded.faces.shape == (1, 3)
    assert loaded.uvs.shape == (3, 2)
    assert np.allclose(loaded.texture.mean(axis=(0, 1)), [190, 65, 32], atol=2)
    assert np.all((loaded.uvs >= 0.0) & (loaded.uvs <= 1.0))


def test_glb_loader_combines_multiple_material_textures_and_untextured_geometry(tmp_path: Path):
    glb = tmp_path / "mixed.glb"
    _write_glb(
        glb,
        [
            ("red_panel", _triangle_mesh((190, 30, 20))),
            ("blue_panel", _triangle_mesh((20, 40, 190))),
            ("plain_panel", _triangle_mesh()),
        ],
    )

    loaded = load_textured_glb(glb)

    assert loaded.texture is not None
    assert loaded.textured_geometry_count == 2
    assert loaded.untextured_geometry_count == 1
    assert loaded.faces.shape == (2, 3)
    assert loaded.untextured_faces.shape == (1, 3)
    assert loaded.texture.shape[1] == 16
    assert loaded.untextured_vertex_colors.shape == (3, 4)
    assert loaded.texture_coverage == pytest.approx(2 / 3)


def test_glb_loader_applies_scene_node_transforms(tmp_path: Path):
    glb = tmp_path / "translated.glb"
    scene = trimesh.Scene()
    scene.add_geometry(
        _triangle_mesh((120, 80, 40)),
        node_name="translated_body",
        geom_name="body",
        transform=trimesh.transformations.translation_matrix([4, 2, 1]),
    )
    exported = scene.export(file_obj=str(glb), file_type="glb")
    if isinstance(exported, bytes):
        glb.write_bytes(exported)

    loaded = load_textured_glb(glb)

    assert loaded.all_vertices.min(axis=0) == pytest.approx([4, 2, 1])
    assert loaded.all_vertices.max(axis=0) == pytest.approx([5, 3, 1])
