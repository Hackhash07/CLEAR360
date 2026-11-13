"""
mesh/tests/test_textured_mesher.py — Unit Tests for M7.2 Textured Mesher Engine
==============================================================================
"""

from pathlib import Path
import tempfile
import numpy as np
import pytest

from mesh.textured_mesher import TexturedMesher, TexturedCameraMesh, TexturedSceneMesh
from pipeline.calibration import CalibrationStore

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@pytest.fixture(scope="module")
def calib_store():
    store = CalibrationStore(PROJECT_ROOT / "config" / "cameras.yaml")
    store.load()
    return store


@pytest.fixture
def mesher(calib_store):
    return TexturedMesher(calib_store=calib_store, stride=2)


def test_mesher_init(mesher):
    """Verify default parameters and initialization."""
    assert mesher.stride == 2
    assert mesher.min_depth == 0.20
    assert mesher.max_depth == 35.0
    assert mesher.max_edge_len == 0.35
    assert mesher.min_grazing_cos == 0.15


def test_single_camera_triangulation(mesher):
    """Verify single camera unprojection, UV mapping, and faces."""
    H, W = 720, 1280
    rgb = np.full((H, W, 3), 180, dtype=np.uint8)
    # Planar ground depth
    depth = np.full((H, W), 5.0, dtype=np.float32)

    cm = mesher.triangulate_camera_textured("front", rgb, depth)
    assert isinstance(cm, TexturedCameraMesh)
    assert cm.camera_name == "front"
    assert len(cm.vertices) > 0
    assert len(cm.faces) > 0
    assert len(cm.uvs) == len(cm.vertices)
    assert cm.uvs[:, 0].min() >= 0.0
    assert cm.uvs[:, 0].max() <= 1.0
    assert cm.uvs[:, 1].min() >= 0.0
    assert cm.uvs[:, 1].max() <= 1.0


def test_depth_discontinuity_rejection(mesher):
    """Verify that step discontinuities (e.g. wall to sky) are rejected."""
    H, W = 720, 1280
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    depth = np.full((H, W), 3.0, dtype=np.float32)
    # Introduce sharp 10m step discontinuity across middle
    depth[:, W // 2:] = 15.0

    cm = mesher.triangulate_camera_textured("front", rgb, depth)
    # Check that no face bridges the 3m and 15m regions
    verts = cm.vertices
    faces = cm.faces
    zA = verts[faces[:, 0], 1]
    zB = verts[faces[:, 1], 1]
    zC = verts[faces[:, 2], 1]
    z_diff = np.maximum.reduce([np.abs(zA - zB), np.abs(zB - zC), np.abs(zC - zA)])
    # In vehicle frame, Y is depth forward. Diff shouldn't bridge the 12m step
    assert np.all(z_diff < 5.0)


def test_texture_atlas_layout(mesher):
    """Verify 2x2 texture atlas dimensions and UV transform quadrants."""
    H, W = 720, 1280
    rgb_dict = {
        "front": np.full((H, W, 3), 10, dtype=np.uint8),
        "rear": np.full((H, W, 3), 20, dtype=np.uint8),
        "left": np.full((H, W, 3), 30, dtype=np.uint8),
        "right": np.full((H, W, 3), 40, dtype=np.uint8),
    }

    atlas, uv_transforms = mesher.create_texture_atlas(rgb_dict)
    assert atlas.shape == (H * 2, W * 2, 3)
    assert "front" in uv_transforms
    assert "rear" in uv_transforms
    assert "left" in uv_transforms
    assert "right" in uv_transforms

    # Quadrant checks
    assert uv_transforms["front"] == (0.0, 0.5, 0.5, 0.5)
    assert uv_transforms["rear"] == (0.5, 0.5, 0.5, 0.5)
    assert uv_transforms["left"] == (0.0, 0.0, 0.5, 0.5)
    assert uv_transforms["right"] == (0.5, 0.0, 0.5, 0.5)


def test_four_camera_reconstruction(mesher):
    """Verify 4-camera fused textured scene reconstruction."""
    H, W = 720, 1280
    rgb_dict = {c: np.full((H, W, 3), 100, dtype=np.uint8) for c in ["front", "rear", "left", "right"]}
    depth_dict = {c: np.full((H, W), 4.0, dtype=np.float32) for c in ["front", "rear", "left", "right"]}

    scene = mesher.reconstruct_textured_scene(rgb_dict, depth_dict)
    assert isinstance(scene, TexturedSceneMesh)
    assert len(scene.vertices) > 0
    assert len(scene.faces) > 0
    assert len(scene.uvs) == len(scene.vertices)
    assert scene.texture_atlas.shape == (H * 2, W * 2, 3)


def test_obj_export(mesher):
    """Verify Wavefront OBJ + MTL + PNG texture export."""
    H, W = 720, 1280
    rgb = np.full((H, W, 3), 200, dtype=np.uint8)
    depth = np.full((H, W), 4.0, dtype=np.float32)
    cm = mesher.triangulate_camera_textured("front", rgb, depth)

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_dir = Path(tmp_dir)
        obj_file = mesher.save_textured_mesh_obj(cm, out_dir, name_prefix="test_mesh")
        assert obj_file.exists()
        mtl_file = out_dir / "test_mesh.mtl"
        tex_file = out_dir / "test_mesh_texture.png"
        assert mtl_file.exists()
        assert tex_file.exists()

        # Check OBJ contents
        with open(obj_file, "r") as f:
            header = [f.readline() for _ in range(5)]
            assert any("mtllib" in line for line in header)


def test_ply_fallback_export(mesher):
    """Verify PLY fallback export with vertex colors."""
    H, W = 720, 1280
    rgb = np.full((H, W, 3), 150, dtype=np.uint8)
    depth = np.full((H, W), 4.0, dtype=np.float32)
    cm = mesher.triangulate_camera_textured("front", rgb, depth)

    with tempfile.TemporaryDirectory() as tmp_dir:
        ply_file = Path(tmp_dir) / "test_fallback.ply"
        mesher.save_textured_mesh_ply(cm, ply_file)
        assert ply_file.exists()
        assert ply_file.stat().st_size > 1000

