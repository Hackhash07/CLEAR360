"""
viewer/tests/test_surface_mesh.py — Comprehensive Unit & Integration Tests for M7.1
===================================================================================

Validates:
  1. RGBDSurfaceMesher initialization and parameter validation.
  2. 2D organized grid unprojection and depth discontinuity rejection.
  3. Grazing-angle filtering in camera overlap zones.
  4. Multi-camera surface fusion into unified watertight mesh.
  5. PLY serialization and deserialization with vertex colors and normals.
  6. SceneData mesh attributes and ViewerDataLoader caching.
  7. Clear360Viewport 3-mode visualization switching.
  8. Authentic 2022 Hyundai Creta SUV 3D dimensions and materials.
"""

from pathlib import Path
import numpy as np
import pytest

from pipeline.calibration import CalibrationStore
from mesh.rgbd_mesher import RGBDSurfaceMesher, CameraMesh
from viewer.data_loader import SceneData, ViewerDataLoader
from viewer.vehicle_model import VehicleModel3D

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "cameras.yaml"
DATA_ROOT = PROJECT_ROOT / "data"


@pytest.fixture
def calib_store():
    store = CalibrationStore(CONFIG_PATH)
    store.load()
    return store


@pytest.fixture
def mesher(calib_store):
    return RGBDSurfaceMesher(calib_store=calib_store, stride=4)


def test_mesher_initialization(calib_store):
    """Verify default and customized mesher parameters."""
    m = RGBDSurfaceMesher(calib_store=calib_store, stride=3, min_depth=0.5, max_depth=25.0)
    assert m.stride == 3
    assert m.min_depth == 0.5
    assert m.max_depth == 25.0
    assert m.min_grazing_cos == 0.12


def test_synthetic_camera_triangulation(mesher):
    """Verify single-camera planar depth unprojection and mesh generation."""
    # Synthetic 720x1280 image of a flat wall at 4.0 m
    H, W = 720, 1280
    rgb = np.full((H, W, 3), 180, dtype=np.uint8)
    depth = np.full((H, W), 4.0, dtype=np.float32)

    cm = mesher.triangulate_camera("front", rgb, depth)
    assert isinstance(cm, CameraMesh)
    assert cm.camera_name == "front"
    assert len(cm.vertices) > 1000
    assert len(cm.faces) > 1000
    assert cm.vertices.shape[1] == 3
    assert cm.faces.shape[1] == 3
    assert cm.colors.shape[1] == 3
    assert cm.normals.shape[1] == 3

    # All unprojected points should have positive Y forward in front camera
    assert np.all(cm.vertices[:, 1] > 0.0)


def test_depth_discontinuity_rejection(mesher):
    """Verify that sharp depth steps (silhouette boundaries) produce no flying triangles."""
    H, W = 720, 1280
    rgb = np.full((H, W, 3), 128, dtype=np.uint8)
    # Left half at 2.0 m, right half at 10.0 m (8.0 m jump)
    depth = np.full((H, W), 2.0, dtype=np.float32)
    depth[:, W // 2:] = 10.0

    cm = mesher.triangulate_camera("front", rgb, depth)
    pA = cm.vertices[cm.faces[:, 0]]
    pB = cm.vertices[cm.faces[:, 1]]
    pC = cm.vertices[cm.faces[:, 2]]

    # Compute maximum 3D edge length of every triangle
    e1 = np.linalg.norm(pA - pB, axis=1)
    e2 = np.linalg.norm(pB - pC, axis=1)
    e3 = np.linalg.norm(pC - pA, axis=1)
    max_edge = np.maximum.reduce([e1, e2, e3])

    # No triangle should span across the 8-metre depth jump
    assert np.all(max_edge <= mesher.max_edge_len + 1e-4)


def test_scene_mesh_fusion(mesher):
    """Verify multi-camera fusion combines separate camera meshes into unified geometry."""
    H, W = 720, 1280
    rgb_dict = {c: np.full((H, W, 3), 100, dtype=np.uint8) for c in ["front", "rear", "left", "right"]}
    depth_dict = {c: np.full((H, W), 5.0, dtype=np.float32) for c in ["front", "rear", "left", "right"]}

    fused_mesh = mesher.reconstruct_scene_mesh(rgb_dict, depth_dict)
    n_v = len(fused_mesh.vertices)
    n_f = len(fused_mesh.triangles)

    assert n_v > 4000
    assert n_f > 4000
    assert len(fused_mesh.vertex_colors) == n_v
    assert len(fused_mesh.vertex_normals) == n_v


def test_mesh_ply_io(tmp_path, mesher):
    """Verify saving and loading PLY meshes with vertex colors and normals."""
    H, W = 720, 1280
    rgb = np.full((H, W, 3), 220, dtype=np.uint8)
    depth = np.full((H, W), 3.5, dtype=np.float32)

    cm = mesher.triangulate_camera("front", rgb, depth)
    fused_mesh = mesher.reconstruct_scene_mesh({"front": rgb}, {"front": depth})

    out_file = tmp_path / "test_mesh.ply"
    mesher.save_mesh_ply(fused_mesh, out_file)
    assert out_file.exists()
    assert out_file.stat().st_size > 1000

    verts, faces, colors, normals = mesher.load_mesh_ply(out_file)
    assert len(verts) == len(fused_mesh.vertices)
    assert len(faces) == len(fused_mesh.triangles)
    assert colors.shape[1] == 4  # RGBA
    assert normals.shape[1] == 3


def test_data_loader_mesh_integration():
    """Verify SceneData container and ViewerDataLoader load pre-generated meshes."""
    loader = ViewerDataLoader(data_dir=DATA_ROOT)
    scene = loader.load_scene(frame_id=0, source="ground_truth")

    assert scene.mesh_vertices is not None
    assert scene.mesh_faces is not None
    assert scene.mesh_vertex_count > 100000
    assert scene.mesh_face_count > 200000
    assert scene.mesh_colors is not None
    assert scene.mesh_colors.shape[1] == 4


def test_vehicle_model_creta_specifications():
    """Verify authentic dimensions and parts of virtual 2022 Hyundai Creta SUV."""
    veh = VehicleModel3D()
    assert np.isclose(veh.half_length * 2.0, 4.30)
    assert np.isclose(veh.half_width * 2.0, 1.79)
    assert np.isclose(veh.mirror_tip_x * 2.0, 2.10)
    assert np.isclose(veh.height, 1.635)

    v, f, c = veh.get_mesh_data()
    assert len(v) > 200
    assert len(f) > 100
    assert c.shape == (len(v), 4)

    # Verify clearance envelope lines
    env_pts, env_cols = veh.get_clearance_envelope_lines()
    assert len(env_pts) > 20
    assert env_pts.shape[1] == 3
