"""
pointcloud/tests/test_unprojector.py
=====================================
Unit tests for point cloud unprojection, fusion, and PLY export.
"""

from pathlib import Path
import numpy as np
import pytest
import open3d as o3d

from pipeline.calibration import CalibrationStore
from pipeline.sync import FrameSetLoader
from pointcloud.unprojector import DepthUnprojector

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
CONFIG_PATH = PROJECT_ROOT / "config" / "cameras.yaml"


@pytest.fixture
def store():
    st = CalibrationStore(CONFIG_PATH)
    st.load()
    return st


@pytest.fixture
def loader():
    return FrameSetLoader(DATA_ROOT)


@pytest.fixture
def unprojector():
    return DepthUnprojector(max_depth=50.0, min_depth=0.1, stride=4, exclude_ego=True)


def test_unprojector_single_camera(store, loader, unprojector):
    """Test unprojection of front camera into Open3D PointCloud."""
    fs = loader.load_frame(0, load_position=False, load_semantic=True)
    front_cam = store.get("front")
    cfd = fs.get("front")

    pcd = unprojector.unproject_to_pointcloud(
        depth=cfd.depth, camera_model=front_cam, rgb=cfd.rgb, semantic_mask=cfd.semantic_mask
    )

    pts = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors)

    assert len(pts) > 1000, f"Expected >1000 points, got {len(pts)}"
    assert pts.shape[1] == 3
    assert cols.shape == pts.shape
    assert np.all((cols >= 0.0) & (cols <= 1.0))

    # Front camera points should all be generally in front of vehicle (Y > 0)
    # or near boundary walls
    assert np.mean(pts[:, 1]) > 0.0


def test_fuse_frame_set_preserves_vertical_geometry(store, loader, unprojector):
    """Test 4-camera fusion and vertical geometry retention (walls and pillar)."""
    fs = loader.load_frame(0, load_position=False, load_semantic=True)
    merged = unprojector.fuse_frame_set(fs, store)
    pts = np.asarray(merged.points)

    assert len(pts) > 10000, "Fused point cloud should contain >10k points"

    # Wall at Y=+9.5 should span Z from 0.0 to 2.5
    wall_north = pts[(np.abs(pts[:, 1] - 9.5) < 0.5)]
    assert len(wall_north) > 100
    z_span = wall_north[:, 2].max() - wall_north[:, 2].min()
    assert z_span > 2.0, f"North wall vertical span {z_span:.2f}m is too low (<2.0m)"

    # Road flatness: Z near 0
    road_pts = pts[(np.abs(pts[:, 2]) < 0.05) & (np.abs(pts[:, 0]) < 4) & (np.abs(pts[:, 1]) < 4)]
    assert len(road_pts) > 1000
    assert np.std(road_pts[:, 2]) < 0.02


def test_unprojector_save_and_downsample(store, loader, unprojector, tmp_path):
    """Test PLY saving and voxel downsampling."""
    fs = loader.load_frame(0, load_position=False, load_semantic=False)
    pcd = unprojector.fuse_frame_set(fs, store)

    ply_path = tmp_path / "test_merged.ply"
    unprojector.save_ply(pcd, ply_path)
    assert ply_path.exists()
    assert ply_path.stat().st_size > 0

    # Load back
    loaded = o3d.io.read_point_cloud(str(ply_path))
    assert len(loaded.points) == len(pcd.points)

    # Downsample
    ds = unprojector.downsample(pcd, voxel_size=0.1)
    assert len(ds.points) < len(pcd.points)
    assert len(ds.points) > 0

