"""
viewer/tests/test_data_loader.py — Unit Tests for Viewer Data Loader
"""

import pytest
import numpy as np
from pathlib import Path
from viewer.data_loader import ViewerDataLoader, SceneData


@pytest.fixture
def data_loader():
    return ViewerDataLoader()


def test_data_loader_available_sources_and_frames(data_loader):
    assert data_loader.available_frames == [0, 5, 10]
    assert "ground_truth" in data_loader.available_sources
    assert "ai_calibrated" in data_loader.available_sources
    assert "ai_raw" in data_loader.available_sources


def test_load_ground_truth_frame_0(data_loader):
    scene = data_loader.load_scene(0, "ground_truth")
    assert isinstance(scene, SceneData)
    assert scene.frame_id == 0
    assert scene.source == "ground_truth"
    assert scene.point_count > 100000
    assert scene.points_rgb.ndim == 2 and scene.points_rgb.shape[1] == 3
    assert scene.colors_rgb.ndim == 2 and scene.colors_rgb.shape[1] == 3
    assert scene.points_semantic.ndim == 2 and scene.points_semantic.shape[1] == 3
    assert len(scene.obstacles) > 0
    assert "ground_plane" in scene.scene_metadata
    assert "occupancy_grid" in scene.scene_metadata


def test_load_calibrated_ai_frame_5(data_loader):
    scene = data_loader.load_scene(5, "ai_calibrated")
    assert scene.frame_id == 5
    assert scene.source == "ai_calibrated"
    assert scene.point_count > 50000
    assert len(scene.occupancy_occupied_points) > 0


def test_load_raw_ai_frame_10(data_loader):
    scene = data_loader.load_scene(10, "ai_raw")
    assert scene.frame_id == 10
    assert scene.source == "ai_raw"
    assert scene.point_count > 10000


def test_caching_behavior(data_loader):
    scene1 = data_loader.load_scene(0, "ground_truth")
    scene2 = data_loader.load_scene(0, "ground_truth")
    # Verify cached instance identity
    assert scene1 is scene2


def test_unknown_source_error(data_loader):
    with pytest.raises(ValueError, match="Unknown source"):
        data_loader.load_scene(0, "non_existent_source")


def test_occupancy_points_bounds(data_loader):
    scene = data_loader.load_scene(0, "ground_truth")
    occ_pts = scene.occupancy_occupied_points
    assert occ_pts.ndim == 2 and occ_pts.shape[1] == 3
    # Check that occupancy points lie within vehicle surround workspace
    assert np.all(occ_pts[:, 0] >= -15.0) and np.all(occ_pts[:, 0] <= 15.0)
    assert np.all(occ_pts[:, 1] >= -15.0) and np.all(occ_pts[:, 1] <= 15.0)
    assert np.all(occ_pts[:, 2] >= -2.0) and np.all(occ_pts[:, 2] <= 5.0)

