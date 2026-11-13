"""
viewer/tests/test_viewport.py — Unit Tests for 3D Viewport Controls & Offscreen Rendering
"""

import pytest
import numpy as np
from PySide6 import QtWidgets
from viewer.viewport import Clear360Viewport
from viewer.data_loader import ViewerDataLoader


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def viewport(qapp):
    vp = Clear360Viewport()
    loader = ViewerDataLoader()
    scene = loader.load_scene(0, "ground_truth")
    vp.set_scene_data(scene)
    return vp


def test_viewport_initialization(viewport):
    assert viewport.current_scene is not None
    assert viewport.current_scene.frame_id == 0
    assert viewport.point_cloud_mode == "rgb"
    assert viewport.show_vehicle is True
    assert viewport.show_bboxes is True


def test_camera_presets(viewport):
    presets = ["chase", "front", "rear", "left", "right", "top", "reset"]
    for p in presets:
        viewport.set_camera_preset(p)
        assert viewport.view.camera is not None


def test_point_cloud_modes(viewport):
    viewport.set_point_cloud_mode("semantic")
    assert viewport.point_cloud_mode == "semantic"
    assert viewport.visual_cloud.visible is True

    viewport.set_point_cloud_mode("none")
    assert viewport.point_cloud_mode == "none"
    assert viewport.visual_cloud.visible is False

    viewport.set_point_cloud_mode("rgb")
    assert viewport.point_cloud_mode == "rgb"
    assert viewport.visual_cloud.visible is True


def test_layer_toggles(viewport):
    viewport.set_occupancy_visibility(occupied=True, free=True)
    assert viewport.show_occupied_voxels is True
    assert viewport.show_free_voxels is True

    viewport.set_vehicle_visible(False)
    assert viewport.visual_vehicle.visible is False

    viewport.set_bboxes_visible(False)
    assert viewport.visual_bboxes.visible is False

    viewport.set_clearance_line_visible(False)
    assert viewport.show_clearance_line is False


def test_obstacle_selection(viewport):
    first_obs_id = viewport.current_scene.obstacles[0]["obstacle_id"]
    viewport.select_obstacle(first_obs_id)
    assert viewport.selected_obstacle_id == first_obs_id
    assert viewport.visual_selected_bbox.visible is True
    assert viewport.visual_clearance_line.visible is True

    viewport.select_obstacle(None)
    assert viewport.selected_obstacle_id is None
    assert viewport.visual_selected_bbox.visible is False


def test_offscreen_rendering(viewport):
    img = viewport.render_offscreen()
    assert isinstance(img, np.ndarray)
    assert img.ndim == 3 and img.shape[2] == 4  # RGBA
    assert img.shape[0] > 0 and img.shape[1] > 0
    # Frame must not be completely black/transparent
    assert np.any(img[:, :, :3] > 0)

