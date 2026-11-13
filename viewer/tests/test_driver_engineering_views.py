"""
viewer/tests/test_driver_engineering_views.py — Tests for M7.2 Driver & Engineering Views
========================================================================================
"""

from pathlib import Path
import numpy as np
import pytest
from PySide6 import QtWidgets, QtCore, QtGui

from viewer.viewport import Clear360Viewport
from viewer.controls import ViewerControlPanel
from viewer.data_loader import SceneData

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


@pytest.fixture
def dummy_scene():
    """Create a mock SceneData with both textured and surface mesh."""
    v = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], dtype=np.float32)
    f = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
    uvs = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    tex = np.zeros((100, 100, 3), dtype=np.uint8)
    cols = np.ones((4, 4), dtype=np.float32)

    return SceneData(
        frame_id=0,
        source="ground_truth",
        points_rgb=np.zeros((10, 3), dtype=np.float32),
        colors_rgb=np.zeros((10, 3), dtype=np.float32),
        points_semantic=np.zeros((10, 3), dtype=np.float32),
        colors_semantic=np.zeros((10, 3), dtype=np.float32),
        occupancy_occupied_points=np.zeros((5, 3), dtype=np.float32),
        occupancy_free_points=np.zeros((5, 3), dtype=np.float32),
        obstacles=[],
        scene_metadata={},
        mesh_vertices=v,
        mesh_faces=f,
        mesh_colors=cols,
        mesh_normals=np.zeros((4, 3), dtype=np.float32),
        textured_mesh_vertices=v,
        textured_mesh_faces=f,
        textured_mesh_uvs=uvs,
        textured_mesh_texture=tex,
    )


def test_driver_view_visibility(qapp, dummy_scene):
    """Verify that Driver View mode hides all engineering clutter and shows textured mesh."""
    vp = Clear360Viewport()
    vp.set_scene_data(dummy_scene)

    vp.set_visualization_mode("driver")
    assert vp.vis_mode == "driver"
    assert vp.visual_textured_mesh.visible is True
    assert vp.visual_vehicle.visible is True

    # Engineering overlays must be hidden
    assert vp.visual_cloud.visible is False
    assert vp.visual_occ_occupied.visible is False
    assert vp.visual_occ_free.visible is False
    assert vp.visual_bboxes.visible is False
    assert vp.visual_clearance_line.visible is False
    assert vp.visual_grid.visible is False
    assert vp.visual_axes.visible is False
    assert vp.visual_envelope.visible is False


def test_engineering_view_visibility(qapp, dummy_scene):
    """Verify that Engineering View mode enables diagnostic layers."""
    vp = Clear360Viewport()
    vp.set_scene_data(dummy_scene)

    vp.set_visualization_mode("semantic_occupancy")
    assert vp.vis_mode == "semantic_occupancy"
    assert vp.visual_textured_mesh.visible is False
    assert vp.visual_occ_occupied.visible is True
    assert vp.visual_occ_free.visible is True
    assert vp.visual_bboxes.visible is True
    assert vp.visual_grid.visible is True


def test_control_panel_driver_mode_toggling(qapp):
    """Verify control panel disables layer toggles in Driver View."""
    cp = ViewerControlPanel()
    # Switch to Driver View (index 3)
    cp.combo_vis_mode.setCurrentIndex(3)
    assert cp.combo_vis_mode.currentData() == "driver"
    assert cp.grp_layers.isEnabled() is False

    # Switch back to Engineering View (index 2)
    cp.combo_vis_mode.setCurrentIndex(2)
    assert cp.combo_vis_mode.currentData() == "semantic_occupancy"
    assert cp.grp_layers.isEnabled() is True


def test_camera_presets(qapp):
    """Verify camera viewpoints in viewport."""
    vp = Clear360Viewport()
    for preset in ["chase", "front", "rear", "left", "right", "top"]:
        vp.set_camera_preset(preset)
        assert vp.view.camera is not None

