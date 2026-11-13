"""
viewer/tests/test_viewport_modes.py — Viewport & Control Panel Integration Tests
================================================================================
"""

from pathlib import Path
import numpy as np
import pytest
from PySide6 import QtWidgets, QtCore

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


def test_viewport_initialization(qapp):
    """Verify Clear360Viewport default state and camera presets."""
    vp = Clear360Viewport()
    assert vp.vis_mode == "surface"
    assert vp.show_vehicle is True
    assert vp.view.camera.fov == 50.0

    # Camera presets
    assert "chase" in vp.CAMERA_PRESETS
    assert "front" in vp.CAMERA_PRESETS
    assert "rear" in vp.CAMERA_PRESETS
    assert "left" in vp.CAMERA_PRESETS
    assert "right" in vp.CAMERA_PRESETS
    assert "top" in vp.CAMERA_PRESETS

    vp.set_camera_preset("front")
    assert vp.view.camera.azimuth == 90.0


def test_viewport_mode_switching(qapp):
    """Verify switching between 3 primary visualization modes."""
    vp = Clear360Viewport()

    # Mode 1: Surface
    vp.set_visualization_mode("surface")
    assert vp.vis_mode == "surface"

    # Mode 2: Point cloud
    vp.set_visualization_mode("pointcloud")
    assert vp.vis_mode == "pointcloud"
    assert vp.visual_surface_mesh.visible is False

    # Mode 3: Semantic occupancy
    vp.set_visualization_mode("semantic_occupancy")
    assert vp.vis_mode == "semantic_occupancy"
    assert vp.visual_surface_mesh.visible is False


def test_control_panel_signals(qapp):
    """Verify control panel emissions for modes and sources."""
    cp = ViewerControlPanel()
    received_modes = []
    received_sources = []

    cp.vis_mode_changed.connect(received_modes.append)
    cp.source_changed.connect(received_sources.append)

    cp.combo_vis_mode.setCurrentIndex(1)  # pointcloud
    assert len(received_modes) == 1
    assert received_modes[0] == "pointcloud"

    cp.combo_source.setCurrentIndex(1)  # ai_calibrated
    assert len(received_sources) == 1
    assert received_sources[0] == "ai_calibrated"
    assert "CALIBRATED AI" in cp.lbl_pipeline_badge.text()

