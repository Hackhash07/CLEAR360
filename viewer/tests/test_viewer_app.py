"""
viewer/tests/test_viewer_app.py — Integration Tests for Main Application Window
"""

import pytest
from PySide6 import QtWidgets, QtCore, QtGui
from viewer.app import Clear360ViewerWindow


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def viewer_window(qapp):
    win = Clear360ViewerWindow()
    win.resize(1280, 800)
    win.show()
    qapp.processEvents()
    yield win
    win.close()


def test_app_window_initial_state(viewer_window):
    assert viewer_window.current_frame_id == 0
    assert viewer_window.current_source == "ground_truth"
    assert viewer_window.viewport.current_scene is not None
    assert viewer_window.info_panel.obstacles is not None
    assert len(viewer_window.info_panel.obstacles) > 0


def test_frame_switching(viewer_window, qapp):
    viewer_window.control_panel.combo_frame.setCurrentIndex(1)  # Frame 5
    qapp.processEvents()
    assert viewer_window.current_frame_id == 5
    assert viewer_window.viewport.current_scene.frame_id == 5

    viewer_window.control_panel.combo_frame.setCurrentIndex(2)  # Frame 10
    qapp.processEvents()
    assert viewer_window.current_frame_id == 10
    assert viewer_window.viewport.current_scene.frame_id == 10


def test_source_switching(viewer_window, qapp):
    # Switch to Calibrated AI
    viewer_window.control_panel.combo_source.setCurrentIndex(1)
    qapp.processEvents()
    assert viewer_window.current_source == "ai_calibrated"
    assert viewer_window.viewport.current_scene.source == "ai_calibrated"

    # Switch to Raw AI
    viewer_window.control_panel.combo_source.setCurrentIndex(2)
    qapp.processEvents()
    assert viewer_window.current_source == "ai_raw"

    # Switch back to Ground Truth
    viewer_window.control_panel.combo_source.setCurrentIndex(0)
    qapp.processEvents()
    assert viewer_window.current_source == "ground_truth"


def test_obstacle_table_selection(viewer_window, qapp):
    table = viewer_window.info_panel.table_obs
    assert table.rowCount() > 0

    # Select first obstacle row
    table.selectRow(0)
    qapp.processEvents()

    selected_id = viewer_window.info_panel.selected_obstacle_id
    assert selected_id is not None
    assert viewer_window.viewport.selected_obstacle_id == selected_id


def test_window_grab_screenshot(viewer_window, qapp):
    qapp.processEvents()
    pixmap = viewer_window.grab()
    assert not pixmap.isNull()
    assert pixmap.width() >= 1280
    assert pixmap.height() >= 800


@pytest.mark.parametrize(
    ("loaded_mode", "mesh_path"),
    [
        ("textured", "scene_mesh.obj"),
        ("mesh", "scene_mesh.obj"),
        ("cloud", "scene_geometry_preview.ply"),
    ],
)
def test_phone_mesh_loaded_uses_actual_viewport_mode(
    viewer_window, monkeypatch, loaded_mode, mesh_path
):
    selected_modes = []
    monkeypatch.setattr(
        viewer_window.viewport,
        "load_external_mesh",
        lambda _path: loaded_mode,
    )
    monkeypatch.setattr(
        viewer_window.phone_panel,
        "set_display_mode",
        selected_modes.append,
    )

    viewer_window._on_phone_mesh_loaded("session-test", mesh_path)

    assert selected_modes == [loaded_mode]
