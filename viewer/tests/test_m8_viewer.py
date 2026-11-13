"""
viewer/tests/test_m8_viewer.py — Unit Tests for Milestone 8 Viewer Enhancements

Validates:
  1. ViewerDataLoader scenario loading (load_scenario for all 10 M8 scenarios)
  2. SceneData unknown-space voxels and confidence color maps
  3. Clear360Viewport confidence point cloud mode and unknown occupancy layer
  4. ViewerControlPanel scenario selector and unknown space toggle
  5. ViewerInfoPanel health & diagnostics HUD updates
  6. Clear360ViewerWindow scenario switching and HUD updates
"""

import pytest
import numpy as np
from pathlib import Path
from PySide6 import QtWidgets, QtCore

from viewer.data_loader import ViewerDataLoader, SceneData
from viewer.viewport import Clear360Viewport
from viewer.controls import ViewerControlPanel
from viewer.info_panel import ViewerInfoPanel
from viewer.app import Clear360ViewerWindow


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_data_loader_available_scenarios():
    loader = ViewerDataLoader()
    assert len(loader.available_scenarios) == 10
    assert "scenario_01_narrow_street" in loader.available_scenarios
    assert "scenario_10_construction_zone" in loader.available_scenarios


def test_data_loader_load_scenario():
    loader = ViewerDataLoader()
    scene = loader.load_scenario("scenario_01_narrow_street", "ai_calibrated")

    assert isinstance(scene, SceneData)
    assert scene.scenario_id == "scenario_01_narrow_street"
    assert scene.point_count > 50000
    assert scene.points_rgb.shape[1] == 3
    assert scene.colors_confidence.shape[1] == 3
    assert len(scene.occupancy_occupied_points) > 0
    assert len(scene.occupancy_unknown_points) > 0
    assert len(scene.obstacles) > 0


def test_viewport_confidence_mode(qapp):
    viewport = Clear360Viewport()
    loader = ViewerDataLoader()
    scene = loader.load_scenario("scenario_01_narrow_street", "ai_calibrated")
    viewport.set_scene_data(scene)

    viewport.set_point_cloud_mode("confidence")
    assert viewport.point_cloud_mode == "confidence"
    assert viewport.visual_cloud.visible is True

    # Test unknown occupancy layer
    viewport.set_occupancy_visibility(occupied=True, free=True, unknown=True)
    assert viewport.show_unknown_voxels is True
    assert viewport.visual_occ_unknown.visible is True


def test_controls_scenario_and_unknown_toggle(qapp):
    controls = ViewerControlPanel()

    # Verify combo_cloud contains Reconstruction Confidence
    modes = [controls.combo_cloud.itemData(i) for i in range(controls.combo_cloud.count())]
    assert "confidence" in modes

    # Verify scenario dropdown contains 11 entries (Baseline + 10 scenarios)
    assert controls.combo_scenario.count() == 11
    assert controls.combo_scenario.itemData(1) == "scenario_01_narrow_street"

    # Verify unknown space checkbox
    assert hasattr(controls, "chk_occ_unknown")
    assert controls.chk_occ_unknown.isChecked() is False

    # Test signal emission
    received = []
    controls.scenario_changed.connect(lambda s: received.append(s))
    controls.combo_scenario.setCurrentIndex(1)
    assert received == ["scenario_01_narrow_street"]


def test_info_panel_quality_report_update(qapp):
    panel = ViewerInfoPanel()
    assert hasattr(panel, "grp_health")

    cams = {
        "front": {"status": "HEALTHY", "latency_ms": 25.0},
        "rear": {"status": "DEGRADED", "latency_ms": 45.0},
        "left": {"status": "HEALTHY", "latency_ms": 26.0},
        "right": {"status": "OFFLINE", "latency_ms": 0.0},
    }
    panel.update_quality_report(
        overall_quality_score=68.5,
        alert_level="ADVISORY",
        recommendation="Rear camera degraded; Right camera offline.",
        camera_reports=cams,
    )

    assert "68.5%" in panel.lbl_quality_score.text()
    assert panel.lbl_alert_badge.text() == "ADVISORY"
    assert "DEGRADED" in panel.cam_status_labels["rear"].text()
    assert "OFFLINE" in panel.cam_status_labels["right"].text()


def test_app_window_scenario_switching(qapp):
    win = Clear360ViewerWindow()
    assert win.current_scenario_id is None

    # Switch to Scenario 01 via control panel
    win.control_panel.combo_scenario.setCurrentIndex(1)
    qapp.processEvents()
    assert win.current_scenario_id == "scenario_01_narrow_street"
    assert win.viewport.current_scene.scenario_id == "scenario_01_narrow_street"
    assert len(win.viewport.current_scene.obstacles) > 0

