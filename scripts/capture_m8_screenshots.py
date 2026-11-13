"""
scripts/capture_m8_screenshots.py — Capture M8 Viewer Screenshots

Captures high-resolution application screenshots of:
  1. Unknown-Space Volumetric Occupancy Grid Visualization
  2. Point Cloud Colored by Reconstruction Confidence (Green/Amber/Red)
  3. Sensor Health Monitoring & Scene Quality Diagnostics HUD
  4. Out-of-Distribution Scenario 01 (Narrow Street) Reconstruction
  5. Out-of-Distribution Scenario 10 (Urban Construction Zone) Reconstruction
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PySide6 import QtWidgets, QtCore, QtGui
from viewer.app import Clear360ViewerWindow


def capture_m8_screenshots():
    out_dir = ROOT / "docs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = Clear360ViewerWindow()
    win.resize(1440, 900)
    win.show()
    app.processEvents()
    time.sleep(0.5)

    print("=== CLEAR360 M8: Capturing Application Screenshots ===")

    # 1. Unknown-Space Visualization (Occupied in red, Free in cyan, Unknown in slate gray)
    win.control_panel.combo_source.setCurrentIndex(1)  # Calibrated AI
    win.control_panel.combo_frame.setCurrentIndex(0)   # Frame 0
    win.control_panel.combo_cloud.setCurrentIndex(3)   # Hidden
    win.control_panel.chk_occ_occupied.setChecked(True)
    win.control_panel.chk_occ_free.setChecked(True)
    win.control_panel.chk_occ_unknown.setChecked(True)
    win.viewport.set_occupancy_visibility(occupied=True, free=True, unknown=True)
    win.viewport.set_camera_preset("chase")
    app.processEvents()
    time.sleep(0.4)
    p1 = win.grab()
    path1 = out_dir / "m8_viewer_unknown_space.png"
    p1.save(str(path1))
    print(f"Saved: {path1} ({p1.width()}x{p1.height()})")

    # 2. Reconstruction Confidence Coloring Mode
    win.control_panel.chk_occ_occupied.setChecked(False)
    win.control_panel.chk_occ_free.setChecked(False)
    win.control_panel.chk_occ_unknown.setChecked(False)
    win.viewport.set_occupancy_visibility(occupied=False, free=False, unknown=False)
    win.control_panel.combo_cloud.setCurrentIndex(2)   # Reconstruction Confidence
    win.viewport.set_point_cloud_mode("confidence")
    win.viewport.set_camera_preset("chase")
    app.processEvents()
    time.sleep(0.4)
    p2 = win.grab()
    path2 = out_dir / "m8_viewer_confidence_mode.png"
    p2.save(str(path2))
    print(f"Saved: {path2} ({p2.width()}x{p2.height()})")

    # 3. Camera Health & Diagnostics HUD View
    health_data = {
        "front": {"status": "HEALTHY", "latency_ms": 28.4},
        "rear": {"status": "DEGRADED", "latency_ms": 32.1},
        "left": {"status": "HEALTHY", "latency_ms": 29.5},
        "right": {"status": "HEALTHY", "latency_ms": 30.2},
    }
    win.info_panel.update_quality_report(
        88.5,
        "ADVISORY",
        "Minor optical glare detected on rear camera. Proceed with caution.",
        health_data,
    )
    app.processEvents()
    time.sleep(0.3)
    p3 = win.grab()
    path3 = out_dir / "m8_viewer_camera_health_hud.png"
    p3.save(str(path3))
    print(f"Saved: {path3} ({p3.width()}x{p3.height()})")

    # 4. Out-of-Distribution Scenario 01: Narrow Indian-Style Urban Street
    win.control_panel.combo_cloud.setCurrentIndex(0)   # Photorealistic RGB
    win.viewport.set_point_cloud_mode("rgb")
    win.control_panel.combo_scenario.setCurrentIndex(1)  # scenario_01_narrow_street
    win.viewport.set_camera_preset("chase")
    app.processEvents()
    time.sleep(0.5)
    p4 = win.grab()
    path4 = out_dir / "m8_scenario_narrow_street.png"
    p4.save(str(path4))
    print(f"Saved: {path4} ({p4.width()}x{p4.height()})")

    # 5. Out-of-Distribution Scenario 10: Urban Roadwork & Construction Zone
    win.control_panel.combo_scenario.setCurrentIndex(10)  # scenario_10_construction_zone
    win.viewport.set_camera_preset("chase")
    app.processEvents()
    time.sleep(0.5)
    p5 = win.grab()
    path5 = out_dir / "m8_scenario_construction_zone.png"
    p5.save(str(path5))
    print(f"Saved: {path5} ({p5.width()}x{p5.height()})")

    print("=== M8 Screenshots Capture Complete ===")


if __name__ == "__main__":
    capture_m8_screenshots()

