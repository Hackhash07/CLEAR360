"""
scripts/capture_m7_1_screenshots.py — Capture Authentic Screenshots for M7.1
=============================================================================

Renders the live CLEAR360 M7.1 application across multiple viewpoints, pipelines,
and visualization modes, saving crisp, publication-quality images to docs/figures/:

  1. docs/figures/m7_1_photoreal_chase.png        — Default Chase 3D (Ground Truth surface)
  2. docs/figures/m7_1_photoreal_front.png        — Front low-angle view (Creta grille, road, walls)
  3. docs/figures/m7_1_photoreal_flank.png        — Left flank perspective (body panels, wheels, kerb)
  4. docs/figures/m7_1_ai_surface_mesh.png        — Calibrated AI Surface Mesh (Depth Anything V2)
  5. docs/figures/m7_1_pointcloud_debug.png       — Point Cloud Debug Mode (RGB points)
  6. docs/figures/m7_1_semantic_occupancy.png     — Semantic / Occupancy Grid Mode
  7. docs/figures/m7_1_scenario_01_narrow.png     — M8 Scenario 01 (Narrow Street) Surface Mesh
"""

from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from PySide6 import QtWidgets, QtCore, QtGui

from viewer.app import Clear360ViewerWindow

FIGURES_DIR = PROJECT_ROOT / "docs" / "figures"


def save_screenshot(window: Clear360ViewerWindow, filename: str) -> Path:
    """Capture full window or viewport and save as PNG."""
    QtWidgets.QApplication.processEvents()
    time.sleep(0.1)
    QtWidgets.QApplication.processEvents()

    # Capture whole application window with QWidget.grab()
    pixmap = window.grab()
    out_path = FIGURES_DIR / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pixmap.save(str(out_path), "PNG")
    print(f"Captured: {out_path} ({pixmap.width()}x{pixmap.height()})")
    return out_path


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    window = Clear360ViewerWindow()
    window.resize(1500, 950)
    window.show()
    QtWidgets.QApplication.processEvents()

    # 1. Default Chase 3D View (Ground Truth Surface Mesh)
    window.control_panel.combo_vis_mode.setCurrentIndex(0)  # "surface"
    window.control_panel.combo_source.setCurrentIndex(0)    # "ground_truth"
    window.control_panel.combo_frame.setCurrentIndex(0)     # Frame 0000
    window.viewport.set_camera_preset("chase")
    save_screenshot(window, "m7_1_photoreal_chase.png")

    # 2. Front low-angle viewpoint
    window.viewport.set_camera_preset("front")
    # Tweak elevation slightly for dynamic low-angle perspective
    window.viewport.view.camera.elevation = 12.0
    window.viewport.view.camera.distance = 6.8
    save_screenshot(window, "m7_1_photoreal_front.png")

    # 3. Left flank viewpoint
    window.viewport.set_camera_preset("left")
    window.viewport.view.camera.elevation = 14.0
    window.viewport.view.camera.distance = 7.5
    save_screenshot(window, "m7_1_photoreal_flank.png")

    # 4. Calibrated AI Surface Mesh (Depth Anything V2)
    window.viewport.set_camera_preset("chase")
    window.control_panel.combo_source.setCurrentIndex(1)  # "ai_calibrated"
    QtWidgets.QApplication.processEvents()
    time.sleep(0.15)
    save_screenshot(window, "m7_1_ai_surface_mesh.png")

    # 5. Point Cloud Debug Mode
    window.control_panel.combo_vis_mode.setCurrentIndex(1)  # "pointcloud"
    window.control_panel.combo_cloud.setCurrentIndex(0)     # RGB
    window.viewport.set_camera_preset("chase")
    QtWidgets.QApplication.processEvents()
    time.sleep(0.15)
    save_screenshot(window, "m7_1_pointcloud_debug.png")

    # 6. Semantic / Occupancy Grid Mode
    window.control_panel.combo_vis_mode.setCurrentIndex(2)  # "semantic_occupancy"
    window.viewport.set_camera_preset("chase")
    QtWidgets.QApplication.processEvents()
    time.sleep(0.15)
    save_screenshot(window, "m7_1_semantic_occupancy.png")

    # 7. Scenario 01: Narrow Street (Ground Truth Surface Mesh)
    window.control_panel.combo_vis_mode.setCurrentIndex(0)  # "surface"
    window.control_panel.combo_source.setCurrentIndex(0)    # "ground_truth"
    window.control_panel.combo_scenario.setCurrentIndex(1)  # Scenario 01
    window.viewport.set_camera_preset("chase")
    QtWidgets.QApplication.processEvents()
    time.sleep(0.2)
    save_screenshot(window, "m7_1_scenario_01_narrow.png")

    print("\n[Done] All 7 authentic screenshots generated successfully.")


if __name__ == "__main__":
    main()

