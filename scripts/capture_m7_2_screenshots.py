"""
scripts/capture_m7_2_screenshots.py — Capture High-Resolution M7.2 Deliverables

Renders and saves authoritative screenshots of the CLEAR360 viewer in both
Driver View (Photorealistic textured mesh + Creta SUV) and Engineering View
across multiple viewpoints and both GT and AI pipelines.
"""

from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from PySide6 import QtWidgets, QtCore, QtGui
import cv2
import numpy as np

from viewer.app import Clear360ViewerWindow

FIGURES_DIR = PROJECT_ROOT / "docs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def capture_screenshots():
    print("=================================================================")
    print("CLEAR360 — M7.2 Screenshot Capture")
    print("=================================================================")

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

    window = Clear360ViewerWindow()
    window.resize(1920, 1080)
    window.show()
    app.processEvents()
    time.sleep(1.0)
    app.processEvents()

    # 1. Driver View — Chase 3D Perspective (Ground Truth)
    print("[1] Capturing Driver View (Chase 3D)...")
    window.control_panel.combo_vis_mode.setCurrentIndex(3)  # driver
    window.viewport.set_camera_preset("chase")
    app.processEvents()
    time.sleep(0.5)
    app.processEvents()

    pix = window.grab()
    pix.save(str(FIGURES_DIR / "m7_2_driver_view_chase.png"))
    print(f"    Saved: {FIGURES_DIR / 'm7_2_driver_view_chase.png'}")

    # 2. Driver View — Front Perspective
    print("[2] Capturing Driver View (Front)...")
    window.viewport.set_camera_preset("front")
    app.processEvents()
    time.sleep(0.5)
    app.processEvents()

    pix = window.grab()
    pix.save(str(FIGURES_DIR / "m7_2_driver_view_front.png"))
    print(f"    Saved: {FIGURES_DIR / 'm7_2_driver_view_front.png'}")

    # 3. Driver View — Top Perspective
    print("[3] Capturing Driver View (Top Perspective)...")
    window.viewport.set_camera_preset("top")
    app.processEvents()
    time.sleep(0.5)
    app.processEvents()

    pix = window.grab()
    pix.save(str(FIGURES_DIR / "m7_2_driver_view_top.png"))
    print(f"    Saved: {FIGURES_DIR / 'm7_2_driver_view_top.png'}")

    # 4. Engineering View — Full telemetry (Point cloud, Occupancy, BBoxes, Clearance)
    print("[4] Capturing Engineering View (Occupancy & Telemetry)...")
    window.control_panel.combo_vis_mode.setCurrentIndex(2)  # semantic_occupancy
    window.viewport.set_camera_preset("chase")
    window.control_panel.chk_occ_occupied.setChecked(True)
    window.control_panel.chk_occ_free.setChecked(True)
    app.processEvents()
    time.sleep(0.5)
    app.processEvents()

    pix = window.grab()
    pix.save(str(FIGURES_DIR / "m7_2_engineering_view.png"))
    print(f"    Saved: {FIGURES_DIR / 'm7_2_engineering_view.png'}")

    # 5. Calibrated AI Pipeline Reconstruction (Driver View)
    print("[5] Capturing Calibrated AI Pipeline (Driver View)...")
    window.control_panel.combo_source.setCurrentIndex(1)  # ai_calibrated
    window.control_panel.combo_vis_mode.setCurrentIndex(3)  # driver
    window.viewport.set_camera_preset("chase")
    app.processEvents()
    time.sleep(0.5)
    app.processEvents()

    pix = window.grab()
    pix.save(str(FIGURES_DIR / "m7_2_ai_calibrated_reconstruction.png"))
    print(f"    Saved: {FIGURES_DIR / 'm7_2_ai_calibrated_reconstruction.png'}")

    window.close()
    print("\nAll M7.2 Screenshots Captured Successfully!")


if __name__ == "__main__":
    capture_screenshots()

