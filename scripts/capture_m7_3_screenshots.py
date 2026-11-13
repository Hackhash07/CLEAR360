"""
scripts/capture_m7_3_screenshots.py — Capture High-Resolution M7.3 UI Screenshots
================================================================================

Renders and saves full desktop application window screenshots showing:
  1. MODE 2: Phone Photo 3D World with Session 001 (4-photo reconstruction, EXIF, quality ratings)
  2. MODE 2: Phone Photo 3D World with Session 003 (12-photo 360° surround reconstruction)
"""

from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from PySide6 import QtWidgets, QtCore, QtGui
from viewer.app import Clear360ViewerWindow

FIGURES_DIR = PROJECT_ROOT / "docs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def capture_phone_ui_screenshots():
    print("=================================================================")
    print("CLEAR360 — M7.3 UI Screenshot Capture")
    print("=================================================================")

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

    window = Clear360ViewerWindow()
    window.resize(1920, 1080)
    window.show()
    app.processEvents()
    time.sleep(1.0)
    app.processEvents()

    # 1. Switch to Mode 2: Phone Photo 3D World
    print("[1] Switching to MODE 2: Phone Photo 3D World...")
    window._set_app_mode("phone")
    app.processEvents()
    time.sleep(0.5)
    app.processEvents()

    # 2. Load Session 001 (4-Photo Reconstruction)
    print("[2] Loading Session 001 in Phone Panel...")
    window._on_phone_session_requested("session_001")
    app.processEvents()
    time.sleep(1.0)
    app.processEvents()

    pix1 = window.grab()
    out1 = FIGURES_DIR / "m7_3_viewer_phone_mode_session_001.png"
    pix1.save(str(out1))
    print(f"    Saved: {out1}")

    # 3. Load Session 003 (12-Photo 360° Scalability Reconstruction)
    print("[3] Loading Session 003 in Phone Panel...")
    window._on_phone_session_requested("session_003")
    app.processEvents()
    time.sleep(1.0)
    app.processEvents()

    pix2 = window.grab()
    out2 = FIGURES_DIR / "m7_3_viewer_phone_mode_session_003.png"
    pix2.save(str(out2))
    print(f"    Saved: {out2}")

    print("=================================================================")
    print("M7.3 SCREENSHOTS CAPTURED SUCCESSFULLY!")
    print("=================================================================")


if __name__ == "__main__":
    capture_phone_ui_screenshots()

