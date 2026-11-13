"""
CLEAR360 – Calibration Tests
==============================
Tests for CameraIntrinsics, CameraModel, and CalibrationStore.

Coordinate conventions
----------------------
World frame : X=RIGHT, Y=FORWARD, Z=UP  (vehicle at origin).
Camera frame: x=right-in-image, y=down-in-image, z=forward-into-scene.
T_cam_world : p_cam = T_cam_world @ p_world_hom

Unprojection formula:
    x_cam = (u - cx) * d / fx
    y_cam = (v - cy) * d / fy
    z_cam = d
    p_world = inv(T_cam_world) @ [x_cam, y_cam, z_cam, 1.0]
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

from pipeline.calibration import (
    CalibrationStore,
    CameraIntrinsics,
    CameraModel,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "cameras.yaml"

# Standard intrinsics matching the CLEAR360 camera rig (120° HFOV approx)
_INTRINSICS = CameraIntrinsics(
    width=1280,
    height=720,
    fx=369.4,
    fy=369.4,
    cx=640.0,
    cy=360.0,
    focal_length_mm=10.39,
    sensor_width_mm=36.0,
)


def _identity_camera(name: str = "test") -> CameraModel:
    """Return a CameraModel with identity T_cam_world for simple geometry tests."""
    T = np.eye(4, dtype=np.float64)
    # position_world = -R.T @ t = 0 for identity
    pos = np.zeros(3, dtype=np.float64)
    return CameraModel(
        name=name,
        intrinsics=_INTRINSICS,
        T_cam_world=T,
        position_world=pos,
    )


# ---------------------------------------------------------------------------
# 1. CameraIntrinsics – K matrix
# ---------------------------------------------------------------------------


def test_intrinsics_K_matrix() -> None:
    """K() must return the correct 3×3 intrinsic matrix."""
    intr = CameraIntrinsics(
        width=1280,
        height=720,
        fx=369.4,
        fy=369.4,
        cx=640.0,
        cy=360.0,
        focal_length_mm=10.39,
        sensor_width_mm=36.0,
    )
    K = intr.K()

    expected = np.array(
        [[369.4, 0.0, 640.0],
         [0.0, 369.4, 360.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    assert K.shape == (3, 3), f"K shape {K.shape} ≠ (3,3)"
    assert K.dtype == np.float64, f"K dtype {K.dtype} ≠ float64"
    npt.assert_allclose(K, expected, rtol=1e-10, atol=1e-10)


# ---------------------------------------------------------------------------
# 2. CameraIntrinsics – HFOV
# ---------------------------------------------------------------------------


def test_intrinsics_hfov() -> None:
    """HFOV for fx=369.4, width=1280 should be ≈ 120° (±1°)."""
    intr = CameraIntrinsics(
        width=1280,
        height=720,
        fx=369.4,
        fy=369.4,
        cx=640.0,
        cy=360.0,
        focal_length_mm=10.39,
        sensor_width_mm=36.0,
    )
    hfov = intr.hfov_deg()
    assert abs(hfov - 120.0) <= 1.0, (
        f"HFOV {hfov:.3f}° is not within ±1° of 120°"
    )


# ---------------------------------------------------------------------------
# 3. CameraModel – unproject centre pixel with identity extrinsics
# ---------------------------------------------------------------------------


def test_camera_model_unproject_centre() -> None:
    """
    Unproject the centre pixel (640, 360) with depth 5.0 m.

    For identity T_cam_world (camera frame == world frame):
        x_cam = (640 - 640) * 5 / 369.4 = 0
        y_cam = (360 - 360) * 5 / 369.4 = 0
        z_cam = 5.0
    → world point should be (0, 0, 5).
    """
    cam = _identity_camera()
    p_world = cam.unproject(u=640.0, v=360.0, depth=5.0)

    npt.assert_allclose(p_world, [0.0, 0.0, 5.0], atol=1e-9)
    assert p_world.shape == (3,)


# ---------------------------------------------------------------------------
# 4. CameraModel – project → unproject round-trip
# ---------------------------------------------------------------------------


def test_camera_model_project_roundtrip() -> None:
    """
    project a 3-D world point, then unproject the resulting (u, v, depth)
    back to 3-D; the two world points must agree within 1e-4 m.
    """
    cam = _identity_camera()

    # Choose a point clearly in front of the camera
    p_original = np.array([0.5, 0.3, 4.0], dtype=np.float64)

    u, v, depth = cam.project(p_original)
    assert not (np.isnan(u) or np.isnan(v) or np.isnan(depth)), (
        "project() returned NaN for a point in front of the camera"
    )

    p_recovered = cam.unproject(u, v, depth)
    dist = float(np.linalg.norm(p_recovered - p_original))
    assert dist < 1e-4, (
        f"Round-trip distance {dist:.2e} m exceeds 1e-4 m tolerance"
    )


# ---------------------------------------------------------------------------
# 5. CalibrationStore – load cameras.yaml
# ---------------------------------------------------------------------------


def test_calibration_store_load() -> None:
    """
    Load config/cameras.yaml and verify:
      - 4 cameras loaded (front, rear, left, right)
      - Each has width=1280, height=720
      - T_cam_world is shape (4,4) float64 with no NaN
    """
    if not CONFIG_PATH.exists():
        pytest.skip(f"cameras.yaml not found at {CONFIG_PATH}")

    store = CalibrationStore(CONFIG_PATH)
    store.load()

    assert len(store.all_cameras()) == 4, (
        f"Expected 4 cameras, got {len(store.all_cameras())}"
    )

    for name in ["front", "rear", "left", "right"]:
        cam = store.get(name)
        assert cam.intrinsics.width == 1280, f"[{name}] width ≠ 1280"
        assert cam.intrinsics.height == 720, f"[{name}] height ≠ 720"

        T = cam.T_cam_world
        assert T.shape == (4, 4), f"[{name}] T_cam_world shape {T.shape} ≠ (4,4)"
        assert T.dtype == np.float64, f"[{name}] T_cam_world dtype {T.dtype} ≠ float64"
        assert not np.any(np.isnan(T)), f"[{name}] T_cam_world contains NaN"


# ---------------------------------------------------------------------------
# 6. CalibrationStore – round-trip save/load
# ---------------------------------------------------------------------------


def test_calibration_store_roundtrip() -> None:
    """
    Load cameras.yaml → save to a tmp file → reload → compare T_cam_world
    values within 1e-9 tolerance.
    """
    if not CONFIG_PATH.exists():
        pytest.skip(f"cameras.yaml not found at {CONFIG_PATH}")

    store1 = CalibrationStore(CONFIG_PATH)
    store1.load()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "cameras_roundtrip.yaml"
        store1.save(tmp_path)

        store2 = CalibrationStore(tmp_path)
        store2.load()

        for name in ["front", "rear", "left", "right"]:
            T1 = store1.get(name).T_cam_world
            T2 = store2.get(name).T_cam_world
            npt.assert_allclose(
                T1, T2,
                atol=1e-9,
                rtol=0,
                err_msg=f"[{name}] T_cam_world differs after save/reload",
            )
