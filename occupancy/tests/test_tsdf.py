"""
occupancy/tests/test_tsdf.py — Unit Tests for 3D TSDF Volume
============================================================

Tests:
  - TSDF volume initialization and memory allocation
  - Integration of depth measurements
  - Zero-crossing surface extraction
"""

from pathlib import Path
import numpy as np
import pytest

from occupancy.tsdf import TSDFVolume3D
from pipeline.calibration import CalibrationStore


@pytest.fixture
def calib_store():
    p = Path("config/cameras.yaml")
    store = CalibrationStore(p)
    store.load()
    return store


@pytest.fixture
def tsdf_volume():
    return TSDFVolume3D(
        bounds_x=(-3.0, 3.0),
        bounds_y=(1.0, 7.0),
        bounds_z=(0.0, 2.0),
        voxel_size=0.10,
        truncation_margin=0.25,
    )


def test_tsdf_initialization(tsdf_volume):
    """Verify TSDF volume initialization values."""
    assert tsdf_volume.total_voxels == tsdf_volume.nx * tsdf_volume.ny * tsdf_volume.nz
    # Initially tsdf should be 1.0 (empty) and weight 0.0
    assert np.all(tsdf_volume.tsdf == 1.0)
    assert np.all(tsdf_volume.weight == 0.0)


def test_tsdf_integration_and_zero_crossing(tsdf_volume, calib_store):
    """Verify TSDF surface zero-crossing extraction for a planar depth map."""
    cam_model = calib_store.get("front") # camera at Y=2.15, looking +Y
    H, W = 720, 1280
    # Flat wall 3.0 m ahead of front camera (Y ~ 5.15 m)
    depth = np.full((H, W), 3.0, dtype=np.float32)

    tsdf_volume.integrate_view(depth, cam_model, weight=1.0)

    # Some weights should be positive
    assert np.any(tsdf_volume.weight > 0)

    # Surface zero-crossing points should lie near Y = 2.15 + 3.0 * cos(-8 deg) ~ 5.12 m
    surf_pts = tsdf_volume.extract_zero_crossing_points(min_weight=1.0)
    assert len(surf_pts) > 0
    mean_y = np.mean(surf_pts[:, 1])
    assert np.isclose(mean_y, 2.15 + 3.0 * np.cos(np.radians(-8.0)), atol=0.25)

