"""
obstacles/tests/test_vehicle_clearance.py — Unit Tests for Vehicle Clearance
============================================================================

Tests:
  - Vehicle footprint definition (Hyundai Creta body + side mirrors)
  - Clearance calculation is NOT distance from vehicle origin (0, 0, 0)
  - Longitudinal clearance from front bumper (Y = +2.15 m) and rear bumper (Y = -2.15 m)
  - Lateral clearance from wing mirror tips (X = +/- 1.05 m)
  - Zero clearance for points touching or inside the vehicle envelope
"""

import numpy as np
import pytest

from obstacles.clustering import VehicleFootprint


@pytest.fixture
def footprint():
    return VehicleFootprint()


def test_clearance_not_origin_distance(footprint):
    """
    CRITICAL REQUIREMENT: Clearance must be calculated from outer vehicle geometry,
    NEVER from the vehicle origin (0, 0, 0).
    """
    # Obstacle directly ahead at Y = 3.0 m, X = 0.0 m, Z = 0.5 m
    # Distance from origin (0, 0, 0) is 3.0 m
    # But distance from front bumper (Y = +2.15 m) is exactly 3.0 - 2.15 = 0.85 m!
    pts_ahead = np.array([[0.0, 3.0, 0.5]])
    clearance = footprint.compute_clearance(pts_ahead)

    assert not np.isclose(clearance, 3.0, atol=0.10), "Clearance must NOT equal distance from origin!"
    assert np.isclose(clearance, 0.85, atol=0.01), f"Expected 0.85m front clearance, got {clearance:.4f}m"


def test_lateral_clearance_accounts_for_mirrors(footprint):
    """
    Verify that lateral clearance accounts for the side mirrors extending to X = +/- 1.05 m,
    not just the 1.79m body width (X = +/- 0.895 m).
    """
    # Obstacle alongside mirror at X = -1.50 m, Y = 0.40 m, Z = 1.0 m
    # Distance from vehicle origin is 1.50 m
    # Distance from body panel (X = -0.895) would be 1.50 - 0.895 = 0.605 m
    # But distance from left mirror tip (X = -1.05 m) is exactly 1.50 - 1.05 = 0.45 m!
    pts_mirror = np.array([[-1.50, 0.40, 1.00]])
    clearance = footprint.compute_clearance(pts_mirror)

    assert not np.isclose(clearance, 0.605, atol=0.02), "Clearance failed to account for mirror extension!"
    assert np.isclose(clearance, 0.45, atol=0.01), f"Expected 0.45m mirror clearance, got {clearance:.4f}m"


def test_rear_bumper_clearance(footprint):
    """Verify clearance from rear bumper (Y = -2.15 m)."""
    # Obstacle behind vehicle at Y = -3.15 m
    pts_behind = np.array([[0.0, -3.15, 0.5]])
    clearance = footprint.compute_clearance(pts_behind)
    assert np.isclose(clearance, 1.00, atol=0.01)


def test_penetrating_points_zero_clearance(footprint):
    """Verify points inside the vehicle envelope yield 0.0 m clearance."""
    pts_inside = np.array([[0.0, 0.0, 0.5]]) # inside vehicle cabin
    clearance = footprint.compute_clearance(pts_inside)
    assert clearance == 0.0

