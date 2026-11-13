"""
fusion/tests/test_bounding_box_metrics.py — Unit Tests for Geometric Integrity & Bounding Box
=============================================================================================

Tests:
  - Road profile flatness and slope fitting
  - Wall distance and verticality evaluation
  - Sky leakage and peripheral boundary expansion quantification
  - Bounding box compliance with physical environment geometry
"""

import numpy as np
import pytest

from fusion.distortion import GeometricDistortionAnalyzer, RoadProfileStats, SkyLeakageStats


def test_road_profile_flat_plane():
    """Verify that a flat horizontal plane yields slope ~0 deg and near-zero RMSE."""
    X = np.linspace(-5, 5, 50)
    Y = np.linspace(-5, 5, 50)
    xx, yy = np.meshgrid(X, Y)
    zz = np.zeros_like(xx) + 0.02 # 2 cm height above nominal

    pts = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    stats = GeometricDistortionAnalyzer.analyze_road_profile(pts)

    assert isinstance(stats, RoadProfileStats)
    assert np.isclose(stats.mean_elevation_z, 0.02, atol=0.005)
    assert stats.slope_angle_deg < 0.10
    assert stats.planarity_rmse_m < 0.005


def test_road_profile_tilted_plane():
    """Verify that a tilted plane's slope angle is accurately measured."""
    X = np.linspace(-5, 5, 50)
    Y = np.linspace(-5, 5, 50)
    xx, yy = np.meshgrid(X, Y)
    # 5 degree slope along Y: z = Y * tan(5 deg)
    slope_rad = np.radians(5.0)
    zz = yy * np.tan(slope_rad)

    pts = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    stats = GeometricDistortionAnalyzer.analyze_road_profile(pts)

    assert np.isclose(stats.slope_angle_deg, 5.0, atol=0.10)
    assert stats.planarity_rmse_m < 0.005


def test_sky_leakage_quantification():
    """Verify that points above wall height (Z > 2.55 m) are correctly counted as sky leakage."""
    pts_real = np.random.uniform(low=[-9, -9, 0], high=[9, 9, 2.5], size=(1000, 3))
    pts_sky = np.random.uniform(low=[-15, -15, 2.6], high=[15, 15, 6.0], size=(400, 3))
    pts_sub = np.random.uniform(low=[-5, -5, -0.5], high=[5, 5, -0.2], size=(100, 3))

    all_pts = np.vstack([pts_real, pts_sky, pts_sub])
    stats = GeometricDistortionAnalyzer.analyze_sky_leakage(all_pts)

    assert isinstance(stats, SkyLeakageStats)
    assert stats.total_points == 1500
    assert stats.sky_leakage_points == 400
    assert np.isclose(stats.sky_leakage_pct, 400 / 1500 * 100.0, atol=0.1)
    assert stats.sub_ground_points == 100
    assert np.isclose(stats.sub_ground_pct, 100 / 1500 * 100.0, atol=0.1)

