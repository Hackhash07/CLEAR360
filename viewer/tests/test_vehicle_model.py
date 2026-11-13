"""
viewer/tests/test_vehicle_model.py — Unit Tests for 3D Vehicle Model & Clearance Math
"""

import pytest
import numpy as np
from viewer.vehicle_model import VehicleModel3D


@pytest.fixture
def vehicle_model():
    return VehicleModel3D()


def test_vehicle_geometry_dimensions(vehicle_model):
    # Length = 4.30 m, Width = 1.79 m, Height = 1.635 m
    assert np.isclose(vehicle_model.half_length * 2.0, 4.30)
    assert np.isclose(vehicle_model.half_width * 2.0, 1.79)
    assert np.isclose(vehicle_model.height, 1.635)
    # Wing mirror tip extension = 1.05 m each side -> total 2.10 m
    assert np.isclose(vehicle_model.mirror_tip_x, 1.05)


def test_vehicle_mesh_generation(vehicle_model):
    verts, faces, colors = vehicle_model.get_mesh_data()
    assert verts.ndim == 2 and verts.shape[1] == 3
    assert faces.ndim == 2 and faces.shape[1] == 3
    assert colors.ndim == 2 and colors.shape[1] == 4
    # All face indices must be valid vertex indices
    assert np.max(faces) < len(verts)
    assert len(verts) > 50


def test_clearance_envelope_lines(vehicle_model):
    pos, colors = vehicle_model.get_clearance_envelope_lines()
    assert pos.ndim == 2 and pos.shape[1] == 3
    # connect='segments' requires even number of points (pairs of endpoints)
    assert len(pos) % 2 == 0
    # Must enclose body and mirror boundaries
    assert np.min(pos[:, 0]) <= -1.05
    assert np.max(pos[:, 0]) >= +1.05
    assert np.min(pos[:, 1]) <= -2.15
    assert np.max(pos[:, 1]) >= +2.15


def test_axes_lines(vehicle_model):
    pts, cols = vehicle_model.get_axes_lines(length=2.0)
    assert len(pts) == 6  # 3 segments (origin to +X, +Y, +Z)
    assert pts.shape[1] == 3
    assert cols.shape == (6, 4)


def test_ground_grid_lines(vehicle_model):
    pts, cols = vehicle_model.get_ground_grid_lines(size=10.0, step=1.0)
    assert len(pts) % 2 == 0
    assert np.allclose(pts[:, 2], 0.0)  # All grid points on ground Z=0


def test_clearance_front_obstacle(vehicle_model):
    # Obstacle 1.0 m in front of vehicle front bumper (Y = +2.15 m)
    obs_min = np.array([-0.5, 3.15, 0.2], dtype=np.float32)
    obs_max = np.array([+0.5, 3.50, 0.8], dtype=np.float32)

    p_veh, p_obs, dist = vehicle_model.compute_clearance_endpoints(obs_min, obs_max)
    assert np.isclose(dist, 1.0, atol=0.01)
    assert np.isclose(p_veh[1], +2.15, atol=0.01)
    assert np.isclose(p_obs[1], +3.15, atol=0.01)


def test_clearance_side_mirror_obstacle(vehicle_model):
    # Obstacle 0.5 m to the right of the right wing mirror (mirror tip X = +1.05 m, Y = 0.40 m)
    obs_min = np.array([1.55, 0.38, 0.95], dtype=np.float32)
    obs_max = np.array([1.80, 0.42, 1.05], dtype=np.float32)

    p_veh, p_obs, dist = vehicle_model.compute_clearance_endpoints(obs_min, obs_max)
    # Distance to mirror tip at 1.05 m should be exactly 1.55 - 1.05 = 0.50 m
    assert np.isclose(dist, 0.50, atol=0.01)
    assert np.isclose(p_veh[0], +1.05, atol=0.01)
    assert np.isclose(p_obs[0], +1.55, atol=0.01)


def test_clearance_rear_obstacle(vehicle_model):
    # Obstacle 1.5 m behind rear bumper (rear bumper Y = -2.15 m)
    obs_min = np.array([-0.5, -3.95, 0.2], dtype=np.float32)
    obs_max = np.array([+0.5, -3.65, 0.8], dtype=np.float32)

    p_veh, p_obs, dist = vehicle_model.compute_clearance_endpoints(obs_min, obs_max)
    assert np.isclose(dist, 1.50, atol=0.01)
    assert np.isclose(p_veh[1], -2.15, atol=0.01)
    assert np.isclose(p_obs[1], -3.65, atol=0.01)

