"""
occupancy/tests/test_occupancy_grid.py — Unit Tests for 3D Voxel Occupancy Grid
================================================================================

Tests:
  - 3-state grid initialization: UNKNOWN (0), FREE (1), OCCUPIED (2)
  - Voxel coordinate conversions (world to grid, grid to world)
  - Ray-based free-space carving (Bresenham / DDA line traversal)
  - Endpoint occupancy marking
  - Occlusion preservation (space beyond surface remains UNKNOWN)
  - Memory consumption calculation and bounds checking
"""

import numpy as np
import pytest

from occupancy.grid3d import OccupancyGrid3D, VoxelState


@pytest.fixture
def empty_grid():
    return OccupancyGrid3D(
        bounds_x=(-5.0, 5.0),
        bounds_y=(-5.0, 5.0),
        bounds_z=(-1.0, 3.0),
        resolution=0.10, # 10 cm for fast unit tests
    )


def test_grid_initialization_all_unknown(empty_grid):
    """Verify newly created grid is 100% UNKNOWN."""
    stats = empty_grid.get_stats()
    assert stats["total_voxels"] == empty_grid.nx * empty_grid.ny * empty_grid.nz
    assert stats["unknown_voxels"] == stats["total_voxels"]
    assert stats["free_voxels"] == 0
    assert stats["occupied_voxels"] == 0
    assert stats["unknown_pct"] == 100.0


def test_coordinate_conversion_roundtrip(empty_grid):
    """Verify world-to-grid and grid-to-world conversion consistency."""
    query_pts = np.array([
        [0.0, 0.0, 0.0],
        [2.35, -1.82, 1.45],
        [-4.50, 4.50, -0.50],
    ])
    indices = empty_grid.world_to_grid(query_pts)
    reconstructed_pts = empty_grid.grid_to_world(indices)

    # Discrepancy should be within half voxel resolution
    diff = np.abs(query_pts - reconstructed_pts)
    assert np.all(diff <= empty_grid.resolution * 0.5)


def test_ray_carving_marks_free_and_occupied(empty_grid):
    """Verify that ray carving marks traversed voxels FREE and destination OCCUPIED."""
    origin = np.array([0.0, 0.0, 1.0])
    # Surface obstacle at (3.0, 0.0, 1.0) -> ray travels along +X
    endpoint = np.array([3.0, 0.0, 1.0])

    empty_grid.carve_ray(
        origin=origin,
        endpoint=endpoint,
        free_margin_m=0.05,
        max_ray_len_m=10.0,
        is_surface_hit=True,
    )

    # 1. Endpoint voxel must be OCCUPIED
    idx_end = empty_grid.world_to_grid(endpoint[None, :])[0]
    assert empty_grid.grid[idx_end[0], idx_end[1], idx_end[2]] == VoxelState.OCCUPIED

    # 2. Midpoint voxels along the ray (e.g. X=1.5, Y=0, Z=1) must be FREE
    midpoint = np.array([1.5, 0.0, 1.0])
    idx_mid = empty_grid.world_to_grid(midpoint[None, :])[0]
    assert empty_grid.grid[idx_mid[0], idx_mid[1], idx_mid[2]] == VoxelState.FREE

    # 3. Space beyond the endpoint (e.g. X=4.0, Y=0, Z=1) must remain UNKNOWN (occluded!)
    occluded_pt = np.array([4.0, 0.0, 1.0])
    idx_occ = empty_grid.world_to_grid(occluded_pt[None, :])[0]
    assert empty_grid.grid[idx_occ[0], idx_occ[1], idx_occ[2]] == VoxelState.UNKNOWN


def test_integrate_camera_view_multiple_rays(empty_grid):
    """Verify batch camera view integration with multi-point endpoints."""
    cam_origin = np.array([0.0, 2.0, 1.0])
    # A planar surface 3m ahead: Y = 5.0, X in [-1, 1], Z in [0.5, 1.5]
    X = np.linspace(-1, 1, 10)
    Z = np.linspace(0.5, 1.5, 10)
    xx, zz = np.meshgrid(X, Z)
    yy = np.full_like(xx, 5.0)
    endpoints = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    empty_grid.integrate_camera_view(cam_origin, endpoints, max_range_m=15.0)

    stats = empty_grid.get_stats()
    assert stats["occupied_voxels"] > 0
    assert stats["free_voxels"] > 0
    # The occupied points should match endpoints within tolerance
    occ_pts = empty_grid.get_occupied_points()
    assert len(occ_pts) > 0
    assert np.all(occ_pts[:, 1] >= 4.9) and np.all(occ_pts[:, 1] <= 5.1)

