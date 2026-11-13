"""
scene/tests/test_scene_filtering.py — Unit Tests for 3D Scene Filtering
=======================================================================

Tests:
  - Voxel downsampling
  - Spatial deduplication
  - Outlier removal (SOR & ROR)
  - Sky and invalid depth filtering without artificial height clamping
  - Ground plane identification via RANSAC
  - Preservation of vertical geometry
"""

import numpy as np
import open3d as o3d
import pytest

from scene.filter import GroundPlaneResult, SceneFilter


@pytest.fixture
def scene_filter():
    return SceneFilter(
        voxel_size=0.05,
        sor_nb_neighbors=15,
        sor_std_ratio=2.0,
        ror_nb_points=8,
        ror_radius=0.15,
        ground_dist_threshold=0.05,
        ground_max_tilt_deg=5.0,
    )


def test_voxel_downsampling(scene_filter):
    """Verify that voxel downsampling reduces point count and enforces regular spacing."""
    pts = np.random.uniform(low=[-2, -2, 0], high=[2, 2, 2], size=(10000, 3))
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))

    ds = scene_filter.downsample_voxel(pcd, voxel_size=0.10)
    assert len(ds.points) < len(pts)
    assert len(ds.points) > 50


def test_deduplication(scene_filter):
    """Verify that identical or near-identical points within tolerance are removed."""
    pts = np.array([
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],          # exact duplicate
        [1.001, 2.001, 3.001],    # within 5 mm tolerance
        [5.0, 5.0, 5.0],          # distinct point
    ])
    dedup_pts, _, _ = scene_filter.remove_duplicates(pts, spatial_tolerance=0.005)
    assert len(dedup_pts) == 2
    assert np.allclose(dedup_pts[0], [1.0, 2.0, 3.0], atol=0.01)
    assert np.allclose(dedup_pts[1], [5.0, 5.0, 5.0], atol=0.01)


def test_sky_filtering_without_height_clamping(scene_filter):
    """Verify that sky points are removed without artificially clipping tall physical obstacles."""
    # Tall obstacle (e.g. 5m light pole or building at X=3, Y=3)
    tall_obstacle_pts = np.column_stack([
        np.full(100, 3.0),
        np.full(100, 3.0),
        np.linspace(0.0, 5.0, 100),
    ])
    # Distinct dark/grey color
    tall_col = np.full((100, 3), 0.3)

    # Sky hallucination points (Z > 2.6m, R > 8.5m, high blue/luminance)
    sky_pts = np.column_stack([
        np.full(50, 9.5),
        np.full(50, 9.5),
        np.linspace(3.0, 6.0, 50),
    ])
    sky_col = np.column_stack([np.full(50, 0.4), np.full(50, 0.7), np.full(50, 0.95)]) # bright blue

    all_pts = np.vstack([tall_obstacle_pts, sky_pts])
    all_col = np.vstack([tall_col, sky_col])

    filtered_pts, filtered_col, _, valid = scene_filter.filter_sky_and_invalid_points(
        all_pts, all_col, is_ground_truth=False
    )

    # The tall 5m obstacle must be preserved!
    tall_remaining = filtered_pts[filtered_pts[:, 2] > 2.6]
    assert len(tall_remaining) > 0
    # The tall obstacle points at X=3, Y=3 should still reach up to 5.0m
    assert tall_remaining[:, 2].max() >= 4.8

    # Sky points at R > 8.5m with blue color must be filtered out
    sky_remaining = filtered_pts[(filtered_pts[:, 0] > 9.0) & (filtered_pts[:, 2] > 2.6)]
    assert len(sky_remaining) == 0


def test_ground_plane_segmentation_ransac(scene_filter):
    """Verify robust ground plane identification and tilt angle calculation."""
    # Road plane at Z = 0.0 with slight 0.5 deg slope and small noise
    X = np.linspace(-5, 5, 40)
    Y = np.linspace(-5, 5, 40)
    xx, yy = np.meshgrid(X, Y)
    zz = np.zeros_like(xx) + 0.01 * np.random.randn(*xx.shape)

    road_pts = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    # Vertical obstacle points (e.g. wall or car)
    obs_pts = np.array([[x, 3.0, z] for x in np.linspace(-2, 2, 20) for z in np.linspace(0.5, 2.0, 20)])

    all_pts = np.vstack([road_pts, obs_pts])

    ground_pts, non_ground_pts, res = scene_filter.segment_ground_plane(all_pts)

    assert isinstance(res, GroundPlaneResult)
    assert len(ground_pts) >= len(road_pts) * 0.95
    assert len(non_ground_pts) >= len(obs_pts) * 0.95
    assert res.tilt_deg < 2.0
    assert abs(res.elevation_z) < 0.05
    assert res.rmse_m < 0.03

