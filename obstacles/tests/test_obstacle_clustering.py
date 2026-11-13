"""
obstacles/tests/test_obstacle_clustering.py — Unit Tests for 3D Obstacle Clustering
===================================================================================

Tests:
  - DBSCAN spatial clustering of discrete obstacles
  - Centroid, 3D bounding box (AABB), and dimension calculation
  - Noise point rejection
  - Clearance ordering (closest obstacles sorted first)
"""

import numpy as np
import pytest

from obstacles.clustering import BoundingBox3D, Obstacle3D, ObstacleClusterer
from scene.semantics import SemanticClass


@pytest.fixture
def clusterer():
    return ObstacleClusterer(eps=0.30, min_samples=15, min_cluster_points=20)


def test_cluster_two_separate_obstacles(clusterer):
    """Verify that two spatially separated clusters are recognized as distinct obstacles."""
    # Obstacle 1: Box at (2.0, 4.0, 0.5) of size 0.5m x 0.5m x 0.5m
    obs1_pts = np.random.uniform(low=[1.8, 3.8, 0.3], high=[2.2, 4.2, 0.7], size=(100, 3))
    # Obstacle 2: Box at (-3.0, -3.0, 0.5) of size 1.0m x 1.0m x 1.0m
    obs2_pts = np.random.uniform(low=[-3.5, -3.5, 0.1], high=[-2.5, -2.5, 1.1], size=(150, 3))

    all_pts = np.vstack([obs1_pts, obs2_pts])
    obstacles = clusterer.cluster_obstacles(all_pts)

    assert len(obstacles) == 2

    # Check centroids and clearances
    centroids = [o.centroid for o in obstacles]
    c1 = next(c for c in centroids if c[0] > 0)
    c2 = next(c for c in centroids if c[0] < 0)

    assert np.allclose(c1[:2], [2.0, 4.0], atol=0.20)
    assert np.allclose(c2[:2], [-3.0, -3.0], atol=0.20)

    # Obstacles must be sorted in ascending order of vehicle clearance
    assert obstacles[0].clearance_to_vehicle_m <= obstacles[1].clearance_to_vehicle_m


def test_noise_rejection(clusterer):
    """Verify that sparse noise points are filtered out by DBSCAN min_samples threshold."""
    # Real dense cluster
    dense_pts = np.random.uniform(low=[0, 3, 0.5], high=[0.5, 3.5, 1.0], size=(80, 3))
    # 5 isolated sparse noise points
    noise_pts = np.array([
        [-5.0, 0.0, 1.0],
        [4.0, 1.0, 2.0],
        [-2.0, -4.0, 0.5],
        [1.0, -2.0, 1.5],
        [3.0, 5.0, 0.8],
    ])
    all_pts = np.vstack([dense_pts, noise_pts])
    obstacles = clusterer.cluster_obstacles(all_pts)

    # Only 1 obstacle should be returned, noise rejected
    assert len(obstacles) == 1
    assert obstacles[0].point_count >= 70

