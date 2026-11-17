"""Regression tests for bounded, vectorized mesh nearest-neighbour queries."""

import numpy as np
from scipy.spatial import cKDTree

from photo_reconstruction.nearest_neighbors import query_nearest_neighbors


def test_nearest_neighbor_queries_match_brute_force_across_chunks():
    rng = np.random.default_rng(42)
    reference = rng.normal(size=(37, 3))
    queries = rng.normal(size=(53, 3))

    tree = cKDTree(reference)
    distances, indices = query_nearest_neighbors(
        tree, queries, chunk_size=7, workers=1
    )

    brute_force = np.linalg.norm(
        queries[:, None, :] - reference[None, :, :], axis=2
    )
    expected_distances = brute_force.min(axis=1)

    assert distances.shape == (len(queries),)
    assert indices.shape == (len(queries),)
    assert np.allclose(distances, expected_distances)
    assert np.all(indices >= 0)
    assert np.all(indices < len(reference))
    assert np.allclose(
        np.linalg.norm(queries - reference[indices], axis=1), distances
    )


def test_nearest_neighbor_queries_handle_empty_vertex_batches():
    tree = cKDTree(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    distances, indices = query_nearest_neighbors(
        tree, np.empty((0, 3)), chunk_size=1, workers=1
    )

    assert distances.shape == (0,)
    assert indices.shape == (0,)


def test_nearest_neighbor_queries_validate_input():
    tree = cKDTree(np.array([[0.0, 0.0, 0.0]]))

    with np.testing.assert_raises(ValueError):
        query_nearest_neighbors(tree, np.zeros((3, 2)))

    with np.testing.assert_raises(ValueError):
        query_nearest_neighbors(
            tree, np.zeros((1, 3)), chunk_size=0
        )
