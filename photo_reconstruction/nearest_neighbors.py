"""Memory-bounded nearest-neighbour queries for reconstruction geometry."""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np


def query_nearest_neighbors(
    tree: Any,
    query_points: np.ndarray,
    *,
    chunk_size: int = 200_000,
    workers: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Query a SciPy cKDTree in bounded batches.

    Chunking limits temporary memory for high-detail meshes while keeping the
    nearest-neighbour search inside SciPy's compiled backend.
    """
    from scipy.spatial import cKDTree

    if not isinstance(tree, cKDTree):
        raise TypeError("tree must be a scipy.spatial.cKDTree instance.")
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least one.")
    if workers < 1:
        raise ValueError("workers must be at least one.")

    queries = np.asarray(query_points, dtype=np.float64)
    if queries.ndim != 2 or queries.shape[1] != 3:
        raise ValueError("query_points must have shape (N, 3).")

    distances = np.empty(len(queries), dtype=np.float64)
    indices = np.empty(len(queries), dtype=np.intp)
    for start in range(0, len(queries), chunk_size):
        stop = min(start + chunk_size, len(queries))
        batch_distances, batch_indices = tree.query(
            queries[start:stop], k=1, workers=workers
        )
        distances[start:stop] = batch_distances
        indices[start:stop] = batch_indices

    return distances, indices
