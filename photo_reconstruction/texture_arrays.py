"""Vectorized array construction for visibility-aware texture mapping."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def build_unwelded_uv_arrays(
    vertices: np.ndarray,
    faces: np.ndarray,
    best_camera_indices: np.ndarray,
    camera_names: list[str],
    camera_models: Mapping[str, Any],
    camera_cell_offsets: Mapping[int, tuple[int, int, float, float]],
    *,
    total_atlas_width: int,
    total_atlas_height: int,
    cell_width: int,
    cell_height: int,
    neutral_u: float,
    neutral_v: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build UV-seam-split mesh arrays while batching projection by camera.

    Each face is assigned to exactly one camera by the visibility-scoring
    stage. Faces without a valid camera map to the neutral atlas swatch.
    Projection math, camera assignment, clipping bounds and output ordering
    match the former per-face implementation.
    """
    verts = np.asarray(vertices, dtype=np.float64)
    tri = np.asarray(faces)
    camera_ids = np.asarray(best_camera_indices, dtype=np.int32)

    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError("vertices must have shape (N, 3)")
    if tri.ndim != 2 or tri.shape[1] != 3:
        raise ValueError("faces must have shape (M, 3)")
    if camera_ids.shape != (len(tri),):
        raise ValueError("best_camera_indices must contain one entry per face")
    if total_atlas_width <= 0 or total_atlas_height <= 0:
        raise ValueError("atlas dimensions must be positive")
    if cell_width <= 0 or cell_height <= 0:
        raise ValueError("atlas cell dimensions must be positive")
    if len(tri) and (np.any(tri < 0) or np.any(tri >= len(verts))):
        raise ValueError("faces contain vertex indices outside the vertices array")

    new_vertices = verts[tri].reshape(-1, 3).astype(np.float32, copy=False)
    new_faces = np.arange(len(tri) * 3, dtype=np.uint32).reshape(-1, 3)
    new_uvs = np.empty((len(tri) * 3, 2), dtype=np.float32)
    new_uvs[:, 0] = np.float32(neutral_u)
    new_uvs[:, 1] = np.float32(neutral_v)

    triangle_columns = np.arange(3, dtype=np.int64)

    for camera_index, (x_offset, y_offset, scale_x, scale_y) in camera_cell_offsets.items():
        face_indices = np.flatnonzero(camera_ids == int(camera_index))
        if face_indices.size == 0:
            continue
        if camera_index < 0 or camera_index >= len(camera_names):
            continue

        camera_name = camera_names[camera_index]
        camera = camera_models.get(camera_name)
        if camera is None:
            continue

        # One vectorized projection call for every face assigned to this camera.
        face_vertex_indices = tri[face_indices].reshape(-1)
        projected, _, _ = camera.project_world_to_image(
            verts[face_vertex_indices],
            check_bounds=False,
        )

        u_atlas = (x_offset + projected[:, 0] * scale_x) / float(total_atlas_width)
        v_atlas = (y_offset + projected[:, 1] * scale_y) / float(total_atlas_height)

        u_min = x_offset / float(total_atlas_width)
        u_max = (x_offset + cell_width) / float(total_atlas_width)
        v_min = y_offset / float(total_atlas_height)
        v_max = (y_offset + cell_height) / float(total_atlas_height)

        u_atlas = np.clip(u_atlas, u_min + 1e-4, u_max - 1e-4)
        v_atlas = np.clip(v_atlas, v_min + 1e-4, v_max - 1e-4)

        flat_slots = (face_indices[:, None] * 3 + triangle_columns).reshape(-1)
        new_uvs[flat_slots] = np.column_stack((u_atlas, v_atlas)).astype(
            np.float32, copy=False
        )

    return new_vertices, new_faces, new_uvs
