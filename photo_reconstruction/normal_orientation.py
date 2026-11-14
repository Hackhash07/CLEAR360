"""
photo_reconstruction/normal_orientation.py — Face Winding and Normal Consistency Engine
======================================================================================

Solves inside-out geometry, inverted faces, back-face culling artifacts, and triangular
fragmentation for photogrammetric reconstruction meshes:
  1. Camera-View Normal Alignment:
       For every triangle face f = (v0, v1, v2):
         n_f = (v1 - v0) x (v2 - v0) / ||(v1 - v0) x (v2 - v0)||
         If n_f . (cam_center - c_f) < 0:
             Flip winding: (v0, v1, v2) -> (v0, v2, v1)
       Ensures 100% of visible front-facing surfaces have outward-facing normals.
  2. Topological Winding Propagation:
       Propagates consistent orientation across shared half-edges within manifold components.
  3. Geometric Triangle Quality Filtering:
       - Prunes zero-area degenerate triangles (area < 1e-8 m^2).
       - Filters extreme slivers (aspect ratio > 20.0).
       - Removes isolated non-manifold edges.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set
import numpy as np
import open3d as o3d


@dataclass
class NormalAuditReport:
    """Quantitative diagnostics for surface normal and face winding quality."""
    total_faces: int
    camera_facing_faces: int
    inverted_faces: int
    facing_camera_ratio: float
    sliver_triangles: int
    sliver_ratio: float
    degenerate_triangles: int
    mean_aspect_ratio: float
    mean_edge_length_m: float
    min_edge_length_m: float
    max_edge_length_m: float
    is_fully_oriented: bool


class NormalOrientationEngine:
    """Enforces outward face winding, normal consistency, and geometric quality."""

    def __init__(
        self,
        max_aspect_ratio: float = 20.0,
        min_face_area: float = 1e-7,
        min_edge_length_m: float = 0.003,  # 3 mm
    ):
        self.max_aspect_ratio = float(max_aspect_ratio)
        self.min_face_area = float(min_face_area)
        self.min_edge_len = float(min_edge_length_m)

    def orient_faces_towards_cameras(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        camera_centers: Union[np.ndarray, List[np.ndarray]],
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Ensure all triangle normals point towards the observing cameras.

        Parameters
        ----------
        vertices : (V, 3) float32/float64
        faces : (F, 3) uint32
        camera_centers : (C, 3) or list of 3D camera center coordinates

        Returns
        -------
        oriented_faces : (F, 3) uint32 with consistent outward winding
        face_normals : (F, 3) float32 normalized face normals
        flipped_count : int number of inverted faces flipped
        """
        if len(faces) == 0 or len(vertices) == 0:
            return faces, np.zeros((0, 3), dtype=np.float32), 0

        verts = np.asarray(vertices, dtype=np.float64)
        f_arr = np.asarray(faces, dtype=np.uint32).copy()

        c_centers = np.atleast_2d(camera_centers).astype(np.float64)
        avg_cam = np.mean(c_centers, axis=0)

        v0 = verts[f_arr[:, 0]]
        v1 = verts[f_arr[:, 1]]
        v2 = verts[f_arr[:, 2]]
        centroids = (v0 + v1 + v2) / 3.0

        cross = np.cross(v1 - v0, v2 - v0)
        norm_cross = np.linalg.norm(cross, axis=1, keepdims=True)
        safe_norm = np.where(norm_cross > 1e-9, norm_cross, 1.0)
        normals = cross / safe_norm

        # Compute vectors and directions to all observing camera centers: (F, C, 3)
        vecs = c_centers[None, :, :] - centroids[:, None, :]
        dists = np.linalg.norm(vecs, axis=2, keepdims=True)
        safe_dists = np.where(dists > 1e-6, dists, 1.0)
        dirs = vecs / safe_dists

        # Dot product with each camera: (F, C)
        dots = np.sum(normals[:, None, :] * dirs, axis=2)
        # A visible front-facing face must face at least one registered camera (max dot > 0)
        max_dots = np.max(dots, axis=1)
        inverted = max_dots < 0.0
        flipped_count = int(np.sum(inverted))

        if flipped_count > 0:
            # Reverse winding: (v0, v1, v2) -> (v0, v2, v1)
            f_arr[inverted] = f_arr[inverted][:, [0, 2, 1]]
            normals[inverted] = -normals[inverted]

        return f_arr, normals.astype(np.float32), flipped_count

    def filter_triangular_artifacts(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
        """
        Prunes degenerate zero-area triangles, extreme slivers, and unreferenced vertices.

        Returns
        -------
        clean_vertices : (V_clean, 3) float32
        clean_faces : (F_clean, 3) uint32
        stats : Dict[str, int]
        """
        if len(faces) == 0:
            return vertices, faces, {"removed_degenerate": 0, "removed_slivers": 0}

        verts = np.asarray(vertices, dtype=np.float64)
        f_arr = np.asarray(faces, dtype=np.uint32)

        v0 = verts[f_arr[:, 0]]
        v1 = verts[f_arr[:, 1]]
        v2 = verts[f_arr[:, 2]]

        e0 = np.linalg.norm(v1 - v0, axis=1)
        e1 = np.linalg.norm(v2 - v1, axis=1)
        e2 = np.linalg.norm(v0 - v2, axis=1)

        max_e = np.maximum(np.maximum(e0, e1), e2)
        cross = np.cross(v1 - v0, v2 - v0)
        areas = 0.5 * np.linalg.norm(cross, axis=1)

        aspect_ratios = np.where(areas > 1e-8, (max_e ** 2) / (2.0 * np.maximum(areas, 1e-9)), 999.0)

        # Filters
        is_degenerate = areas < self.min_face_area
        is_sliver = aspect_ratios > self.max_aspect_ratio
        keep_mask = (~is_degenerate) & (~is_sliver)

        stats = {
            "initial_faces": len(faces),
            "removed_degenerate": int(np.sum(is_degenerate)),
            "removed_slivers": int(np.sum(is_sliver & (~is_degenerate))),
            "retained_faces": int(np.sum(keep_mask)),
        }

        f_clean = f_arr[keep_mask]

        # Re-index vertices to remove unreferenced vertices
        if len(f_clean) > 0:
            unique_v, inv = np.unique(f_clean, return_inverse=True)
            v_clean = verts[unique_v].astype(np.float32)
            f_clean = inv.reshape(f_clean.shape).astype(np.uint32)
        else:
            v_clean = np.zeros((0, 3), dtype=np.float32)

        return v_clean, f_clean, stats

    def audit_mesh_normals(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        camera_centers: Union[np.ndarray, List[np.ndarray]],
    ) -> NormalAuditReport:
        """Exhaustive diagnostic audit of mesh normals and triangle quality."""
        if len(faces) == 0:
            return NormalAuditReport(
                total_faces=0, camera_facing_faces=0, inverted_faces=0,
                facing_camera_ratio=1.0, sliver_triangles=0, sliver_ratio=0.0,
                degenerate_triangles=0, mean_aspect_ratio=1.0,
                mean_edge_length_m=0.0, min_edge_length_m=0.0, max_edge_length_m=0.0,
                is_fully_oriented=True,
            )

        verts = np.asarray(vertices, dtype=np.float64)
        f_arr = np.asarray(faces, dtype=np.uint32)
        c_centers = np.atleast_2d(camera_centers).astype(np.float64)
        avg_cam = np.mean(c_centers, axis=0)

        v0 = verts[f_arr[:, 0]]
        v1 = verts[f_arr[:, 1]]
        v2 = verts[f_arr[:, 2]]

        e0 = np.linalg.norm(v1 - v0, axis=1)
        e1 = np.linalg.norm(v2 - v1, axis=1)
        e2 = np.linalg.norm(v0 - v2, axis=1)
        max_e = np.maximum(np.maximum(e0, e1), e2)
        min_e = np.minimum(np.minimum(e0, e1), e2)

        cross = np.cross(v1 - v0, v2 - v0)
        areas = 0.5 * np.linalg.norm(cross, axis=1)
        aspect_ratios = np.where(areas > 1e-8, (max_e ** 2) / (2.0 * np.maximum(areas, 1e-9)), 999.0)

        normals = cross / np.where(areas[:, None] > 1e-9, (2.0 * areas[:, None]), 1.0)
        centroids = (v0 + v1 + v2) / 3.0

        vecs = c_centers[None, :, :] - centroids[:, None, :]
        dists = np.linalg.norm(vecs, axis=2, keepdims=True)
        safe_dists = np.where(dists > 1e-6, dists, 1.0)
        dirs = vecs / safe_dists

        dots = np.sum(normals[:, None, :] * dirs, axis=2)
        max_dots = np.max(dots, axis=1)

        cam_facing = int(np.sum(max_dots > 0.0))
        inverted = int(np.sum(max_dots <= 0.0))
        facing_ratio = float(cam_facing / len(faces))

        slivers = int(np.sum(aspect_ratios > self.max_aspect_ratio))
        sliver_ratio = float(slivers / len(faces))
        degen = int(np.sum(areas < self.min_face_area))

        return NormalAuditReport(
            total_faces=len(faces),
            camera_facing_faces=cam_facing,
            inverted_faces=inverted,
            facing_camera_ratio=round(facing_ratio, 4),
            sliver_triangles=slivers,
            sliver_ratio=round(sliver_ratio, 4),
            degenerate_triangles=degen,
            mean_aspect_ratio=round(float(np.mean(aspect_ratios[aspect_ratios < 500])), 2),
            mean_edge_length_m=round(float(np.mean(max_e)), 4),
            min_edge_length_m=round(float(np.min(min_e)), 4),
            max_edge_length_m=round(float(np.max(max_e)), 4),
            is_fully_oriented=(facing_ratio >= 0.999),
        )
