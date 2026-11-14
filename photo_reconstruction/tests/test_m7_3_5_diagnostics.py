"""
photo_reconstruction/tests/test_m7_3_5_diagnostics.py — Unit Tests for M7.3.5 Mesh Diagnostics & Repair
=======================================================================================================
"""

import numpy as np
import pytest
from photo_reconstruction.mesh_diagnostics import MeshQualityAuditor, PolygonSoupRepair


def test_mesh_quality_auditor_open_surface():
    """Verify open surface classification on planar quad mesh."""
    # Simple quad with 2 triangles
    verts = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float32)
    faces = np.array([
        [0, 1, 2],
        [0, 2, 3],
    ], dtype=np.uint32)

    rep = MeshQualityAuditor.audit_mesh(verts, faces, expected_closed=False)
    assert rep.vertex_count == 4
    assert rep.face_count == 2
    assert not rep.is_watertight
    assert rep.surface_type == "open_surface"
    assert rep.boundary_edges == 4
    assert rep.non_manifold_edges == 0
    assert rep.degenerate_faces == 0
    assert rep.sliver_faces == 0


def test_mesh_quality_auditor_closed_cube():
    """Verify closed volume classification on watertight cube."""
    # 8 vertices of unit cube
    verts = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
    ], dtype=np.float32)
    # 12 triangles (2 per face)
    faces = np.array([
        [0, 2, 1], [0, 3, 2], # bottom
        [4, 5, 6], [4, 6, 7], # top
        [0, 1, 5], [0, 5, 4], # front
        [2, 3, 7], [2, 7, 6], # back
        [0, 4, 7], [0, 7, 3], # left
        [1, 2, 6], [1, 6, 5], # right
    ], dtype=np.uint32)

    rep = MeshQualityAuditor.audit_mesh(verts, faces, expected_closed=True)
    assert rep.vertex_count == 8
    assert rep.face_count == 12
    assert rep.is_watertight
    assert rep.surface_type == "closed_volume"
    assert rep.boundary_edges == 0
    assert rep.non_manifold_edges == 0


def test_polygon_soup_repair_duplicates_and_degenerates():
    """Verify excision of duplicate faces, degenerate triangles, and vertex welding."""
    verts = np.array([
        [0.0, 0.0, 0.0],  # 0
        [1.0, 0.0, 0.0],  # 1
        [1.0, 1.0, 0.0],  # 2
        [0.0, 1.0, 0.0],  # 3
        [0.0, 0.0, 0.00001], # 4 (coincident with 0, within weld_epsilon)
    ], dtype=np.float32)

    faces = np.array([
        [0, 1, 2],       # valid face 1
        [0, 1, 2],       # duplicate of face 1
        [0, 2, 1],       # duplicate with reversed winding
        [0, 1, 0],       # degenerate face (collinear edge)
        [4, 2, 3],       # valid face 2 using near-coincident vertex
    ], dtype=np.uint32)

    cam_center = np.array([0.5, 0.5, 2.0])
    r_verts, r_faces, stats = PolygonSoupRepair.repair_mesh(
        verts, faces, camera_centers=cam_center, weld_epsilon=1e-4
    )

    assert stats["duplicate_faces_removed"] >= 1
    assert stats["degenerate_faces_removed"] >= 1
    assert len(r_faces) == 2  # Only 2 distinct physical faces remain
    assert stats["non_manifold_faces_removed"] == 0
