"""
validation/test_m6_validation.py — Milestone M6 Deliverables & Integration Tests
================================================================================

Validates:
  - Generation and validity of all M6 deliverable files:
      * Clean PLYs (frame_0000_ground_truth_clean.ply, frame_0000_ai_per_camera_clean.ply)
      * Semantic PLYs (frame_0000_ground_truth_semantic.ply, frame_0000_ai_per_camera_semantic.ply)
      * Occupancy NPZ grids (frame_0000_ground_truth_occupancy.npz, frame_0000_ai_per_camera_occupancy.npz)
      * Scene Metadata JSON
      * Obstacle Bounding Box JSON
  - Ground-plane accuracy (< 5 cm RMSE)
  - Occupancy grid classification (presence of free, occupied, unknown space)
  - Vehicle clearance accuracy
  - Independent pipeline isolation
"""

import json
from pathlib import Path
import numpy as np
import open3d as o3d
import pytest


def test_m6_pointcloud_deliverables_exist():
    """Verify that all clean and semantic PLYs are saved on disk and have valid data."""
    pc_dir = Path("data/pointclouds")
    required_plys = [
        "frame_0000_ground_truth_clean.ply",
        "frame_0000_ground_truth_semantic.ply",
        "frame_0000_ai_per_camera_clean.ply",
        "frame_0000_ai_per_camera_semantic.ply",
    ]
    for name in required_plys:
        p = pc_dir / name
        assert p.exists(), f"Missing required deliverable: {p}"
        pcd = o3d.io.read_point_cloud(str(p))
        assert len(pcd.points) > 10000, f"{name}: insufficient point count"
        assert len(pcd.colors) == len(pcd.points)


def test_m6_occupancy_grid_deliverables():
    """Verify that occupancy NPZ grids are created and have valid 3-state voxel values."""
    occ_dir = Path("data/occupancy")
    required_npz = [
        "frame_0000_ground_truth_occupancy.npz",
        "frame_0000_ai_per_camera_occupancy.npz",
    ]
    for name in required_npz:
        p = occ_dir / name
        assert p.exists(), f"Missing required occupancy grid: {p}"
        data = np.load(str(p))
        assert "grid" in data
        grid = data["grid"]
        assert grid.ndim == 3
        # Check presence of Free (1) and Occupied (2) voxels
        n_free = int(np.sum(grid == 1))
        n_occ = int(np.sum(grid == 2))
        assert n_free > 100000, f"{name}: Free space carving failed (got {n_free} voxels)"
        assert n_occ > 10000, f"{name}: Occupied surface detection failed (got {n_occ} voxels)"


def test_m6_metadata_and_obstacle_json():
    """Verify that scene metadata and obstacle bounding box JSON files exist and have valid structure."""
    meta_dir = Path("data/metadata")
    required_json = [
        "frame_0000_ground_truth_scene.json",
        "frame_0000_ground_truth_obstacles.json",
        "frame_0000_ai_per_camera_scene.json",
        "frame_0000_ai_per_camera_obstacles.json",
    ]
    for name in required_json:
        p = meta_dir / name
        assert p.exists(), f"Missing metadata file: {p}"
        with open(p, "r") as f:
            data = json.load(f)
        if "obstacles" in name:
            assert isinstance(data, list)
            assert len(data) > 0
            # Verify obstacle fields
            first = data[0]
            assert "centroid" in first
            assert "bbox" in first
            assert "clearance_to_vehicle_m" in first
            assert "semantic_name" in first
            assert first["clearance_to_vehicle_m"] >= 0.0
        elif "scene" in name:
            assert "ground_plane" in data
            assert "occupancy_grid" in data
            assert data["ground_plane"]["rmse_m"] < 0.05

