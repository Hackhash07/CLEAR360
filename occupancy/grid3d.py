"""
occupancy/grid3d.py — 3D Voxel Occupancy Mapping with Ray-Based Free-Space Carving
================================================================================

Represents the 3D environment as a discretized regular voxel grid with 3 states:
  - UNKNOWN  (0) : Unobserved or occluded space
  - FREE     (1) : Observed empty space traversed by optical rays
  - OCCUPIED (2) : Observed physical obstacle or surface boundary

Ray-Based Free-Space Carving:
  - Optical rays originate from calibrated camera centers (T_world_cam translation).
  - Traversed voxels up to the measured distance (d - delta) are marked FREE.
  - The surface endpoint voxel is marked OCCUPIED.
  - Space beyond the endpoint remains UNKNOWN (occlusion preserved).
  - Configurable resolution (default 0.05 m = 5 cm).
"""

from __future__ import annotations

from enum import IntEnum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import open3d as o3d


class VoxelState(IntEnum):
    UNKNOWN = 0
    FREE = 1
    OCCUPIED = 2


class OccupancyGrid3D:
    """
    3D Voxel Occupancy Grid with ray carving.

    Parameters
    ----------
    bounds_x : Tuple[float, float]
        [min_x, max_x] in metres (default [-12.0, 12.0]).
    bounds_y : Tuple[float, float]
        [min_y, max_y] in metres (default [-12.0, 12.0]).
    bounds_z : Tuple[float, float]
        [min_z, max_z] in metres (default [-0.5, 3.5]).
    resolution : float
        Voxel cube side length in metres (default 0.05 m = 5 cm).
    """

    def __init__(
        self,
        bounds_x: Tuple[float, float] = (-12.0, 12.0),
        bounds_y: Tuple[float, float] = (-12.0, 12.0),
        bounds_z: Tuple[float, float] = (-0.5, 3.5),
        resolution: float = 0.05,
    ):
        self.bounds_x = tuple(bounds_x)
        self.bounds_y = tuple(bounds_y)
        self.bounds_z = tuple(bounds_z)
        self.resolution = float(resolution)

        self.min_bound = np.array([bounds_x[0], bounds_y[0], bounds_z[0]], dtype=np.float64)
        self.max_bound = np.array([bounds_x[1], bounds_y[1], bounds_z[1]], dtype=np.float64)

        # Grid dimensions in voxels
        self.dims = np.ceil((self.max_bound - self.min_bound) / self.resolution).astype(np.int64)
        self.nx, self.ny, self.nz = int(self.dims[0]), int(self.dims[1]), int(self.dims[2])

        # Allocate 3D grid as uint8 (0=UNKNOWN, 1=FREE, 2=OCCUPIED)
        self.grid = np.zeros((self.nx, self.ny, self.nz), dtype=np.uint8)

    @property
    def total_voxels(self) -> int:
        return self.nx * self.ny * self.nz

    @property
    def memory_mb(self) -> float:
        return self.grid.nbytes / (1024.0 * 1024.0)

    def world_to_grid(self, pts_world: np.ndarray) -> np.ndarray:
        """Convert continuous 3D world coordinates (metres) to discrete voxel indices."""
        indices = np.floor((pts_world - self.min_bound) / self.resolution).astype(np.int64)
        return indices

    def grid_to_world(self, indices: np.ndarray) -> np.ndarray:
        """Convert discrete voxel indices to continuous 3D voxel center coordinates (metres)."""
        pts_world = (indices.astype(np.float64) + 0.5) * self.resolution + self.min_bound
        return pts_world

    def in_bounds_mask(self, indices: np.ndarray) -> np.ndarray:
        """Return boolean mask of voxel indices falling within the grid envelope."""
        return (
            (indices[:, 0] >= 0) & (indices[:, 0] < self.nx)
            & (indices[:, 1] >= 0) & (indices[:, 1] < self.ny)
            & (indices[:, 2] >= 0) & (indices[:, 2] < self.nz)
        )

    def set_occupied_endpoints(self, endpoints: np.ndarray) -> int:
        """
        Mark the voxel containing each endpoint as OCCUPIED.
        Returns count of voxels marked.
        """
        if len(endpoints) == 0:
            return 0

        idx = self.world_to_grid(endpoints)
        valid = self.in_bounds_mask(idx)
        valid_idx = idx[valid]

        if len(valid_idx) == 0:
            return 0

        # Mark occupied
        self.grid[valid_idx[:, 0], valid_idx[:, 1], valid_idx[:, 2]] = VoxelState.OCCUPIED
        return int(np.sum(valid))

    def carve_ray(
        self,
        origin: np.ndarray,
        endpoint: np.ndarray,
        free_margin_m: float = 0.05,
        max_ray_len_m: float = 25.0,
        is_surface_hit: bool = True,
    ) -> None:
        """
        Carve a single 3D ray using 3D line traversal.

        Parameters
        ----------
        origin : (3,) camera position in world frame
        endpoint : (3,) observed surface or ray termination in world frame
        free_margin_m : float, buffer before endpoint where space is marked free
        max_ray_len_m : float, maximum distance to carve
        is_surface_hit : bool, whether endpoint marks an obstacle surface (OCCUPIED)
        """
        diff = endpoint - origin
        dist = np.linalg.norm(diff)
        if dist < 1e-4:
            return

        direction = diff / dist
        carve_dist = min(dist, max_ray_len_m)

        # Sampling step along ray (half voxel resolution ensures no skipped voxels)
        step = self.resolution * 0.5
        n_steps = int(max(1, np.floor((carve_dist - free_margin_m) / step)))

        if n_steps > 0:
            s_vals = np.linspace(0.1, max(0.1, carve_dist - free_margin_m), n_steps)
            ray_pts = origin[None, :] + s_vals[:, None] * direction[None, :]
            idx_free = self.world_to_grid(ray_pts)
            valid_free = self.in_bounds_mask(idx_free)

            if np.any(valid_free):
                vf = idx_free[valid_free]
                # Only mark FREE if current voxel is UNKNOWN (do not overwrite existing OCCUPIED voxels)
                curr_vals = self.grid[vf[:, 0], vf[:, 1], vf[:, 2]]
                mask_unknown = curr_vals == VoxelState.UNKNOWN
                if np.any(mask_unknown):
                    v_to_free = vf[mask_unknown]
                    self.grid[v_to_free[:, 0], v_to_free[:, 1], v_to_free[:, 2]] = VoxelState.FREE

        # Mark endpoint OCCUPIED if valid surface hit
        if is_surface_hit and dist <= max_ray_len_m:
            idx_end = self.world_to_grid(endpoint[None, :])
            if self.in_bounds_mask(idx_end)[0]:
                self.grid[idx_end[0, 0], idx_end[0, 1], idx_end[0, 2]] = VoxelState.OCCUPIED

    def integrate_camera_view(
        self,
        camera_origin: np.ndarray,
        endpoints: np.ndarray,
        free_margin_m: float = 0.05,
        max_range_m: float = 25.0,
        subsample_stride: int = 1,
    ) -> None:
        """
        Vectorized ray-carving integration for a dense multi-point camera cloud.

        To guarantee real-time performance on millions of points:
          1. Quantizes endpoints to unique grid voxels (eliminates redundant rays).
          2. Traverses rays from camera origin to all unique surface voxels.
          3. Marks traversed voxels FREE and destination voxels OCCUPIED.
        """
        if len(endpoints) == 0:
            return

        sampled_pts = endpoints[::subsample_stride]
        idx = self.world_to_grid(sampled_pts)
        valid = self.in_bounds_mask(idx)
        valid_pts = sampled_pts[valid]

        if len(valid_pts) == 0:
            return

        # Deduplicate endpoints to voxel centers to avoid redundant ray tracing
        unique_idx = np.unique(self.world_to_grid(valid_pts), axis=0)
        unique_endpoints = self.grid_to_world(unique_idx)

        # Pre-mark all endpoints as OCCUPIED
        self.grid[unique_idx[:, 0], unique_idx[:, 1], unique_idx[:, 2]] = VoxelState.OCCUPIED

        # Ray-trace to carve free space
        diffs = unique_endpoints - camera_origin[None, :]
        dists = np.linalg.norm(diffs, axis=1)
        valid_rays = (dists > 0.20) & (dists <= max_range_m)

        dirs = diffs[valid_rays] / dists[valid_rays, None]
        ray_dists = dists[valid_rays]

        # Batch sample along rays
        step = self.resolution * 0.75
        for d_target, d_vec in zip(ray_dists, dirs):
            max_s = max(0.1, d_target - free_margin_m)
            s_samples = np.arange(0.20, max_s, step)
            if len(s_samples) == 0:
                continue

            r_pts = camera_origin[None, :] + s_samples[:, None] * d_vec[None, :]
            r_idx = self.world_to_grid(r_pts)
            r_val = self.in_bounds_mask(r_idx)
            if not np.any(r_val):
                continue

            r_idx_val = r_idx[r_val]
            # Fast update: mark UNKNOWN voxels as FREE
            curr_state = self.grid[r_idx_val[:, 0], r_idx_val[:, 1], r_idx_val[:, 2]]
            to_update = r_idx_val[curr_state == VoxelState.UNKNOWN]
            if len(to_update) > 0:
                self.grid[to_update[:, 0], to_update[:, 1], to_update[:, 2]] = VoxelState.FREE

    def get_occupied_points(self) -> np.ndarray:
        """Return (N, 3) float64 array of center coordinates of all OCCUPIED voxels."""
        occ_idx = np.argwhere(self.grid == VoxelState.OCCUPIED)
        if len(occ_idx) == 0:
            return np.zeros((0, 3), dtype=np.float64)
        return self.grid_to_world(occ_idx)

    def get_free_points(self, stride: int = 4) -> np.ndarray:
        """Return (N, 3) float64 array of center coordinates of FREE voxels (with stride)."""
        free_idx = np.argwhere(self.grid == VoxelState.FREE)
        if len(free_idx) == 0:
            return np.zeros((0, 3), dtype=np.float64)
        return self.grid_to_world(free_idx[::stride])

    def to_point_cloud(
        self,
        include_free: bool = False,
        free_stride: int = 8,
    ) -> o3d.geometry.PointCloud:
        """
        Convert the occupancy grid into a visualizable Open3D PointCloud.
        Occupied voxels are colored red/grey; Free voxels are colored translucent cyan.
        """
        occ_pts = self.get_occupied_points()
        occ_col = np.tile(np.array([0.8, 0.2, 0.2]), (len(occ_pts), 1))

        if include_free:
            free_pts = self.get_free_points(stride=free_stride)
            free_col = np.tile(np.array([0.2, 0.7, 0.9]), (len(free_pts), 1))
            all_pts = np.vstack([occ_pts, free_pts])
            all_col = np.vstack([occ_col, free_col])
        else:
            all_pts = occ_pts
            all_col = occ_col

        pcd = o3d.geometry.PointCloud()
        if len(all_pts) > 0:
            pcd.points = o3d.utility.Vector3dVector(all_pts)
            pcd.colors = o3d.utility.Vector3dVector(all_col)
        return pcd

    def get_stats(self) -> Dict[str, Union[int, float]]:
        """Return comprehensive occupancy grid statistics."""
        n_unknown = int(np.sum(self.grid == VoxelState.UNKNOWN))
        n_free = int(np.sum(self.grid == VoxelState.FREE))
        n_occupied = int(np.sum(self.grid == VoxelState.OCCUPIED))
        total = self.total_voxels

        return {
            "total_voxels": total,
            "nx": self.nx,
            "ny": self.ny,
            "nz": self.nz,
            "resolution_m": self.resolution,
            "memory_mb": self.memory_mb,
            "unknown_voxels": n_unknown,
            "free_voxels": n_free,
            "occupied_voxels": n_occupied,
            "unknown_pct": 100.0 * n_unknown / total,
            "free_pct": 100.0 * n_free / total,
            "occupied_pct": 100.0 * n_occupied / total,
        }

    def save_npz(self, path: Union[str, Path]) -> Path:
        """Save grid state and metadata to compressed NPZ archive."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(p),
            grid=self.grid,
            bounds_x=self.bounds_x,
            bounds_y=self.bounds_y,
            bounds_z=self.bounds_z,
            resolution=self.resolution,
        )
        return p

    @classmethod
    def load_npz(cls, path: Union[str, Path]) -> "OccupancyGrid3D":
        """Load grid state and metadata from NPZ archive."""
        data = np.load(str(path))
        grid = cls(
            bounds_x=tuple(data["bounds_x"]),
            bounds_y=tuple(data["bounds_y"]),
            bounds_z=tuple(data["bounds_z"]),
            resolution=float(data["resolution"]),
        )
        grid.grid = data["grid"]
        return grid

