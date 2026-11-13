"""
occupancy/tsdf.py — Truncated Signed Distance Function (TSDF) Surface Representation
====================================================================================

Implements a 3D TSDF Volume for fine, sub-voxel continuous surface reconstruction:
  - Each voxel stores a signed distance value d in [-1, +1] (normalized by truncation margin)
    and an integration weight w.
  - Multi-camera views are incrementally fused into the common volume via weighted running average.
  - Surface zero-crossings yield smooth continuous boundary meshes or point clouds.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import open3d as o3d

from pipeline.calibration import CameraModel


class TSDFVolume3D:
    """
    3D Truncated Signed Distance Function (TSDF) Volume.

    Parameters
    ----------
    bounds_x : Tuple[float, float]
        [min_x, max_x] in metres (default [-10.0, 10.0]).
    bounds_y : Tuple[float, float]
        [min_y, max_y] in metres (default [-10.0, 10.0]).
    bounds_z : Tuple[float, float]
        [min_z, max_z] in metres (default [-0.2, 2.8]).
    voxel_size : float
        Grid resolution in metres (default 0.05 m = 5 cm).
    truncation_margin : float
        Truncation band distance in metres (default 0.15 m = 3 * voxel_size).
    """

    def __init__(
        self,
        bounds_x: Tuple[float, float] = (-10.0, 10.0),
        bounds_y: Tuple[float, float] = (-10.0, 10.0),
        bounds_z: Tuple[float, float] = (-0.2, 2.8),
        voxel_size: float = 0.05,
        truncation_margin: float = 0.15,
    ):
        self.bounds_x = tuple(bounds_x)
        self.bounds_y = tuple(bounds_y)
        self.bounds_z = tuple(bounds_z)
        self.voxel_size = float(voxel_size)
        self.truncation_margin = float(truncation_margin)

        self.min_bound = np.array([bounds_x[0], bounds_y[0], bounds_z[0]], dtype=np.float64)
        self.max_bound = np.array([bounds_x[1], bounds_y[1], bounds_z[1]], dtype=np.float64)

        self.dims = np.ceil((self.max_bound - self.min_bound) / self.voxel_size).astype(np.int64)
        self.nx, self.ny, self.nz = int(self.dims[0]), int(self.dims[1]), int(self.dims[2])

        # TSDF values in [-1, +1] initialized to 1.0 (empty/free space)
        self.tsdf = np.ones((self.nx, self.ny, self.nz), dtype=np.float32)
        # Weights initialized to 0
        self.weight = np.zeros((self.nx, self.ny, self.nz), dtype=np.float32)

    @property
    def total_voxels(self) -> int:
        return self.nx * self.ny * self.nz

    def integrate_view(
        self,
        depth: np.ndarray,
        camera_model: CameraModel,
        weight: float = 1.0,
        max_depth: float = 25.0,
    ) -> None:
        """
        Integrate a single camera depth map into the TSDF volume.

        Parameters
        ----------
        depth : (H, W) float32 in metres
        camera_model : CameraModel with intrinsics and T_cam_world
        weight : float, confidence weight
        max_depth : float, depth limit
        """
        H, W = depth.shape
        intr = camera_model.intrinsics
        T_cam_world = camera_model.T_cam_world

        # Voxel 3D grid centers in world coordinates (slice-by-slice to maintain low peak RAM)
        mu = self.truncation_margin
        step = self.voxel_size

        x_coords = np.arange(self.nx) * step + self.min_bound[0] + 0.5 * step
        y_coords = np.arange(self.ny) * step + self.min_bound[1] + 0.5 * step

        xx, yy = np.meshgrid(x_coords, y_coords, indexing="ij")
        n_xy = self.nx * self.ny

        for iz in range(self.nz):
            z_val = iz * step + self.min_bound[2] + 0.5 * step
            pts_world = np.column_stack([xx.ravel(), yy.ravel(), np.full(n_xy, z_val)])

            # Transform to camera frame: p_cam = T_cam_world @ [X, Y, Z, 1]^T
            hom = np.hstack([pts_world, np.ones((n_xy, 1), dtype=np.float64)])
            p_cam = (T_cam_world @ hom.T).T[:, :3]

            z_cam = p_cam[:, 2]
            valid_z = (z_cam > 0.1) & (z_cam < max_depth)

            u = np.zeros(n_xy, dtype=np.float64)
            v = np.zeros(n_xy, dtype=np.float64)

            u[valid_z] = intr.fx * (p_cam[valid_z, 0] / z_cam[valid_z]) + intr.cx
            v[valid_z] = intr.fy * (p_cam[valid_z, 1] / z_cam[valid_z]) + intr.cy

            in_image = (
                valid_z
                & (u >= 0) & (u < W)
                & (v >= 0) & (v < H)
            )

            if not np.any(in_image):
                continue

            u_int = np.clip(np.round(u[in_image]).astype(np.int64), 0, W - 1)
            v_int = np.clip(np.round(v[in_image]).astype(np.int64), 0, H - 1)
            z_c = z_cam[in_image]

            measured_depth = depth[v_int, u_int]
            valid_depth = (measured_depth > 0.1) & (measured_depth < max_depth)

            if not np.any(valid_depth):
                continue

            # Signed distance: positive in front of surface, negative behind surface
            sdf = measured_depth[valid_depth] - z_c[valid_depth]
            # Only integrate voxels within the truncation band: sdf > -mu
            in_band = sdf > -mu
            if not np.any(in_band):
                continue

            tsdf_vals = np.clip(sdf[in_band] / mu, -1.0, 1.0)

            # Map back to volume indices
            xy_valid_indices = np.where(in_image)[0][valid_depth][in_band]
            ix = xy_valid_indices // self.ny
            iy = xy_valid_indices % self.ny

            # Update running average
            curr_tsdf = self.tsdf[ix, iy, iz]
            curr_w = self.weight[ix, iy, iz]

            new_w = curr_w + weight
            new_tsdf = (curr_tsdf * curr_w + tsdf_vals * weight) / new_w

            self.tsdf[ix, iy, iz] = new_tsdf
            self.weight[ix, iy, iz] = new_w

    def extract_zero_crossing_points(
        self,
        min_weight: float = 1.0,
        tsdf_threshold: float = 0.15,
    ) -> np.ndarray:
        """
        Extract surface points at the zero-crossing boundary (|tsdf| < threshold with positive weight).
        """
        surface_mask = (np.abs(self.tsdf) < tsdf_threshold) & (self.weight >= min_weight)
        indices = np.argwhere(surface_mask)
        if len(indices) == 0:
            return np.zeros((0, 3), dtype=np.float64)

        pts_world = (indices.astype(np.float64) + 0.5) * self.voxel_size + self.min_bound
        return pts_world

    def to_point_cloud(self, min_weight: float = 1.0) -> o3d.geometry.PointCloud:
        """Convert zero-crossing surface into an Open3D PointCloud."""
        pts = self.extract_zero_crossing_points(min_weight=min_weight)
        pcd = o3d.geometry.PointCloud()
        if len(pts) > 0:
            pcd.points = o3d.utility.Vector3dVector(pts)
            # Color by surface weight
            pcd.paint_uniform_color([0.2, 0.8, 0.3])
        return pcd

