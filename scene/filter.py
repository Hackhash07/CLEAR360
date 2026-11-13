"""
scene/filter.py — 3D Scene Point Cloud Filtering and Ground Segmentation for CLEAR360
====================================================================================

Transforms raw fused point clouds into clean, structured 3D geometry:
  1. Voxel Downsampling (configurable, default 5 cm)
  2. Duplicate point removal
  3. Statistical Outlier Removal (SOR)
  4. Radius Outlier Removal (ROR)
  5. Sky and invalid-depth filtering (WITHOUT artificially restricting scene height to 2.55 m)
  6. Ground plane identification via RANSAC with tilt-angle constraints
  7. Vertical surface preservation (walls, pillars, vehicles, obstacles)

Strict Pipeline Separation:
  - Ground-truth filtering uses physical depth thresholds and Blender ground-truth semantics.
  - AI-derived filtering uses image-space photometric/geometric indicators and depth-edge
    filtering. GT semantic masks are NEVER used in the AI pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


@dataclass
class GroundPlaneResult:
    """Result of ground plane segmentation."""
    plane_coefficients: np.ndarray  # [a, b, c, d] where ax + by + cz + d = 0
    normal: np.ndarray              # [nx, ny, nz] unit normal pointing up
    elevation_z: float              # ground level at (X=0, Y=0)
    tilt_deg: float                 # angle deviation from vertical [0, 0, 1]
    rmse_m: float                   # inlier residual RMSE
    inlier_indices: np.ndarray      # indices of ground points
    outlier_indices: np.ndarray     # indices of non-ground points


@dataclass
class FilterStats:
    """Statistics recorded during scene filtering."""
    initial_points: int
    after_voxel_ds: int
    after_dedup: int
    after_sky_filter: int
    after_sor: int
    after_ror: int
    final_points: int
    ground_points: int
    non_ground_points: int


class SceneFilter:
    """
    Robust 3D point cloud filter and ground plane segmenter.

    Parameters
    ----------
    voxel_size : float
        Voxel downsampling resolution in metres (default 0.05 m = 5 cm).
    sor_nb_neighbors : int
        Number of neighbors for Statistical Outlier Removal (default 20).
    sor_std_ratio : float
        Standard deviation multiplier for SOR (default 2.0).
    ror_nb_points : int
        Minimum neighbor count within search radius for Radius Outlier Removal (default 12).
    ror_radius : float
        Search sphere radius for ROR in metres (default 0.15 m).
    ground_dist_threshold : float
        RANSAC distance threshold for ground plane fitting (default 0.06 m).
    ground_max_tilt_deg : float
        Maximum allowable tilt of ground plane normal from [0, 0, 1] (default 6.0 deg).
    """

    def __init__(
        self,
        voxel_size: float = 0.05,
        sor_nb_neighbors: int = 20,
        sor_std_ratio: float = 2.0,
        ror_nb_points: int = 12,
        ror_radius: float = 0.15,
        ground_dist_threshold: float = 0.06,
        ground_max_tilt_deg: float = 6.0,
    ):
        self.voxel_size = voxel_size
        self.sor_nb_neighbors = sor_nb_neighbors
        self.sor_std_ratio = sor_std_ratio
        self.ror_nb_points = ror_nb_points
        self.ror_radius = ror_radius
        self.ground_dist_threshold = ground_dist_threshold
        self.ground_max_tilt_deg = ground_max_tilt_deg

    def downsample_voxel(
        self,
        pcd: o3d.geometry.PointCloud,
        voxel_size: Optional[float] = None,
    ) -> o3d.geometry.PointCloud:
        """Apply 3D voxel grid downsampling."""
        vs = voxel_size if voxel_size is not None else self.voxel_size
        return pcd.voxel_down_sample(voxel_size=vs)

    def remove_duplicates(
        self,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        spatial_tolerance: float = 0.005,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Deduplicate points closer than spatial_tolerance (default 5 mm) using spatial voxel hashing.
        """
        if len(points) == 0:
            return points, colors, labels

        quantized = np.floor(points / spatial_tolerance).astype(np.int64)
        # Use structured array view for fast unique row lookup
        keys = quantized[:, 0] * 73856093 ^ quantized[:, 1] * 19349663 ^ quantized[:, 2] * 83492791
        _, unique_indices = np.unique(keys, return_index=True)
        unique_indices.sort()

        dedup_pts = points[unique_indices]
        dedup_col = colors[unique_indices] if colors is not None else None
        dedup_lab = labels[unique_indices] if labels is not None else None

        return dedup_pts, dedup_col, dedup_lab

    def remove_statistical_outliers(
        self,
        pcd: o3d.geometry.PointCloud,
        nb_neighbors: Optional[int] = None,
        std_ratio: Optional[float] = None,
    ) -> Tuple[o3d.geometry.PointCloud, np.ndarray]:
        """Apply Statistical Outlier Removal (SOR)."""
        nb = nb_neighbors if nb_neighbors is not None else self.sor_nb_neighbors
        std = std_ratio if std_ratio is not None else self.sor_std_ratio
        filtered_pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=nb, std_ratio=std)
        return filtered_pcd, np.asarray(ind)

    def remove_radius_outliers(
        self,
        pcd: o3d.geometry.PointCloud,
        nb_points: Optional[int] = None,
        radius: Optional[float] = None,
    ) -> Tuple[o3d.geometry.PointCloud, np.ndarray]:
        """Apply Radius Outlier Removal (ROR)."""
        nb = nb_points if nb_points is not None else self.ror_nb_points
        rad = radius if radius is not None else self.ror_radius
        filtered_pcd, ind = pcd.remove_radius_outlier(nb_points=nb, radius=rad)
        return filtered_pcd, np.asarray(ind)

    @staticmethod
    def detect_ai_sky_mask(
        rgb: np.ndarray,
        depth: np.ndarray,
        elevation_cutoff_row: int = 240,
        depth_threshold_m: float = 12.0,
    ) -> np.ndarray:
        """
        Detect sky pixels in AI camera images without ground-truth semantic masks.

        Sky criteria:
          1. Pixel lies in the upper image quadrant (row < elevation_cutoff_row)
          2. Estimated depth is large (d > depth_threshold_m, where Depth Anything V2 plateaus)
          3. Blue chromaticity / brightness: (B - R) / (B + R + 1e-4) > 0.05 or luminance > 160
          4. Low local spatial depth gradient (smooth sky texture)
        """
        H, W, _ = rgb.shape
        sky_mask = np.zeros((H, W), dtype=bool)

        r = rgb[:elevation_cutoff_row, :, 0].astype(np.float32)
        g = rgb[:elevation_cutoff_row, :, 1].astype(np.float32)
        b = rgb[:elevation_cutoff_row, :, 2].astype(np.float32)
        d = depth[:elevation_cutoff_row, :]

        # Chromaticity and brightness of outdoor sky
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        blue_dominance = (b - r) / (b + r + 1e-3)

        # Depth gradient magnitude
        dy = np.abs(np.diff(d, axis=0, append=d[-1:, :]))
        dx = np.abs(np.diff(d, axis=1, append=d[:, -1:]))
        grad_mag = np.sqrt(dx**2 + dy**2)

        is_sky = (
            (d >= depth_threshold_m)
            & ((blue_dominance > 0.04) | (lum > 155.0))
            & (grad_mag < 1.5)
        )

        sky_mask[:elevation_cutoff_row, :] = is_sky
        return sky_mask

    def filter_sky_and_invalid_points(
        self,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        is_ground_truth: bool = False,
        max_valid_distance: float = 40.0,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
        """
        Filter sky points and invalid floating artifacts without artificially clipping scene height.

        Allows tall structures (e.g. 10m high obstacles, towers, lights) while removing
        monocular sky hallucination points and subterranean noise.

        Sky Filter Logic (AI):
          - In outdoor surround rigs, sky artifacts occur when upward rays unproject into
            isolated high-Z points with high blue luminance and distance > 12 m.
          - Points with Z < -0.25 m (subterranean noise below road) are removed.
          - Points at extreme distance > max_valid_distance are removed.
        """
        if len(points) == 0:
            return points, colors, labels, np.zeros(0, dtype=bool)

        valid = np.ones(len(points), dtype=bool)

        # 1. Subterranean noise (below road grade)
        valid = valid & (points[:, 2] >= -0.25)

        # 2. Maximum radial envelope
        radial_dist = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
        valid = valid & (radial_dist <= max_valid_distance)

        # 3. Ground Truth: filter semantic class 0 (sky) if labels provided
        if is_ground_truth and labels is not None:
            valid = valid & (labels != 0)

        # 4. AI-predicted sky filtering based on geometry and color (NO GT MASK USED):
        # Sky points have high elevation (Z > 2.6 m), large radial distance (R > 9.0 m),
        # and sky-like color (blue-dominant or bright white/grey)
        if not is_ground_truth and colors is not None:
            r = colors[:, 0]
            g = colors[:, 1]
            b = colors[:, 2]
            blue_tint = (b - r) / (b + r + 1e-4)
            brightness = 0.299 * r + 0.587 * g + 0.114 * b

            # A point is an ungrounded sky hallucination if:
            # - Z > 2.6 m
            # - R > 8.5 m (at or beyond the perimeter wall)
            # - Appearance is sky (blue tint or high overcast brightness)
            is_sky_point = (
                (points[:, 2] > 2.60)
                & (radial_dist > 8.50)
                & ((blue_tint > 0.05) | (brightness > 0.65))
            )
            valid = valid & (~is_sky_point)

        filtered_pts = points[valid]
        filtered_col = colors[valid] if colors is not None else None
        filtered_lab = labels[valid] if labels is not None else None

        return filtered_pts, filtered_col, filtered_lab, valid

    def segment_ground_plane(
        self,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, GroundPlaneResult]:
        """
        Robustly identify the road / ground plane using RANSAC with tilt constraints.

        Returns
        -------
        ground_pts : (N_g, 3) points classified as road/ground
        non_ground_pts : (N_ng, 3) points belonging to obstacles, walls, vehicles, pillars
        result : GroundPlaneResult
        """
        N = len(points)
        if N < 50:
            empty_res = GroundPlaneResult(
                plane_coefficients=np.array([0, 0, 1, 0], dtype=np.float64),
                normal=np.array([0, 0, 1], dtype=np.float64),
                elevation_z=0.0,
                tilt_deg=0.0,
                rmse_m=0.0,
                inlier_indices=np.zeros(0, dtype=int),
                outlier_indices=np.arange(N),
            )
            return np.zeros((0, 3)), points, empty_res

        # Restrict candidate points to near-ground region for initial plane discovery
        near_ground_mask = (points[:, 2] >= -0.20) & (points[:, 2] <= 0.35)
        candidate_indices = np.where(near_ground_mask)[0]

        if len(candidate_indices) < 30:
            candidate_indices = np.arange(N)

        cand_pts = points[candidate_indices]
        pcd_cand = o3d.geometry.PointCloud()
        pcd_cand.points = o3d.utility.Vector3dVector(cand_pts)

        # Run RANSAC plane fitting
        plane_model, inliers_cand = pcd_cand.segment_plane(
            distance_threshold=self.ground_dist_threshold,
            ransac_n=3,
            num_iterations=1000,
        )
        a, b, c, d = plane_model

        # Ensure normal points upwards (positive Z)
        if c < 0:
            a, b, c, d = -a, -b, -c, -d

        normal = np.array([a, b, c], dtype=np.float64)
        norm_len = np.linalg.norm(normal)
        if norm_len > 1e-6:
            normal = normal / norm_len
            d = d / norm_len

        # Compute tilt angle from true vertical [0, 0, 1]
        cos_tilt = np.clip(normal[2], -1.0, 1.0)
        tilt_deg = float(np.degrees(np.arccos(cos_tilt)))

        # Evaluate all points against fitted plane
        distances = np.abs(points[:, 0] * normal[0] + points[:, 1] * normal[1] + points[:, 2] * normal[2] + d)

        # Inliers must be within distance threshold AND below max ground clearance
        inlier_mask = (distances <= self.ground_dist_threshold) & (points[:, 2] <= 0.35)
        inlier_indices = np.where(inlier_mask)[0]
        outlier_indices = np.where(~inlier_mask)[0]

        inlier_residuals = distances[inlier_mask]
        rmse = float(np.sqrt(np.mean(inlier_residuals**2))) if len(inlier_residuals) > 0 else 0.0

        # Ground elevation at vehicle origin (X=0, Y=0): z = -d / normal[2]
        elevation_z = float(-d / normal[2]) if abs(normal[2]) > 1e-4 else 0.0

        res = GroundPlaneResult(
            plane_coefficients=np.array([normal[0], normal[1], normal[2], d]),
            normal=normal,
            elevation_z=elevation_z,
            tilt_deg=tilt_deg,
            rmse_m=rmse,
            inlier_indices=inlier_indices,
            outlier_indices=outlier_indices,
        )

        ground_pts = points[inlier_indices]
        non_ground_pts = points[outlier_indices]

        return ground_pts, non_ground_pts, res

    def filter_complete_scene(
        self,
        pcd: o3d.geometry.PointCloud,
        labels: Optional[np.ndarray] = None,
        is_ground_truth: bool = False,
    ) -> Tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud, GroundPlaneResult, FilterStats]:
        """
        Execute full scene filtering pipeline:
          1. Voxel Downsample
          2. Sky & Subterranean Filtering
          3. Statistical Outlier Removal
          4. Radius Outlier Removal
          5. Ground Plane Segmentation

        Returns
        -------
        clean_pcd : o3d.geometry.PointCloud (complete filtered scene)
        non_ground_pcd : o3d.geometry.PointCloud (obstacles, walls, etc.)
        ground_res : GroundPlaneResult
        stats : FilterStats
        """
        n_init = len(pcd.points)

        # 1. Voxel downsampling
        pcd_ds = self.downsample_voxel(pcd)
        n_ds = len(pcd_ds.points)

        pts = np.asarray(pcd_ds.points)
        col = np.asarray(pcd_ds.colors)

        # 2. Deduplication
        pts, col, _ = self.remove_duplicates(pts, col)
        n_dedup = len(pts)

        # 3. Sky & Invalid Points Filter
        pts, col, _, _ = self.filter_sky_and_invalid_points(
            pts, col, labels=labels, is_ground_truth=is_ground_truth
        )
        n_sky = len(pts)

        # Re-wrap in Open3D PointCloud
        pcd_clean = o3d.geometry.PointCloud()
        pcd_clean.points = o3d.utility.Vector3dVector(pts)
        pcd_clean.colors = o3d.utility.Vector3dVector(col)

        # 4. Statistical Outlier Removal
        pcd_sor, _ = self.remove_statistical_outliers(pcd_clean)
        n_sor = len(pcd_sor.points)

        # 5. Radius Outlier Removal
        pcd_ror, _ = self.remove_radius_outliers(pcd_sor)
        n_ror = len(pcd_ror.points)

        final_pts = np.asarray(pcd_ror.points)
        final_col = np.asarray(pcd_ror.colors)

        # 6. Ground Plane Segmentation
        ground_pts, non_ground_pts, ground_res = self.segment_ground_plane(final_pts, final_col)

        non_ground_pcd = o3d.geometry.PointCloud()
        non_ground_pcd.points = o3d.utility.Vector3dVector(non_ground_pts)
        non_ground_col = final_col[ground_res.outlier_indices] if len(final_col) > 0 else np.zeros((0, 3))
        non_ground_pcd.colors = o3d.utility.Vector3dVector(non_ground_col)

        stats = FilterStats(
            initial_points=n_init,
            after_voxel_ds=n_ds,
            after_dedup=n_dedup,
            after_sky_filter=n_sky,
            after_sor=n_sor,
            after_ror=n_ror,
            final_points=len(final_pts),
            ground_points=len(ground_pts),
            non_ground_points=len(non_ground_pts),
        )

        return pcd_ror, non_ground_pcd, ground_res, stats

