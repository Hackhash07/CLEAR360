"""
fusion/distortion.py — Geometric Distortion and Bounding Box Investigation for CLEAR360
======================================================================================

Investigates why the calibrated AI-predicted point cloud bounding box is wider
than ground truth ([-15.97, 13.99] x [-14.19, 15.55] x [-0.63, 6.37] m vs [-9.36, 9.36] x [-9.36, 9.36] x [0.00, 2.51] m).

Performs rigorous geometric sanity checks:
    1. Road height profile: Flatness (RMSE), slope angle, mean elevation
    2. Wall verticality and distance: North, South, East, West plane fits
    3. Pillar geometry: Radius and centroid error vs nominal (R=0.20 m at [-6.0, 4.0])
    4. Parked vehicle dimensions: Extent vs nominal (1.75 x 4.2 x 1.5 m)
    5. Peripheral distortion and sky/horizon leakage quantification
    6. Cloud-to-cloud residual error metrics and heatmap generation
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


@dataclass
class RoadProfileStats:
    num_points: int
    mean_elevation_z: float
    elevation_std_z: float
    planarity_rmse_m: float
    slope_angle_deg: float
    plane_normal: List[float]
    plane_intercept: float


@dataclass
class WallStats:
    wall_name: str
    nominal_distance_m: float
    fitted_distance_m: float
    distance_error_m: float
    normal_tilt_deg: float  # deviation from vertical (0 deg = perfectly vertical)
    num_points: int


@dataclass
class PillarStats:
    nominal_center: List[float]
    fitted_center: List[float]
    center_error_m: float
    nominal_radius_m: float
    fitted_radius_m: float
    radius_error_m: float
    height_extent_m: float
    num_points: int


@dataclass
class SkyLeakageStats:
    total_points: int
    sky_leakage_points: int       # points with Z > 2.55 m (above wall height)
    sky_leakage_pct: float
    sub_ground_points: int        # points with Z < -0.15 m
    sub_ground_pct: float
    perimeter_leakage_points: int # points with |X| > 9.5 m or |Y| > 9.5 m
    perimeter_leakage_pct: float
    in_boundary_points: int
    in_boundary_pct: float


@dataclass
class CloudComparisonStats:
    all_points_mae_m: float
    all_points_rmse_m: float
    all_points_median_m: float
    all_points_p95_m: float
    in_boundary_mae_m: float
    in_boundary_rmse_m: float
    in_boundary_median_m: float
    in_boundary_p95_m: float


class GeometricDistortionAnalyzer:
    """
    Analyzes geometric deformation, wall verticality, road flatness,
    and bounding-box dilation in reconstructed point clouds.
    """

    @staticmethod
    def analyze_road_profile(
        points: np.ndarray,
        z_min: float = -0.60,
        z_max: float = 0.35,
        radial_max: float = 8.5,
    ) -> RoadProfileStats:
        """
        Analyze ground plane flatness, elevation, and tilt angle.
        """
        # Filter candidate road points within near radius to avoid walls
        r = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
        mask = (points[:, 2] >= z_min) & (points[:, 2] <= z_max) & (r <= radial_max)
        road_pts = points[mask]

        if len(road_pts) < 100:
            return RoadProfileStats(
                num_points=len(road_pts),
                mean_elevation_z=float("nan"),
                elevation_std_z=float("nan"),
                planarity_rmse_m=float("nan"),
                slope_angle_deg=float("nan"),
                plane_normal=[0.0, 0.0, 1.0],
                plane_intercept=0.0,
            )

        # Fit plane Z = a*X + b*Y + c
        A = np.column_stack([road_pts[:, 0], road_pts[:, 1], np.ones(len(road_pts))])
        z = road_pts[:, 2]
        coeff, residuals, rank, s = np.linalg.lstsq(A, z, rcond=None)
        a, b, c = coeff

        # Normal vector: [-a, -b, 1] / sqrt(a^2 + b^2 + 1)
        norm_mag = np.sqrt(a**2 + b**2 + 1.0)
        n = np.array([-a, -b, 1.0]) / norm_mag

        # Slope angle: angle between n and [0, 0, 1]
        cos_slope = np.clip(n[2], -1.0, 1.0)
        slope_deg = float(np.degrees(np.arccos(np.abs(cos_slope))))

        # Planarity RMSE
        z_fitted = A @ coeff
        rmse = float(np.sqrt(np.mean((z - z_fitted) ** 2)))

        return RoadProfileStats(
            num_points=len(road_pts),
            mean_elevation_z=float(np.mean(z)),
            elevation_std_z=float(np.std(z)),
            planarity_rmse_m=rmse,
            slope_angle_deg=slope_deg,
            plane_normal=n.tolist(),
            plane_intercept=float(c),
        )

    @staticmethod
    def analyze_walls(points: np.ndarray) -> Dict[str, WallStats]:
        """
        Analyze North, South, East, West wall distances and verticality.
        Nominal enclosure: 4 walls at distance 9.5 m, height 2.5 m.
        """
        wall_definitions = {
            "north": {"coord": 1, "target": 9.5, "range": (8.0, 12.0), "perp_max": 8.0},
            "south": {"coord": 1, "target": -9.5, "range": (-12.0, -8.0), "perp_max": 8.0},
            "east":  {"coord": 0, "target": 9.5, "range": (8.0, 12.0), "perp_max": 8.0},
            "west":  {"coord": 0, "target": -9.5, "range": (-12.0, -8.0), "perp_max": 8.0},
        }

        results = {}
        for wname, wcfg in wall_definitions.items():
            coord_idx = wcfg["coord"]
            other_coord = 1 - coord_idx
            lo, hi = wcfg["range"]
            target = wcfg["target"]

            # Filter points for this wall
            mask = (
                (points[:, coord_idx] >= lo)
                & (points[:, coord_idx] <= hi)
                & (np.abs(points[:, other_coord]) <= wcfg["perp_max"])
                & (points[:, 2] >= 0.20)
                & (points[:, 2] <= 2.60)
            )
            wall_pts = points[mask]

            if len(wall_pts) < 50:
                results[wname] = WallStats(
                    wall_name=wname,
                    nominal_distance_m=abs(target),
                    fitted_distance_m=float("nan"),
                    distance_error_m=float("nan"),
                    normal_tilt_deg=float("nan"),
                    num_points=len(wall_pts),
                )
                continue

            # Fit plane to wall points using SVD / PCA
            centroid = wall_pts.mean(axis=0)
            shifted = wall_pts - centroid
            u, s, vh = np.linalg.svd(shifted, full_matrices=False)
            normal = vh[2]  # direction of minimum variance

            # Wall normal should be horizontal (n_z ~ 0). Deviation from vertical wall:
            tilt_deg = float(np.degrees(np.arcsin(np.clip(abs(normal[2]), 0.0, 1.0))))

            # Fitted distance from origin
            fitted_dist = float(abs(centroid[coord_idx]))
            dist_err = fitted_dist - abs(target)

            results[wname] = WallStats(
                wall_name=wname,
                nominal_distance_m=abs(target),
                fitted_distance_m=fitted_dist,
                distance_error_m=dist_err,
                normal_tilt_deg=tilt_deg,
                num_points=len(wall_pts),
            )

        return results

    @staticmethod
    def analyze_pillar(
        points: np.ndarray,
        nominal_xy: Tuple[float, float] = (-6.0, 4.0),
        nominal_r: float = 0.20,
    ) -> PillarStats:
        """
        Analyze cylindrical pillar radius and centroid.
        """
        x0, y0 = nominal_xy
        r_dist = np.sqrt((points[:, 0] - x0) ** 2 + (points[:, 1] - y0) ** 2)
        mask = (r_dist <= 0.60) & (points[:, 2] >= 0.20) & (points[:, 2] <= 2.40)
        pillar_pts = points[mask]

        if len(pillar_pts) < 30:
            return PillarStats(
                nominal_center=list(nominal_xy),
                fitted_center=[float("nan"), float("nan")],
                center_error_m=float("nan"),
                nominal_radius_m=nominal_r,
                fitted_radius_m=float("nan"),
                radius_error_m=float("nan"),
                height_extent_m=float("nan"),
                num_points=len(pillar_pts),
            )

        fitted_center = [float(np.mean(pillar_pts[:, 0])), float(np.mean(pillar_pts[:, 1]))]
        center_err = float(np.sqrt((fitted_center[0] - x0) ** 2 + (fitted_center[1] - y0) ** 2))

        # Radial distances from fitted center
        rads = np.sqrt(
            (pillar_pts[:, 0] - fitted_center[0]) ** 2
            + (pillar_pts[:, 1] - fitted_center[1]) ** 2
        )
        fitted_r = float(np.median(rads))
        rad_err = fitted_r - nominal_r
        h_extent = float(pillar_pts[:, 2].max() - pillar_pts[:, 2].min())

        return PillarStats(
            nominal_center=list(nominal_xy),
            fitted_center=fitted_center,
            center_error_m=center_err,
            nominal_radius_m=nominal_r,
            fitted_radius_m=fitted_r,
            radius_error_m=rad_err,
            height_extent_m=h_extent,
            num_points=len(pillar_pts),
        )

    @staticmethod
    def analyze_sky_leakage(
        points: np.ndarray,
        z_wall_top: float = 2.55,
        z_ground_floor: float = -0.15,
        xy_perimeter: float = 9.50,
    ) -> SkyLeakageStats:
        """
        Quantify sky leakage, sub-ground noise, and boundary over-extension.
        """
        total = len(points)
        if total == 0:
            return SkyLeakageStats(0, 0, 0.0, 0, 0.0, 0, 0.0, 0, 0.0)

        sky_mask = points[:, 2] > z_wall_top
        sub_mask = points[:, 2] < z_ground_floor
        perim_mask = (np.abs(points[:, 0]) > xy_perimeter) | (np.abs(points[:, 1]) > xy_perimeter)

        in_bounds_mask = (
            (points[:, 2] >= z_ground_floor)
            & (points[:, 2] <= z_wall_top)
            & (np.abs(points[:, 0]) <= xy_perimeter)
            & (np.abs(points[:, 1]) <= xy_perimeter)
        )

        n_sky = int(np.sum(sky_mask))
        n_sub = int(np.sum(sub_mask))
        n_perim = int(np.sum(perim_mask))
        n_in = int(np.sum(in_bounds_mask))

        return SkyLeakageStats(
            total_points=total,
            sky_leakage_points=n_sky,
            sky_leakage_pct=100.0 * n_sky / total,
            sub_ground_points=n_sub,
            sub_ground_pct=100.0 * n_sub / total,
            perimeter_leakage_points=n_perim,
            perimeter_leakage_pct=100.0 * n_perim / total,
            in_boundary_points=n_in,
            in_boundary_pct=100.0 * n_in / total,
        )

    @staticmethod
    def compare_point_clouds(
        ai_points: np.ndarray,
        gt_points: np.ndarray,
        z_wall_top: float = 2.55,
        xy_perimeter: float = 9.50,
    ) -> Tuple[CloudComparisonStats, np.ndarray]:
        """
        Compare AI-predicted point cloud directly against GT point cloud.
        Returns accuracy statistics and per-point Euclidean distance array.
        """
        tree_gt = cKDTree(gt_points)
        all_dists, _ = tree_gt.query(ai_points, k=1)

        all_mae = float(np.mean(all_dists))
        all_rmse = float(np.sqrt(np.mean(all_dists**2)))
        all_median = float(np.median(all_dists))
        all_p95 = float(np.percentile(all_dists, 95))

        # Filter in-boundary points (excluding sky leakage & peripheral distortion)
        in_boundary = (
            (ai_points[:, 2] >= -0.15)
            & (ai_points[:, 2] <= z_wall_top)
            & (np.abs(ai_points[:, 0]) <= xy_perimeter)
            & (np.abs(ai_points[:, 1]) <= xy_perimeter)
        )

        if np.any(in_boundary):
            in_dists = all_dists[in_boundary]
            in_mae = float(np.mean(in_dists))
            in_rmse = float(np.sqrt(np.mean(in_dists**2)))
            in_median = float(np.median(in_dists))
            in_p95 = float(np.percentile(in_dists, 95))
        else:
            in_mae = all_mae
            in_rmse = all_rmse
            in_median = all_median
            in_p95 = all_p95

        stats = CloudComparisonStats(
            all_points_mae_m=all_mae,
            all_points_rmse_m=all_rmse,
            all_points_median_m=all_median,
            all_points_p95_m=all_p95,
            in_boundary_mae_m=in_mae,
            in_boundary_rmse_m=in_rmse,
            in_boundary_median_m=in_median,
            in_boundary_p95_m=in_p95,
        )
        return stats, all_dists

