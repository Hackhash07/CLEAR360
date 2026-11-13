"""
fusion/overlap.py — Pairwise Camera Overlap Consistency and Alignment Analyzer
==============================================================================

Evaluates geometric consistency in the four surround camera overlap regions:
    1. Front-Left  (FL) : Overlap between 'front' and 'left' cameras
    2. Front-Right (FR) : Overlap between 'front' and 'right' cameras
    3. Rear-Left   (RL) : Overlap between 'rear'  and 'left' cameras
    4. Rear-Right  (RR) : Overlap between 'rear'  and 'right' cameras

Metrics computed per overlap region:
    - Overlapping point counts (N_cam_A, N_cam_B)
    - Nearest-neighbor Euclidean distance:
        * Median NN distance (m)
        * Mean NN distance (m)
        * 95th percentile NN distance (m)
        * Max NN distance (m)
        * Inlier ratio (< 0.10 m, < 0.20 m, < 0.50 m)
    - Point-to-plane residual using Open3D surface normals
    - Separation of co-observed geometry vs occlusion/unobserved boundaries
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from pipeline.calibration import CalibrationStore, CameraModel


@dataclass
class OverlapMetrics:
    """
    Quantitative geometric consistency metrics for a single camera pair.

    Attributes
    ----------
    region_name         : str ('front_left', 'front_right', 'rear_left', 'rear_right')
    cam_a               : str
    cam_b               : str
    num_points_a        : int
    num_points_b        : int
    num_points_total    : int
    median_nn_dist_m    : float
    mean_nn_dist_m      : float
    p95_nn_dist_m       : float
    max_nn_dist_m       : float
    inlier_ratio_10cm   : float  # fraction with NN dist < 0.10 m
    inlier_ratio_20cm   : float  # fraction with NN dist < 0.20 m
    inlier_ratio_50cm   : float  # fraction with NN dist < 0.50 m
    median_pt2plane_m   : Optional[float]
    mean_pt2plane_m     : Optional[float]
    p95_pt2plane_m      : Optional[float]
    co_observed_median_m: float  # median error among inliers (< 0.50 m, excludes occlusions)
    """
    region_name: str
    cam_a: str
    cam_b: str
    num_points_a: int
    num_points_b: int
    num_points_total: int
    median_nn_dist_m: float
    mean_nn_dist_m: float
    p95_nn_dist_m: float
    max_nn_dist_m: float
    inlier_ratio_10cm: float
    inlier_ratio_20cm: float
    inlier_ratio_50cm: float
    median_pt2plane_m: Optional[float]
    mean_pt2plane_m: Optional[float]
    p95_pt2plane_m: Optional[float]
    co_observed_median_m: float

    def to_dict(self) -> dict:
        return asdict(self)


# Overlap angular wedges (azimuth in degrees: atan2(Y, X)):
# World coordinates: X = Right, Y = Forward
# Front: +Y (90 deg), FOV 120 deg -> [30, 150]
# Left : -X (180 deg), FOV 120 deg -> [120, 240]
# Rear : -Y (270 deg / -90 deg), FOV 120 deg -> [210, 330]
# Right: +X (0 deg), FOV 120 deg -> [-60, +60] or [300, 360] U [0, 60]
OVERLAP_REGIONS: Dict[str, Dict] = {
    "front_left": {
        "cam_a": "front",
        "cam_b": "left",
        "azimuth_min_deg": 120.0,
        "azimuth_max_deg": 150.0,
    },
    "front_right": {
        "cam_a": "front",
        "cam_b": "right",
        "azimuth_min_deg": 30.0,
        "azimuth_max_deg": 60.0,
    },
    "rear_left": {
        "cam_a": "rear",
        "cam_b": "left",
        "azimuth_min_deg": 210.0,
        "azimuth_max_deg": 240.0,
    },
    "rear_right": {
        "cam_a": "rear",
        "cam_b": "right",
        "azimuth_min_deg": 300.0,
        "azimuth_max_deg": 330.0,
    },
}


class OverlapAnalyzer:
    """
    Measures and visualizes geometric consistency in camera overlap regions.

    Parameters
    ----------
    calibration_store : CalibrationStore
        Calibrated camera models with intrinsics and extrinsics.
    occlusion_dist_threshold : float
        Distance threshold (metres) beyond which a point is classified as an
        occlusion/unobserved area rather than geometric misalignment (default 0.50 m).
    """

    def __init__(
        self,
        calibration_store: CalibrationStore,
        occlusion_dist_threshold: float = 0.50,
    ):
        self.calibration_store = calibration_store
        self.occlusion_dist_threshold = occlusion_dist_threshold

    def filter_points_by_azimuth(
        self,
        points: np.ndarray,
        azimuth_min_deg: float,
        azimuth_max_deg: float,
    ) -> np.ndarray:
        """
        Filter 3D points in world coordinates by horizontal azimuth angle.

        Azimuth angle theta = atan2(Y, X) converted to [0, 360) degrees.
        """
        if len(points) == 0:
            return np.zeros(0, dtype=bool)

        x = points[:, 0]
        y = points[:, 1]
        azimuth = np.degrees(np.arctan2(y, x)) % 360.0

        if azimuth_min_deg <= azimuth_max_deg:
            mask = (azimuth >= azimuth_min_deg) & (azimuth <= azimuth_max_deg)
        else:
            # Wrap around 360 / 0 boundary
            mask = (azimuth >= azimuth_min_deg) | (azimuth <= azimuth_max_deg)

        return mask

    def is_visible_in_camera(
        self,
        points_world: np.ndarray,
        camera_name: str,
        margin_px: float = 0.0,
    ) -> np.ndarray:
        """
        Test whether world points project inside the camera's image frustum.

        Parameters
        ----------
        points_world : (N, 3) float64
        camera_name : str
        margin_px : float, pixel border inset/margin

        Returns
        -------
        mask : (N,) bool
        """
        if len(points_world) == 0:
            return np.zeros(0, dtype=bool)

        cam = self.calibration_store.get(camera_name)
        T_cam_world = cam.T_cam_world
        intr = cam.intrinsics

        N = len(points_world)
        hom = np.hstack([points_world, np.ones((N, 1), dtype=np.float64)])
        p_cam = (T_cam_world @ hom.T).T[:, :3]

        z_cam = p_cam[:, 2]
        # In front of camera
        valid_z = z_cam > 0.05

        u = np.zeros(N, dtype=np.float64)
        v = np.zeros(N, dtype=np.float64)

        u[valid_z] = intr.fx * (p_cam[valid_z, 0] / z_cam[valid_z]) + intr.cx
        v[valid_z] = intr.fy * (p_cam[valid_z, 1] / z_cam[valid_z]) + intr.cy

        in_bounds = (
            valid_z
            & (u >= margin_px)
            & (u < (intr.width - margin_px))
            & (v >= margin_px)
            & (v < (intr.height - margin_px))
        )
        return in_bounds

    def extract_overlap_points(
        self,
        cam_a_pcd: o3d.geometry.PointCloud,
        cam_b_pcd: o3d.geometry.PointCloud,
        cam_a_name: str,
        cam_b_name: str,
        azimuth_min_deg: float,
        azimuth_max_deg: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract points from camera A and camera B that fall into the shared overlap region.
        """
        pts_a = np.asarray(cam_a_pcd.points)
        pts_b = np.asarray(cam_b_pcd.points)

        if len(pts_a) == 0 or len(pts_b) == 0:
            return np.zeros((0, 3)), np.zeros((0, 3))

        # Filter by azimuth wedge
        mask_a_azim = self.filter_points_by_azimuth(pts_a, azimuth_min_deg, azimuth_max_deg)
        mask_b_azim = self.filter_points_by_azimuth(pts_b, azimuth_min_deg, azimuth_max_deg)

        pts_a_wedge = pts_a[mask_a_azim]
        pts_b_wedge = pts_b[mask_b_azim]

        # Verify frustum visibility in opposing camera
        vis_a_in_b = self.is_visible_in_camera(pts_a_wedge, cam_b_name)
        vis_b_in_a = self.is_visible_in_camera(pts_b_wedge, cam_a_name)

        overlap_a = pts_a_wedge[vis_a_in_b]
        overlap_b = pts_b_wedge[vis_b_in_a]

        return overlap_a, overlap_b

    def analyze_pair(
        self,
        cam_a_pcd: o3d.geometry.PointCloud,
        cam_b_pcd: o3d.geometry.PointCloud,
        region_name: str,
    ) -> OverlapMetrics:
        """
        Analyze geometric consistency for a specific overlap pair.
        """
        cfg = OVERLAP_REGIONS[region_name]
        cam_a_name = cfg["cam_a"]
        cam_b_name = cfg["cam_b"]

        overlap_a, overlap_b = self.extract_overlap_points(
            cam_a_pcd,
            cam_b_pcd,
            cam_a_name,
            cam_b_name,
            cfg["azimuth_min_deg"],
            cfg["azimuth_max_deg"],
        )

        num_a = len(overlap_a)
        num_b = len(overlap_b)
        num_total = num_a + num_b

        if num_a == 0 or num_b == 0:
            return OverlapMetrics(
                region_name=region_name,
                cam_a=cam_a_name,
                cam_b=cam_b_name,
                num_points_a=num_a,
                num_points_b=num_b,
                num_points_total=num_total,
                median_nn_dist_m=float("nan"),
                mean_nn_dist_m=float("nan"),
                p95_nn_dist_m=float("nan"),
                max_nn_dist_m=float("nan"),
                inlier_ratio_10cm=0.0,
                inlier_ratio_20cm=0.0,
                inlier_ratio_50cm=0.0,
                median_pt2plane_m=None,
                mean_pt2plane_m=None,
                p95_pt2plane_m=None,
                co_observed_median_m=float("nan"),
            )

        # Build KD-trees for bidirectional nearest-neighbor queries
        tree_b = cKDTree(overlap_b)
        dists_a_to_b, idx_b = tree_b.query(overlap_a, k=1)

        tree_a = cKDTree(overlap_a)
        dists_b_to_a, idx_a = tree_a.query(overlap_b, k=1)

        # Symmetrized distances
        all_dists = np.concatenate([dists_a_to_b, dists_b_to_a])

        median_nn = float(np.median(all_dists))
        mean_nn = float(np.mean(all_dists))
        p95_nn = float(np.percentile(all_dists, 95))
        max_nn = float(np.max(all_dists))

        inliers_10cm = float(np.mean(all_dists < 0.10))
        inliers_20cm = float(np.mean(all_dists < 0.20))
        inliers_50cm = float(np.mean(all_dists < 0.50))

        # Co-observed geometry (excluding occlusion shadow regions > threshold)
        co_observed_dists = all_dists[all_dists < self.occlusion_dist_threshold]
        if len(co_observed_dists) > 0:
            co_observed_median = float(np.median(co_observed_dists))
        else:
            co_observed_median = median_nn

        # Point-to-plane residual using Open3D surface normals
        pcd_b_overlap = o3d.geometry.PointCloud()
        pcd_b_overlap.points = o3d.utility.Vector3dVector(overlap_b)
        pcd_b_overlap.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.20, max_nn=30)
        )
        normals_b = np.asarray(pcd_b_overlap.normals)

        if len(normals_b) == len(overlap_b):
            matched_b_pts = overlap_b[idx_b]
            matched_b_normals = normals_b[idx_b]
            diff = overlap_a - matched_b_pts
            pt2plane_residuals = np.abs(np.sum(diff * matched_b_normals, axis=1))

            # Filter valid normals (norm ~ 1.0)
            valid_norms = np.linalg.norm(matched_b_normals, axis=1) > 0.5
            if np.any(valid_norms):
                valid_res = pt2plane_residuals[valid_norms]
                median_pt2plane = float(np.median(valid_res))
                mean_pt2plane = float(np.mean(valid_res))
                p95_pt2plane = float(np.percentile(valid_res, 95))
            else:
                median_pt2plane, mean_pt2plane, p95_pt2plane = None, None, None
        else:
            median_pt2plane, mean_pt2plane, p95_pt2plane = None, None, None

        return OverlapMetrics(
            region_name=region_name,
            cam_a=cam_a_name,
            cam_b=cam_b_name,
            num_points_a=num_a,
            num_points_b=num_b,
            num_points_total=num_total,
            median_nn_dist_m=median_nn,
            mean_nn_dist_m=mean_nn,
            p95_nn_dist_m=p95_nn,
            max_nn_dist_m=max_nn,
            inlier_ratio_10cm=inliers_10cm,
            inlier_ratio_20cm=inliers_20cm,
            inlier_ratio_50cm=inliers_50cm,
            median_pt2plane_m=median_pt2plane,
            mean_pt2plane_m=mean_pt2plane,
            p95_pt2plane_m=p95_pt2plane,
            co_observed_median_m=co_observed_median,
        )

    def analyze_all_regions(
        self,
        camera_clouds: Dict[str, o3d.geometry.PointCloud],
    ) -> Dict[str, OverlapMetrics]:
        """
        Analyze all 4 overlap regions (front_left, front_right, rear_left, rear_right).
        """
        results = {}
        for region_name, cfg in OVERLAP_REGIONS.items():
            cam_a = cfg["cam_a"]
            cam_b = cfg["cam_b"]
            if cam_a in camera_clouds and cam_b in camera_clouds:
                metrics = self.analyze_pair(
                    camera_clouds[cam_a],
                    camera_clouds[cam_b],
                    region_name,
                )
                results[region_name] = metrics
        return results

    def generate_overlap_heatmap(
        self,
        camera_clouds: Dict[str, o3d.geometry.PointCloud],
        max_error_thresh_m: float = 0.20,
    ) -> o3d.geometry.PointCloud:
        """
        Generate a PointCloud of overlap points color-coded by alignment error:
        Green (< 0.05 m) -> Yellow (0.10 m) -> Red (> 0.20 m).
        """
        all_pts: List[np.ndarray] = []
        all_cols: List[np.ndarray] = []

        for region_name, cfg in OVERLAP_REGIONS.items():
            cam_a = cfg["cam_a"]
            cam_b = cfg["cam_b"]
            if cam_a not in camera_clouds or cam_b not in camera_clouds:
                continue

            pts_a, pts_b = self.extract_overlap_points(
                camera_clouds[cam_a],
                camera_clouds[cam_b],
                cam_a,
                cam_b,
                cfg["azimuth_min_deg"],
                cfg["azimuth_max_deg"],
            )

            if len(pts_a) == 0 or len(pts_b) == 0:
                continue

            tree_b = cKDTree(pts_b)
            dists_a, _ = tree_b.query(pts_a, k=1)

            # Map distance to RGB heatmap: 0.0 -> Green, 0.10 -> Yellow, >= 0.20 -> Red
            normalized = np.clip(dists_a / max_error_thresh_m, 0.0, 1.0)
            colors_a = np.zeros((len(pts_a), 3), dtype=np.float64)

            # Red channel rises from 0 to 1
            colors_a[:, 0] = np.clip(2.0 * normalized, 0.0, 1.0)
            # Green channel falls from 1 to 0
            colors_a[:, 1] = np.clip(2.0 * (1.0 - normalized), 0.0, 1.0)
            colors_a[:, 2] = 0.1

            all_pts.append(pts_a)
            all_cols.append(colors_a)

        pcd = o3d.geometry.PointCloud()
        if all_pts:
            pcd.points = o3d.utility.Vector3dVector(np.vstack(all_pts))
            pcd.colors = o3d.utility.Vector3dVector(np.vstack(all_cols))
        return pcd

