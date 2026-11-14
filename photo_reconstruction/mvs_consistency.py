"""
photo_reconstruction/mvs_consistency.py — Multi-View Stereo Geometric Verification Engine
========================================================================================

Elevates monocular neural depth predictions into genuine Multi-View Stereo (MVS) representations
by enforcing multi-camera forward-backward geometric consistency across co-observed views.

Mathematical Formulation:
-------------------------
For a candidate 3D point P_w generated from reference view i at pixel (u_i, v_i) with depth d_i:
    1. Forward projection into neighboring camera j:
         P_{cam, j} = T_{cam_world, j} @ P_w
         (u_j, v_j) = project_j(P_{cam, j})
         d_j^{obs} = D_j(v_j, u_j)
         Forward relative depth residual:
           Delta_d^{fwd} = |P_{cam, j}[2] - d_j^{obs}| / P_{cam, j}[2]

    2. Backward unprojection verification:
         P_{w, j}^{back} = unproject_j(u_j, v_j, d_j^{obs})
         3D spatial discrepancy:
           Delta_3D = || P_w - P_{w, j}^{back} ||
         Two-way pixel reprojection error:
           Delta_px = || (u_i, v_i) - project_i(P_{w, j}^{back}) ||

Consistency Classification:
    - High Confidence (K >= 2): Confirmed by 2 or more independent neighboring views.
    - Medium Confidence (K == 1): Confirmed by 1 neighboring view.
    - Single-View Observation: Valid pixel, but unobserved by other cameras due to occlusions.
    - Rejected: Inconsistent depth or geometric contradiction across co-observed views.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from photo_reconstruction.camera_model import PhoneCameraModel


@dataclass
class MVSConsistencyResult:
    """Comprehensive container for geometrically verified point cloud and diagnostics."""
    points: np.ndarray              # (N, 3) float32 verified 3D world coordinates
    colors: np.ndarray              # (N, 3) float32 RGB in [0, 1]
    support_counts: np.ndarray      # (N,) int32 number of independent confirming views
    confidences: np.ndarray         # (N,) float32 geometric confidence in [0, 1]
    total_candidates: int
    retained_count: int
    rejected_count: int
    high_confidence_count: int      # K >= 2
    medium_confidence_count: int    # K == 1
    mean_depth_residual: float      # Mean relative depth residual across verified points
    mean_spatial_residual: float    # Mean 3D spatial residual (m) across verified points


class MVSGeometricConsistencyFilter:
    """Rigorous multi-camera forward-backward geometric consistency validator."""

    def __init__(
        self,
        rel_depth_threshold: float = 0.035,   # 3.5% relative depth error
        spatial_dist_threshold: float = 0.20, # 20 cm 3D spatial discrepancy
        reproj_threshold_px: float = 2.5,     # 2.5 px two-way reprojection error
        min_consistent_views: int = 1,        # 1 other camera (K >= 1) for retaining
        strict_dual_view: bool = False,       # True requires K >= 2
    ):
        """
        Parameters
        ----------
        rel_depth_threshold : float
        spatial_dist_threshold : float (m)
        reproj_threshold_px : float (px)
        min_consistent_views : int
        strict_dual_view : bool
        """
        self.rel_depth_threshold = float(rel_depth_threshold)
        self.spatial_dist_threshold = float(spatial_dist_threshold)
        self.reproj_threshold_px = float(reproj_threshold_px)
        self.min_consistent_views = 2 if strict_dual_view else int(min_consistent_views)

    def filter_points_with_colors(
        self,
        camera_models: Dict[str, Union[PhoneCameraModel, np.ndarray]],
        depth_maps: Dict[str, np.ndarray],
        image_dir: Optional[Path] = None,
        stride: int = 8,
        sky_masks: Optional[Dict[str, np.ndarray]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Filter multi-view depth maps into a high-fidelity 3D point cloud.

        Returns
        -------
        verified_points : (N, 3) float32
        verified_colors : (N, 3) float32
        view_counts : (N,) int32
        """
        res = self.evaluate_consistency(
            camera_models=camera_models,
            depth_maps=depth_maps,
            image_dir=image_dir,
            stride=stride,
            sky_masks=sky_masks,
        )
        return res.points, res.colors, res.support_counts

    def filter_points(
        self,
        camera_poses: Dict[str, Union[PhoneCameraModel, np.ndarray]],
        depth_maps: Dict[str, np.ndarray],
        stride: int = 8,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Backward-compatible helper returning (points, support_counts)."""
        pts, _, counts = self.filter_points_with_colors(
            camera_models=camera_poses,
            depth_maps=depth_maps,
            image_dir=None,
            stride=stride,
        )
        return pts, counts

    def evaluate_consistency(
        self,
        camera_models: Dict[str, Union[PhoneCameraModel, np.ndarray]],
        depth_maps: Dict[str, np.ndarray],
        image_dir: Optional[Path] = None,
        stride: int = 8,
        sky_masks: Optional[Dict[str, np.ndarray]] = None,
    ) -> MVSConsistencyResult:
        """
        Execute full forward-backward geometric consistency evaluation.
        """
        cam_names = sorted(list(camera_models.keys()))
        if len(cam_names) < 2:
            return MVSConsistencyResult(
                points=np.zeros((0, 3), dtype=np.float32),
                colors=np.zeros((0, 3), dtype=np.float32),
                support_counts=np.zeros(0, dtype=np.int32),
                confidences=np.zeros(0, dtype=np.float32),
                total_candidates=0,
                retained_count=0,
                rejected_count=0,
                high_confidence_count=0,
                medium_confidence_count=0,
                mean_depth_residual=0.0,
                mean_spatial_residual=0.0,
            )

        # Ensure all models are PhoneCameraModel
        models: Dict[str, PhoneCameraModel] = {}
        for name, cam in camera_models.items():
            if isinstance(cam, PhoneCameraModel):
                models[name] = cam
            elif isinstance(cam, np.ndarray) and cam.shape == (4, 4):
                H, W = depth_maps[name].shape[:2]
                f = 1.20 * max(W, H)
                models[name] = PhoneCameraModel(
                    name=name,
                    model_name="PINHOLE",
                    width=W,
                    height=H,
                    params=np.array([f, f, W / 2.0, H / 2.0]),
                    fx=f,
                    fy=f,
                    cx=W / 2.0,
                    cy=H / 2.0,
                    T_world_cam=cam,
                    T_cam_world=np.linalg.inv(cam),
                )

        verified_pts_list: List[np.ndarray] = []
        verified_cols_list: List[np.ndarray] = []
        verified_counts_list: List[np.ndarray] = []
        verified_conf_list: List[np.ndarray] = []
        depth_residuals: List[float] = []
        spatial_residuals: List[float] = []

        total_candidates = 0

        for ref_name in cam_names:
            if ref_name not in depth_maps:
                continue

            ref_depth = depth_maps[ref_name]
            ref_cam = models[ref_name]
            H, W = ref_depth.shape[:2]

            # Scale camera model if depth map resolution differs from calibrated sensor
            if ref_cam.width != W or ref_cam.height != H:
                ref_cam = ref_cam.rescale(W, H)

            # Load original RGB image for authentic color sampling
            img_rgb = None
            if image_dir is not None:
                img_path = Path(image_dir) / ref_name
                if img_path.is_file():
                    bgr = cv2.imread(str(img_path))
                    if bgr is not None:
                        img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                        if img_rgb.shape[:2] != (H, W):
                            img_rgb = cv2.resize(img_rgb, (W, H), interpolation=cv2.INTER_LINEAR)

            sky = sky_masks.get(ref_name) if sky_masks else None
            if sky is not None and sky.shape[:2] != (H, W):
                sky = cv2.resize(sky.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0

            # Grid sampling with configurable stride
            s = max(2, int(stride))
            v_grid, u_grid = np.mgrid[0:H:s, 0:W:s]
            d = ref_depth[v_grid, u_grid]

            valid = (d > 0.20) & (d < 50.0)
            if sky is not None:
                sky_sub = sky[v_grid, u_grid]
                valid = valid & (~sky_sub)

            if not np.any(valid):
                continue

            u_sub = u_grid[valid]
            v_sub = v_grid[valid]
            d_sub = d[valid]

            pix_2d = np.column_stack([u_sub, v_sub])
            # Unproject candidate points to world space
            pts_world = ref_cam.unproject_image_to_world(pix_2d, d_sub)
            num_pts = len(pts_world)
            total_candidates += num_pts

            # Forward-backward consistency tracking across other views
            consistency_counts = np.zeros(num_pts, dtype=np.int32)
            accum_res = np.zeros(num_pts, dtype=np.float64)

            for other_name in cam_names:
                if other_name == ref_name or other_name not in depth_maps:
                    continue

                other_depth = depth_maps[other_name]
                other_cam = models[other_name]
                other_H, other_W = other_depth.shape[:2]
                if other_cam.width != other_W or other_cam.height != other_H:
                    other_cam = other_cam.rescale(other_W, other_H)

                # 1. Forward project into neighboring camera
                proj_pix, depth_cam, valid_proj = other_cam.project_world_to_image(
                    pts_world, check_bounds=True
                )
                if not np.any(valid_proj):
                    continue

                valid_indices = np.where(valid_proj)[0]
                u_other = np.clip(np.round(proj_pix[valid_indices, 0]).astype(int), 0, other_W - 1)
                v_other = np.clip(np.round(proj_pix[valid_indices, 1]).astype(int), 0, other_H - 1)

                observed_d = other_depth[v_other, u_other].astype(np.float64)
                valid_obs = (observed_d > 0.20) & (observed_d < 50.0)

                test_indices = valid_indices[valid_obs]
                if len(test_indices) == 0:
                    continue

                z_pred = depth_cam[test_indices]
                z_obs = observed_d[valid_obs]

                # Forward relative depth difference
                rel_diff = np.abs(z_pred - z_obs) / np.maximum(z_pred, 1e-4)

                # 2. Backward unprojection
                pts_back_cam = other_cam.unproject_image_to_world(
                    proj_pix[test_indices], z_obs
                )
                dist_3d = np.linalg.norm(pts_world[test_indices] - pts_back_cam, axis=1)

                # Two-way reprojection check back to reference view
                reproj_back, _, valid_back = ref_cam.project_world_to_image(
                    pts_back_cam, check_bounds=False
                )
                reproj_err = np.linalg.norm(reproj_back - pix_2d[test_indices], axis=1)

                # Consistent if relative depth, 3D distance, AND reprojection error are satisfied
                is_consistent = (
                    (rel_diff <= self.rel_depth_threshold)
                    & (dist_3d <= self.spatial_dist_threshold)
                    & (reproj_err <= self.reproj_threshold_px)
                )

                cons_idx = test_indices[is_consistent]
                consistency_counts[cons_idx] += 1
                accum_res[cons_idx] += rel_diff[is_consistent]
                depth_residuals.extend(rel_diff[is_consistent].tolist())
                spatial_residuals.extend(dist_3d[is_consistent].tolist())

            # Filter by minimum consistent views
            retained_mask = consistency_counts >= self.min_consistent_views
            if not np.any(retained_mask):
                continue

            v_pts = pts_world[retained_mask]
            v_counts = consistency_counts[retained_mask]
            v_conf = np.clip(v_counts.astype(np.float32) / 3.0, 0.3, 1.0)

            # Sample RGB colors
            if img_rgb is not None:
                u_ret = u_sub[retained_mask]
                v_ret = v_sub[retained_mask]
                v_cols = img_rgb[v_ret, u_ret].astype(np.float32) / 255.0
            else:
                v_cols = np.ones((len(v_pts), 3), dtype=np.float32) * 0.75

            verified_pts_list.append(v_pts.astype(np.float32))
            verified_cols_list.append(v_cols.astype(np.float32))
            verified_counts_list.append(v_counts.astype(np.int32))
            verified_conf_list.append(v_conf.astype(np.float32))

        if verified_pts_list:
            all_pts = np.vstack(verified_pts_list)
            all_cols = np.vstack(verified_cols_list)
            all_counts = np.concatenate(verified_counts_list)
            all_confs = np.concatenate(verified_conf_list)
            retained_count = len(all_pts)
            high_conf = int(np.sum(all_counts >= 2))
            med_conf = int(np.sum(all_counts == 1))
        else:
            all_pts = np.zeros((0, 3), dtype=np.float32)
            all_cols = np.zeros((0, 3), dtype=np.float32)
            all_counts = np.zeros(0, dtype=np.int32)
            all_confs = np.zeros(0, dtype=np.float32)
            retained_count = 0
            high_conf = 0
            med_conf = 0

        mean_d_res = float(np.mean(depth_residuals)) if depth_residuals else 0.0
        mean_3d_res = float(np.mean(spatial_residuals)) if spatial_residuals else 0.0

        return MVSConsistencyResult(
            points=all_pts,
            colors=all_cols,
            support_counts=all_counts,
            confidences=all_confs,
            total_candidates=total_candidates,
            retained_count=retained_count,
            rejected_count=total_candidates - retained_count,
            high_confidence_count=high_conf,
            medium_confidence_count=med_conf,
            mean_depth_residual=mean_d_res,
            mean_spatial_residual=mean_3d_res,
        )
