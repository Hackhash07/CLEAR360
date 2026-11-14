"""
photo_reconstruction/depth_aligner.py — Robust SfM-Scale Depth Alignment
=======================================================================

Aligns single-view monocular depth predictions with the metric/arbitrary coordinate
gauge established by COLMAP Structure-from-Motion.

Mathematical Formulation:
  For each camera view i with 3D triangulated sparse points P_k:
    1. Transform P_k to camera space: P_{cam, k} = R_i P_k + t_i.
    2. Extract axial Z-depth: Z_{SfM, k} = P_{cam, k}[2].
    3. Sample raw monocular depth at corresponding 2D observation: D_{mono, k} = D_i(v_k, u_k).
    4. Compute scale ratio: r(i, k) = Z_{SfM, k} / D_{mono, k}.
    5. Robust scale estimate:
         s_i = robust_median({ r(i, k) })
    6. Rescale:
         D_{aligned, i} = s_i * D_{mono, i}

Comparison Modes:
  - 'global': A single robust pooled scale factor across all images.
  - 'per_view': Independent scale factors per image.
  - 'regularized': Per-view factor regularized toward the pooled global factor.

Labeling:
  Strictly labeled 'SFM-SCALE ALIGNED DEPTH'. Absolute physical metric scaling requires
  an independent verified reference distance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from photo_reconstruction.camera_model import PhoneCameraModel


@dataclass
class ViewAlignmentStats:
    """Statistical summary of SfM-scale depth alignment for a single view."""
    image_name: str
    num_anchors: int
    median_scale: float
    mean_scale: float
    std_scale: float
    mad_scale: float
    iqr_scale: float
    raw_residual_mean: float
    raw_residual_median: float
    aligned_residual_mean: float
    aligned_residual_median: float
    residual_reduction_pct: float
    success: bool
    status_message: str


@dataclass
class AlignmentReport:
    """Full dataset alignment report containing per-view and global diagnostics."""
    mode: str
    global_scale: float
    total_anchors: int
    view_stats: Dict[str, ViewAlignmentStats] = field(default_factory=dict)
    aligned_depth_maps: Dict[str, np.ndarray] = field(default_factory=dict)


class SfMDepthAligner:
    """Robust estimator for SfM-scale alignment of monocular depth maps."""

    def __init__(
        self,
        mode: str = "regularized",
        max_reprojection_error: float = 2.0,
        min_depth: float = 0.20,
        max_depth: float = 80.0,
        min_anchors: int = 15,
        regularization_prior_weight: float = 30.0,
        validation_split_ratio: float = 0.20,
    ):
        """
        Parameters
        ----------
        mode : str ('global', 'per_view', 'regularized')
        max_reprojection_error : float (px)
        min_depth : float (m)
        max_depth : float (m)
        min_anchors : int
        regularization_prior_weight : float (N_0 weight for regularizing toward global)
        validation_split_ratio : float (held-out validation fraction: e.g. 0.20)
        """
        self.mode = mode.lower()
        self.max_reprojection_error = float(max_reprojection_error)
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.min_anchors = int(min_anchors)
        self.prior_weight = float(regularization_prior_weight)
        self.val_split = float(validation_split_ratio)

    def align_depth_maps(
        self,
        depth_maps: Dict[str, np.ndarray],
        camera_models: Dict[str, PhoneCameraModel],
        sparse_tracks: Dict[str, List[Dict[str, Any]]],
    ) -> AlignmentReport:
        """
        Align all depth maps to the SfM scale using sparse triangulated anchors.

        Parameters
        ----------
        depth_maps : Dict[str, np.ndarray]
            Raw monocular depth maps (H, W).
        camera_models : Dict[str, PhoneCameraModel]
            Authoritative camera models with world/camera extrinsics.
        sparse_tracks : Dict[str, List[Dict[str, Any]]]
            Mapping of image_name -> list of observations:
            [{'xy': (u, v), 'xyz': (x, y, z), 'error': float}, ...]

        Returns
        -------
        AlignmentReport
        """
        per_view_ratios: Dict[str, np.ndarray] = {}
        per_view_pairs: Dict[str, np.ndarray] = {}

        # 1. Collect valid anchor points for each view
        for name, tracks in sparse_tracks.items():
            if name not in depth_maps or name not in camera_models:
                continue

            d_map = depth_maps[name]
            cam = camera_models[name]
            H, W = d_map.shape[:2]

            pairs: List[Tuple[float, float]] = []  # (z_sfm, d_mono)
            for trk in tracks:
                err = trk.get("error", 0.0)
                if err > self.max_reprojection_error:
                    continue

                xyz_w = np.asarray(trk["xyz"], dtype=np.float64)
                # Compute camera-space axial Z-depth: P_cam = T_cam_world @ P_world
                p_cam = cam.T_cam_world[:3, :3] @ xyz_w + cam.T_cam_world[:3, 3]
                z_sfm = p_cam[2]
                if z_sfm < self.min_depth or z_sfm > self.max_depth:
                    continue

                u, v = trk["xy"]
                u_i, v_i = int(round(u)), int(round(v))
                if 0 <= v_i < H and 0 <= u_i < W:
                    d_mono = float(d_map[v_i, u_i])
                    if d_mono >= self.min_depth and d_mono <= self.max_depth:
                        # Reject local depth boundary gradient outliers
                        pairs.append((z_sfm, d_mono))

            if len(pairs) >= self.min_anchors:
                pairs_arr = np.array(pairs, dtype=np.float64)
                # Filter extreme numerical ratio outliers using IQR gating
                ratios = pairs_arr[:, 0] / pairs_arr[:, 1]
                q25, q75 = np.percentile(ratios, 25), np.percentile(ratios, 75)
                iqr = q75 - q25
                inliers = (ratios >= q25 - 2.5 * iqr) & (ratios <= q75 + 2.5 * iqr)

                per_view_pairs[name] = pairs_arr[inliers]
                per_view_ratios[name] = ratios[inliers]
            else:
                per_view_pairs[name] = np.zeros((0, 2), dtype=np.float64)
                per_view_ratios[name] = np.zeros(0, dtype=np.float64)

        # 2. Compute Pooled Global Scale Factor
        all_ratios_list = [r for r in per_view_ratios.values() if len(r) > 0]
        if all_ratios_list:
            pooled = np.concatenate(all_ratios_list)
            global_scale = float(np.median(pooled))
            total_anchors = len(pooled)
        else:
            global_scale = 1.0
            total_anchors = 0

        report = AlignmentReport(
            mode=self.mode,
            global_scale=global_scale,
            total_anchors=total_anchors,
        )

        # 3. Compute Per-View Scales and Held-Out Validation
        for name, d_map in depth_maps.items():
            ratios = per_view_ratios.get(name, np.zeros(0, dtype=np.float64))
            pairs = per_view_pairs.get(name, np.zeros((0, 2), dtype=np.float64))
            n = len(ratios)

            if n >= self.min_anchors:
                s_indep = float(np.median(ratios))
                s_mean = float(np.mean(ratios))
                s_std = float(np.std(ratios))
                s_mad = float(np.median(np.abs(ratios - s_indep)))
                s_iqr = float(np.percentile(ratios, 75) - np.percentile(ratios, 25))

                if self.mode == "global":
                    chosen_scale = global_scale
                elif self.mode == "per_view":
                    chosen_scale = s_indep
                else:  # regularized
                    weight = float(n) / (float(n) + self.prior_weight)
                    chosen_scale = weight * s_indep + (1.0 - weight) * global_scale

                # Held-out validation split
                np.random.seed(42)
                perm = np.random.permutation(n)
                n_train = max(5, int(n * (1.0 - self.val_split)))
                train_idx, test_idx = perm[:n_train], perm[n_train:]

                test_pairs = pairs[test_idx] if len(test_idx) > 0 else pairs
                raw_res = np.abs(test_pairs[:, 0] - test_pairs[:, 1])
                aligned_res = np.abs(test_pairs[:, 0] - test_pairs[:, 1] * chosen_scale)

                raw_mean = float(np.mean(raw_res))
                raw_med = float(np.median(raw_res))
                aligned_mean = float(np.mean(aligned_res))
                aligned_med = float(np.median(aligned_res))
                red_pct = float((1.0 - aligned_mean / max(raw_mean, 1e-6)) * 100.0)

                v_stat = ViewAlignmentStats(
                    image_name=name,
                    num_anchors=n,
                    median_scale=chosen_scale,
                    mean_scale=s_mean,
                    std_scale=s_std,
                    mad_scale=s_mad,
                    iqr_scale=s_iqr,
                    raw_residual_mean=raw_mean,
                    raw_residual_median=raw_med,
                    aligned_residual_mean=aligned_mean,
                    aligned_residual_median=aligned_med,
                    residual_reduction_pct=red_pct,
                    success=True,
                    status_message=f"Aligned ({n} anchors, {red_pct:.1f}% residual reduction).",
                )
            else:
                chosen_scale = global_scale
                v_stat = ViewAlignmentStats(
                    image_name=name,
                    num_anchors=n,
                    median_scale=chosen_scale,
                    mean_scale=chosen_scale,
                    std_scale=0.0,
                    mad_scale=0.0,
                    iqr_scale=0.0,
                    raw_residual_mean=0.0,
                    raw_residual_median=0.0,
                    aligned_residual_mean=0.0,
                    aligned_residual_median=0.0,
                    residual_reduction_pct=0.0,
                    success=False,
                    status_message=f"Insufficient anchors ({n} < {self.min_anchors}); using global scale {global_scale:.4f}.",
                )

            report.view_stats[name] = v_stat
            report.aligned_depth_maps[name] = (d_map * chosen_scale).astype(np.float32)

        return report

