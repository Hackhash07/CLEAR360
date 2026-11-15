"""
photo_reconstruction/depth_aligner.py — held-out SfM depth calibration.

Fits an affine mapping from monocular depth to the SfM camera-Z gauge per view:
    z_sfm = scale * depth_mono + offset

Calibration uses training anchors only. A separate held-out set determines whether
that view is trustworthy enough for multi-view geometric fusion. Rejected views are
not returned as aligned depth maps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from photo_reconstruction.camera_model import PhoneCameraModel


@dataclass
class ViewAlignmentStats:
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
    depth_offset: float = 0.0
    validation_median_relative_error: float = 1.0
    validation_p90_relative_error: float = 1.0


@dataclass
class AlignmentReport:
    mode: str
    global_scale: float
    total_anchors: int
    view_stats: Dict[str, ViewAlignmentStats] = field(default_factory=dict)
    aligned_depth_maps: Dict[str, np.ndarray] = field(default_factory=dict)


class SfMDepthAligner:
    """Robust affine calibration with genuine held-out acceptance checks."""

    def __init__(
        self,
        mode: str = "regularized",
        max_reprojection_error: float = 2.0,
        min_depth: float = 0.20,
        max_depth: float = 80.0,
        min_anchors: int = 15,
        regularization_prior_weight: float = 30.0,
        validation_split_ratio: float = 0.20,
        max_validation_median_relative_error: float = 0.04,
        max_validation_p90_relative_error: float = 0.12,
    ):
        self.mode = mode.lower()
        if self.mode not in {"global", "per_view", "regularized"}:
            raise ValueError(f"Unsupported depth alignment mode: {mode}")
        self.max_reprojection_error = float(max_reprojection_error)
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.min_anchors = int(min_anchors)
        self.prior_weight = float(regularization_prior_weight)
        self.val_split = float(validation_split_ratio)
        self.max_val_median_rel = float(max_validation_median_relative_error)
        self.max_val_p90_rel = float(max_validation_p90_relative_error)

    @staticmethod
    def _robust_affine_fit(
        mono_depth: np.ndarray,
        sfm_depth: np.ndarray,
        seed: int = 42,
    ) -> Tuple[float, float]:
        """Fit y = scale*x + offset using median pairwise slopes and robust refit."""
        x = np.asarray(mono_depth, dtype=np.float64).ravel()
        y = np.asarray(sfm_depth, dtype=np.float64).ravel()
        valid = np.isfinite(x) & np.isfinite(y)
        x, y = x[valid], y[valid]
        if len(x) < 5 or float(np.ptp(x)) < 1e-5:
            raise ValueError("Insufficient depth variation for affine calibration")

        rng = np.random.default_rng(seed)
        n_pairs = min(8192, max(512, len(x) * 12))
        i = rng.integers(0, len(x), size=n_pairs)
        j = rng.integers(0, len(x), size=n_pairs)
        dx = x[j] - x[i]
        dy = y[j] - y[i]
        min_dx = max(1e-4, float(np.std(x)) * 1e-3)
        keep = (np.abs(dx) > min_dx) & np.isfinite(dy / np.where(dx == 0, 1.0, dx))
        slopes = dy[keep] / dx[keep]
        slopes = slopes[np.isfinite(slopes) & (slopes > 0.0)]
        if len(slopes) < 10:
            raise ValueError("Could not estimate a positive depth scale")
        scale = float(np.median(slopes))
        offset = float(np.median(y - scale * x))

        residual = y - (scale * x + offset)
        residual_center = float(np.median(residual))
        residual_mad = float(1.4826 * np.median(np.abs(residual - residual_center)))
        y_scale = max(float(np.median(np.abs(y))), 1e-3)
        cutoff = max(3.5 * residual_mad, 0.01 * y_scale, 1e-4)
        inliers = np.abs(residual - residual_center) <= cutoff

        if int(np.sum(inliers)) >= max(8, int(0.5 * len(x))):
            A = np.column_stack([x[inliers], np.ones(int(np.sum(inliers)))])
            fitted, *_ = np.linalg.lstsq(A, y[inliers], rcond=None)
            fitted_scale, fitted_offset = float(fitted[0]), float(fitted[1])
            if np.isfinite(fitted_scale) and fitted_scale > 0 and np.isfinite(fitted_offset):
                scale, offset = fitted_scale, fitted_offset

        return scale, offset

    @staticmethod
    def _split_indices(n: int, name: str, val_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
        seed = 42 + sum(name.encode("utf-8"))
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        n_train = min(n - 3, max(8, int(round(n * (1.0 - val_ratio)))))
        return perm[:n_train], perm[n_train:]

    def align_depth_maps(
        self,
        depth_maps: Dict[str, np.ndarray],
        camera_models: Dict[str, PhoneCameraModel],
        sparse_tracks: Dict[str, List[Dict[str, Any]]],
    ) -> AlignmentReport:
        """Calibrate each view on training anchors; return only validated depth maps."""
        pairs_by_view: Dict[str, np.ndarray] = {}
        ratios_by_view: Dict[str, np.ndarray] = {}
        split_by_view: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        train_models: Dict[str, Tuple[float, float]] = {}

        for name, d_map in depth_maps.items():
            if name not in camera_models:
                pairs_by_view[name] = np.zeros((0, 2), dtype=np.float64)
                continue

            d_map = np.asarray(d_map, dtype=np.float32)
            if d_map.ndim != 2:
                pairs_by_view[name] = np.zeros((0, 2), dtype=np.float64)
                continue
            H, W = d_map.shape
            cam = camera_models[name]
            pairs: List[Tuple[float, float]] = []

            for track in sparse_tracks.get(name, []):
                try:
                    error = float(track.get("error", 0.0))
                    if not np.isfinite(error) or error > self.max_reprojection_error:
                        continue
                    xyz_w = np.asarray(track["xyz"], dtype=np.float64).reshape(3)
                    if not np.all(np.isfinite(xyz_w)):
                        continue
                    p_cam = cam.T_cam_world[:3, :3] @ xyz_w + cam.T_cam_world[:3, 3]
                    z_sfm = float(p_cam[2])
                    u, v = track["xy"]
                    ui, vi = int(round(float(u))), int(round(float(v)))
                    if not (self.min_depth <= z_sfm <= self.max_depth):
                        continue
                    if not (0 <= vi < H and 0 <= ui < W):
                        continue
                    d_mono = float(d_map[vi, ui])
                    if np.isfinite(d_mono) and self.min_depth <= d_mono <= self.max_depth:
                        pairs.append((z_sfm, d_mono))
                except (KeyError, TypeError, ValueError, IndexError):
                    continue

            arr = np.asarray(pairs, dtype=np.float64).reshape(-1, 2)
            if len(arr) >= self.min_anchors:
                ratios = arr[:, 0] / np.maximum(arr[:, 1], 1e-8)
                q25, q75 = np.percentile(ratios, [25, 75])
                iqr = q75 - q25
                keep = (ratios >= q25 - 2.5 * iqr) & (ratios <= q75 + 2.5 * iqr)
                arr = arr[keep]
            pairs_by_view[name] = arr
            if len(arr) >= self.min_anchors:
                train_idx, val_idx = self._split_indices(len(arr), name, self.val_split)
                split_by_view[name] = (train_idx, val_idx)
                try:
                    train = arr[train_idx]
                    train_models[name] = self._robust_affine_fit(
                        train[:, 1], train[:, 0], seed=42 + sum(name.encode("utf-8"))
                    )
                except ValueError:
                    pass
            else:
                split_by_view[name] = (np.arange(len(arr)), np.zeros(0, dtype=np.int64))

        pooled_train = [
            pairs_by_view[name][split_by_view[name][0]]
            for name in train_models
            if len(split_by_view[name][0]) >= 5
        ]
        if pooled_train:
            pooled = np.vstack(pooled_train)
            global_scale, global_offset = self._robust_affine_fit(
                pooled[:, 1], pooled[:, 0], seed=2026
            )
            total_anchors = int(sum(len(x) for x in pooled_train))
        else:
            global_scale, global_offset, total_anchors = 1.0, 0.0, 0

        report = AlignmentReport(
            mode=self.mode,
            global_scale=float(global_scale),
            total_anchors=total_anchors,
        )

        for name, d_map in depth_maps.items():
            arr = pairs_by_view.get(name, np.zeros((0, 2), dtype=np.float64))
            n = len(arr)
            if name not in split_by_view or name not in train_models:
                stats = ViewAlignmentStats(
                    image_name=name, num_anchors=n,
                    median_scale=float(global_scale), mean_scale=float(global_scale),
                    std_scale=0.0, mad_scale=0.0, iqr_scale=0.0,
                    raw_residual_mean=0.0, raw_residual_median=0.0,
                    aligned_residual_mean=0.0, aligned_residual_median=0.0,
                    residual_reduction_pct=0.0, success=False,
                    status_message=f"Rejected: only {n} usable SfM anchors; {self.min_anchors} required.",
                )
                report.view_stats[name] = stats
                continue

            train_idx, val_idx = split_by_view[name]
            local_scale, local_offset = train_models[name]
            if self.mode == "global":
                scale, offset = global_scale, global_offset
            elif self.mode == "per_view":
                scale, offset = local_scale, local_offset
            else:
                weight = float(len(train_idx)) / (float(len(train_idx)) + self.prior_weight)
                scale = weight * local_scale + (1.0 - weight) * global_scale
                offset = weight * local_offset + (1.0 - weight) * global_offset

            validation = arr[val_idx] if len(val_idx) else arr[train_idx]
            z_sfm, d_mono = validation[:, 0], validation[:, 1]
            raw_res = np.abs(z_sfm - d_mono)
            aligned_res = np.abs(z_sfm - (scale * d_mono + offset))
            reference_depth = max(float(np.median(np.abs(z_sfm))), 1e-3)
            med_rel = float(np.median(aligned_res) / reference_depth)
            p90_rel = float(np.percentile(aligned_res, 90) / reference_depth)
            raw_mean = float(np.mean(raw_res))
            aligned_mean = float(np.mean(aligned_res))
            raw_med = float(np.median(raw_res))
            aligned_med = float(np.median(aligned_res))
            reduction = float((1.0 - aligned_mean / max(raw_mean, 1e-8)) * 100.0)

            accepted = (
                np.isfinite(med_rel)
                and np.isfinite(p90_rel)
                and med_rel <= self.max_val_median_rel
                and p90_rel <= self.max_val_p90_rel
            )
            status = (
                f"Accepted: holdout median={med_rel:.1%}, p90={p90_rel:.1%}."
                if accepted else
                f"Rejected: holdout median={med_rel:.1%}, p90={p90_rel:.1%}; "
                f"limits are {self.max_val_median_rel:.1%}/{self.max_val_p90_rel:.1%}."
            )

            ratios = arr[:, 0] / np.maximum(arr[:, 1], 1e-8)
            stats = ViewAlignmentStats(
                image_name=name,
                num_anchors=n,
                median_scale=float(scale),
                mean_scale=float(np.mean(ratios)),
                std_scale=float(np.std(ratios)),
                mad_scale=float(np.median(np.abs(ratios - np.median(ratios)))),
                iqr_scale=float(np.percentile(ratios, 75) - np.percentile(ratios, 25)),
                raw_residual_mean=raw_mean,
                raw_residual_median=raw_med,
                aligned_residual_mean=aligned_mean,
                aligned_residual_median=aligned_med,
                residual_reduction_pct=reduction,
                success=bool(accepted),
                status_message=status,
                depth_offset=float(offset),
                validation_median_relative_error=med_rel,
                validation_p90_relative_error=p90_rel,
            )
            report.view_stats[name] = stats

            if accepted:
                aligned = scale * np.asarray(d_map, dtype=np.float32) + offset
                aligned = np.clip(aligned, self.min_depth, self.max_depth).astype(np.float32)
                report.aligned_depth_maps[name] = aligned

        return report
