"""
depth/benchmark.py — Benchmark Depth Anything V2 Against Ground-Truth Depth
===========================================================================

Evaluates metric AI depth predictions against existing ground-truth Z-depth maps.

Metrics Computed:
  - MAE (Mean Absolute Error, in metres)
  - RMSE (Root Mean Squared Error, in metres)
  - AbsRel (Mean Absolute Relative Error: mean(|pred - gt| / gt))
  - Log10 Error: mean(|log10(pred) - log10(gt)|)
  - Delta 1.25 (fraction of pixels with max(pred/gt, gt/pred) < 1.25)
  - Delta 1.25^2 (< 1.5625)
  - Delta 1.25^3 (< 1.9531)

Evaluation Mask Rules (NEVER ALIGN OR SCALE PREDICTIONS):
  - Valid GT depth range: 0.5 m <= gt <= 50.0 m (excludes sensor clip and infinite sky)
  - Non-sky objects: semantic_mask != 0 (or gt < 75.0 m)
  - Pred > 0.0 m
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.sync import FrameSetLoader
from depth.estimator import DepthEstimator


def compute_depth_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> dict[str, float]:
    """
    Compute standard depth estimation benchmark metrics on masked pixels.

    Parameters
    ----------
    pred : np.ndarray (H, W) float32
        Predicted metric depth in metres.
    gt : np.ndarray (H, W) float32
        Ground-truth metric depth in metres.
    mask : np.ndarray (H, W) bool, optional
        Boolean evaluation mask. If None, valid range [0.5, 50.0] is used.

    Returns
    -------
    dict with keys: 'mae', 'rmse', 'abs_rel', 'log10', 'delta_1', 'delta_2', 'delta_3', 'valid_pixels'
    """
    if mask is None:
        mask = (gt >= 0.5) & (gt <= 50.0) & (pred > 0.0)
    else:
        mask = mask & (gt >= 0.5) & (gt <= 50.0) & (pred > 0.0) & (~np.isnan(pred)) & (~np.isnan(gt))

    num_valid = int(np.sum(mask))
    if num_valid == 0:
        return {
            "mae": float("nan"),
            "rmse": float("nan"),
            "abs_rel": float("nan"),
            "log10": float("nan"),
            "delta_1": float("nan"),
            "delta_2": float("nan"),
            "delta_3": float("nan"),
            "valid_pixels": 0,
        }

    p = pred[mask].astype(np.float64)
    g = gt[mask].astype(np.float64)

    # MAE
    mae = float(np.mean(np.abs(p - g)))

    # RMSE
    rmse = float(np.sqrt(np.mean((p - g) ** 2)))

    # AbsRel: mean(|pred - gt| / gt)
    abs_rel = float(np.mean(np.abs(p - g) / g))

    # Log10
    log10 = float(np.mean(np.abs(np.log10(p) - np.log10(g))))

    # Threshold accuracy (delta)
    ratio = np.maximum(p / g, g / p)
    delta_1 = float(np.mean(ratio < 1.25))
    delta_2 = float(np.mean(ratio < (1.25 ** 2)))
    delta_3 = float(np.mean(ratio < (1.25 ** 3)))

    return {
        "mae": mae,
        "rmse": rmse,
        "abs_rel": abs_rel,
        "log10": log10,
        "delta_1": delta_1,
        "delta_2": delta_2,
        "delta_3": delta_3,
        "valid_pixels": num_valid,
    }


def format_metrics_table(camera_metrics: dict[str, dict[str, float]], title: str = "DEPTH BENCHMARK RESULTS") -> str:
    """Format dictionary of per-camera metrics into an ASCII / markdown table."""
    lines = []
    lines.append("=" * 86)
    lines.append(f" {title}")
    lines.append("=" * 86)
    header = f"{'Camera':<12} | {'MAE (m)':<9} | {'RMSE (m)':<9} | {'AbsRel':<8} | {'δ < 1.25':<9} | {'δ < 1.25²':<9} | {'Valid Px':<10}"
    lines.append(header)
    lines.append("-" * 86)

    for cam, m in camera_metrics.items():
        if m["valid_pixels"] == 0:
            lines.append(f"{cam:<12} | {'N/A':<9} | {'N/A':<9} | {'N/A':<8} | {'N/A':<9} | {'N/A':<9} | {0:<10}")
        else:
            line = (
                f"{cam:<12} | "
                f"{m['mae']:<9.3f} | "
                f"{m['rmse']:<9.3f} | "
                f"{m['abs_rel']:<8.3f} | "
                f"{m['delta_1']*100:<8.1f}% | "
                f"{m['delta_2']*100:<8.1f}% | "
                f"{m['valid_pixels']:<10,}"
            )
            lines.append(line)

    lines.append("=" * 86)
    return "\n".join(lines)


def run_benchmark(
    frame_ids: list[int],
    estimator: DepthEstimator,
    loader: FrameSetLoader,
    min_eval_depth: float = 0.5,
    max_eval_depth: float = 50.0,
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    """
    Run benchmark across specified frames and compute per-camera and overall metrics.

    Parameters
    ----------
    frame_ids : list of int
        List of frame indices to evaluate.
    estimator : DepthEstimator
        Initialized DepthEstimator instance.
    loader : FrameSetLoader
        Initialized FrameSetLoader instance.
    min_eval_depth : float
        Near clipping threshold for evaluation (default 0.5 m).
    max_eval_depth : float
        Far clipping threshold for evaluation (default 50.0 m).

    Returns
    -------
    per_camera : dict[str, dict[str, float]]
        Metrics averaged per camera.
    aggregate : dict[str, float]
        Overall metrics pooling all pixels across cameras and frames.
    """
    cameras = ["front", "rear", "left", "right"]

    pooled_pred = {cam: [] for cam in cameras}
    pooled_gt = {cam: [] for cam in cameras}

    for fid in frame_ids:
        fs = loader.load_frame(fid, load_position=False, load_semantic=True)
        for cam_name in cameras:
            cfd = fs.get(cam_name)
            pred_depth, _ = estimator.estimate_depth(cfd.rgb, is_rgb=True)
            gt_depth = cfd.depth

            # Evaluation mask: non-sky (semantic != 0), within depth limits
            mask = (gt_depth >= min_eval_depth) & (gt_depth <= max_eval_depth)
            if cfd.semantic_mask is not None:
                mask = mask & (cfd.semantic_mask != 0)

            valid_p = pred_depth[mask]
            valid_g = gt_depth[mask]

            pooled_pred[cam_name].append(valid_p)
            pooled_gt[cam_name].append(valid_g)

    # Compute per-camera results
    per_camera = {}
    all_pred = []
    all_gt = []

    for cam_name in cameras:
        cat_p = np.concatenate(pooled_pred[cam_name]) if pooled_pred[cam_name] else np.array([])
        cat_g = np.concatenate(pooled_gt[cam_name]) if pooled_gt[cam_name] else np.array([])

        m = compute_depth_metrics(cat_p, cat_g, mask=np.ones_like(cat_g, dtype=bool))
        per_camera[cam_name] = m

        if len(cat_p) > 0:
            all_pred.append(cat_p)
            all_gt.append(cat_g)

    # Aggregate result
    if all_pred:
        cat_all_p = np.concatenate(all_pred)
        cat_all_g = np.concatenate(all_gt)
        aggregate = compute_depth_metrics(cat_all_p, cat_all_g, mask=np.ones_like(cat_all_g, dtype=bool))
    else:
        aggregate = {}

    return per_camera, aggregate


def main():
    parser = argparse.ArgumentParser(description="Run Depth Anything V2 Benchmark")
    parser.add_argument("--frames", type=int, default=1, help="Number of frames to evaluate (default: 1 for Frame 0)")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use ('cpu', 'mps', 'cuda')")
    args = parser.parse_args()

    loader = FrameSetLoader(PROJECT_ROOT / "data")
    avail = loader.available_frames()
    if not avail:
        print("No frames available in data/. Run generate_synthetic_dataset.py first.")
        sys.exit(1)

    eval_frames = avail[: args.frames]
    print(f"Evaluating {len(eval_frames)} frame(s): {eval_frames}")

    estimator = DepthEstimator(device=args.device)
    per_cam, agg = run_benchmark(eval_frames, estimator, loader)

    # Add aggregate row to table
    full_table_dict = dict(per_cam)
    full_table_dict["OVERALL"] = agg

    table_str = format_metrics_table(full_table_dict, title=f"DEPTH ANYTHING V2 METRIC BENCHMARK ({len(eval_frames)} FRAME(S))")
    print("\n" + table_str)


if __name__ == "__main__":
    main()

