"""
depth/visualization.py — Visualization & Error Heatmaps for Depth Anything V2
=============================================================================

Generates side-by-side comparison panels for Frame 0 (and arbitrary frames):
  1. Four original RGB camera images
  2. Four Ground-Truth depth maps (consistent colormap scale)
  3. Four AI-predicted metric depth maps (identical colormap scale)
  4. Four absolute error heatmaps (|AI - GT| in metres)
  5. 4x4 master multi-camera comparison grid
  6. Detailed textual comparison report

Color Scales (strictly synchronized for fair comparison):
  - Depth maps: Turbo colormap, fixed metric range [0.5, 35.0] metres
  - Error heatmaps: Hot / Magma colormap, fixed error range [0.0, 10.0] metres
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.sync import FrameSetLoader
from depth.estimator import DepthEstimator
from depth.benchmark import compute_depth_metrics


def colorize_depth(depth: np.ndarray, min_depth: float = 0.5, max_depth: float = 35.0) -> np.ndarray:
    """
    Colorize depth map with Turbo colormap using fixed metric range [min_depth, max_depth].
    Returns (H, W, 3) BGR uint8 image suitable for OpenCV.
    """
    clipped = np.clip(depth, min_depth, max_depth)
    normalized = ((clipped - min_depth) / (max_depth - min_depth) * 255.0).astype(np.uint8)
    # Apply COLORMAP_TURBO (or JET if unavailable)
    try:
        colored = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    except Exception:
        colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)

    # Sky / background (> 75m) colored in dark navy
    sky_mask = depth > 75.0
    colored[sky_mask] = (40, 20, 20)
    return colored


def colorize_error(error: np.ndarray, max_error: float = 10.0, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Colorize absolute depth error (|pred - gt|) using Hot colormap in [0, max_error] metres.
    Masked-out pixels (e.g. sky) are colored neutral dark gray.
    """
    clipped = np.clip(error, 0.0, max_error)
    normalized = (clipped / max_error * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_HOT)

    if mask is not None:
        invalid = ~mask
        colored[invalid] = (50, 50, 50)  # Dark gray for un-evaluated regions

    return colored


def add_header(img: np.ndarray, text: str, bg_color=(20, 20, 25), text_color=(240, 240, 240)) -> np.ndarray:
    """Add a labelled banner header above an image."""
    h, w = img.shape[:2]
    header_h = 40
    banner = np.full((header_h, w, 3), bg_color, dtype=np.uint8)
    cv2.putText(banner, text, (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, text_color, 2)
    return np.vstack([banner, img])


def generate_frame_comparison(
    frame_id: int,
    estimator: DepthEstimator,
    loader: FrameSetLoader,
    output_dir: Optional[Path] = None,
) -> tuple[Path, dict]:
    """
    Generate complete 4-camera comparison mosaic and error analysis for a given frame.

    Saves:
      - 4x4 master grid: {output_dir}/depth_comparison_frame_{frame_id:04d}.png
      - Per-camera error maps and comparison metrics.
    """
    if output_dir is None:
        output_dir = PROJECT_ROOT / "data" / "depth_eval"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cameras = ["front", "rear", "left", "right"]
    fs = loader.load_frame(frame_id, load_position=False, load_semantic=True)

    # Prepare outputs
    rgb_imgs = []
    gt_depth_imgs = []
    ai_depth_imgs = []
    error_imgs = []
    metrics_summary = {}

    # Downscale resolution for mosaic visualization (e.g. 640x360 per tile)
    tile_w, tile_h = 640, 360

    print(f"\n[Visualization] Processing Frame {frame_id:04d} across 4 cameras...")

    for cam in cameras:
        cfd = fs.get(cam)

        # AI Depth prediction
        ai_depth, dt = estimator.estimate_depth(cfd.rgb, is_rgb=True)
        gt_depth = cfd.depth

        # Evaluation mask: valid foreground 0.5 to 50m, non-sky
        eval_mask = (gt_depth >= 0.5) & (gt_depth <= 50.0)
        if cfd.semantic_mask is not None:
            eval_mask = eval_mask & (cfd.semantic_mask != 0)

        # Compute error
        abs_err = np.abs(ai_depth - gt_depth)

        # Metrics for this frame
        m = compute_depth_metrics(ai_depth, gt_depth, mask=eval_mask)
        metrics_summary[cam] = m

        # Save AI depth as numpy array in data/ai_depth/
        ai_depth_dir = PROJECT_ROOT / "data" / "ai_depth" / cam
        ai_depth_dir.mkdir(parents=True, exist_ok=True)
        np.save(str(ai_depth_dir / f"frame_{frame_id:04d}.npy"), ai_depth)

        # Render visualizations
        # 1. RGB
        rgb_bgr = cv2.cvtColor(cfd.rgb, cv2.COLOR_RGB2BGR)
        rgb_tile = cv2.resize(rgb_bgr, (tile_w, tile_h))
        rgb_tile = add_header(rgb_tile, f"{cam.upper()} — RGB Input")
        rgb_imgs.append(rgb_tile)

        # 2. GT Depth
        gt_color = colorize_depth(gt_depth, min_depth=0.5, max_depth=35.0)
        gt_tile = cv2.resize(gt_color, (tile_w, tile_h))
        gt_tile = add_header(gt_tile, f"{cam.upper()} — Ground-Truth Depth [0.5–35m]")
        gt_depth_imgs.append(gt_tile)

        # 3. AI Depth
        ai_color = colorize_depth(ai_depth, min_depth=0.5, max_depth=35.0)
        ai_tile = cv2.resize(ai_color, (tile_w, tile_h))
        ai_tile = add_header(ai_tile, f"{cam.upper()} — AI Depth Anything V2 ({dt*1000:.0f}ms)")
        ai_depth_imgs.append(ai_tile)

        # 4. Error Heatmap
        err_color = colorize_error(abs_err, max_error=10.0, mask=eval_mask)
        err_tile = cv2.resize(err_color, (tile_w, tile_h))
        err_tile = add_header(err_tile, f"{cam.upper()} — Absolute Error |AI - GT| (MAE={m['mae']:.2f}m)", bg_color=(40, 15, 15))
        error_imgs.append(err_tile)

        print(f"  [{cam}] MAE={m['mae']:.2f}m, RMSE={m['rmse']:.2f}m, AbsRel={m['abs_rel']:.3f}, delta_1={m['delta_1']*100:.1f}%")

    # Combine into 4x4 matrix:
    # Row 1: RGB
    # Row 2: GT Depth
    # Row 3: AI Depth
    # Row 4: Error Heatmap
    row_rgb = np.hstack(rgb_imgs)
    row_gt = np.hstack(gt_depth_imgs)
    row_ai = np.hstack(ai_depth_imgs)
    row_err = np.hstack(error_imgs)

    master_grid = np.vstack([row_rgb, row_gt, row_ai, row_err])

    # Add master title banner
    title_bar = np.full((60, master_grid.shape[1], 3), (15, 15, 18), dtype=np.uint8)
    cv2.putText(
        title_bar,
        f"CLEAR360 M4: DEPTH ANYTHING V2 METRIC EVALUATION — FRAME {frame_id:04d} (4-CAMERA SURROUND)",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.95,
        (255, 255, 255),
        2,
    )
    master_vis = np.vstack([title_bar, master_grid])

    out_file = output_dir / f"depth_comparison_frame_{frame_id:04d}.png"
    cv2.imwrite(str(out_file), master_vis)
    print(f"\n[Visualization] Master 4x4 comparison saved to: {out_file}")

    return out_file, metrics_summary

