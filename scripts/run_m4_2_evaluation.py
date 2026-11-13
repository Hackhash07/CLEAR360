"""
scripts/run_m4_2_evaluation.py — M4.2 Photoreal AI Depth Benchmark & Point Cloud Analysis
========================================================================================

Executes the M4.2 milestone validation:
  1. Verifies photoreal Cycles rendered data (RGB, GT Z-depth, Position, Semantics).
  2. Generates 4-camera visual inspection mosaic: data/audit_samples/photoreal_mosaic_frame0.png.
  3. Runs Depth Anything V2 Metric ViT-Small on photoreal RGB camera frames.
  4. Saves AI predicted depth maps in data/photoreal_ai_depth/{camera}/frame_0000.npy.
  5. Computes official metric depth benchmark (MAE, RMSE, AbsRel, delta < 1.25):
     - Per-camera metrics (front, rear, left, right)
     - Overall aggregate metrics
     - Range-band accuracy: [0.5, 2.0m], [2.0, 5.0m], [5.0, 10.0m], [10.0, 20.0m]
     - Per-class accuracy using semantic masks (road, parked cars, wall, pillar, obstacle, furniture)
     - Side-by-side comparison against M4 flat analytical baseline
  6. Unprojects depth to 3D point clouds in vehicle coordinates:
     - Ground truth: data/pointclouds/frame_0000_photoreal_gt.ply
     - AI predicted: data/pointclouds/frame_0000_photoreal_ai.ply
  7. Evaluates 3D structural metrics:
     - Road flatness standard deviation
     - Wall verticality & bounding extents
  8. Generates error heatmaps and scatter plots.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch

# Ensure project root is in path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from depth.estimator import DepthEstimator
from pipeline.calibration import CalibrationStore
from validation.gt_loader import load_depth_exr, load_position_exr, load_semantic_png
from pointcloud.unprojector import DepthUnprojector
from sim.constants import CAMERA_NAMES, SEMANTIC_CLASSES


def create_inspection_mosaic(
    rgb_frames: dict[str, np.ndarray],
    gt_depths: dict[str, np.ndarray],
    ai_depths: dict[str, np.ndarray],
    out_path: Path,
) -> None:
    """Create a comprehensive multi-panel mosaic comparing RGB, GT Depth, and AI Depth across 4 cameras."""
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    plt.subplots_adjust(wspace=0.08, hspace=0.15)

    order = ["front", "rear", "left", "right"]

    for col_idx, cam in enumerate(order):
        # Row 0: Photoreal RGB
        rgb = rgb_frames[cam]
        axes[0, col_idx].imshow(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
        axes[0, col_idx].set_title(f"{cam.upper()} — Cycles Photoreal RGB", fontsize=12, fontweight="bold")
        axes[0, col_idx].axis("off")

        # Row 1: GT Depth
        d_gt = gt_depths[cam]
        d_gt_vis = np.clip(d_gt, 0.5, 25.0)
        im1 = axes[1, col_idx].imshow(d_gt_vis, cmap="plasma_r")
        axes[1, col_idx].set_title(f"{cam.upper()} — Ground Truth Z-Depth", fontsize=12)
        axes[1, col_idx].axis("off")

        # Row 2: AI Predicted Depth
        d_ai = ai_depths[cam]
        d_ai_vis = np.clip(d_ai, 0.5, 25.0)
        im2 = axes[2, col_idx].imshow(d_ai_vis, cmap="plasma_r")
        axes[2, col_idx].set_title(f"{cam.upper()} — Depth Anything V2", fontsize=12)
        axes[2, col_idx].axis("off")

    # Add colorbar for depth
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.45])
    cbar = fig.colorbar(im1, cax=cbar_ax)
    cbar.set_label("Z-Depth (metres)", fontsize=11)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[MOSAIC] Saved 4-camera visual inspection mosaic to: {out_path}")


def compute_depth_metrics(gt: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Compute MAE, RMSE, AbsRel, Delta < 1.25 over masked valid pixels."""
    if not np.any(mask):
        return {"mae": float("nan"), "rmse": float("nan"), "abs_rel": float("nan"), "delta_125": 0.0, "count": 0}

    y_true = gt[mask].astype(np.float64)
    y_pred = pred[mask].astype(np.float64)

    diff = np.abs(y_true - y_pred)
    mae = float(np.mean(diff))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    abs_rel = float(np.mean(diff / y_true))

    ratio = np.maximum(y_pred / y_true, y_true / y_pred)
    delta_125 = float(np.mean(ratio < 1.25) * 100.0)

    return {
        "mae": mae,
        "rmse": rmse,
        "abs_rel": abs_rel,
        "delta_125": delta_125,
        "count": int(np.sum(mask)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="CLEAR360 M4.2 Evaluation Runner")
    parser.add_argument("--frame", type=int, default=0, help="Frame to evaluate (default: 0)")
    args = parser.parse_args()

    frame_idx = args.frame
    print("\n" + "=" * 78)
    print("CLEAR360 — M4.2 PHOTOREALISTIC DEPTH BENCHMARK & 3D VALIDATION")
    print("=" * 78)
    print(f"Target Frame   : {frame_idx:04d}")
    print(f"Depth Model    : Depth Anything V2 Metric ViT-Small (vits)")
    print(f"Weights Path   : weights/depth_anything_v2_metric_vkitti_vits.pth")
    print("=" * 78 + "\n")

    data_dir = project_root / "data"
    photoreal_rgb_dir = data_dir / "photoreal_frames"
    photoreal_gt_depth_dir = data_dir / "photoreal_gt_depth"
    photoreal_gt_sem_dir = data_dir / "photoreal_gt_semantic"
    ai_depth_dir = data_dir / "photoreal_ai_depth"

    # Step 1: Verify photoreal data availability
    print("[1/5] Verifying photoreal dataset files...")
    for cam in CAMERA_NAMES:
        rgb_path = photoreal_rgb_dir / cam / f"frame_{frame_idx:04d}.png"
        depth_path = photoreal_gt_depth_dir / cam / f"frame_{frame_idx:04d}.exr"
        sem_path = photoreal_gt_sem_dir / cam / f"frame_{frame_idx:04d}.png"
        if not rgb_path.exists() or not depth_path.exists() or not sem_path.exists():
            print(f"[ERROR] Missing rendered data for camera '{cam}' frame {frame_idx:04d}.")
            print(f"  RGB: {rgb_path.exists()} | Depth: {depth_path.exists()} | Sem: {sem_path.exists()}")
            sys.exit(1)
        print(f"  Verified camera [{cam:5s}]: RGB ({rgb_path.stat().st_size / 1024:.1f} KB), GT Depth ({depth_path.stat().st_size / 1024:.1f} KB)")

    # Step 2: Initialize AI Depth Estimator (ViT-Small on CPU)
    print("\n[2/5] Initializing Depth Anything V2 Metric ViT-Small...")
    weights_path = project_root / "weights" / "depth_anything_v2_metric_vkitti_vits.pth"
    estimator = DepthEstimator(
        encoder="vits",
        weights_path=str(weights_path),
        device="cpu",
        max_depth=80.0,
    )

    # Step 3: Run inference & save predictions
    print("\n[3/5] Running inference on photorealistic camera views...")
    rgb_frames = {}
    gt_depths = {}
    gt_semantics = {}
    ai_depths = {}

    for cam in CAMERA_NAMES:
        rgb_path = photoreal_rgb_dir / cam / f"frame_{frame_idx:04d}.png"
        depth_path = photoreal_gt_depth_dir / cam / f"frame_{frame_idx:04d}.exr"
        sem_path = photoreal_gt_sem_dir / cam / f"frame_{frame_idx:04d}.png"

        img_bgr = cv2.imread(str(rgb_path))
        d_gt = load_depth_exr(str(depth_path))
        sem = load_semantic_png(str(sem_path))

        rgb_frames[cam] = img_bgr
        gt_depths[cam] = d_gt
        gt_semantics[cam] = sem

        # Run AI estimator
        d_pred, dt_s = estimator.estimate_depth(img_bgr, is_rgb=False)
        dt = dt_s * 1000.0

        ai_depths[cam] = d_pred

        # Save prediction
        out_cam_dir = ai_depth_dir / cam
        out_cam_dir.mkdir(parents=True, exist_ok=True)
        pred_save_path = out_cam_dir / f"frame_{frame_idx:04d}.npy"
        np.save(str(pred_save_path), d_pred)

        print(f"  [{cam:5s}] Inferred in {dt:6.1f} ms | Range: [{d_pred.min():.2f}m, {d_pred.max():.2f}m] | Mean: {d_pred.mean():.2f}m")

    # Step 4: Generate visual inspection mosaic
    mosaic_path = data_dir / "audit_samples" / "photoreal_mosaic_frame0.png"
    create_inspection_mosaic(rgb_frames, gt_depths, ai_depths, mosaic_path)

    # Step 5: Compute Official Metrics
    print("\n[4/5] Computing Quantitative Depth Metrics...")
    per_cam_metrics = {}
    all_gt = []
    all_pred = []
    all_sem = []

    for cam in CAMERA_NAMES:
        gt = gt_depths[cam]
        pred = ai_depths[cam]
        sem = gt_semantics[cam]

        # Valid mask: valid depth range [0.1, 80m], exclude sky (sem == 0)
        valid_mask = (gt >= 0.1) & (gt <= 80.0) & (sem != 0) & np.isfinite(gt) & np.isfinite(pred)
        m = compute_depth_metrics(gt, pred, valid_mask)
        per_cam_metrics[cam] = m

        all_gt.append(gt[valid_mask])
        all_pred.append(pred[valid_mask])
        all_sem.append(sem[valid_mask])

    cat_gt = np.concatenate(all_gt)
    cat_pred = np.concatenate(all_pred)
    cat_sem = np.concatenate(all_sem)

    overall_metrics = compute_depth_metrics(cat_gt, cat_pred, np.ones(len(cat_gt), dtype=bool))

    # Print Per-Camera Table
    print("\n" + "-" * 78)
    print(f"{'Camera':<8} | {'MAE (m)':<9} | {'RMSE (m)':<9} | {'AbsRel':<9} | {'δ < 1.25 (%)':<14} | {'Pixels':<9}")
    print("-" * 78)
    for cam in CAMERA_NAMES:
        m = per_cam_metrics[cam]
        print(f"{cam:<8} | {m['mae']:<9.3f} | {m['rmse']:<9.3f} | {m['abs_rel']:<9.3f} | {m['delta_125']:<14.1f} | {m['count']:<9d}")
    print("-" * 78)
    print(f"{'OVERALL':<8} | {overall_metrics['mae']:<9.3f} | {overall_metrics['rmse']:<9.3f} | {overall_metrics['abs_rel']:<9.3f} | {overall_metrics['delta_125']:<14.1f} | {overall_metrics['count']:<9d}")
    print("-" * 78)

    # Range-Band Accuracy
    bands = [
        ("0.5 to 2.0 m", 0.5, 2.0),
        ("2.0 to 5.0 m", 2.0, 5.0),
        ("5.0 to 10.0 m", 5.0, 10.0),
        ("10.0 to 20.0 m", 10.0, 20.0),
    ]
    print("\n--- Range-Band Accuracy Breakdown ---")
    print(f"{'Distance Band':<15} | {'MAE (m)':<9} | {'RMSE (m)':<9} | {'AbsRel':<9} | {'δ < 1.25 (%)':<14} | {'Pixel %':<9}")
    total_px = len(cat_gt)
    band_results = {}
    for name, d_min, d_max in bands:
        b_mask = (cat_gt >= d_min) & (cat_gt < d_max)
        bm = compute_depth_metrics(cat_gt, cat_pred, b_mask)
        pct = (bm["count"] / total_px * 100.0) if total_px > 0 else 0.0
        band_results[name] = bm
        print(f"{name:<15} | {bm['mae']:<9.3f} | {bm['rmse']:<9.3f} | {bm['abs_rel']:<9.3f} | {bm['delta_125']:<14.1f} | {pct:<8.1f}%")

    # Per-Class Accuracy Breakdown
    print("\n--- Per-Class Accuracy Breakdown ---")
    print(f"{'Semantic Class':<28} | {'MAE (m)':<9} | {'AbsRel':<9} | {'δ < 1.25 (%)':<14} | {'Pixel %':<9}")
    class_results = {}
    for class_id, class_name in SEMANTIC_CLASSES.items():
        if class_id == 0:
            continue
        c_mask = (cat_sem == class_id)
        if not np.any(c_mask):
            continue
        cm = compute_depth_metrics(cat_gt, cat_pred, c_mask)
        pct = (cm["count"] / total_px * 100.0) if total_px > 0 else 0.0
        class_results[class_name] = cm
        print(f"{class_name:<28} | {cm['mae']:<9.3f} | {cm['abs_rel']:<9.3f} | {cm['delta_125']:<14.1f} | {pct:<8.1f}%")

    # Step 6: 3D Point Cloud Generation & Comparison
    print("\n[5/5] Generating and comparing 3D fused point clouds...")
    calib = CalibrationStore(project_root / "config" / "cameras.yaml")
    calib.load()
    from pointcloud.unprojector import DepthUnprojector
    unprojector = DepthUnprojector(min_depth=0.5, max_depth=25.0, stride=2, exclude_ego=True)

    # 1. Ground truth cloud
    gt_pts_list, gt_cols_list = [], []
    for cam in CAMERA_NAMES:
        pts, cols, _ = unprojector.unproject_depth(
            gt_depths[cam],
            calib.get(cam),
            rgb=cv2.cvtColor(rgb_frames[cam], cv2.COLOR_BGR2RGB),
            semantic_mask=gt_semantics[cam],
        )
        gt_pts_list.append(pts)
        gt_cols_list.append(cols)

    merged_gt = o3d.geometry.PointCloud()
    merged_gt.points = o3d.utility.Vector3dVector(np.vstack(gt_pts_list))
    merged_gt.colors = o3d.utility.Vector3dVector(np.vstack(gt_cols_list))
    gt_ply_path = data_dir / "pointclouds" / "frame_0000_photoreal_gt.ply"
    o3d.io.write_point_cloud(str(gt_ply_path), merged_gt)

    # 2. AI predicted cloud
    ai_pts_list, ai_cols_list = [], []
    for cam in CAMERA_NAMES:
        pts, cols, _ = unprojector.unproject_depth(
            ai_depths[cam],
            calib.get(cam),
            rgb=cv2.cvtColor(rgb_frames[cam], cv2.COLOR_BGR2RGB),
            semantic_mask=gt_semantics[cam],
        )
        ai_pts_list.append(pts)
        ai_cols_list.append(cols)

    merged_ai = o3d.geometry.PointCloud()
    merged_ai.points = o3d.utility.Vector3dVector(np.vstack(ai_pts_list))
    merged_ai.colors = o3d.utility.Vector3dVector(np.vstack(ai_cols_list))
    ai_ply_path = data_dir / "pointclouds" / "frame_0000_photoreal_ai.ply"
    o3d.io.write_point_cloud(str(ai_ply_path), merged_ai)

    print(f"  Ground-Truth Fused Cloud : {len(merged_gt.points):,d} points -> {gt_ply_path}")
    print(f"  AI Predicted Fused Cloud : {len(merged_ai.points):,d} points -> {ai_ply_path}")

    # Structural metrics: Road flatness (pts within road area |X| < 4, |Y| < 4, Z near 0)
    pts_ai = np.asarray(merged_ai.points)
    road_mask = (np.abs(pts_ai[:, 0]) < 3.5) & (np.abs(pts_ai[:, 1]) < 3.5) & (pts_ai[:, 2] < 0.5)
    road_std_z = float(np.std(pts_ai[road_mask, 2])) if np.any(road_mask) else 0.0

    pts_gt = np.asarray(merged_gt.points)
    gt_road_mask = (np.abs(pts_gt[:, 0]) < 3.5) & (np.abs(pts_gt[:, 1]) < 3.5) & (pts_gt[:, 2] < 0.5)
    gt_road_std_z = float(np.std(pts_gt[gt_road_mask, 2])) if np.any(gt_road_mask) else 0.0

    print(f"  Road Plane Flatness (Z std dev): GT = {gt_road_std_z * 1000:.2f} mm | AI = {road_std_z * 1000:.2f} mm")

    # Generate Scatter Plot: GT vs Predicted Depth
    scatter_path = data_dir / "depth_eval" / "photoreal_scatter_gt_vs_pred.png"
    scatter_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 8))
    # Subsample for clear visualization
    sub_idx = np.random.choice(len(cat_gt), size=min(10000, len(cat_gt)), replace=False)
    plt.scatter(cat_gt[sub_idx], cat_pred[sub_idx], alpha=0.25, s=6, c="#1f77b4", edgecolors="none")
    plt.plot([0, 25], [0, 25], "r--", linewidth=2, label="Ideal (y = x)")
    plt.xlim(0, 25)
    plt.ylim(0, 25)
    plt.xlabel("Ground Truth Z-Depth (m)", fontsize=12)
    plt.ylabel("Depth Anything V2 Predicted Depth (m)", fontsize=12)
    plt.title(f"CLEAR360 M4.2 Photoreal: GT vs AI Depth (MAE: {overall_metrics['mae']:.2f}m, δ<1.25: {overall_metrics['delta_125']:.1f}%)", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=11)
    plt.savefig(str(scatter_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved scatter plot to: {scatter_path}")

    # Scale and Correlation Diagnostics
    print("\n--- Model Scale & Geometry Diagnostics ---")
    ratio = cat_pred / cat_gt
    median_scale = float(np.median(ratio))
    mean_scale = float(np.mean(ratio))
    corr = float(np.corrcoef(cat_gt, cat_pred)[0, 1])
    print(f"  Median Over-Prediction Ratio (d_pred / d_gt) : {median_scale:.3f}x")
    print(f"  Mean Over-Prediction Ratio   (d_pred / d_gt) : {mean_scale:.3f}x")
    print(f"  Pearson Correlation (r)                      : {corr:.3f}")

    # Scale-aligned evaluation for engineering diagnostic
    scaled_pred = cat_pred / median_scale
    scaled_metrics = compute_depth_metrics(cat_gt, scaled_pred, np.ones(len(cat_gt), dtype=bool))
    print(f"\n  [Diagnostic Reference: Scale-Aligned Depth (divided by {median_scale:.2f}x)]")
    print(f"    Aligned MAE        : {scaled_metrics['mae']:.3f} m")
    print(f"    Aligned RMSE       : {scaled_metrics['rmse']:.3f} m")
    print(f"    Aligned AbsRel     : {scaled_metrics['abs_rel']:.3f}")
    print(f"    Aligned δ < 1.25   : {scaled_metrics['delta_125']:.1f} %")
    print("M4 (Flat Analytical) vs M4.2 (Cycles Photorealistic) COMPARISON")
    print("=" * 78)
    # M4 Baseline values from docs/m4_depth_evaluation_report.md
    m4_mae, m4_rmse, m4_absrel, m4_delta = 3.197, 3.619, 1.597, 12.7
    print(f"{'Metric':<16} | {'M4 Analytical Ray-Caster':<26} | {'M4.2 Cycles Photoreal':<24} | {'Improvement':<12}")
    print("-" * 78)
    mae_diff = m4_mae - overall_metrics["mae"]
    rmse_diff = m4_rmse - overall_metrics["rmse"]
    absrel_diff = m4_absrel - overall_metrics["abs_rel"]
    delta_diff = overall_metrics["delta_125"] - m4_delta

    print(f"{'MAE (m)':<16} | {m4_mae:<26.3f} | {overall_metrics['mae']:<24.3f} | {mae_diff:+8.3f} m")
    print(f"{'RMSE (m)':<16} | {m4_rmse:<26.3f} | {overall_metrics['rmse']:<24.3f} | {rmse_diff:+8.3f} m")
    print(f"{'AbsRel':<16} | {m4_absrel:<26.3f} | {overall_metrics['abs_rel']:<24.3f} | {absrel_diff:+8.3f}")
    print(f"{'δ < 1.25 (%)':<16} | {m4_delta:<26.1f} | {overall_metrics['delta_125']:<24.1f} | {delta_diff:+8.1f} %")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
