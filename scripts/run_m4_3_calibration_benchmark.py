"""
scripts/run_m4_3_calibration_benchmark.py — M4.3 Metric Calibration & Benchmark
=================================================================================

Implements Parts C, D, and E of M4.3:
  1. Infers Depth Anything V2 on dedicated calibration dataset.
  2. Fits 4 independent calibration models strictly on calibration data:
     - Global Multiplicative Scale
     - Per-Camera Multiplicative Scale
     - Depth Affine (a * d + b)
     - Disparity Affine (alpha / d + beta)
  3. Saves calibration parameters to config/depth_calibration.yaml.
  4. Evaluates all models on:
     - Calibration dataset (in-sample)
     - Held-out evaluation dataset (out-of-sample Frame 0000)
  5. Unprojects calibrated depth into 3D point cloud:
     - data/pointclouds/frame_0000_calibrated_ai.ply
  6. Evaluates 3D geometric dimensions and obstacle distances against Ground Truth.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
import yaml

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from depth.estimator import DepthEstimator
from pipeline.calibration import CalibrationStore
from pointcloud.unprojector import DepthUnprojector
from sim.constants import CAMERA_NAMES, SEMANTIC_CLASSES
from validation.gt_loader import load_depth_exr, load_semantic_png


def compute_metrics(gt: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> dict[str, float]:
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


def main():
    print("=" * 82)
    print("CLEAR360 M4.3: METRIC DEPTH CALIBRATION & HELD-OUT EVALUATION")
    print("=" * 82)

    data_dir = project_root / "data"
    calib_ds_dir = data_dir / "calibration_dataset"
    calib_rgb_dir = calib_ds_dir / "rgb"
    calib_gt_dir = calib_ds_dir / "gt_depth"
    calib_sem_dir = calib_ds_dir / "gt_semantic"

    # Held-out dataset (Frame 0000)
    eval_rgb_dir = data_dir / "photoreal_frames"
    eval_gt_dir = data_dir / "photoreal_gt_depth"
    eval_sem_dir = data_dir / "photoreal_gt_semantic"

    # Step 1: Initialize Depth Anything V2
    print("\n[1/6] Loading Depth Anything V2 Metric ViT-Small...")
    estimator = DepthEstimator(encoder="vits", device="cpu", max_depth=80.0)

    # Step 2: Infer on Calibration Dataset
    print("\n[2/6] Running inference on dedicated calibration dataset...")
    calib_data = {}
    calib_cat_gt = []
    calib_cat_pred = []
    calib_per_cam_gt = {c: [] for c in CAMERA_NAMES}
    calib_per_cam_pred = {c: [] for c in CAMERA_NAMES}

    for cam in CAMERA_NAMES:
        rgb_path = calib_rgb_dir / cam / "calib.png"
        depth_path = calib_gt_dir / cam / "calib.exr"
        sem_path = calib_sem_dir / cam / "calib.png"

        img = cv2.imread(str(rgb_path))
        d_gt = load_depth_exr(depth_path)
        sem = load_semantic_png(sem_path)

        d_pred, dt = estimator.estimate_depth(img, is_rgb=False)

        # Mask valid pixels: exclude sky (sem == 0), valid GT [0.5, 30m]
        mask = (d_gt >= 0.5) & (d_gt <= 30.0) & (sem != 0) & np.isfinite(d_gt) & np.isfinite(d_pred)
        gt_v = d_gt[mask]
        pred_v = d_pred[mask]

        calib_data[cam] = {"img": img, "gt": d_gt, "pred": d_pred, "mask": mask}
        calib_per_cam_gt[cam] = gt_v
        calib_per_cam_pred[cam] = pred_v
        calib_cat_gt.append(gt_v)
        calib_cat_pred.append(pred_v)

        med_ratio = np.median(pred_v / gt_v)
        print(f"  [{cam:5s}] Inferred | Valid px: {len(gt_v):,d} | Median Pred/GT: {med_ratio:.3f}x")

    all_calib_gt = np.concatenate(calib_cat_gt)
    all_calib_pred = np.concatenate(calib_cat_pred)

    # Step 3: Fit Calibration Models strictly on Calibration Dataset
    print("\n[3/6] Fitting Calibration Models on Calibration Dataset...")

    # Model 1: Global Multiplicative Scale
    global_scale = float(np.median(all_calib_pred / all_calib_gt))

    # Model 2: Per-Camera Multiplicative Scale
    per_cam_scales = {}
    for cam in CAMERA_NAMES:
        per_cam_scales[cam] = float(np.median(calib_per_cam_pred[cam] / calib_per_cam_gt[cam]))

    # Model 3: Depth Affine (d_cal = a * d_pred + b)
    A_aff = np.stack([all_calib_pred, np.ones_like(all_calib_pred)], axis=1)
    aff_params, _, _, _ = np.linalg.lstsq(A_aff, all_calib_gt, rcond=None)
    a_aff, b_aff = float(aff_params[0]), float(aff_params[1])

    # Model 4: Disparity (Inverse-Depth) Affine (1/d_cal = alpha * (1/d_pred) + beta)
    inv_c_gt = 1.0 / all_calib_gt
    inv_c_pred = 1.0 / all_calib_pred
    A_inv = np.stack([inv_c_pred, np.ones_like(inv_c_pred)], axis=1)
    inv_params, _, _, _ = np.linalg.lstsq(A_inv, inv_c_gt, rcond=None)
    alpha_inv, beta_inv = float(inv_params[0]), float(inv_params[1])

    print(f"  Model 1: Global Multiplicative Scale      : s_global = {global_scale:.4f}")
    print(f"  Model 2: Per-Camera Multiplicative Scales : {per_cam_scales}")
    print(f"  Model 3: Depth Affine                     : d_cal = {a_aff:.4f} * d_pred + ({b_aff:.4f})")
    print(f"  Model 4: Disparity Affine                 : 1/d_cal = {alpha_inv:.4f} * (1/d_pred) + ({beta_inv:.4f})")

    # Save to config/depth_calibration.yaml
    calib_yaml_path = project_root / "config" / "depth_calibration.yaml"
    calib_config = {
        "depth_calibration_version": "1.0",
        "reference_checkpoint": "depth_anything_v2_metric_vkitti_vits.pth",
        "global_scale": global_scale,
        "per_camera_scales": per_cam_scales,
        "depth_affine": {"a": a_aff, "b": b_aff},
        "disparity_affine": {"alpha": alpha_inv, "beta": beta_inv},
    }
    with open(calib_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(calib_config, f, default_flow_style=False)
    print(f"  Saved calibration config to: {calib_yaml_path}")

    # Helper calibration functions
    def apply_calib(pred: np.ndarray, cam_name: str, model_type: str) -> np.ndarray:
        if model_type == "original":
            return pred
        elif model_type == "global_scale":
            return pred / global_scale
        elif model_type == "per_camera_scale":
            return pred / per_cam_scales[cam_name]
        elif model_type == "depth_affine":
            return np.maximum(0.1, a_aff * pred + b_aff)
        elif model_type == "disparity_affine":
            disp = alpha_inv * (1.0 / np.maximum(0.1, pred)) + beta_inv
            return 1.0 / np.clip(disp, 1e-4, 10.0)
        raise ValueError(model_type)

    models = ["original", "global_scale", "per_camera_scale", "depth_affine", "disparity_affine"]

    # Step 4: Evaluate on Calibration Dataset (In-Sample)
    print("\n[4/6] Evaluating Models on Dedicated Calibration Dataset (In-Sample)...")
    print(f"{'Model':<22} | {'MAE (m)':<9} | {'RMSE (m)':<9} | {'AbsRel':<9} | {'δ < 1.25 (%)':<14}")
    print("-" * 72)
    for m_type in models:
        c_preds = []
        for cam in CAMERA_NAMES:
            p = apply_calib(calib_data[cam]["pred"], cam, m_type)
            c_preds.append(p[calib_data[cam]["mask"]])
        all_cp = np.concatenate(c_preds)
        res = compute_metrics(all_calib_gt, all_cp, np.ones(len(all_calib_gt), dtype=bool))
        print(f"{m_type:<22} | {res['mae']:<9.3f} | {res['rmse']:<9.3f} | {res['abs_rel']:<9.3f} | {res['delta_125']:<14.1f}")

    # Step 5: Evaluate on Held-Out Evaluation Dataset (Frame 0000)
    print("\n[5/6] Evaluating Models on HELD-OUT Evaluation Dataset (Out-of-Sample Frame 0000)...")
    eval_gt_dict = {}
    eval_pred_dict = {}
    eval_sem_dict = {}
    eval_rgb_dict = {}

    eval_cat_gt = []
    eval_cat_masks = []
    eval_per_cam_gt = {}
    eval_per_cam_pred = {}

    for cam in CAMERA_NAMES:
        gt_p = eval_gt_dir / cam / "frame_0000.exr"
        sem_p = eval_sem_dir / cam / "frame_0000.png"
        rgb_p = eval_rgb_dir / cam / "frame_0000.png"
        pred_p = data_dir / "photoreal_ai_depth" / cam / "frame_0000.npy"

        d_gt = load_depth_exr(gt_p)
        sem = load_semantic_png(sem_p)
        rgb = cv2.imread(str(rgb_p))
        d_pred = np.load(pred_p)

        mask = (d_gt >= 0.5) & (d_gt <= 50.0) & (sem != 0) & np.isfinite(d_gt) & np.isfinite(d_pred)

        eval_gt_dict[cam] = d_gt
        eval_pred_dict[cam] = d_pred
        eval_sem_dict[cam] = sem
        eval_rgb_dict[cam] = rgb

        eval_per_cam_gt[cam] = d_gt[mask]
        eval_per_cam_pred[cam] = d_pred[mask]
        eval_cat_gt.append(d_gt[mask])

    all_eval_gt = np.concatenate(eval_cat_gt)

    print("-" * 72)
    print(f"{'Model':<22} | {'MAE (m)':<9} | {'RMSE (m)':<9} | {'AbsRel':<9} | {'δ < 1.25 (%)':<14}")
    print("-" * 72)
    heldout_results = {}
    for m_type in models:
        ep_list = []
        for cam in CAMERA_NAMES:
            p = apply_calib(eval_pred_dict[cam], cam, m_type)
            ep_list.append(p[(eval_gt_dict[cam] >= 0.5) & (eval_gt_dict[cam] <= 50.0) & (eval_sem_dict[cam] != 0)])
        all_ep = np.concatenate(ep_list)
        res = compute_metrics(all_eval_gt, all_ep, np.ones(len(all_eval_gt), dtype=bool))
        heldout_results[m_type] = res
        print(f"{m_type:<22} | {res['mae']:<9.3f} | {res['rmse']:<9.3f} | {res['abs_rel']:<9.3f} | {res['delta_125']:<14.1f}")
    print("-" * 72)

    # Per-Camera breakdown on Held-out set for Per-Camera Scale Model
    print("\n--- Held-Out Per-Camera Breakdown for Per-Camera Calibrated Model ---")
    print(f"{'Camera':<8} | {'Scale Used':<12} | {'MAE (m)':<9} | {'RMSE (m)':<9} | {'AbsRel':<9} | {'δ < 1.25 (%)':<14}")
    print("-" * 72)
    for cam in CAMERA_NAMES:
        p_cal = apply_calib(eval_pred_dict[cam], cam, "per_camera_scale")
        mask = (eval_gt_dict[cam] >= 0.5) & (eval_gt_dict[cam] <= 50.0) & (eval_sem_dict[cam] != 0)
        c_m = compute_metrics(eval_gt_dict[cam], p_cal, mask)
        print(f"{cam:<8} | {per_cam_scales[cam]:<12.3f} | {c_m['mae']:<9.3f} | {c_m['rmse']:<9.3f} | {c_m['abs_rel']:<9.3f} | {c_m['delta_125']:<14.1f}")

    # Step 6: 3D Point Cloud Generation & Geometric Analysis
    print("\n[6/6] Generating 3D Point Clouds from Calibrated Depths...")
    calib_store = CalibrationStore(project_root / "config" / "cameras.yaml")
    calib_store.load()
    unprojector = DepthUnprojector(min_depth=0.5, max_depth=35.0, stride=2, exclude_ego=True)

    # Build calibrated depth dict using per-camera calibration
    calib_depths = {cam: apply_calib(eval_pred_dict[cam], cam, "per_camera_scale") for cam in CAMERA_NAMES}

    # Save calibrated depth maps to data/photoreal_calib_depth/
    calib_depth_out_dir = data_dir / "photoreal_calib_depth"
    for cam in CAMERA_NAMES:
        (calib_depth_out_dir / cam).mkdir(parents=True, exist_ok=True)
        np.save(str(calib_depth_out_dir / cam / "frame_0000.npy"), calib_depths[cam])

    calib_pts_list, calib_cols_list = [], []
    for cam in CAMERA_NAMES:
        pts, cols, _ = unprojector.unproject_depth(
            calib_depths[cam],
            calib_store.get(cam),
            rgb=cv2.cvtColor(eval_rgb_dict[cam], cv2.COLOR_BGR2RGB),
            semantic_mask=eval_sem_dict[cam],
        )
        calib_pts_list.append(pts)
        calib_cols_list.append(cols)

    merged_calib_ai = o3d.geometry.PointCloud()
    merged_calib_ai.points = o3d.utility.Vector3dVector(np.vstack(calib_pts_list))
    merged_calib_ai.colors = o3d.utility.Vector3dVector(np.vstack(calib_cols_list))
    calib_ply_path = data_dir / "pointclouds" / "frame_0000_calibrated_ai.ply"
    o3d.io.write_point_cloud(str(calib_ply_path), merged_calib_ai)

    # Read GT and uncalibrated clouds for comparison
    gt_pcd = o3d.io.read_point_cloud(str(data_dir / "pointclouds" / "frame_0000_photoreal_gt.ply"))
    uncal_pcd = o3d.io.read_point_cloud(str(data_dir / "pointclouds" / "frame_0000_photoreal_ai.ply"))

    pts_cal = np.asarray(merged_calib_ai.points)
    pts_gt = np.asarray(gt_pcd.points)
    pts_uncal = np.asarray(uncal_pcd.points)

    print(f"\n  Point Cloud Extents (Vehicle Frame: X=Right, Y=Forward, Z=Up):")
    print(f"    Ground Truth Cloud       : X in [{pts_gt[:,0].min():.2f}, {pts_gt[:,0].max():.2f}] m | Y in [{pts_gt[:,1].min():.2f}, {pts_gt[:,1].max():.2f}] m | Z in [{pts_gt[:,2].min():.2f}, {pts_gt[:,2].max():.2f}] m")
    print(f"    Uncalibrated AI Cloud    : X in [{pts_uncal[:,0].min():.2f}, {pts_uncal[:,0].max():.2f}] m | Y in [{pts_uncal[:,1].min():.2f}, {pts_uncal[:,1].max():.2f}] m | Z in [{pts_uncal[:,2].min():.2f}, {pts_uncal[:,2].max():.2f}] m")
    print(f"    CALIBRATED AI Cloud      : X in [{pts_cal[:,0].min():.2f}, {pts_cal[:,0].max():.2f}] m | Y in [{pts_cal[:,1].min():.2f}, {pts_cal[:,1].max():.2f}] m | Z in [{pts_cal[:,2].min():.2f}, {pts_cal[:,2].max():.2f}] m")

    # Obstacle distance verification at known locations:
    # 1. North Wall (nominal Y = +9.5 m)
    nw_pts_gt = pts_gt[(np.abs(pts_gt[:, 0]) < 2.0) & (pts_gt[:, 1] > 8.0) & (pts_gt[:, 2] > 0.5)]
    nw_pts_cal = pts_cal[(np.abs(pts_cal[:, 0]) < 2.0) & (pts_cal[:, 1] > 6.0) & (pts_cal[:, 2] > 0.5)]
    mean_y_gt = float(np.mean(nw_pts_gt[:, 1])) if len(nw_pts_gt) > 0 else 9.5
    mean_y_cal = float(np.mean(nw_pts_cal[:, 1])) if len(nw_pts_cal) > 0 else 0.0

    # 2. South Wall (nominal Y = -9.5 m)
    sw_pts_gt = pts_gt[(np.abs(pts_gt[:, 0]) < 2.0) & (pts_gt[:, 1] < -8.0) & (pts_gt[:, 2] > 0.5)]
    sw_pts_cal = pts_cal[(np.abs(pts_cal[:, 0]) < 2.0) & (pts_cal[:, 1] < -6.0) & (pts_cal[:, 2] > 0.5)]
    mean_sw_gt = float(np.mean(sw_pts_gt[:, 1])) if len(sw_pts_gt) > 0 else -9.5
    mean_sw_cal = float(np.mean(sw_pts_cal[:, 1])) if len(sw_pts_cal) > 0 else 0.0

    print(f"\n  Obstacle Landmark Positions:")
    print(f"    North Wall Position (Y)  : Ground Truth = {mean_y_gt:.2f} m | Calibrated AI = {mean_y_cal:.2f} m (Error: {abs(mean_y_gt - mean_y_cal):.2f} m)")
    print(f"    South Wall Position (Y)  : Ground Truth = {mean_sw_gt:.2f} m | Calibrated AI = {mean_sw_cal:.2f} m (Error: {abs(mean_sw_gt - mean_sw_cal):.2f} m)")


if __name__ == "__main__":
    main()

