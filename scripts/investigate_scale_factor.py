"""
scripts/investigate_scale_factor.py — Part B: Deep Scale Factor Investigation
=============================================================================

Investigates the 6.72x scale discrepancy between Depth Anything V2 and Ground Truth:
  1. Per-camera statistics: median, mean, std, CV (std/mean).
  2. Scale vs Distance bins.
  3. Scale vs Semantic object classes.
  4. Scale vs Image position (radial distance from center and vertical image coordinate).
  5. Multiplicative scale vs Affine vs Inverse-Depth Affine models.
"""

from pathlib import Path
import sys
import numpy as np
import cv2

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from validation.gt_loader import load_depth_exr, load_semantic_png
from sim.constants import CAMERA_NAMES, SEMANTIC_CLASSES


def main():
    data_dir = project_root / "data"
    photoreal_gt_depth_dir = data_dir / "photoreal_gt_depth"
    photoreal_gt_sem_dir = data_dir / "photoreal_gt_semantic"
    ai_depth_dir = data_dir / "photoreal_ai_depth"

    frame_idx = 0
    per_cam_data = {}

    print("=" * 78)
    print("CLEAR360 M4.3 PART B: PREDICTED / GROUND-TRUTH DEPTH RATIO INVESTIGATION")
    print("=" * 78)

    for cam in CAMERA_NAMES:
        gt_path = photoreal_gt_depth_dir / cam / f"frame_{frame_idx:04d}.exr"
        sem_path = photoreal_gt_sem_dir / cam / f"frame_{frame_idx:04d}.png"
        pred_path = ai_depth_dir / cam / f"frame_{frame_idx:04d}.npy"

        d_gt = load_depth_exr(gt_path)
        sem = load_semantic_png(sem_path)
        d_pred = np.load(pred_path)

        # Valid mask: exclude sky (sem == 0), valid GT [0.5, 50m], valid finite pred
        mask = (d_gt >= 0.5) & (d_gt <= 50.0) & (sem != 0) & np.isfinite(d_gt) & np.isfinite(d_pred)

        gt_vals = d_gt[mask]
        pred_vals = d_pred[mask]
        sem_vals = sem[mask]

        # Coordinates
        h, w = d_gt.shape
        v_coords, u_coords = np.indices((h, w))
        u_vals = u_coords[mask]
        v_vals = v_coords[mask]

        ratios = pred_vals / gt_vals

        per_cam_data[cam] = {
            "gt": gt_vals,
            "pred": pred_vals,
            "sem": sem_vals,
            "u": u_vals,
            "v": v_vals,
            "ratios": ratios,
        }

    # 1. Per-Camera Ratio Table
    print("\n--- 1. Per-Camera Predicted/GT Depth Ratio Statistics ---")
    print(f"{'Camera':<8} | {'Median Ratio':<14} | {'Mean Ratio':<12} | {'Std Dev':<10} | {'CV (Std/Mean)':<14} | {'Pixel Count':<12}")
    print("-" * 78)
    all_ratios = []
    all_gt = []
    all_pred = []
    all_sem = []
    all_u = []
    all_v = []

    for cam in CAMERA_NAMES:
        r = per_cam_data[cam]["ratios"]
        all_ratios.append(r)
        all_gt.append(per_cam_data[cam]["gt"])
        all_pred.append(per_cam_data[cam]["pred"])
        all_sem.append(per_cam_data[cam]["sem"])
        all_u.append(per_cam_data[cam]["u"])
        all_v.append(per_cam_data[cam]["v"])

        med = float(np.median(r))
        mean = float(np.mean(r))
        std = float(np.std(r))
        cv = (std / mean) if mean > 0 else 0.0
        print(f"{cam:<8} | {med:<14.3f} | {mean:<12.3f} | {std:<10.3f} | {cv:<14.3f} | {len(r):<12d}")

    cat_r = np.concatenate(all_ratios)
    cat_gt = np.concatenate(all_gt)
    cat_pred = np.concatenate(all_pred)
    cat_sem = np.concatenate(all_sem)
    cat_u = np.concatenate(all_u)
    cat_v = np.concatenate(all_v)

    tot_med = float(np.median(cat_r))
    tot_mean = float(np.mean(cat_r))
    tot_std = float(np.std(cat_r))
    tot_cv = tot_std / tot_mean
    print("-" * 78)
    print(f"{'OVERALL':<8} | {tot_med:<14.3f} | {tot_mean:<12.3f} | {tot_std:<10.3f} | {tot_cv:<14.3f} | {len(cat_r):<12d}")
    print("-" * 78)

    # 2. Ratio vs Distance
    print("\n--- 2. Scale Ratio vs Ground Truth Distance ---")
    dist_bins = [
        ("0.5 - 1.5 m", 0.5, 1.5),
        ("1.5 - 3.0 m", 1.5, 3.0),
        ("3.0 - 5.0 m", 3.0, 5.0),
        ("5.0 - 8.0 m", 5.0, 8.0),
        ("8.0 - 15.0 m", 8.0, 15.0),
    ]
    print(f"{'Distance Bin':<15} | {'Median Ratio':<14} | {'Mean Ratio':<12} | {'Std Dev':<10} | {'CV':<8} | {'Pixel %':<8}")
    print("-" * 78)
    for name, d_min, d_max in dist_bins:
        b_mask = (cat_gt >= d_min) & (cat_gt < d_max)
        if np.any(b_mask):
            br = cat_r[b_mask]
            med = float(np.median(br))
            mean = float(np.mean(br))
            std = float(np.std(br))
            cv = std / mean
            pct = len(br) / len(cat_r) * 100.0
            print(f"{name:<15} | {med:<14.3f} | {mean:<12.3f} | {std:<10.3f} | {cv:<8.3f} | {pct:<7.1f}%")

    # 3. Ratio vs Semantic Class
    print("\n--- 3. Scale Ratio vs Semantic Object Category ---")
    print(f"{'Semantic Category':<28} | {'Median Ratio':<14} | {'Mean Ratio':<12} | {'Std Dev':<10} | {'CV':<8} | {'Pixel %':<8}")
    print("-" * 78)
    for class_id, class_name in SEMANTIC_CLASSES.items():
        if class_id == 0:
            continue
        c_mask = (cat_sem == class_id)
        if np.any(c_mask):
            cr = cat_r[c_mask]
            med = float(np.median(cr))
            mean = float(np.mean(cr))
            std = float(np.std(cr))
            cv = std / mean
            pct = len(cr) / len(cat_r) * 100.0
            print(f"{class_name:<28} | {med:<14.3f} | {mean:<12.3f} | {std:<10.3f} | {cv:<8.3f} | {pct:<7.1f}%")

    # 4. Ratio vs Image Position
    print("\n--- 4. Scale Ratio vs Image Radial Position (Center vs Periphery) ---")
    # Normalized radial distance from optical center (640, 360)
    cx, cy = 640.0, 360.0
    max_radius = np.sqrt(cx**2 + cy**2)
    radius = np.sqrt((cat_u - cx)**2 + (cat_v - cy)**2) / max_radius

    rad_bins = [
        ("Center (0.0 - 0.3)", 0.0, 0.3),
        ("Mid-field (0.3 - 0.6)", 0.3, 0.6),
        ("Outer (0.6 - 0.85)", 0.6, 0.85),
        ("Periphery (0.85 - 1.0)", 0.85, 1.0),
    ]
    print(f"{'Image Zone':<24} | {'Median Ratio':<14} | {'Mean Ratio':<12} | {'Std Dev':<10} | {'CV':<8} | {'Pixel %':<8}")
    print("-" * 78)
    for name, r_min, r_max in rad_bins:
        rmask = (radius >= r_min) & (radius < r_max)
        if np.any(rmask):
            rr = cat_r[rmask]
            med = float(np.median(rr))
            mean = float(np.mean(rr))
            std = float(np.std(rr))
            cv = std / mean
            pct = len(rr) / len(cat_r) * 100.0
            print(f"{name:<24} | {med:<14.3f} | {mean:<12.3f} | {std:<10.3f} | {cv:<8.3f} | {pct:<7.1f}%")

    # 5. Multiplicative vs Affine vs Inverse-Depth Affine Models
    print("\n--- 5. Model Calibration Comparison: Multiplicative vs Affine vs Disparity Affine ---")
    
    # Model A: Constant Scale: d_est = d_pred / s  (or d_gt ≈ s_inv * d_pred)
    s_mult = tot_med  # 6.720
    pred_mult = cat_pred / s_mult
    mae_mult = np.mean(np.abs(cat_gt - pred_mult))
    rmse_mult = np.sqrt(np.mean((cat_gt - pred_mult)**2))

    # Model B: Depth Affine: d_gt ≈ a * d_pred + b
    A_mat = np.stack([cat_pred, np.ones_like(cat_pred)], axis=1)
    affine_params, _, _, _ = np.linalg.lstsq(A_mat, cat_gt, rcond=None)
    a_aff, b_aff = affine_params[0], affine_params[1]
    pred_affine = a_aff * cat_pred + b_aff
    mae_aff = np.mean(np.abs(cat_gt - pred_affine))
    rmse_aff = np.sqrt(np.mean((cat_gt - pred_affine)**2))

    # Model C: Disparity (Inverse Depth) Affine: 1/d_gt ≈ alpha * (1/d_pred) + beta
    inv_gt = 1.0 / cat_gt
    inv_pred = 1.0 / cat_pred
    A_inv = np.stack([inv_pred, np.ones_like(inv_pred)], axis=1)
    inv_params, _, _, _ = np.linalg.lstsq(A_inv, inv_gt, rcond=None)
    alpha_inv, beta_inv = inv_params[0], inv_params[1]
    pred_disp_aff = 1.0 / np.clip(alpha_inv * inv_pred + beta_inv, 1e-4, 10.0)
    mae_disp = np.mean(np.abs(cat_gt - pred_disp_aff))
    rmse_disp = np.sqrt(np.mean((cat_gt - pred_disp_aff)**2))

    print(f"Model A (Pure Multiplicative: d_cal = d_pred / {s_mult:.3f}):")
    print(f"  MAE  : {mae_mult:.4f} m | RMSE : {rmse_mult:.4f} m")
    print(f"\nModel B (Depth Affine: d_cal = {a_aff:.4f} * d_pred + ({b_aff:.4f})):")
    print(f"  MAE  : {mae_aff:.4f} m | RMSE : {rmse_aff:.4f} m")
    print(f"\nModel C (Disparity Affine: 1/d_cal = {alpha_inv:.4f} * (1/d_pred) + ({beta_inv:.4f})):")
    print(f"  MAE  : {mae_disp:.4f} m | RMSE : {rmse_disp:.4f} m")


if __name__ == "__main__":
    main()

