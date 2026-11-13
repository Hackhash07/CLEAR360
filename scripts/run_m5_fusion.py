"""
scripts/run_m5_fusion.py — Milestone M5 Multi-Camera 3D Fusion & Consistency Benchmark
======================================================================================

Executes the complete M5 fusion evaluation pipeline:
  1. Ground-Truth (GT) 3D Point Cloud Reconstruction:
     - Individual camera point clouds (front, rear, left, right)
     - 4-camera unified point cloud: data/pointclouds/frame_0000_gt_merged.ply
     - 5 cm voxel-downsampled cloud: data/pointclouds/frame_0000_gt_merged_ds005.ply
  2. AI-Predicted 3D Point Cloud Reconstructions:
     - Raw AI: data/pointclouds/frame_0000_ai_raw_merged.ply
     - Global-Scale: data/pointclouds/frame_0000_ai_global_merged.ply
     - Per-Camera Scale: data/pointclouds/frame_0000_ai_per_camera_merged.ply
  3. Camera Overlap Consistency Analysis (Front-Left, Front-Right, Rear-Left, Rear-Right):
     - Overlapping point counts
     - Nearest-neighbor distances (median, mean, 95th percentile, max)
     - Point-to-plane residuals
     - Inlier ratios and occlusion separation
     - Overlap visualizations (camera color-coded & error heatmap)
  4. Geometric Distortion & Bounding Box Investigation:
     - Road flatness and slope
     - Wall verticality and distances
     - Pillar radius and centroid
     - Sky leakage and peripheral expansion quantification
     - Cloud-to-cloud comparison
  5. Multi-Scene Evaluation across 3 synchronized scene variants.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Tuple, Union
import yaml

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fusion.engine import (
    CAMERA_OVERLAP_COLORS,
    FusionResult,
    PointCloudFusionEngine,
)
from fusion.overlap import (
    OVERLAP_REGIONS,
    OverlapAnalyzer,
    OverlapMetrics,
)
from fusion.distortion import (
    CloudComparisonStats,
    GeometricDistortionAnalyzer,
    PillarStats,
    RoadProfileStats,
    SkyLeakageStats,
    WallStats,
)
from pipeline.calibration import CalibrationStore
from pipeline.sync import CameraFrameData, FrameSet, FrameSetLoader
from sim.constants import CAMERA_NAMES
from validation.gt_loader import load_depth_exr, load_semantic_png


def load_gt_frame_data(
    data_dir: Path,
    frame_id: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Load GT RGB, Depth EXR, and Semantic PNG maps for a given frame."""
    rgbs = {}
    depths = {}
    semantics = {}

    rgb_dir = data_dir / "photoreal_frames"
    depth_dir = data_dir / "photoreal_gt_depth"
    sem_dir = data_dir / "photoreal_gt_semantic"

    for cam in CAMERA_NAMES:
        rgb_file = rgb_dir / cam / f"frame_{frame_id:04d}.png"
        depth_file = depth_dir / cam / f"frame_{frame_id:04d}.exr"
        sem_file = sem_dir / cam / f"frame_{frame_id:04d}.png"

        if not rgb_file.exists() or not depth_file.exists():
            raise FileNotFoundError(f"Missing required GT data for camera '{cam}' frame {frame_id}")

        bgr = cv2.imread(str(rgb_file), cv2.IMREAD_COLOR)
        rgbs[cam] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        depths[cam] = load_depth_exr(depth_file)
        if sem_file.exists():
            semantics[cam] = load_semantic_png(sem_file)

    return rgbs, depths, semantics


def main() -> None:
    parser = argparse.ArgumentParser(description="CLEAR360 M5 Multi-Camera 3D Fusion & Benchmark")
    parser.add_argument("--frame", type=int, default=0, help="Primary frame index to analyze (default: 0)")
    parser.add_argument("--voxel-size", type=float, default=0.05, help="Voxel size in metres (default: 0.05 m)")
    args = parser.parse_args()

    frame_id = args.frame
    voxel_size = args.voxel_size
    data_dir = PROJECT_ROOT / "data"
    output_dir = data_dir / "pointclouds"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = PROJECT_ROOT / "docs" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 78)
    print("CLEAR360 — MILESTONE M5: MULTI-CAMERA 3D FUSION & OVERLAP CONSISTENCY")
    print("=" * 78)
    print(f"Target Frame     : {frame_id:04d}")
    print(f"Voxel Resolution : {voxel_size * 100:.1f} cm")
    print(f"Cameras          : {', '.join(CAMERA_NAMES)}")
    print("=" * 78 + "\n")

    # 1. Load Calibration Stores
    calib_yaml = PROJECT_ROOT / "config" / "cameras.yaml"
    calib_store = CalibrationStore(calib_yaml)
    calib_store.load()
    print(f"[1/5] Loaded authoritative camera rig calibration from {calib_yaml.name}")

    depth_calib_yaml = PROJECT_ROOT / "config" / "depth_calibration.yaml"
    with open(depth_calib_yaml, "r") as f:
        depth_calib = yaml.safe_load(f)
    global_scale = float(depth_calib["global_scale"])
    per_cam_scales = {k: float(v) for k, v in depth_calib["per_camera_scales"].items()}
    print(f"      Global scale s_global = {global_scale:.4f}")
    for k, v in per_cam_scales.items():
        print(f"      Per-camera scale s_{k} = {v:.4f}")

    engine = PointCloudFusionEngine(calib_store, max_depth=50.0, min_depth=0.1, stride=1, exclude_ego=True)
    overlap_analyzer = OverlapAnalyzer(calib_store, occlusion_dist_threshold=0.50)

    # 2. PART 1: Ground-Truth 3D Reconstruction
    print(f"\n[2/5] PART 1: Ground-Truth 3D Multi-Camera Reconstruction...")
    gt_rgbs, gt_depths, gt_semantics = load_gt_frame_data(data_dir, frame_id)

    gt_result = engine.fuse_depth_maps(
        depth_maps=gt_depths,
        rgb_maps=gt_rgbs,
        semantic_masks=gt_semantics,
        frame_id=frame_id,
        pipeline_type="ground_truth",
        use_camera_colors=False,
    )
    print(f"      {gt_result.summary()}")

    # Save individual camera GT point clouds
    for cam_name, cam_pcd in gt_result.camera_clouds.items():
        cam_ply_path = output_dir / f"frame_{frame_id:04d}_gt_{cam_name}.ply"
        engine.save_ply(cam_pcd, cam_ply_path)
        print(f"      Saved individual GT cloud: {cam_ply_path.name} ({len(cam_pcd.points):,} pts)")

    # Save merged GT point cloud and downsampled version
    gt_merged_path = output_dir / f"frame_{frame_id:04d}_gt_merged.ply"
    engine.save_ply(gt_result.point_cloud, gt_merged_path)

    gt_ds = engine.downsample_voxel(gt_result.point_cloud, voxel_size=voxel_size)
    gt_ds_path = output_dir / f"frame_{frame_id:04d}_gt_merged_ds005.ply"
    engine.save_ply(gt_ds, gt_ds_path)
    print(f"      Saved: {gt_merged_path.name} ({gt_result.num_points:,} pts)")
    print(f"      Saved: {gt_ds_path.name} ({len(gt_ds.points):,} pts)")

    # 3. PART 2: Overlap Analysis on Ground Truth
    print(f"\n[3/5] PART 2: Camera Overlap Geometric Consistency (Ground Truth)...")
    gt_overlap_metrics = overlap_analyzer.analyze_all_regions(gt_result.camera_clouds)

    print("-" * 88)
    print(f"{'Region':15s} | {'Pair':12s} | {'Pts A':8s} | {'Pts B':8s} | {'Med NN (m)':10s} | {'Mean NN':10s} | {'P95 (m)':8s} | {'Pt2Plane':8s}")
    print("-" * 88)
    for reg, m in gt_overlap_metrics.items():
        p2p_str = f"{m.median_pt2plane_m:.4f}" if m.median_pt2plane_m is not None else "N/A"
        print(
            f"{reg:15s} | {m.cam_a + '-' + m.cam_b:12s} | {m.num_points_a:8,d} | {m.num_points_b:8,d} | "
            f"{m.median_nn_dist_m:10.4f} | {m.mean_nn_dist_m:10.4f} | {m.p95_nn_dist_m:8.4f} | {p2p_str:8s}"
        )
    print("-" * 88)

    # Generate Overlap Visualizations
    gt_overlap_colored = engine.fuse_depth_maps(
        depth_maps=gt_depths,
        rgb_maps=gt_rgbs,
        semantic_masks=gt_semantics,
        frame_id=frame_id,
        pipeline_type="ground_truth",
        use_camera_colors=True,
    )
    overlap_cam_colors_path = output_dir / f"frame_{frame_id:04d}_overlap_cameras.ply"
    engine.save_ply(gt_overlap_colored.point_cloud, overlap_cam_colors_path)
    print(f"      Saved camera color-coded overlap visualization: {overlap_cam_colors_path.name}")

    gt_overlap_heatmap = overlap_analyzer.generate_overlap_heatmap(gt_result.camera_clouds, max_error_thresh_m=0.15)
    overlap_heatmap_path = output_dir / f"frame_{frame_id:04d}_overlap_heatmap.ply"
    engine.save_ply(gt_overlap_heatmap, overlap_heatmap_path)
    print(f"      Saved overlap alignment heatmap: {overlap_heatmap_path.name} ({len(gt_overlap_heatmap.points):,} pts)")

    # 4. PART 3: AI-Predicted 3D Reconstruction (Raw, Global, Per-Camera)
    print(f"\n[4/5] PART 3: AI-Predicted 3D Point Cloud Reconstructions...")

    # Load raw AI depth maps
    raw_ai_depths = {}
    for cam in CAMERA_NAMES:
        raw_ai_file = data_dir / "photoreal_ai_depth" / cam / f"frame_{frame_id:04d}.npy"
        if not raw_ai_file.exists():
            raise FileNotFoundError(f"Missing raw AI depth: {raw_ai_file}")
        raw_ai_depths[cam] = np.load(raw_ai_file).astype(np.float32)

    # a) Raw AI Reconstruction
    raw_ai_result = engine.fuse_depth_maps(
        depth_maps=raw_ai_depths,
        rgb_maps=gt_rgbs,
        semantic_masks=gt_semantics,
        frame_id=frame_id,
        pipeline_type="ai_raw",
    )
    raw_ai_path = output_dir / f"frame_{frame_id:04d}_ai_raw_merged.ply"
    engine.save_ply(raw_ai_result.point_cloud, raw_ai_path)
    print(f"      a) Raw AI       : {raw_ai_result.summary()}")
    print(f"         Saved: {raw_ai_path.name}")

    # b) Global-Scale Calibrated AI Reconstruction
    global_ai_depths = {cam: raw_ai_depths[cam] / global_scale for cam in CAMERA_NAMES}
    global_ai_result = engine.fuse_depth_maps(
        depth_maps=global_ai_depths,
        rgb_maps=gt_rgbs,
        semantic_masks=gt_semantics,
        frame_id=frame_id,
        pipeline_type="ai_global",
    )
    global_ai_path = output_dir / f"frame_{frame_id:04d}_ai_global_merged.ply"
    engine.save_ply(global_ai_result.point_cloud, global_ai_path)
    print(f"      b) Global Calib : {global_ai_result.summary()}")
    print(f"         Saved: {global_ai_path.name}")

    # c) Per-Camera Calibrated AI Reconstruction
    per_cam_ai_depths = {cam: raw_ai_depths[cam] / per_cam_scales[cam] for cam in CAMERA_NAMES}
    per_cam_ai_result = engine.fuse_depth_maps(
        depth_maps=per_cam_ai_depths,
        rgb_maps=gt_rgbs,
        semantic_masks=gt_semantics,
        frame_id=frame_id,
        pipeline_type="ai_per_camera",
    )
    per_cam_ai_path = output_dir / f"frame_{frame_id:04d}_ai_per_camera_merged.ply"
    engine.save_ply(per_cam_ai_result.point_cloud, per_cam_ai_path)
    per_cam_ds = engine.downsample_voxel(per_cam_ai_result.point_cloud, voxel_size=voxel_size)
    per_cam_ds_path = output_dir / f"frame_{frame_id:04d}_ai_per_camera_merged_ds005.ply"
    engine.save_ply(per_cam_ds, per_cam_ds_path)
    print(f"      c) Per-Cam Calib: {per_cam_ai_result.summary()}")
    print(f"         Saved: {per_cam_ai_path.name}")
    print(f"         Saved: {per_cam_ds_path.name} ({len(per_cam_ds.points):,} pts)")

    # Overlap analysis on Calibrated AI
    print(f"\n      Overlap Consistency on Per-Camera Calibrated AI Point Cloud:")
    ai_overlap_metrics = overlap_analyzer.analyze_all_regions(per_cam_ai_result.camera_clouds)
    print("-" * 88)
    print(f"{'Region':15s} | {'Pair':12s} | {'Pts A':8s} | {'Pts B':8s} | {'Med NN (m)':10s} | {'Mean NN':10s} | {'P95 (m)':8s} | {'Co-Obs Med':10s}")
    print("-" * 88)
    for reg, m in ai_overlap_metrics.items():
        print(
            f"{reg:15s} | {m.cam_a + '-' + m.cam_b:12s} | {m.num_points_a:8,d} | {m.num_points_b:8,d} | "
            f"{m.median_nn_dist_m:10.4f} | {m.mean_nn_dist_m:10.4f} | {m.p95_nn_dist_m:8.4f} | {m.co_observed_median_m:10.4f}"
        )
    print("-" * 88)

    # 5. PART 4: Geometric Distortion & Bounding Box Investigation
    print(f"\n[5/5] PART 4: Geometric Distortion & Bounding Box Investigation...")
    gt_pts = np.asarray(gt_result.point_cloud.points)
    ai_pts = np.asarray(per_cam_ai_result.point_cloud.points)

    # Road Profile
    gt_road = GeometricDistortionAnalyzer.analyze_road_profile(gt_pts)
    ai_road = GeometricDistortionAnalyzer.analyze_road_profile(ai_pts)
    print(f"      Road Profile:")
    print(f"        GT : Mean Z = {gt_road.mean_elevation_z:.4f} m, Slope = {gt_road.slope_angle_deg:.2f} deg, RMSE = {gt_road.planarity_rmse_m:.4f} m")
    print(f"        AI : Mean Z = {ai_road.mean_elevation_z:.4f} m, Slope = {ai_road.slope_angle_deg:.2f} deg, RMSE = {ai_road.planarity_rmse_m:.4f} m")

    # Walls
    gt_walls = GeometricDistortionAnalyzer.analyze_walls(gt_pts)
    ai_walls = GeometricDistortionAnalyzer.analyze_walls(ai_pts)
    print(f"      Wall Distances & Verticality:")
    for w in ["north", "south", "east", "west"]:
        gw = gt_walls[w]
        aw = ai_walls[w]
        print(
            f"        Wall {w.upper():5s} | GT: dist={gw.fitted_distance_m:.3f} m, tilt={gw.normal_tilt_deg:.2f}° | "
            f"AI: dist={aw.fitted_distance_m:.3f} m (err={aw.distance_error_m:+.3f} m), tilt={aw.normal_tilt_deg:.2f}°"
        )

    # Pillar
    gt_pillar = GeometricDistortionAnalyzer.analyze_pillar(gt_pts)
    ai_pillar = GeometricDistortionAnalyzer.analyze_pillar(ai_pts)
    print(f"      Pillar Geometry (Nominal R=0.20 m at [-6.0, 4.0]):")
    print(f"        GT : Center = ({gt_pillar.fitted_center[0]:.3f}, {gt_pillar.fitted_center[1]:.3f}), R = {gt_pillar.fitted_radius_m:.3f} m (err={gt_pillar.radius_error_m:+.3f} m)")
    print(f"        AI : Center = ({ai_pillar.fitted_center[0]:.3f}, {ai_pillar.fitted_center[1]:.3f}) (err={ai_pillar.center_error_m:.3f} m), R = {ai_pillar.fitted_radius_m:.3f} m (err={ai_pillar.radius_error_m:+.3f} m)")

    # Sky Leakage & Peripheral Expansion
    ai_leakage = GeometricDistortionAnalyzer.analyze_sky_leakage(ai_pts)
    print(f"      Sky Leakage & Perimeter Expansion:")
    print(f"        Total Points                  : {ai_leakage.total_points:,}")
    print(f"        In-Boundary Points (Physical) : {ai_leakage.in_boundary_points:,} ({ai_leakage.in_boundary_pct:.2f}%)")
    print(f"        Sky Hallucination (Z > 2.55 m): {ai_leakage.sky_leakage_points:,} ({ai_leakage.sky_leakage_pct:.2f}%)")
    print(f"        Sub-Ground Noise (Z < -0.15 m): {ai_leakage.sub_ground_points:,} ({ai_leakage.sub_ground_pct:.2f}%)")
    print(f"        Perimeter Expansion (|XY|>9.5): {ai_leakage.perimeter_leakage_points:,} ({ai_leakage.perimeter_leakage_pct:.2f}%)")

    # Cloud-to-Cloud Accuracy
    cmp_stats, ai_dists = GeometricDistortionAnalyzer.compare_point_clouds(ai_pts, gt_pts)
    print(f"      Cloud-to-Cloud Alignment Accuracy:")
    print(f"        All Points  : MAE = {cmp_stats.all_points_mae_m:.4f} m, RMSE = {cmp_stats.all_points_rmse_m:.4f} m, Median = {cmp_stats.all_points_median_m:.4f} m, P95 = {cmp_stats.all_points_p95_m:.4f} m")
    print(f"        In-Boundary : MAE = {cmp_stats.in_boundary_mae_m:.4f} m, RMSE = {cmp_stats.in_boundary_rmse_m:.4f} m, Median = {cmp_stats.in_boundary_median_m:.4f} m, P95 = {cmp_stats.in_boundary_p95_m:.4f} m")

    # Generate 2D Diagnostic Plots
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    # Plot 1: Top-down Bird's-Eye View (X-Y)
    stride_plot = 5
    axes[0].scatter(gt_pts[::stride_plot, 0], gt_pts[::stride_plot, 1], s=0.5, c="gray", alpha=0.3, label="GT Enclosure")
    sc = axes[0].scatter(
        ai_pts[::stride_plot, 0],
        ai_pts[::stride_plot, 1],
        c=np.clip(ai_dists[::stride_plot], 0.0, 1.0),
        cmap="turbo",
        s=0.8,
        alpha=0.6,
        label="Calibrated AI Points"
    )
    plt.colorbar(sc, ax=axes[0], label="Distance to GT (m)")
    axes[0].set_title("CLEAR360 Bird's-Eye View (X-Y Plane)")
    axes[0].set_xlabel("X (Right, metres)")
    axes[0].set_ylabel("Y (Forward, metres)")
    axes[0].set_xlim(-16, 16)
    axes[0].set_ylim(-16, 16)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(loc="upper right")

    # Plot 2: Side Profile (Y-Z Plane) showing Wall Verticality & Sky Leakage
    axes[1].scatter(gt_pts[::stride_plot, 1], gt_pts[::stride_plot, 2], s=0.5, c="gray", alpha=0.3, label="GT Geometry")
    sc2 = axes[1].scatter(
        ai_pts[::stride_plot, 1],
        ai_pts[::stride_plot, 2],
        c=np.clip(ai_dists[::stride_plot], 0.0, 1.0),
        cmap="turbo",
        s=0.8,
        alpha=0.6,
        label="Calibrated AI Points"
    )
    plt.colorbar(sc2, ax=axes[1], label="Distance to GT (m)")
    axes[1].axhline(0.0, color="k", linestyle=":", label="Ground Plane (Z=0)")
    axes[1].axhline(2.5, color="red", linestyle="--", label="Wall Height (Z=2.5 m)")
    axes[1].set_title("CLEAR360 Side Profile (Y-Z Plane) — Verticality & Sky Leakage")
    axes[1].set_xlabel("Y (Forward, metres)")
    axes[1].set_ylabel("Z (Up, metres)")
    axes[1].set_xlim(-16, 16)
    axes[1].set_ylim(-2, 7)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(loc="upper right")

    fig_path = figures_dir / f"m5_fusion_frame_{frame_id:04d}_analysis.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"      Saved diagnostic figure: {fig_path.name}")

    print("\n" + "=" * 78)
    print("M5 SINGLE-FRAME FUSION EVALUATION COMPLETE")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
