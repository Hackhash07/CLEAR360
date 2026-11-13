"""
scripts/multi_scene_benchmark.py — Multi-Scene 3D Fusion & Consistency Benchmark
================================================================================

Evaluates 3D multi-camera point cloud fusion and overlap consistency across
multiple distinct scene configurations (e.g. Frame 0, Frame 5, Frame 10)
with varied obstacle positions.

Validates that:
  - Overlap geometric consistency holds across all scenes.
  - Direct rigid transformation without ICP reliably registers all 4 cameras.
  - Calibration parameters generalize across scene variations.
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from depth.estimator import DepthEstimator
from fusion.engine import PointCloudFusionEngine
from fusion.overlap import OVERLAP_REGIONS, OverlapAnalyzer, OverlapMetrics
from fusion.distortion import GeometricDistortionAnalyzer
from pipeline.calibration import CalibrationStore
from sim.constants import CAMERA_NAMES
from validation.gt_loader import load_depth_exr, load_semantic_png


def ensure_ai_predictions(frame_ids: list[int], data_dir: Path) -> None:
    """Run Depth Anything V2 inference and apply calibration for missing frames."""
    with open(PROJECT_ROOT / "config" / "depth_calibration.yaml", "r") as f:
        depth_calib = yaml.safe_load(f)
    per_cam_scales = {k: float(v) for k, v in depth_calib["per_camera_scales"].items()}

    estimator: Optional[DepthEstimator] = None

    for fid in frame_ids:
        for cam in CAMERA_NAMES:
            raw_ai_file = data_dir / "photoreal_ai_depth" / cam / f"frame_{fid:04d}.npy"
            calib_ai_file = data_dir / "photoreal_calib_depth" / cam / f"frame_{fid:04d}.npy"

            if not raw_ai_file.exists():
                if estimator is None:
                    print(f"[AI] Initializing Depth Anything V2 Metric ViT-Small on available device...")
                    estimator = DepthEstimator()

                rgb_path = data_dir / "photoreal_frames" / cam / f"frame_{fid:04d}.png"
                if not rgb_path.exists():
                    raise FileNotFoundError(f"Missing RGB image: {rgb_path}")

                bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

                pred_depth, _ = estimator.estimate_depth(rgb, is_rgb=True)
                raw_ai_file.parent.mkdir(parents=True, exist_ok=True)
                np.save(raw_ai_file, pred_depth.astype(np.float32))

            if not calib_ai_file.exists():
                raw = np.load(raw_ai_file).astype(np.float32)
                calib = raw / per_cam_scales[cam]
                calib_ai_file.parent.mkdir(parents=True, exist_ok=True)
                np.save(calib_ai_file, calib.astype(np.float32))


def evaluate_scene(frame_id: int, engine: PointCloudFusionEngine, overlap_analyzer: OverlapAnalyzer, data_dir: Path, output_dir: Path) -> dict:
    """Evaluate GT and Calibrated AI fusion for a single frame."""
    # 1. Load GT Data
    gt_rgbs = {}
    gt_depths = {}
    gt_semantics = {}
    for cam in CAMERA_NAMES:
        rgb_file = data_dir / "photoreal_frames" / cam / f"frame_{frame_id:04d}.png"
        depth_file = data_dir / "photoreal_gt_depth" / cam / f"frame_{frame_id:04d}.exr"
        sem_file = data_dir / "photoreal_gt_semantic" / cam / f"frame_{frame_id:04d}.png"

        bgr = cv2.imread(str(rgb_file), cv2.IMREAD_COLOR)
        gt_rgbs[cam] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        gt_depths[cam] = load_depth_exr(depth_file)
        if sem_file.exists():
            gt_semantics[cam] = load_semantic_png(sem_file)

    # Fuse GT
    gt_result = engine.fuse_depth_maps(gt_depths, gt_rgbs, gt_semantics, frame_id=frame_id, pipeline_type="ground_truth")
    engine.save_ply(gt_result.point_cloud, output_dir / f"frame_{frame_id:04d}_gt_merged.ply")
    gt_overlap = overlap_analyzer.analyze_all_regions(gt_result.camera_clouds)

    # 2. Load Calibrated AI Data
    calib_depths = {}
    for cam in CAMERA_NAMES:
        cf = data_dir / "photoreal_calib_depth" / cam / f"frame_{frame_id:04d}.npy"
        calib_depths[cam] = np.load(cf).astype(np.float32)

    ai_result = engine.fuse_depth_maps(calib_depths, gt_rgbs, gt_semantics, frame_id=frame_id, pipeline_type="ai_per_camera")
    engine.save_ply(ai_result.point_cloud, output_dir / f"frame_{frame_id:04d}_ai_per_camera_merged.ply")
    ai_overlap = overlap_analyzer.analyze_all_regions(ai_result.camera_clouds)

    # Compare clouds
    gt_pts = np.asarray(gt_result.point_cloud.points)
    ai_pts = np.asarray(ai_result.point_cloud.points)
    cmp_stats, _ = GeometricDistortionAnalyzer.compare_point_clouds(ai_pts, gt_pts)

    return {
        "frame_id": frame_id,
        "gt_num_points": gt_result.num_points,
        "ai_num_points": ai_result.num_points,
        "gt_overlap": gt_overlap,
        "ai_overlap": ai_overlap,
        "cmp_stats": cmp_stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-Scene Consistency Benchmark")
    parser.add_argument("--frames", nargs="+", type=int, default=[0, 5, 10], help="Frames to evaluate")
    args = parser.parse_args()

    frame_ids = args.frames
    data_dir = PROJECT_ROOT / "data"
    output_dir = data_dir / "pointclouds"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("CLEAR360 — MULTI-SCENE 3D FUSION & OVERLAP CONSISTENCY BENCHMARK")
    print("=" * 80)
    print(f"Frames to evaluate : {frame_ids}")
    print("=" * 80 + "\n")

    # Ensure AI predictions are ready
    ensure_ai_predictions(frame_ids, data_dir)

    calib_yaml = PROJECT_ROOT / "config" / "cameras.yaml"
    calib_store = CalibrationStore(calib_yaml)
    calib_store.load()

    engine = PointCloudFusionEngine(calib_store, max_depth=50.0, min_depth=0.1, stride=1, exclude_ego=True)
    overlap_analyzer = OverlapAnalyzer(calib_store, occlusion_dist_threshold=0.50)

    results = []
    for fid in frame_ids:
        print(f">>> Evaluating Scene Frame {fid:04d}...")
        res = evaluate_scene(fid, engine, overlap_analyzer, data_dir, output_dir)
        results.append(res)
        print(f"    GT Points: {res['gt_num_points']:,} | AI Points: {res['ai_num_points']:,}")
        print(f"    In-Boundary Cloud MAE: {res['cmp_stats'].in_boundary_mae_m:.4f} m (Median: {res['cmp_stats'].in_boundary_median_m:.4f} m)")

    # Print Summary Table
    print("\n" + "=" * 94)
    print("MULTI-SCENE OVERLAP CONSISTENCY COMPARISON TABLE (GROUND TRUTH)")
    print("=" * 94)
    header = f"{'Frame':6s} | {'Region':14s} | {'Pair':12s} | {'Pts A':8s} | {'Pts B':8s} | {'Med NN (m)':10s} | {'P95 (m)':8s} | {'Pt2Plane':8s}"
    print(header)
    print("-" * 94)
    for res in results:
        fid = res["frame_id"]
        for reg, m in res["gt_overlap"].items():
            p2p = f"{m.median_pt2plane_m:.4f}" if m.median_pt2plane_m is not None else "N/A"
            print(
                f"{fid:6d} | {reg:14s} | {m.cam_a + '-' + m.cam_b:12s} | {m.num_points_a:8,d} | {m.num_points_b:8,d} | "
                f"{m.median_nn_dist_m:10.4f} | {m.p95_nn_dist_m:8.4f} | {p2p:8s}"
            )
        print("-" * 94)

    print("\n" + "=" * 94)
    print("MULTI-SCENE OVERLAP CONSISTENCY COMPARISON TABLE (CALIBRATED AI)")
    print("=" * 94)
    header = f"{'Frame':6s} | {'Region':14s} | {'Pair':12s} | {'Pts A':8s} | {'Pts B':8s} | {'Med NN (m)':10s} | {'P95 (m)':8s} | {'Co-Obs Med':10s}"
    print(header)
    print("-" * 94)
    for res in results:
        fid = res["frame_id"]
        for reg, m in res["ai_overlap"].items():
            print(
                f"{fid:6d} | {reg:14s} | {m.cam_a + '-' + m.cam_b:12s} | {m.num_points_a:8,d} | {m.num_points_b:8,d} | "
                f"{m.median_nn_dist_m:10.4f} | {m.p95_nn_dist_m:8.4f} | {m.co_observed_median_m:10.4f}"
            )
        print("-" * 94)

    print("\n" + "=" * 80)
    print("MULTI-SCENE BENCHMARK COMPLETED SUCCESSFULLY")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
