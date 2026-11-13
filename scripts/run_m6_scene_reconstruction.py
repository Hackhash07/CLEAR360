"""
scripts/run_m6_scene_reconstruction.py — Milestone M6 3D Scene Reconstruction & Mapping
========================================================================================

Executes the complete M6 evaluation across:
  - 3 Synchronized Scenes: Frame 0 (baseline), Frame 5 (variant 1), Frame 10 (variant 2)
  - 3 Separate Pipelines:
      A. Ground Truth (GT)
      B. Raw AI
      C. Per-Camera Calibrated AI

Deliverables generated:
  - Clean PLYs: data/pointclouds/frame_XXXX_<pipeline>_clean.ply
  - Semantic PLYs: data/pointclouds/frame_XXXX_<pipeline>_semantic.ply
  - Occupancy Grids: data/occupancy/frame_XXXX_<pipeline>_occupancy.npz
  - Scene Metadata: data/metadata/frame_XXXX_<pipeline>_scene.json
  - Obstacle Bounding Boxes: data/metadata/frame_XXXX_<pipeline>_obstacles.json
  - Diagnostic Visualizations: docs/figures/m6_*.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import yaml

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import open3d as o3d

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fusion.engine import PointCloudFusionEngine
from occupancy.grid3d import OccupancyGrid3D, VoxelState
from occupancy.tsdf import TSDFVolume3D
from obstacles.clustering import BoundingBox3D, Obstacle3D, ObstacleClusterer, VehicleFootprint
from obstacles.tracker import DynamicObstacleTracker
from pipeline.calibration import CalibrationStore
from scene.filter import GroundPlaneResult, SceneFilter
from scene.semantics import (
    SemanticClass,
    classify_ai_scene_semantics,
    create_semantic_point_cloud,
    labels_to_colors,
)
from sim.constants import CAMERA_NAMES
from validation.gt_loader import load_depth_exr, load_semantic_png


def load_frame_data(
    data_dir: Path,
    frame_id: int,
    pipeline_type: str,
    global_scale: float,
    per_cam_scales: dict[str, float],
) -> Tuple[dict[str, np.ndarray], dict[str, np.ndarray], Optional[dict[str, np.ndarray]]]:
    """Load RGB, Depth, and optional GT Semantics for a given frame and pipeline."""
    rgbs = {}
    depths = {}
    semantics = {} if pipeline_type == "ground_truth" else None

    for cam in CAMERA_NAMES:
        rgb_file = data_dir / "photoreal_frames" / cam / f"frame_{frame_id:04d}.png"
        bgr = cv2.imread(str(rgb_file), cv2.IMREAD_COLOR)
        rgbs[cam] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        if pipeline_type == "ground_truth":
            depth_file = data_dir / "photoreal_gt_depth" / cam / f"frame_{frame_id:04d}.exr"
            depths[cam] = load_depth_exr(depth_file)
            sem_file = data_dir / "photoreal_gt_semantic" / cam / f"frame_{frame_id:04d}.png"
            if sem_file.exists():
                semantics[cam] = load_semantic_png(sem_file)
        elif pipeline_type == "ai_raw":
            raw_file = data_dir / "photoreal_ai_depth" / cam / f"frame_{frame_id:04d}.npy"
            depths[cam] = np.load(raw_file).astype(np.float32)
        elif pipeline_type == "ai_per_camera":
            calib_file = data_dir / "photoreal_calib_depth" / cam / f"frame_{frame_id:04d}.npy"
            depths[cam] = np.load(calib_file).astype(np.float32)

    return rgbs, depths, semantics


def process_scene_pipeline(
    frame_id: int,
    pipeline_type: str,
    calib_store: CalibrationStore,
    fusion_engine: PointCloudFusionEngine,
    scene_filter: SceneFilter,
    obstacle_clusterer: ObstacleClusterer,
    data_dir: Path,
    output_dir: Path,
    meta_dir: Path,
    occupancy_dir: Path,
    global_scale: float,
    per_cam_scales: dict[str, float],
) -> dict:
    """Process a single pipeline on a single frame."""
    t0 = time.perf_counter()

    # 1. Load Data
    rgbs, depths, semantics = load_frame_data(
        data_dir, frame_id, pipeline_type, global_scale, per_cam_scales
    )

    # 2. Multi-Camera Rigid Fusion
    is_gt = pipeline_type == "ground_truth"
    fusion_res = fusion_engine.fuse_depth_maps(
        depth_maps=depths,
        rgb_maps=rgbs,
        semantic_masks=semantics if is_gt else None,
        frame_id=frame_id,
        pipeline_type=pipeline_type,
    )
    t_fusion = time.perf_counter() - t0

    # 3. Scene Filtering (Voxel DS, SOR, ROR, Sky Filter, Ground Segmentation)
    t1 = time.perf_counter()
    raw_pcd = fusion_res.point_cloud
    clean_pcd, non_ground_pcd, ground_res, filter_stats = scene_filter.filter_complete_scene(
        pcd=raw_pcd,
        labels=None, # strictly no GT mask leakage into filter
        is_ground_truth=is_gt,
    )
    t_filter = time.perf_counter() - t1

    clean_pts = np.asarray(clean_pcd.points)
    clean_col = np.asarray(clean_pcd.colors)

    # 4. Semantic Labeling
    t2 = time.perf_counter()
    if is_gt:
        # Ground-truth semantic assignment
        # Build KD-tree of original raw GT points (which had semantic labels) to transfer labels
        gt_labels_all = []
        gt_pts_all = []
        for cam_name in CAMERA_NAMES:
            if semantics and cam_name in semantics:
                c_pts, _, c_lab = fusion_engine.unproject_camera(
                    cam_name, depths[cam_name], rgbs[cam_name], semantics[cam_name]
                )
                if len(c_pts) > 0 and c_lab is not None:
                    gt_pts_all.append(c_pts)
                    gt_labels_all.append(c_lab)
        if gt_pts_all:
            tree_gt = o3d.geometry.KDTreeFlann(
                o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.vstack(gt_pts_all)))
            )
            raw_labels_cat = np.concatenate(gt_labels_all)
            semantic_labels = np.zeros(len(clean_pts), dtype=np.int32)
            for i, p in enumerate(clean_pts):
                [_, idx, _] = tree_gt.search_knn_vector_3d(p, 1)
                semantic_labels[i] = raw_labels_cat[idx[0]]
        else:
            semantic_labels = np.full(len(clean_pts), SemanticClass.UNKNOWN, dtype=np.int32)
    else:
        # AI-derived semantic classification (geometric & appearance heuristics; ZERO GT leakage)
        semantic_labels = classify_ai_scene_semantics(
            points=clean_pts,
            ground_inliers=ground_res.inlier_indices,
            colors=clean_col,
        )
    t_semantic = time.perf_counter() - t2

    # Save clean PLY and semantic PLY
    clean_ply_path = output_dir / f"frame_{frame_id:04d}_{pipeline_type}_clean.ply"
    fusion_engine.save_ply(clean_pcd, clean_ply_path)

    semantic_pcd = create_semantic_point_cloud(clean_pts, semantic_labels)
    semantic_ply_path = output_dir / f"frame_{frame_id:04d}_{pipeline_type}_semantic.ply"
    fusion_engine.save_ply(semantic_pcd, semantic_ply_path)

    # 5. 3D Occupancy Mapping with Ray-Based Free-Space Carving
    t3 = time.perf_counter()
    occ_grid = OccupancyGrid3D(
        bounds_x=(-12.0, 12.0),
        bounds_y=(-12.0, 12.0),
        bounds_z=(-0.5, 3.5),
        resolution=0.05,
    )

    # Camera origins in vehicle frame
    cam_origins = {
        "front": np.array([0.0, 2.15, 0.90]),
        "rear":  np.array([0.0, -2.15, 0.90]),
        "left":  np.array([-0.95, 0.0, 1.25]),
        "right": np.array([+0.95, 0.0, 1.25]),
    }

    # Carve rays for each camera view using unprojected endpoints
    for cam_name, cam_cloud in fusion_res.camera_clouds.items():
        c_pts = np.asarray(cam_cloud.points)
        if len(c_pts) > 0 and cam_name in cam_origins:
            occ_grid.integrate_camera_view(
                camera_origin=cam_origins[cam_name],
                endpoints=c_pts,
                free_margin_m=0.05,
                max_range_m=25.0,
                subsample_stride=2,
            )
    t_occ = time.perf_counter() - t3

    # Save occupancy grid
    occ_npz_path = occupancy_dir / f"frame_{frame_id:04d}_{pipeline_type}_occupancy.npz"
    occ_grid.save_npz(occ_npz_path)
    occ_stats = occ_grid.get_stats()

    # 6. TSDF Volume Surface Reconstruction
    t4 = time.perf_counter()
    tsdf = TSDFVolume3D(
        bounds_x=(-10.0, 10.0),
        bounds_y=(-10.0, 10.0),
        bounds_z=(-0.2, 2.8),
        voxel_size=0.05,
        truncation_margin=0.15,
    )
    for cam_name in CAMERA_NAMES:
        cam_model = calib_store.get(cam_name)
        tsdf.integrate_view(depths[cam_name], cam_model, weight=1.0, max_depth=20.0)
    tsdf_surface_pts = tsdf.extract_zero_crossing_points()
    t_tsdf = time.perf_counter() - t4

    # 7. Obstacle Clustering & Footprint Clearance
    t5 = time.perf_counter()
    non_ground_pts = np.asarray(non_ground_pcd.points)
    non_ground_labels = semantic_labels[ground_res.outlier_indices] if len(semantic_labels) == len(clean_pts) else None

    obstacles = obstacle_clusterer.cluster_obstacles(
        points=non_ground_pts,
        labels=non_ground_labels,
    )
    t_cluster = time.perf_counter() - t5
    t_total = time.perf_counter() - t0

    # Save Obstacle Bounding Box JSON
    obs_json_path = meta_dir / f"frame_{frame_id:04d}_{pipeline_type}_obstacles.json"
    with open(obs_json_path, "w") as f:
        json.dump([o.to_dict() for o in obstacles], f, indent=2)

    # Save Scene Metadata JSON
    scene_meta = {
        "frame_id": frame_id,
        "pipeline_type": pipeline_type,
        "total_processing_time_s": t_total,
        "timing_breakdown_s": {
            "fusion": t_fusion,
            "filter": t_filter,
            "semantic": t_semantic,
            "occupancy": t_occ,
            "tsdf": t_tsdf,
            "clustering": t_cluster,
        },
        "points": {
            "initial_raw": filter_stats.initial_points,
            "after_filter": filter_stats.final_points,
            "ground_points": filter_stats.ground_points,
            "non_ground_points": filter_stats.non_ground_points,
        },
        "ground_plane": {
            "elevation_z_m": ground_res.elevation_z,
            "tilt_deg": ground_res.tilt_deg,
            "rmse_m": ground_res.rmse_m,
            "normal": ground_res.normal.tolist(),
        },
        "occupancy_grid": occ_stats,
        "tsdf": {
            "surface_points_count": len(tsdf_surface_pts),
            "voxel_size_m": tsdf.voxel_size,
        },
        "obstacles_detected": len(obstacles),
        "closest_obstacle_clearance_m": obstacles[0].clearance_to_vehicle_m if obstacles else float("inf"),
    }
    scene_json_path = meta_dir / f"frame_{frame_id:04d}_{pipeline_type}_scene.json"
    with open(scene_json_path, "w") as f:
        json.dump(scene_meta, f, indent=2)

    return {
        "frame_id": frame_id,
        "pipeline_type": pipeline_type,
        "clean_points": len(clean_pts),
        "ground_res": ground_res,
        "filter_stats": filter_stats,
        "occ_stats": occ_stats,
        "obstacles": obstacles,
        "timing": scene_meta["timing_breakdown_s"],
        "total_time_s": t_total,
        "clean_pts": clean_pts,
        "semantic_labels": semantic_labels,
    }


def generate_diagnostic_visualizations(
    frame_id: int,
    res_gt: dict,
    res_ai: dict,
    figures_dir: Path,
) -> None:
    """Generate 2D bird's-eye view diagnostic plots of occupancy, semantics, and obstacles."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    footprint = VehicleFootprint()

    for idx, (res, title) in enumerate([(res_gt, "Ground Truth (Blender Cycles)"), (res_ai, "Calibrated AI (Depth Anything V2)")]):
        ax = axes[idx]
        pts = res["clean_pts"]
        labels = res["semantic_labels"]
        obstacles = res["obstacles"]

        # Downsampled scatter for background
        stride = 6
        pts_sub = pts[::stride]
        labs_sub = labels[::stride]
        cols_sub = labels_to_colors(labs_sub)

        ax.scatter(pts_sub[:, 0], pts_sub[:, 1], c=cols_sub, s=1.0, alpha=0.5)

        # Draw Hyundai Creta Vehicle Footprint
        body_w = footprint.body_max[0] - footprint.body_min[0]
        body_l = footprint.body_max[1] - footprint.body_min[1]
        body_rect = patches.Rectangle(
            (footprint.body_min[0], footprint.body_min[1]),
            body_w, body_l,
            linewidth=2.0, edgecolor="black", facecolor=(0.8, 0.8, 0.8, 0.8),
            label="Ego Vehicle (Body)"
        )
        ax.add_patch(body_rect)

        # Draw Side Mirrors
        l_mirr = patches.Rectangle(
            (footprint.left_mirror_min[0], footprint.left_mirror_min[1]),
            footprint.left_mirror_max[0] - footprint.left_mirror_min[0],
            footprint.left_mirror_max[1] - footprint.left_mirror_min[1],
            linewidth=1.5, edgecolor="red", facecolor="silver", label="Side Mirrors"
        )
        r_mirr = patches.Rectangle(
            (footprint.right_mirror_min[0], footprint.right_mirror_min[1]),
            footprint.right_mirror_max[0] - footprint.right_mirror_min[0],
            footprint.right_mirror_max[1] - footprint.right_mirror_min[1],
            linewidth=1.5, edgecolor="red", facecolor="silver"
        )
        ax.add_patch(l_mirr)
        ax.add_patch(r_mirr)

        # Draw Obstacle Bounding Boxes & Clearance Vectors
        for obs in obstacles[:8]:  # Draw top 8 obstacles
            b = obs.bbox
            bw = b.max_point[0] - b.min_point[0]
            bl = b.max_point[1] - b.min_point[1]
            box_rect = patches.Rectangle(
                (b.min_point[0], b.min_point[1]),
                bw, bl,
                linewidth=1.8, edgecolor="blue" if not obs.is_dynamic else "orange",
                facecolor="none", linestyle="--"
            )
            ax.add_patch(box_rect)

            # Draw label with clearance
            ax.text(
                obs.centroid[0], obs.centroid[1],
                f"{obs.semantic_name[:8]}\n{obs.clearance_to_vehicle_m:.2f}m",
                fontsize=7, color="black", weight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7)
            )

        ax.set_title(f"M6 3D Scene Reconstruction — {title} (Frame {frame_id:04d})")
        ax.set_xlabel("X (Right, metres)")
        ax.set_ylabel("Y (Forward, metres)")
        ax.set_xlim(-12, 12)
        ax.set_ylim(-12, 12)
        ax.grid(True, linestyle=":", alpha=0.6)
        if idx == 0:
            ax.legend(loc="upper right")

    fig_path = figures_dir / f"m6_scene_reconstruction_frame_{frame_id:04d}.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"      Saved diagnostic figure: {fig_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="CLEAR360 M6 Scene Reconstruction & Mapping")
    parser.add_argument("--frames", nargs="+", type=int, default=[0, 5, 10], help="Frames to evaluate")
    args = parser.parse_args()

    frame_ids = args.frames
    data_dir = PROJECT_ROOT / "data"
    output_dir = data_dir / "pointclouds"
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = data_dir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    occupancy_dir = data_dir / "occupancy"
    occupancy_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = PROJECT_ROOT / "docs" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 84)
    print("CLEAR360 — MILESTONE M6: 3D SCENE RECONSTRUCTION, FILTERING & MAPPING")
    print("=" * 84)
    print(f"Target Frames : {frame_ids}")
    print(f"Pipelines     : Ground Truth (GT), Raw AI, Per-Camera Calibrated AI")
    print("=" * 84 + "\n")

    # Load Calibrations
    calib_yaml = PROJECT_ROOT / "config" / "cameras.yaml"
    calib_store = CalibrationStore(calib_yaml)
    calib_store.load()

    with open(PROJECT_ROOT / "config" / "depth_calibration.yaml", "r") as f:
        depth_calib = yaml.safe_load(f)
    global_scale = float(depth_calib["global_scale"])
    per_cam_scales = {k: float(v) for k, v in depth_calib["per_camera_scales"].items()}

    # Initialize Modules
    fusion_engine = PointCloudFusionEngine(calib_store, max_depth=50.0, min_depth=0.1, stride=1, exclude_ego=True)
    scene_filter = SceneFilter(voxel_size=0.05, sor_nb_neighbors=20, sor_std_ratio=2.0, ror_nb_points=12, ror_radius=0.15)
    obstacle_clusterer = ObstacleClusterer(eps=0.30, min_samples=25, min_cluster_points=30)
    tracker = DynamicObstacleTracker(max_association_distance_m=2.0)

    all_results = {}

    for fid in frame_ids:
        print(f"\n" + "-" * 70)
        print(f">>> PROCESSING SCENE FRAME {fid:04d}...")
        print("-" * 70)
        all_results[fid] = {}

        for pipe in ["ground_truth", "ai_raw", "ai_per_camera"]:
            print(f"  [{pipe.upper()}] Running reconstruction & mapping...")
            res = process_scene_pipeline(
                frame_id=fid,
                pipeline_type=pipe,
                calib_store=calib_store,
                fusion_engine=fusion_engine,
                scene_filter=scene_filter,
                obstacle_clusterer=obstacle_clusterer,
                data_dir=data_dir,
                output_dir=output_dir,
                meta_dir=meta_dir,
                occupancy_dir=occupancy_dir,
                global_scale=global_scale,
                per_cam_scales=per_cam_scales,
            )
            all_results[fid][pipe] = res

            print(
                f"      Filtered Points: {res['clean_points']:,} | Ground RMSE: {res['ground_res'].rmse_m*100:.1f} cm | "
                f"Tilt: {res['ground_res'].tilt_deg:.2f}° | Occupied Voxels: {res['occ_stats']['occupied_voxels']:,} | "
                f"Free Voxels: {res['occ_stats']['free_voxels']:,} | Obstacles: {len(res['obstacles'])} | "
                f"Time: {res['total_time_s']:.2f} s"
            )

        # Update dynamic tracker
        tracks = tracker.update(all_results[fid]["ai_per_camera"]["obstacles"], timestamp_s=float(fid) * 0.1)
        print(f"      Active Dynamic Tracks: {len(tracks)}")
        for tr in tracks:
            print(f"        Track {tr.track_id}: Centroid ({tr.latest_centroid[0]:.2f}, {tr.latest_centroid[1]:.2f}, {tr.latest_centroid[2]:.2f}) m | Speed: {tr.speed_mps:.2f} m/s")

        # Generate 2D diagnostic figure comparing GT vs Calibrated AI
        generate_diagnostic_visualizations(fid, all_results[fid]["ground_truth"], all_results[fid]["ai_per_camera"], figures_dir)

    print("\n" + "=" * 84)
    print("M6 MULTI-FRAME SCENE RECONSTRUCTION COMPLETE")
    print("=" * 84 + "\n")


if __name__ == "__main__":
    main()

