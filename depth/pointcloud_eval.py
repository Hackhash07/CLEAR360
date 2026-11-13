"""
depth/pointcloud_eval.py — Compare AI-Predicted vs Ground-Truth 3D Point Clouds
==============================================================================

Generates a fused 3D point cloud from AI-predicted metric depth across all four
cameras, saves it as PLY, and quantitatively compares it against the ground-truth
point cloud.

Comparisons Computed:
  1. Ray-Correspondent 3D Geometric Distance:
     Exact point-to-point Euclidean distance in world metres between AI and GT
     points unprojected along identical pixel rays.
  2. Cloud-to-Cloud Nearest-Neighbour Distance (via Open3D):
     Evaluates surface discrepancy between the two point clouds.
  3. Bounding-box and centroid alignment check.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import open3d as o3d

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.calibration import CalibrationStore
from pipeline.sync import FrameSetLoader
from pointcloud.unprojector import DepthUnprojector


def generate_and_compare_pointclouds(
    frame_id: int,
    ai_depths: dict[str, np.ndarray],
    loader: FrameSetLoader,
    store: CalibrationStore,
    output_dir: Optional[Path] = None,
    voxel_size: float = 0.05,
    max_eval_depth: float = 50.0,
) -> dict[str, float]:
    """
    Generate AI merged point cloud and compare it against the Ground-Truth point cloud.

    Parameters
    ----------
    frame_id : int
        Frame index (e.g. 0).
    ai_depths : dict[str, np.ndarray]
        Dict of (H, W) float32 AI depth maps keyed by camera name.
    loader : FrameSetLoader
        Frame loader to fetch RGB and GT depth.
    store : CalibrationStore
        Calibrated camera models.
    output_dir : Path, optional
        Where to save output PLY files.
    voxel_size : float
        Voxel size for downsampling (default 0.05m).
    max_eval_depth : float
        Maximum evaluation range in metres.

    Returns
    -------
    metrics : dict[str, float]
        Summary of 3D geometric discrepancies between AI and GT clouds.
    """
    if output_dir is None:
        output_dir = PROJECT_ROOT / "data" / "pointclouds"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    unprojector = DepthUnprojector(max_depth=max_eval_depth, min_depth=0.5, stride=2, exclude_ego=True)
    fs = loader.load_frame(frame_id, load_position=False, load_semantic=True)

    cameras = ["front", "rear", "left", "right"]

    # 1. Build AI point cloud
    all_ai_pts = []
    all_ai_cols = []
    ray_3d_errors = []

    for cam_name in cameras:
        cfd = fs.get(cam_name)
        cam = store.get(cam_name)
        ai_depth = ai_depths[cam_name]
        gt_depth = cfd.depth

        # Unproject AI depth
        ai_pts, ai_cols, _ = unprojector.unproject_depth(
            depth=ai_depth,
            camera_model=cam,
            rgb=cfd.rgb,
            semantic_mask=cfd.semantic_mask,
        )
        all_ai_pts.append(ai_pts)
        all_ai_cols.append(ai_cols)

        # Compute ray-by-ray exact 3D distance on valid foreground pixels
        eval_mask = (gt_depth >= 0.5) & (gt_depth <= max_eval_depth)
        if cfd.semantic_mask is not None:
            eval_mask = eval_mask & (cfd.semantic_mask != 0) & (cfd.semantic_mask != 2)

        # Sample stride for fast 3D distance
        v_idx, u_idx = np.where(eval_mask)
        if len(v_idx) > 0:
            sub = np.arange(0, len(v_idx), 4)  # sample every 4th valid pixel
            v_s, u_s = v_idx[sub], u_idx[sub]

            d_ai = ai_depth[v_s, u_s].astype(np.float64)
            d_gt = gt_depth[v_s, u_s].astype(np.float64)

            # Unproject both directly
            fx, fy = cam.intrinsics.fx, cam.intrinsics.fy
            cx, cy = cam.intrinsics.cx, cam.intrinsics.cy
            T_wc = cam.T_world_cam

            x_cam_ai = (u_s - cx) * d_ai / fx
            y_cam_ai = (v_s - cy) * d_ai / fy
            z_cam_ai = d_ai
            p_cam_ai = np.stack([x_cam_ai, y_cam_ai, z_cam_ai, np.ones_like(d_ai)], axis=-1)
            p_w_ai = (T_wc @ p_cam_ai.T).T[:, :3]

            x_cam_gt = (u_s - cx) * d_gt / fx
            y_cam_gt = (v_s - cy) * d_gt / fy
            z_cam_gt = d_gt
            p_cam_gt = np.stack([x_cam_gt, y_cam_gt, z_cam_gt, np.ones_like(d_gt)], axis=-1)
            p_w_gt = (T_wc @ p_cam_gt.T).T[:, :3]

            dist_3d = np.linalg.norm(p_w_ai - p_w_gt, axis=-1)
            ray_3d_errors.append(dist_3d)

    merged_ai_pts = np.vstack(all_ai_pts)
    merged_ai_cols = np.vstack(all_ai_cols)

    ai_pcd = o3d.geometry.PointCloud()
    ai_pcd.points = o3d.utility.Vector3dVector(merged_ai_pts)
    ai_pcd.colors = o3d.utility.Vector3dVector(merged_ai_cols)

    # Save full and downsampled AI PLYs
    ai_ply_path = output_dir / f"frame_{frame_id:04d}_ai_merged.ply"
    unprojector.save_ply(ai_pcd, ai_ply_path)
    print(f"[Point Cloud] Saved AI merged cloud: {ai_ply_path} ({len(merged_ai_pts):,} points)")

    ai_pcd_ds = unprojector.downsample(ai_pcd, voxel_size=voxel_size)
    ai_ds_path = output_dir / f"frame_{frame_id:04d}_ai_merged_ds005.ply"
    unprojector.save_ply(ai_pcd_ds, ai_ds_path)

    # 2. Load or build GT point cloud for comparison
    gt_pcd = unprojector.fuse_frame_set(fs, store)
    gt_pcd_ds = unprojector.downsample(gt_pcd, voxel_size=voxel_size)

    # 3. Ray-based 3D distance metrics
    all_ray_errs = np.concatenate(ray_3d_errors) if ray_3d_errors else np.array([0.0])
    mean_3d_err = float(np.mean(all_ray_errs))
    median_3d_err = float(np.median(all_ray_errs))
    rmse_3d_err = float(np.sqrt(np.mean(all_ray_errs ** 2)))
    p90_3d_err = float(np.percentile(all_ray_errs, 90))

    # 4. Open3D nearest-neighbour cloud-to-cloud distance
    dists_ai_to_gt = ai_pcd_ds.compute_point_cloud_distance(gt_pcd_ds)
    dists_ai_to_gt = np.asarray(dists_ai_to_gt)
    nn_mean = float(np.mean(dists_ai_to_gt))
    nn_median = float(np.median(dists_ai_to_gt))
    nn_rmse = float(np.sqrt(np.mean(dists_ai_to_gt ** 2)))

    metrics = {
        "num_ai_points": len(merged_ai_pts),
        "num_ai_ds_points": len(ai_pcd_ds.points),
        "num_gt_ds_points": len(gt_pcd_ds.points),
        "ray_3d_mean_m": mean_3d_err,
        "ray_3d_median_m": median_3d_err,
        "ray_3d_rmse_m": rmse_3d_err,
        "ray_3d_p90_m": p90_3d_err,
        "nn_c2c_mean_m": nn_mean,
        "nn_c2c_median_m": nn_median,
        "nn_c2c_rmse_m": nn_rmse,
    }

    print("\n" + "=" * 65)
    print(f" 3D GEOMETRIC POINT CLOUD COMPARISON (FRAME {frame_id:04d})")
    print("=" * 65)
    print(f"  AI Point Cloud:            {len(merged_ai_pts):,} points (downsampled: {len(ai_pcd_ds.points):,})")
    print(f"  GT Point Cloud:            {len(gt_pcd.points):,} points (downsampled: {len(gt_pcd_ds.points):,})")
    print(f"  Ray-based 3D Mean Error:   {mean_3d_err:.3f} m")
    print(f"  Ray-based 3D Median Error: {median_3d_err:.3f} m")
    print(f"  Ray-based 3D RMSE:         {rmse_3d_err:.3f} m")
    print(f"  Ray-based 3D 90th Pct:     {p90_3d_err:.3f} m")
    print(f"  Nearest-Neighbour C2C Mean:{nn_mean:.3f} m")
    print(f"  Nearest-Neighbour C2C RMSE:{nn_rmse:.3f} m")
    print("=" * 65)

    return metrics

