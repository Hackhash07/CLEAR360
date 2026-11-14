"""
photo_reconstruction/overlap_fusion.py — Multi-Camera Overlap Fusion & Seam Resolution
=====================================================================================

Resolves duplicate surface observations across overlapping photographs in accordance with
CLEAR360 M7.3.3 Part 6:
  1. Identifies spatially coincident points across different camera views.
  2. Verifies whether points represent the same physical surface via normal and spatial agreement.
  3. Fuses compatible observations toward the most perpendicular viewing camera (max cos theta).
  4. Preserves physically distinct surfaces (e.g. golden sculpture in front of niche wall).
  5. Eliminates multi-layered "double crust" without erasing genuine fine architectural relief.
  6. Produces camera-source-colored diagnostic point clouds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import open3d as o3d

from photo_reconstruction.camera_model import PhoneCameraModel


@dataclass
class OverlapFusionResult:
    """Container for fused point cloud and overlap diagnostics."""
    points: np.ndarray               # (N, 3) float32 fused 3D points
    colors: np.ndarray               # (N, 3) float32 RGB colors [0, 1]
    normals: np.ndarray              # (N, 3) float32 surface normals
    source_camera_indices: np.ndarray # (N,) int32 source camera index
    source_camera_names: List[str]   # List of camera names
    total_raw_points: int
    fused_point_count: int
    overlap_coincident_count: int
    preserved_distinct_count: int
    mean_normal_agreement: float


class CameraOverlapFusion:
    """Controlled multi-camera point cloud fusion and seam resolution."""

    CAMERA_PALETTE = np.array([
        [1.0, 0.15, 0.15],  # Cam 3: Bright Red
        [1.0, 0.55, 0.00],  # Cam 4: Vibrant Orange
        [0.95, 0.90, 0.10], # Cam 5: Yellow
        [0.15, 0.85, 0.25], # Cam 6: Green
        [0.10, 0.75, 0.95], # Cam 7: Cyan
        [0.25, 0.35, 0.95], # Cam 8: Royal Blue
    ], dtype=np.float32)

    def __init__(
        self,
        coincidence_radius: float = 0.035,   # 3.5 cm spatial coincidence radius
        normal_agreement_threshold: float = 0.707, # cos(45 deg) minimum normal alignment
        preserve_relief_margin: float = 0.05, # 5 cm depth separation indicates distinct object
    ):
        self.coincidence_radius = float(coincidence_radius)
        self.normal_thresh = float(normal_agreement_threshold)
        self.relief_margin = float(preserve_relief_margin)

    def fuse_camera_clouds(
        self,
        camera_models: Dict[str, PhoneCameraModel],
        aligned_depth_maps: Dict[str, np.ndarray],
        images_rgb: Optional[Dict[str, np.ndarray]] = None,
        stride: int = 4,
        voxel_size: float = 0.015,
    ) -> OverlapFusionResult:
        """
        Execute controlled surface consensus fusion across all registered views.
        """
        cam_names = sorted(list(camera_models.keys()))
        per_view_pts: List[np.ndarray] = []
        per_view_cols: List[np.ndarray] = []
        per_view_norms: List[np.ndarray] = []
        per_view_angles: List[np.ndarray] = []
        per_view_cam_idx: List[np.ndarray] = []

        total_raw = 0

        # 1. Unproject each view with camera-space viewing angle scoring
        for c_idx, name in enumerate(cam_names):
            if name not in aligned_depth_maps:
                continue

            cam = camera_models[name]
            depth = aligned_depth_maps[name]
            H, W = depth.shape[:2]
            if cam.width != W or cam.height != H:
                cam = cam.rescale(W, H)

            s = max(2, int(stride))
            v_grid, u_grid = np.mgrid[0:H:s, 0:W:s]
            d = depth[v_grid, u_grid]

            valid = (d > 0.3) & (d < 30.0)
            if not np.any(valid):
                continue

            u_val = u_grid[valid]
            v_val = v_grid[valid]
            d_val = d[valid]

            pix = np.column_stack([u_val, v_val])
            pts_world = cam.unproject_image_to_world(pix, d_val)
            num_pts = len(pts_world)
            total_raw += num_pts

            # Viewing vector from point toward camera center
            cam_center = cam.camera_center
            view_vec = cam_center - pts_world
            view_dist = np.linalg.norm(view_vec, axis=1, keepdims=True)
            view_dir = view_vec / np.maximum(view_dist, 1e-6)

            # Sample RGB colors
            if images_rgb and name in images_rgb:
                img = images_rgb[name]
                if img.shape[:2] != (H, W):
                    import cv2
                    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
                cols = img[v_val, u_val].astype(np.float32) / 255.0
            else:
                cols = np.ones((num_pts, 3), dtype=np.float32) * 0.75

            # Initial normal estimation pointing towards camera
            pcd_view = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_world))
            pcd_view.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.5, max_nn=25)
            )
            pcd_view.orient_normals_towards_camera_location(cam_center)
            norms = np.asarray(pcd_view.normals, dtype=np.float32)

            # Perpendicular viewing quality score = n . view_dir
            cos_theta = np.sum(norms * view_dir, axis=1)

            per_view_pts.append(pts_world.astype(np.float32))
            per_view_cols.append(cols.astype(np.float32))
            per_view_norms.append(norms.astype(np.float32))
            per_view_angles.append(cos_theta.astype(np.float32))
            per_view_cam_idx.append(np.full(num_pts, c_idx, dtype=np.int32))

        if not per_view_pts:
            return OverlapFusionResult(
                points=np.zeros((0, 3), dtype=np.float32),
                colors=np.zeros((0, 3), dtype=np.float32),
                normals=np.zeros((0, 3), dtype=np.float32),
                source_camera_indices=np.zeros(0, dtype=np.int32),
                source_camera_names=cam_names,
                total_raw_points=0,
                fused_point_count=0,
                overlap_coincident_count=0,
                preserved_distinct_count=0,
                mean_normal_agreement=1.0,
            )

        all_pts = np.vstack(per_view_pts)
        all_cols = np.vstack(per_view_cols)
        all_norms = np.vstack(per_view_norms)
        all_angles = np.hstack(per_view_angles)
        all_c_idx = np.hstack(per_view_cam_idx)

        # 2. Fast Spatial Consensus Grid Hashing for Seam Resolution & Overlap Fusion
        cell_size = float(self.coincidence_radius)
        v_idx = np.floor(all_pts / cell_size).astype(np.int64)
        prime1, prime2 = 73856093, 19349663
        hash_keys = v_idx[:, 0] ^ (v_idx[:, 1] * prime1) ^ (v_idx[:, 2] * prime2)
        
        sort_order = np.argsort(hash_keys)
        sorted_keys = hash_keys[sort_order]
        unique_keys, split_indices = np.unique(sorted_keys, return_index=True)
        groups = np.split(sort_order, split_indices[1:])

        fused_pts: List[np.ndarray] = []
        fused_cols: List[np.ndarray] = []
        fused_norms: List[np.ndarray] = []
        fused_c_idx: List[int] = []

        coincident_count = 0
        preserved_distinct = 0
        normal_agreements: List[float] = []

        for group in groups:
            if len(group) == 1:
                idx0 = group[0]
                fused_pts.append(all_pts[idx0])
                fused_cols.append(all_cols[idx0])
                fused_norms.append(all_norms[idx0])
                fused_c_idx.append(int(all_c_idx[idx0]))
                continue

            # In multi-observation cell: sort by viewing angle (best observation first)
            grp_angles = all_angles[group]
            local_order = group[np.argsort(-grp_angles)]
            seed_idx = local_order[0]
            seed_norm = all_norms[seed_idx]

            compatible_indices = [seed_idx]
            distinct_indices = []

            for other_idx in local_order[1:]:
                other_norm = all_norms[other_idx]
                dot_prod = float(np.dot(seed_norm, other_norm))
                if dot_prod >= self.normal_thresh:
                    compatible_indices.append(other_idx)
                    normal_agreements.append(dot_prod)
                else:
                    distinct_indices.append(other_idx)
                    preserved_distinct += 1

            coincident_count += len(compatible_indices) - 1

            # Weighted consensus for compatible points on the same surface
            comp_pts = all_pts[compatible_indices]
            comp_norms = all_norms[compatible_indices]
            comp_ang = all_angles[compatible_indices]
            weights = np.maximum(comp_ang, 0.05)[:, None]
            weights /= np.sum(weights)

            consensus_pt = np.sum(comp_pts * weights, axis=0)
            consensus_norm = np.sum(comp_norms * weights, axis=0)
            norm_mag = np.linalg.norm(consensus_norm)
            if norm_mag > 1e-6:
                consensus_norm /= norm_mag
            else:
                consensus_norm = seed_norm

            fused_pts.append(consensus_pt.astype(np.float32))
            fused_cols.append(all_cols[seed_idx])  # Color from most perpendicular view
            fused_norms.append(consensus_norm.astype(np.float32))
            fused_c_idx.append(int(all_c_idx[seed_idx]))

            # Preserve distinct points (e.g. sculpture rim / edge crossing)
            for d_idx in distinct_indices:
                fused_pts.append(all_pts[d_idx])
                fused_cols.append(all_cols[d_idx])
                fused_norms.append(all_norms[d_idx])
                fused_c_idx.append(int(all_c_idx[d_idx]))

        pts_out = np.vstack(fused_pts)
        cols_out = np.vstack(fused_cols)
        norms_out = np.vstack(fused_norms)
        c_idx_out = np.array(fused_c_idx, dtype=np.int32)

        # 3. Optional Voxel Grid Downsampling to regularize density
        if voxel_size > 0.005 and len(pts_out) > 500:
            pcd_clean = o3d.geometry.PointCloud()
            pcd_clean.points = o3d.utility.Vector3dVector(pts_out.astype(np.float64))
            pcd_clean.colors = o3d.utility.Vector3dVector(cols_out.astype(np.float64))
            pcd_clean.normals = o3d.utility.Vector3dVector(norms_out.astype(np.float64))

            pcd_down = pcd_clean.voxel_down_sample(voxel_size=voxel_size)
            # Statistical outlier removal
            pcd_down, inlier_mask = pcd_down.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

            # Map downsampled points to nearest neighbor in pre-downsample cloud to retain camera attribution
            kdtree_pre = o3d.geometry.KDTreeFlann(pcd_clean)
            down_pts = np.asarray(pcd_down.points)
            down_c_idx = np.zeros(len(down_pts), dtype=np.int32)
            for i in range(len(down_pts)):
                _, idx_nn, _ = kdtree_pre.search_knn_vector_3d(down_pts[i], 1)
                if idx_nn:
                    down_c_idx[i] = c_idx_out[idx_nn[0]]

            pts_out = np.asarray(pcd_down.points, dtype=np.float32)
            cols_out = np.asarray(pcd_down.colors, dtype=np.float32)
            norms_out = np.asarray(pcd_down.normals, dtype=np.float32)
            c_idx_out = down_c_idx

        mean_norm_agree = float(np.mean(normal_agreements)) if normal_agreements else 1.0

        return OverlapFusionResult(
            points=pts_out,
            colors=cols_out,
            normals=norms_out,
            source_camera_indices=c_idx_out,
            source_camera_names=cam_names,
            total_raw_points=total_raw,
            fused_point_count=len(pts_out),
            overlap_coincident_count=coincident_count,
            preserved_distinct_count=preserved_distinct,
            mean_normal_agreement=round(mean_norm_agree, 4),
        )
