"""
photo_reconstruction/dense_reconstruction.py — Multi-View Dense 3D Point Cloud Engine
======================================================================================

Generates dense, filtered 3D point clouds from registered camera poses and photographs.
Applies Open3D voxel grid downsampling, statistical outlier removal, and normal estimation.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import cv2
import numpy as np
import open3d as o3d


class DenseReconstructor:
    """Produces filtered dense 3D point clouds from registered camera images."""

    def __init__(self, voxel_size: float = 0.04, nb_neighbors: int = 20, std_ratio: float = 2.0):
        self.voxel_size = float(voxel_size)
        self.nb_neighbors = int(nb_neighbors)
        self.std_ratio = float(std_ratio)

    def reconstruct_dense(
        self,
        image_dir: Path,
        camera_poses: Dict[str, np.ndarray],
        depth_maps: Optional[Dict[str, np.ndarray]] = None,
        sparse_points: Optional[np.ndarray] = None,
        sparse_colors: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Produce a dense 3D point cloud with genuine photographic colors.

        Parameters
        ----------
        image_dir : Path
            Directory containing source photographs.
        camera_poses : Dict[str, np.ndarray]
            Map of filename -> 4x4 T_world_cam pose matrix.
        depth_maps : Optional[Dict[str, np.ndarray]]
            Optional pre-computed or AI-predicted dense depth maps.
        sparse_points : Optional[np.ndarray]
            Sparse SfM 3D points to include.
        sparse_colors : Optional[np.ndarray]
            Sparse point colors.

        Returns
        -------
        points : np.ndarray (N, 3) float32
        colors : np.ndarray (N, 3) float32 in [0, 1]
        normals : np.ndarray (N, 3) float32
        """
        all_pts: List[np.ndarray] = []
        all_cols: List[np.ndarray] = []

        # Include sparse SfM points if provided
        if sparse_points is not None and len(sparse_points) > 0:
            all_pts.append(sparse_points)
            if sparse_colors is not None and len(sparse_colors) == len(sparse_points):
                all_cols.append(sparse_colors)
            else:
                all_cols.append(np.ones((len(sparse_points), 3), dtype=np.float32) * 0.7)

        # If dense depth maps are supplied (e.g. from Depth Anything V2 or stereo)
        if depth_maps:
            for img_name, T_world_cam in camera_poses.items():
                if img_name not in depth_maps:
                    continue

                img_path = image_dir / img_name
                if not img_path.exists():
                    continue

                img_bgr = cv2.imread(str(img_path))
                if img_bgr is None:
                    continue
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                depth = depth_maps[img_name]

                H, W = depth.shape
                f = 1.20 * max(W, H)
                cx = W / 2.0
                cy = H / 2.0

                s = 6  # High-density stride for detailed point cloud generation
                v_grid, u_grid = np.mgrid[0:H:s, 0:W:s]
                d = depth[v_grid, u_grid]

                valid = (d > 0.2) & (d < 30.0)
                if not np.any(valid):
                    continue

                u_val = u_grid[valid]
                v_val = v_grid[valid]
                d_val = d[valid]

                x_c = (u_val - cx) * d_val / f
                y_c = (v_val - cy) * d_val / f
                z_c = d_val

                pts_cam = np.column_stack([x_c, y_c, z_c, np.ones_like(z_c)])
                pts_world = (T_world_cam @ pts_cam.T).T[:, :3]

                cols = img_rgb[v_val, u_val].astype(np.float32) / 255.0

                all_pts.append(pts_world.astype(np.float32))
                all_cols.append(cols.astype(np.float32))

        if not all_pts:
            return (
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.float32),
            )

        merged_pts = np.vstack(all_pts)
        merged_cols = np.vstack(all_cols)

        # Build Open3D point cloud for cleaning and normal estimation
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(merged_pts.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(merged_cols.astype(np.float64))

        # 1. Voxel Grid Downsampling
        if self.voxel_size > 0.005 and len(pcd.points) > 100:
            pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)

        # 2. Statistical Outlier Removal
        if len(pcd.points) > self.nb_neighbors:
            pcd, _ = pcd.remove_statistical_outlier(
                nb_neighbors=self.nb_neighbors,
                std_ratio=self.std_ratio,
            )

        # 3. Normal Estimation
        if len(pcd.points) >= 3:
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=self.voxel_size * 2.5, max_nn=30
                )
            )
            pcd.orient_normals_consistent_tangent_plane(k=15)
            normals = np.asarray(pcd.normals, dtype=np.float32)
        else:
            normals = np.zeros((len(pcd.points), 3), dtype=np.float32)

        pts_clean = np.asarray(pcd.points, dtype=np.float32)
        cols_clean = np.asarray(pcd.colors, dtype=np.float32)

        return pts_clean, cols_clean, normals

    @staticmethod
    def save_point_cloud_ply(
        points: np.ndarray,
        colors: np.ndarray,
        out_path: Path,
        normals: Optional[np.ndarray] = None,
    ) -> Path:
        """Save point cloud to standard PLY format."""
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
        if normals is not None and len(normals) == len(points):
            pcd.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))

        o3d.io.write_point_cloud(str(out), pcd)
        return out

