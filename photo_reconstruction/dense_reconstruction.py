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

    @staticmethod
    def detect_sky_mask(img_bgr: np.ndarray) -> np.ndarray:
        """
        Detect open sky and cloud regions connected to the top of an outdoor photograph
        to prevent unprojecting sky/clouds as solid 3D geometry.
        """
        H, W = img_bgr.shape[:2]
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        H_c, S_c, V_c = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        # Clear blue sky (Hue 85-135 in OpenCV 0-180 scale)
        is_blue_sky = (H_c >= 85) & (H_c <= 135) & (S_c >= 20) & (V_c >= 70)
        # Overcast / white cloud sky (high brightness, low saturation)
        is_cloud_sky = (V_c >= 170) & (S_c <= 50)
        candidate = is_blue_sky | is_cloud_sky

        # Fast flood fill from top border only
        scale = max(1, max(H, W) // 300)
        small = cv2.resize(candidate.astype(np.uint8), (W // scale, H // scale), interpolation=cv2.INTER_NEAREST)
        h_s, w_s = small.shape
        ff_mask = np.zeros((h_s + 2, w_s + 2), dtype=np.uint8)
        filled = small.copy() * 255

        for x in range(0, w_s, max(1, w_s // 30)):
            if filled[0, x] == 255:
                cv2.floodFill(filled, ff_mask, (x, 0), 128)

        sky_small = (filled == 128)
        # If less than 2% of image connects to sky, treat as indoor/no-sky
        if np.mean(sky_small) < 0.02:
            return np.zeros((H, W), dtype=bool)

        return cv2.resize(sky_small.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0

    @staticmethod
    def align_ground_upright(
        points: np.ndarray,
        normals: np.ndarray,
        camera_poses: Dict[str, np.ndarray],
        sparse_points: Optional[np.ndarray] = None,
        distance_threshold: float = 0.25,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], Optional[np.ndarray], np.ndarray]:
        """
        Auto-align scene orientation so that ground is horizontal and scene stays upright:
        1. Derives true physical world UP vector from camera orientations (camera -Y in world).
        2. Detects dominant plane using RANSAC.
        3. Tests if detected plane is a horizontal ground plane (|dot(n, up)| >= 0.707) or a vertical wall.
        4. If ground plane: rotates ground normal to +Z [0, 0, 1] and sets ground elevation to Z = 0.
        5. If vertical wall or no plane: rotates camera UP vector to +Z [0, 0, 1] so walls stay vertical.
        """
        if len(points) < 50:
            return points, normals, camera_poses, sparse_points, np.eye(4)

        # 1. Determine authentic physical world UP vector from camera orientations
        # In camera frame, +X is right, +Y is down, +Z is forward.
        # Thus camera UP in camera coordinates is [0, -1, 0].
        # In world coordinates: cam_up = pose[:3, :3] @ [0, -1, 0] = -pose[:3, 1].
        avg_up = None
        if camera_poses and len(camera_poses) > 0:
            cam_ups = np.array([-p[:3, 1] for p in camera_poses.values()], dtype=np.float64)
            avg_up = np.mean(cam_ups, axis=0)
            norm_up = np.linalg.norm(avg_up)
            if norm_up > 1e-6:
                avg_up = avg_up / norm_up
            else:
                avg_up = None
        if avg_up is None:
            # In COLMAP convention, -Y is up
            avg_up = np.array([0.0, -1.0, 0.0], dtype=np.float64)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))

        try:
            plane_model, inliers = pcd.segment_plane(
                distance_threshold=distance_threshold,
                ransac_n=3,
                num_iterations=1000,
            )
        except Exception:
            inliers = []

        is_ground_plane = False
        plane_normal = None
        z_offset = None

        # 2. Check if segmented plane is a horizontal ground plane
        if len(inliers) >= max(30, int(len(points) * 0.05)):
            a, b, c, d = plane_model
            n = np.array([a, b, c], dtype=np.float64)
            n_norm = np.linalg.norm(n)
            if n_norm > 1e-6:
                n = n / n_norm
                dot_up = float(np.dot(n, avg_up))
                # Only treat as ground if normal is roughly parallel to camera UP (within 45 degrees)
                if abs(dot_up) >= 0.707:
                    is_ground_plane = True
                    if dot_up < 0:
                        n = -n
                        d = -d
                    plane_normal = n
                    z_offset = -d

        if is_ground_plane and plane_normal is not None:
            source_vec = plane_normal
        else:
            # If plane is a vertical wall (e.g. courtyard wall behind fountain), DO NOT rotate wall to floor!
            # Instead, directly align the camera UP vector with +Z [0, 0, 1] to keep walls vertical.
            source_vec = avg_up

        target = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        v = np.cross(source_vec, target)
        s = np.linalg.norm(v)
        cos_val = float(np.dot(source_vec, target))

        if s > 1e-6:
            vx = np.array([
                [0, -v[2], v[1]],
                [v[2], 0, -v[0]],
                [-v[1], v[0], 0]
            ])
            R_align = np.eye(3) + vx + (vx @ vx) * ((1.0 - cos_val) / (s ** 2))
        else:
            R_align = np.eye(3) if cos_val > 0 else -np.eye(3)

        pts_rot = (R_align @ points.T).T

        if is_ground_plane and z_offset is not None:
            ground_z = z_offset
        else:
            # Place lowest 1st percentile of points at Z = 0
            ground_z = float(np.percentile(pts_rot[:, 2], 1.0))

        pts_rot[:, 2] -= ground_z
        norms_rot = (R_align @ normals.T).T if normals is not None and len(normals) == len(points) else normals
        sparse_rot = None
        if sparse_points is not None and len(sparse_points) > 0:
            sparse_rot = (R_align @ sparse_points.T).T
            sparse_rot[:, 2] -= ground_z

        T_align = np.eye(4, dtype=np.float64)
        T_align[:3, :3] = R_align
        T_align[:3, 3] = [0.0, 0.0, -ground_z]

        new_poses = {}
        for name, pose in camera_poses.items():
            new_poses[name] = T_align @ pose

        return pts_rot.astype(np.float32), norms_rot.astype(np.float32), new_poses, sparse_rot, T_align

    def reconstruct_dense(
        self,
        image_dir: Path,
        camera_poses: Dict[str, np.ndarray],
        depth_maps: Optional[Dict[str, np.ndarray]] = None,
        sparse_points: Optional[np.ndarray] = None,
        sparse_colors: Optional[np.ndarray] = None,
        use_mvs_verification: bool = False,
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
            mvs_done = False
            if use_mvs_verification and len(camera_poses) >= 2:
                try:
                    from photo_reconstruction.mvs_consistency import MVSGeometricConsistencyFilter
                    mvs_filter = MVSGeometricConsistencyFilter(
                        rel_depth_threshold=0.10,
                        min_consistent_views=1,
                    )
                    stride = max(4, int(self.voxel_size * 200))
                    mvs_pts, mvs_cols, _ = mvs_filter.filter_points_with_colors(
                        camera_poses=camera_poses,
                        depth_maps=depth_maps,
                        image_dir=image_dir,
                        stride=stride,
                    )
                    if len(mvs_pts) >= 500:
                        all_pts.append(mvs_pts)
                        all_cols.append(mvs_cols)
                        mvs_done = True
                except Exception:
                    mvs_done = False

            if not mvs_done:
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

                    # Detect outdoor sky mask to avoid turning clouds/sky into solid 3D geometry
                    sky_mask = self.detect_sky_mask(img_bgr)
                    if sky_mask.shape[:2] != (H, W):
                        sky_mask = cv2.resize(sky_mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0

                    # Adaptive sampling stride: yields ~30k-60k points per view regardless of 12MP-48MP resolution
                    s = max(6, int(max(W, H) / 200))
                    v_grid, u_grid = np.mgrid[0:H:s, 0:W:s]
                    d = depth[v_grid, u_grid]
                    sky_sub = sky_mask[v_grid, u_grid]

                    valid_d = d[d > 0.2]
                    if len(valid_d) == 0:
                        continue
                    med_d = float(np.median(valid_d))
                    max_d = min(25.0, max(6.0, med_d * 3.5))

                    # Filter out depth outliers AND sky regions
                    valid = (d > 0.2) & (d < max_d) & (~sky_sub)
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
            if camera_poses:
                cam_centers = [pose[:3, 3] for pose in camera_poses.values()]
                avg_cam = np.mean(cam_centers, axis=0)
                pcd.orient_normals_towards_camera_location(avg_cam)
            elif len(pcd.points) > 15:
                pcd.orient_normals_consistent_tangent_plane(k=min(15, len(pcd.points) - 1))
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

