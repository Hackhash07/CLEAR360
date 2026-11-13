"""
pointcloud/unprojector.py — Modular Open3D Point Cloud Generator for CLEAR360
================================================================================

Reads ground-truth depth maps, unprojects pixels into camera-space 3D coordinates,
transforms them into the common vehicle coordinate frame using calibrated extrinsics,
and fuses the four camera point clouds into a single coloured Open3D PointCloud.

Coordinate Conventions (CRITICAL — do not change):
  World / Vehicle frame : X = RIGHT, Y = FORWARD, Z = UP
  Camera frame (OpenCV) : x = right, y = down, z = forward (into scene)
  T_cam_world           : 4x4 float64, transforms world points to camera frame
  T_world_cam           : inv(T_cam_world), transforms camera points to world frame
  Depth                 : Camera-space Z-depth in metres (NOT Euclidean)

Unprojection Formula:
  Given pixel (u, v) and camera Z-depth d (metres):
    x_cam = (u - cx) * d / fx
    y_cam = (v - cy) * d / fy
    z_cam = d
    p_cam = [x_cam, y_cam, z_cam, 1]
    p_world = T_world_cam @ p_cam

Usage:
  from pointcloud.unprojector import DepthUnprojector
  unprojector = DepthUnprojector()
  cloud = unprojector.unproject_frame(rgb, depth, camera_model)
  merged = unprojector.fuse_frame_set(frame_set, calibration_store)
  unprojector.save_ply(merged, "output.ply")
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

try:
    import open3d as o3d
    _O3D_AVAILABLE = True
except ImportError:
    _O3D_AVAILABLE = False
    o3d = None  # type: ignore[assignment]


def _require_open3d() -> None:
    """Raise ImportError with installation instructions if Open3D is missing."""
    if not _O3D_AVAILABLE:
        raise ImportError(
            "Open3D is required for point cloud operations.\n"
            "Install it with:  pip install open3d>=0.18"
        )


class DepthUnprojector:
    """
    Converts depth maps into coloured Open3D point clouds in the world/vehicle
    coordinate frame.

    All operations use the pinhole camera model and calibrated extrinsic matrices.
    No AI estimation is performed — this module works exclusively with measured
    or ground-truth depth.

    Parameters
    ----------
    max_depth : float
        Maximum depth to include (metres). Points with Z-depth > max_depth are
        discarded. Default 50.0 m (walls are at ~9.5 m, sky is at 80.0 m).
    min_depth : float
        Minimum depth to include (metres). Discard artefacts very close to
        the camera sensor. Default 0.1 m.
    stride : int
        Pixel stride for downsampling. stride=1 uses every pixel, stride=2
        uses every other pixel (4× fewer points). Default 1.
    exclude_ego : bool
        If True, exclude pixels with semantic class 2 (ego vehicle).
        Default True — we render the ego vehicle separately in the viewer.
    """

    # Semantic class indices (matching sim/constants.py)
    SEMANTIC_CLASSES = {
        0: "sky_unknown",
        1: "road",
        2: "ego_vehicle",
        3: "parked_vehicle",
        4: "wall",
        5: "pillar",
        6: "dynamic_obstacle",
        7: "kerb_speed_breaker",
    }

    def __init__(
        self,
        max_depth: float = 50.0,
        min_depth: float = 0.1,
        stride: int = 1,
        exclude_ego: bool = True,
    ):
        _require_open3d()
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.stride = max(1, stride)
        self.exclude_ego = exclude_ego

    def unproject_depth(
        self,
        depth: np.ndarray,
        camera_model,
        rgb: Optional[np.ndarray] = None,
        semantic_mask: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Unproject a depth map into world-frame 3D points.

        Parameters
        ----------
        depth : (H, W) float32
            Camera Z-depth in metres.
        camera_model : pipeline.calibration.CameraModel
            Camera with intrinsics and T_cam_world.
        rgb : (H, W, 3) uint8, optional
            RGB image for colouring points. If None, points are white.
        semantic_mask : (H, W) uint8, optional
            Semantic class per pixel. Used for ego-vehicle exclusion.

        Returns
        -------
        points_world : (N, 3) float64 — 3D points in world/vehicle frame
        colors : (N, 3) float64 — RGB colours in [0, 1]
        labels : (N,) uint8 or None — semantic labels if mask provided

        Unprojection Formula (vectorised):
            u_grid, v_grid = meshgrid of pixel coordinates
            x_cam = (u - cx) * depth / fx        # camera-frame X
            y_cam = (v - cy) * depth / fy        # camera-frame Y
            z_cam = depth                         # camera-frame Z (= depth by definition)
            p_cam_hom = stack([x_cam, y_cam, z_cam, 1])   # (N, 4) homogeneous
            p_world = (T_world_cam @ p_cam_hom.T).T[:, :3]  # transform to world frame
        """
        H, W = depth.shape
        fx = camera_model.intrinsics.fx
        fy = camera_model.intrinsics.fy
        cx = camera_model.intrinsics.cx
        cy = camera_model.intrinsics.cy
        T_world_cam = camera_model.T_world_cam  # 4x4, camera→world

        # Build pixel coordinate grids with optional stride
        v_coords = np.arange(0, H, self.stride)
        u_coords = np.arange(0, W, self.stride)
        u_grid, v_grid = np.meshgrid(u_coords, v_coords)  # (Hs, Ws)

        # Sample depth at stride positions
        depth_sampled = depth[v_grid, u_grid]  # (Hs, Ws)

        # Build validity mask
        valid = (depth_sampled >= self.min_depth) & (depth_sampled <= self.max_depth)

        # Exclude ego vehicle pixels if requested
        if self.exclude_ego and semantic_mask is not None:
            sem_sampled = semantic_mask[v_grid, u_grid]
            valid = valid & (sem_sampled != 2)

        # Flatten to valid pixels only
        u_valid = u_grid[valid].astype(np.float64)
        v_valid = v_grid[valid].astype(np.float64)
        d_valid = depth_sampled[valid].astype(np.float64)

        # Unprojection: pixel + depth → camera-frame 3D point
        # x_cam = (u - cx) * d / fx
        # y_cam = (v - cy) * d / fy
        # z_cam = d
        x_cam = (u_valid - cx) * d_valid / fx
        y_cam = (v_valid - cy) * d_valid / fy
        z_cam = d_valid

        N = len(d_valid)
        p_cam_hom = np.ones((N, 4), dtype=np.float64)
        p_cam_hom[:, 0] = x_cam
        p_cam_hom[:, 1] = y_cam
        p_cam_hom[:, 2] = z_cam

        # Transform to world frame: p_world = T_world_cam @ p_cam
        p_world = (T_world_cam @ p_cam_hom.T).T[:, :3]  # (N, 3)

        # Colours
        if rgb is not None:
            rgb_sampled = rgb[v_grid, u_grid]  # (Hs, Ws, 3)
            colors = rgb_sampled[valid].astype(np.float64) / 255.0  # (N, 3)
        else:
            colors = np.ones((N, 3), dtype=np.float64)

        # Labels
        labels = None
        if semantic_mask is not None:
            sem_sampled = semantic_mask[v_grid, u_grid]
            labels = sem_sampled[valid]

        return p_world, colors, labels

    def unproject_to_pointcloud(
        self,
        depth: np.ndarray,
        camera_model,
        rgb: Optional[np.ndarray] = None,
        semantic_mask: Optional[np.ndarray] = None,
    ) -> o3d.geometry.PointCloud:
        """
        Unproject depth map and return an Open3D PointCloud in the world frame.

        Parameters and conventions: see unproject_depth().
        """
        points, colors, labels = self.unproject_depth(
            depth, camera_model, rgb, semantic_mask
        )

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        return pcd

    def fuse_frame_set(
        self,
        frame_set,
        calibration_store,
        camera_names: Optional[list[str]] = None,
    ) -> o3d.geometry.PointCloud:
        """
        Fuse all four camera views from a single FrameSet into one merged
        world-frame point cloud.

        This uses DIRECT EXTRINSIC TRANSFORM — no ICP alignment needed because
        all cameras are rigidly mounted with known calibrations.

        Parameters
        ----------
        frame_set : pipeline.sync.FrameSet
            A complete 4-camera frame set with RGB, depth, and optionally
            semantic masks loaded.
        calibration_store : pipeline.calibration.CalibrationStore
            Loaded camera calibrations.
        camera_names : list[str], optional
            Which cameras to include. Default: ['front','rear','left','right'].

        Returns
        -------
        o3d.geometry.PointCloud
            Merged, coloured point cloud in the world/vehicle coordinate frame.
        """
        if camera_names is None:
            camera_names = ["front", "rear", "left", "right"]

        all_points = []
        all_colors = []

        for cam_name in camera_names:
            cfd = frame_set.get(cam_name)
            cam = calibration_store.get(cam_name)

            points, colors, _ = self.unproject_depth(
                depth=cfd.depth,
                camera_model=cam,
                rgb=cfd.rgb,
                semantic_mask=cfd.semantic_mask,
            )

            all_points.append(points)
            all_colors.append(colors)

        merged_points = np.vstack(all_points)
        merged_colors = np.vstack(all_colors)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(merged_points)
        pcd.colors = o3d.utility.Vector3dVector(merged_colors)

        return pcd

    @staticmethod
    def save_ply(pcd: o3d.geometry.PointCloud, path: str | Path) -> None:
        """Save a point cloud to PLY file."""
        _require_open3d()
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_point_cloud(str(p), pcd, write_ascii=False)

    @staticmethod
    def downsample(pcd: o3d.geometry.PointCloud, voxel_size: float = 0.05) -> o3d.geometry.PointCloud:
        """Voxel downsample a point cloud."""
        _require_open3d()
        return pcd.voxel_down_sample(voxel_size=voxel_size)

    def get_stats(self, pcd: o3d.geometry.PointCloud) -> dict:
        """Return summary statistics for a point cloud."""
        _require_open3d()
        points = np.asarray(pcd.points)
        return {
            "num_points": len(points),
            "bbox_min": points.min(axis=0).tolist(),
            "bbox_max": points.max(axis=0).tolist(),
            "centroid": points.mean(axis=0).tolist(),
        }

