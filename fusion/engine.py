"""
fusion/engine.py — Multi-Camera 3D Point Cloud Fusion Engine for CLEAR360
========================================================================

Transforms depth maps from all 4 surround cameras (front, rear, left, right)
into a single, unified 3D point cloud in the common vehicle/world frame:
    X = RIGHT, Y = FORWARD, Z = UP (vehicle centered at origin).

Uses direct rigid transformations from calibrated extrinsics (T_cam_world).
ICP is NOT used as primary alignment because camera positions and orientations
are rigidly fixed and known from calibration.

Maintains two strictly separate reconstruction pipelines:
    1. Ground-Truth (GT) Reconstruction (Blender Cycles depth)
    2. AI-Predicted Reconstruction:
       a) Raw AI (Depth Anything V2 metric output)
       b) Global-Scale Calibrated AI
       c) Per-Camera Scale Calibrated AI
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import open3d as o3d

from pointcloud.unprojector import DepthUnprojector
from pipeline.calibration import CalibrationStore, CameraModel
from pipeline.sync import FrameSet, CameraFrameData


# Distinct camera colors for multi-camera overlap inspection:
# Front: Red [1, 0, 0], Left: Green [0, 0.8, 0], Rear: Blue [0, 0.2, 1], Right: Yellow [1, 0.8, 0]
CAMERA_OVERLAP_COLORS: Dict[str, np.ndarray] = {
    "front": np.array([0.95, 0.15, 0.15], dtype=np.float64),
    "left":  np.array([0.15, 0.85, 0.15], dtype=np.float64),
    "rear":  np.array([0.15, 0.35, 0.95], dtype=np.float64),
    "right": np.array([0.95, 0.85, 0.15], dtype=np.float64),
}


@dataclass
class FusionResult:
    """
    Result of a multi-camera point cloud fusion run.

    Attributes
    ----------
    pipeline_type : str ('ground_truth', 'ai_raw', 'ai_global', 'ai_per_camera')
    frame_id      : int
    num_points    : int
    bbox_min      : np.ndarray [min_x, min_y, min_z]
    bbox_max      : np.ndarray [max_x, max_y, max_z]
    centroid      : np.ndarray [mean_x, mean_y, mean_z]
    point_cloud   : o3d.geometry.PointCloud
    camera_clouds : Dict[str, o3d.geometry.PointCloud]
    """
    pipeline_type: str
    frame_id: int
    num_points: int
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    centroid: np.ndarray
    point_cloud: o3d.geometry.PointCloud
    camera_clouds: Dict[str, o3d.geometry.PointCloud] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"[{self.pipeline_type.upper()}] Frame {self.frame_id:04d}: "
            f"{self.num_points:,} points | "
            f"BBox X: [{self.bbox_min[0]:.2f}, {self.bbox_max[0]:.2f}] m, "
            f"Y: [{self.bbox_min[1]:.2f}, {self.bbox_max[1]:.2f}] m, "
            f"Z: [{self.bbox_min[2]:.2f}, {self.bbox_max[2]:.2f}] m"
        )


class PointCloudFusionEngine:
    """
    Rigid multi-camera point cloud fusion engine.

    Parameters
    ----------
    calibration_store : CalibrationStore
        Calibrated camera models (intrinsics + extrinsics).
    max_depth : float
        Maximum depth cut-off in metres (default 50.0 m).
    min_depth : float
        Minimum depth cut-off in metres (default 0.1 m).
    stride : int
        Pixel stride for unprojection (default 1 = full resolution).
    exclude_ego : bool
        Whether to exclude ego vehicle pixels (default True).
    """

    def __init__(
        self,
        calibration_store: CalibrationStore,
        max_depth: float = 50.0,
        min_depth: float = 0.1,
        stride: int = 1,
        exclude_ego: bool = True,
    ):
        self.calibration_store = calibration_store
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.stride = stride
        self.exclude_ego = exclude_ego
        self.unprojector = DepthUnprojector(
            max_depth=max_depth,
            min_depth=min_depth,
            stride=stride,
            exclude_ego=exclude_ego,
        )

    def unproject_camera(
        self,
        camera_name: str,
        depth: np.ndarray,
        rgb: Optional[np.ndarray] = None,
        semantic_mask: Optional[np.ndarray] = None,
        use_camera_color: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Unproject a single camera depth map directly into vehicle/world frame.

        Parameters
        ----------
        camera_name : str ('front', 'rear', 'left', 'right')
        depth : (H, W) float32 in metres
        rgb : (H, W, 3) uint8, optional
        semantic_mask : (H, W) uint8, optional
        use_camera_color : bool, if True assigns fixed camera tint for overlap inspection

        Returns
        -------
        points_world : (N, 3) float64
        colors : (N, 3) float64 in [0, 1]
        labels : (N,) uint8 or None
        """
        cam_model = self.calibration_store.get(camera_name)
        points_world, colors, labels = self.unprojector.unproject_depth(
            depth=depth,
            camera_model=cam_model,
            rgb=rgb,
            semantic_mask=semantic_mask,
        )

        if use_camera_color and len(points_world) > 0:
            cam_color = CAMERA_OVERLAP_COLORS.get(camera_name, np.array([0.5, 0.5, 0.5]))
            colors = np.tile(cam_color, (len(points_world), 1))

        return points_world, colors, labels

    def fuse_frame_set(
        self,
        frame_set: FrameSet,
        camera_names: Optional[List[str]] = None,
        use_camera_colors: bool = False,
        pipeline_type: str = "ground_truth",
    ) -> FusionResult:
        """
        Fuse all 4 cameras from a FrameSet into a unified world-frame point cloud.

        Parameters
        ----------
        frame_set : FrameSet
            Synchronized 4-camera frame data.
        camera_names : List[str], optional
            Cameras to fuse (default: ['front', 'rear', 'left', 'right']).
        use_camera_colors : bool
            Color points by camera identifier for overlap inspection.
        pipeline_type : str
            Provenance descriptor ('ground_truth', 'ai_raw', etc.).

        Returns
        -------
        FusionResult containing merged Open3D PointCloud and individual clouds.
        """
        if camera_names is None:
            camera_names = ["front", "rear", "left", "right"]

        all_points: List[np.ndarray] = []
        all_colors: List[np.ndarray] = []
        camera_clouds: Dict[str, o3d.geometry.PointCloud] = {}

        for cam_name in camera_names:
            cfd = frame_set.get(cam_name)
            pts, col, _ = self.unproject_camera(
                camera_name=cam_name,
                depth=cfd.depth,
                rgb=cfd.rgb,
                semantic_mask=cfd.semantic_mask,
                use_camera_color=use_camera_colors,
            )

            # Store individual camera cloud
            cam_pcd = o3d.geometry.PointCloud()
            if len(pts) > 0:
                cam_pcd.points = o3d.utility.Vector3dVector(pts)
                cam_pcd.colors = o3d.utility.Vector3dVector(col)
                all_points.append(pts)
                all_colors.append(col)
            camera_clouds[cam_name] = cam_pcd

        if all_points:
            merged_pts = np.vstack(all_points)
            merged_col = np.vstack(all_colors)
        else:
            merged_pts = np.zeros((0, 3), dtype=np.float64)
            merged_col = np.zeros((0, 3), dtype=np.float64)

        merged_pcd = o3d.geometry.PointCloud()
        if len(merged_pts) > 0:
            merged_pcd.points = o3d.utility.Vector3dVector(merged_pts)
            merged_pcd.colors = o3d.utility.Vector3dVector(merged_col)
            bbox_min = merged_pts.min(axis=0)
            bbox_max = merged_pts.max(axis=0)
            centroid = merged_pts.mean(axis=0)
        else:
            bbox_min = np.zeros(3)
            bbox_max = np.zeros(3)
            centroid = np.zeros(3)

        return FusionResult(
            pipeline_type=pipeline_type,
            frame_id=frame_set.frame_id,
            num_points=len(merged_pts),
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            centroid=centroid,
            point_cloud=merged_pcd,
            camera_clouds=camera_clouds,
        )

    def fuse_depth_maps(
        self,
        depth_maps: Dict[str, np.ndarray],
        rgb_maps: Optional[Dict[str, np.ndarray]] = None,
        semantic_masks: Optional[Dict[str, np.ndarray]] = None,
        frame_id: int = 0,
        pipeline_type: str = "ai_predicted",
        use_camera_colors: bool = False,
    ) -> FusionResult:
        """
        Fuse depth maps provided as a dictionary mapping camera name -> (H, W) float32 array.

        Parameters
        ----------
        depth_maps : Dict[str, np.ndarray]
            Map of camera_name -> depth map (H, W) in metres.
        rgb_maps : Dict[str, np.ndarray], optional
            Map of camera_name -> RGB image (H, W, 3) uint8.
        semantic_masks : Dict[str, np.ndarray], optional
            Map of camera_name -> semantic mask (H, W) uint8.
        frame_id : int
        pipeline_type : str
        use_camera_colors : bool

        Returns
        -------
        FusionResult
        """
        all_points: List[np.ndarray] = []
        all_colors: List[np.ndarray] = []
        camera_clouds: Dict[str, o3d.geometry.PointCloud] = {}

        for cam_name in ["front", "rear", "left", "right"]:
            if cam_name not in depth_maps:
                continue

            depth = depth_maps[cam_name]
            rgb = rgb_maps.get(cam_name) if rgb_maps else None
            sem = semantic_masks.get(cam_name) if semantic_masks else None

            pts, col, _ = self.unproject_camera(
                camera_name=cam_name,
                depth=depth,
                rgb=rgb,
                semantic_mask=sem,
                use_camera_color=use_camera_colors,
            )

            cam_pcd = o3d.geometry.PointCloud()
            if len(pts) > 0:
                cam_pcd.points = o3d.utility.Vector3dVector(pts)
                cam_pcd.colors = o3d.utility.Vector3dVector(col)
                all_points.append(pts)
                all_colors.append(col)
            camera_clouds[cam_name] = cam_pcd

        if all_points:
            merged_pts = np.vstack(all_points)
            merged_col = np.vstack(all_colors)
        else:
            merged_pts = np.zeros((0, 3), dtype=np.float64)
            merged_col = np.zeros((0, 3), dtype=np.float64)

        merged_pcd = o3d.geometry.PointCloud()
        if len(merged_pts) > 0:
            merged_pcd.points = o3d.utility.Vector3dVector(merged_pts)
            merged_pcd.colors = o3d.utility.Vector3dVector(merged_col)
            bbox_min = merged_pts.min(axis=0)
            bbox_max = merged_pts.max(axis=0)
            centroid = merged_pts.mean(axis=0)
        else:
            bbox_min = np.zeros(3)
            bbox_max = np.zeros(3)
            centroid = np.zeros(3)

        return FusionResult(
            pipeline_type=pipeline_type,
            frame_id=frame_id,
            num_points=len(merged_pts),
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            centroid=centroid,
            point_cloud=merged_pcd,
            camera_clouds=camera_clouds,
        )

    @staticmethod
    def downsample_voxel(pcd: o3d.geometry.PointCloud, voxel_size: float = 0.05) -> o3d.geometry.PointCloud:
        """Apply 3D voxel grid downsampling (default 5 cm)."""
        return pcd.voxel_down_sample(voxel_size=voxel_size)

    @staticmethod
    def save_ply(pcd: o3d.geometry.PointCloud, path: Union[str, Path]) -> Path:
        """Save PointCloud to PLY file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_point_cloud(str(p), pcd, write_ascii=False)
        return p

