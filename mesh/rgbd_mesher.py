"""
mesh/rgbd_mesher.py — Organized RGB-D Surface Mesh Reconstruction Engine (M7.1)
================================================================================

Reconstructs photorealistic, continuous 3D surface meshes from synchronized
multi-camera RGB images and metric depth maps.

Pipeline per camera:
  1. Organize pixels on a 2D sampling grid with configurable stride (e.g. s=2).
  2. Unproject valid metric depth pixels to camera coordinates via authoritative intrinsics.
  3. Transform coordinates to common vehicle frame via camera extrinsics T_world_cam.
  4. Triangulate neighboring pixels into 3D triangles.
  5. Reject triangles crossing depth discontinuities (silhouette edge preservation).
  6. Reject extreme grazing-angle triangles to resolve camera overlap seams cleanly.
  7. Compute smooth surface normals.
  8. Assign genuine RGB colors from original camera imagery.
  9. Combine all 4 camera meshes into a single watertight scene surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import time

import numpy as np
import open3d as o3d

from pipeline.calibration import CalibrationStore, CameraModel


@dataclass
class CameraMesh:
    """Container for single-camera reconstructed 3D surface mesh."""
    camera_name: str
    vertices: np.ndarray        # (V, 3) float32
    faces: np.ndarray           # (F, 3) uint32
    colors: np.ndarray          # (V, 3) float32 in [0, 1]
    normals: np.ndarray         # (V, 3) float32


class RGBDSurfaceMesher:
    """
    High-performance, vectorized RGB-D surface mesh reconstructor.
    """

    def __init__(
        self,
        calib_store: Optional[CalibrationStore] = None,
        stride: int = 2,
        min_depth: float = 0.20,
        max_depth: float = 35.0,
        max_edge_len: float = 0.35,
        max_depth_step_base: float = 0.20,
        depth_step_ratio: float = 0.08,
        min_grazing_cos: float = 0.12,
        exclude_ego: bool = True,
    ):
        """
        Parameters
        ----------
        calib_store : CalibrationStore
            Authoritative camera calibration store.
        stride : int
            Pixel sampling stride (1=full 1280x720, 2=640x360, 3=426x240, 4=320x180).
        min_depth : float
            Near clipping plane in metres.
        max_depth : float
            Far clipping plane in metres.
        max_edge_len : float
            Maximum triangle 3D edge length in metres (rejects flying triangles).
        max_depth_step_base : float
            Base depth discontinuity threshold in metres.
        depth_step_ratio : float
            Adaptive depth discontinuity scaling factor with distance.
        min_grazing_cos : float
            Cosine of maximum viewing angle (rejects stretched grazing triangles).
        exclude_ego : bool
            Whether to exclude ego-vehicle pixels using semantic mask (class 2).
        """
        if calib_store is None:
            config_path = Path(__file__).resolve().parent.parent / "config" / "cameras.yaml"
            self.calib_store = CalibrationStore(config_path)
            self.calib_store.load()
        else:
            self.calib_store = calib_store

        self.stride = max(1, int(stride))
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.max_edge_len = float(max_edge_len)
        self.max_depth_step_base = float(max_depth_step_base)
        self.depth_step_ratio = float(depth_step_ratio)
        self.min_grazing_cos = float(min_grazing_cos)
        self.exclude_ego = bool(exclude_ego)

    def triangulate_camera(
        self,
        camera_name: str,
        rgb: np.ndarray,
        depth: np.ndarray,
        semantic_mask: Optional[np.ndarray] = None,
    ) -> CameraMesh:
        """
        Reconstruct a 3D surface mesh from a single camera's RGB image and depth map.

        Parameters
        ----------
        camera_name : str ('front', 'rear', 'left', 'right')
        rgb : np.ndarray (H, W, 3) uint8 or float
        depth : np.ndarray (H, W) float32 in metres
        semantic_mask : Optional[np.ndarray] (H, W) uint8

        Returns
        -------
        CameraMesh with world-space vertices, faces, colors, and normals.
        """
        cam_model: CameraModel = self.calib_store.get(camera_name)
        H, W = depth.shape
        fx = cam_model.intrinsics.fx
        fy = cam_model.intrinsics.fy
        cx = cam_model.intrinsics.cx
        cy = cam_model.intrinsics.cy
        T_world_cam = cam_model.T_world_cam
        cam_pos_world = T_world_cam[:3, 3]

        # 1. Organized 2D Pixel Grid
        s = self.stride
        Hs = H // s
        Ws = W // s
        v_coords = np.arange(0, H, s)[:Hs]
        u_coords = np.arange(0, W, s)[:Ws]
        u_grid, v_grid = np.meshgrid(u_coords, v_coords)

        d_grid = depth[v_grid, u_grid].astype(np.float32)

        # 2. Validity Mask
        valid = (d_grid >= self.min_depth) & (d_grid <= self.max_depth)
        if self.exclude_ego and semantic_mask is not None:
            sem_sampled = semantic_mask[v_grid, u_grid]
            valid = valid & (sem_sampled != 2)  # Class 2 is ego vehicle

        # 3. Vectorized Unprojection to World Frame
        x_cam = (u_grid - cx) * d_grid / fx
        y_cam = (v_grid - cy) * d_grid / fy
        z_cam = d_grid
        pts_cam = np.stack([x_cam, y_cam, z_cam, np.ones_like(z_cam)], axis=-1)
        pts_world = np.einsum("ij,abj->abi", T_world_cam, pts_cam)[..., :3]

        # Colors from original RGB image
        if rgb.dtype == np.uint8:
            cols_grid = rgb[v_grid, u_grid].astype(np.float32) / 255.0
        else:
            cols_grid = rgb[v_grid, u_grid].astype(np.float32)

        # 4. Vertex Indexing
        v_idx = np.full((Hs, Ws), -1, dtype=np.int32)
        v_idx[valid] = np.arange(np.sum(valid), dtype=np.int32)
        verts = pts_world[valid].astype(np.float32)
        cols = cols_grid[valid].astype(np.float32)

        if len(verts) < 3:
            return CameraMesh(
                camera_name=camera_name,
                vertices=np.zeros((0, 3), dtype=np.float32),
                faces=np.zeros((0, 3), dtype=np.uint32),
                colors=np.zeros((0, 3), dtype=np.float32),
                normals=np.zeros((0, 3), dtype=np.float32),
            )

        # 5. Organized Grid Cell Triangulation
        # Each 2x2 grid cell (i, j) forms two candidate triangles:
        # A = (i, j),   B = (i, j+1)
        # C = (i+1, j), D = (i+1, j+1)
        A = v_idx[:-1, :-1]
        B = v_idx[:-1, 1:]
        C = v_idx[1:, :-1]
        D = v_idx[1:, 1:]

        # Initial vertex validity
        val_t1 = (A >= 0) & (B >= 0) & (C >= 0)
        val_t2 = (B >= 0) & (C >= 0) & (D >= 0)

        # 6. Depth Discontinuity & Edge Length Rejection
        dA = d_grid[:-1, :-1]
        dB = d_grid[:-1, 1:]
        dC = d_grid[1:, :-1]
        dD = d_grid[1:, 1:]

        # Triangle 1 depth step: max(|dA-dB|, |dB-dC|, |dC-dA|)
        diff_t1 = np.maximum.reduce([np.abs(dA - dB), np.abs(dB - dC), np.abs(dC - dA)])
        thresh_t1 = self.max_depth_step_base + self.depth_step_ratio * np.minimum.reduce([dA, dB, dC])
        val_t1 = val_t1 & (diff_t1 < thresh_t1)

        # Triangle 2 depth step: max(|dB-dC|, |dC-dD|, |dD-dB|)
        diff_t2 = np.maximum.reduce([np.abs(dB - dC), np.abs(dC - dD), np.abs(dD - dB)])
        thresh_t2 = self.max_depth_step_base + self.depth_step_ratio * np.minimum.reduce([dB, dC, dD])
        val_t2 = val_t2 & (diff_t2 < thresh_t2)

        # Construct candidate face indices
        f1 = np.column_stack([A[val_t1], C[val_t1], B[val_t1]])
        f2 = np.column_stack([B[val_t2], C[val_t2], D[val_t2]])
        faces = np.vstack([f1, f2]).astype(np.uint32)

        if len(faces) == 0:
            return CameraMesh(
                camera_name=camera_name,
                vertices=verts,
                faces=np.zeros((0, 3), dtype=np.uint32),
                colors=cols,
                normals=np.zeros((len(verts), 3), dtype=np.float32),
            )

        # 7. 3D Euclidean Edge Length Check (vectorized on faces)
        pA = verts[faces[:, 0]]
        pB = verts[faces[:, 1]]
        pC = verts[faces[:, 2]]

        e1_sq = np.sum((pA - pB) ** 2, axis=1)
        e2_sq = np.sum((pB - pC) ** 2, axis=1)
        e3_sq = np.sum((pC - pA) ** 2, axis=1)
        max_e_sq = self.max_edge_len ** 2
        valid_edges = (e1_sq < max_e_sq) & (e2_sq < max_e_sq) & (e3_sq < max_e_sq)

        faces = faces[valid_edges]
        if len(faces) == 0:
            return CameraMesh(
                camera_name=camera_name,
                vertices=verts,
                faces=np.zeros((0, 3), dtype=np.uint32),
                colors=cols,
                normals=np.zeros((len(verts), 3), dtype=np.float32),
            )

        # 8. Grazing-Angle Rejection (Eliminates stretched edge textures in overlap zones)
        # Face normal = cross(pB - pA, pC - pA)
        pA = verts[faces[:, 0]]
        pB = verts[faces[:, 1]]
        pC = verts[faces[:, 2]]
        face_normals = np.cross(pB - pA, pC - pA)
        fn_norm = np.linalg.norm(face_normals, axis=1, keepdims=True) + 1e-8
        face_normals /= fn_norm

        # Center of each triangle
        face_centers = (pA + pB + pC) / 3.0
        # Viewing rays from camera center to triangle centers
        view_rays = cam_pos_world - face_centers
        vr_norm = np.linalg.norm(view_rays, axis=1, keepdims=True) + 1e-8
        view_rays /= vr_norm

        cos_angles = np.sum(face_normals * view_rays, axis=1)
        # Filter out extreme grazing angles (< 0.12, i.e., > 83 degrees)
        valid_cos = np.abs(cos_angles) >= self.min_grazing_cos
        faces = faces[valid_cos]

        # 9. Compute Per-Vertex Normals via Open3D
        mesh_o3d = o3d.geometry.TriangleMesh()
        mesh_o3d.vertices = o3d.utility.Vector3dVector(verts.astype(np.float64))
        mesh_o3d.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
        mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(cols.astype(np.float64))
        mesh_o3d.compute_vertex_normals()

        normals = np.asarray(mesh_o3d.vertex_normals, dtype=np.float32)

        return CameraMesh(
            camera_name=camera_name,
            vertices=verts,
            faces=faces,
            colors=cols,
            normals=normals,
        )

    def reconstruct_scene_mesh(
        self,
        rgb_dict: Dict[str, np.ndarray],
        depth_dict: Dict[str, np.ndarray],
        semantic_dict: Optional[Dict[str, np.ndarray]] = None,
        camera_names: Optional[List[str]] = None,
    ) -> o3d.geometry.TriangleMesh:
        """
        Reconstruct and fuse complete 4-camera surround 3D surface mesh.

        Parameters
        ----------
        rgb_dict : Dict[str, np.ndarray]
            Map of camera_name -> (H, W, 3) RGB uint8 image.
        depth_dict : Dict[str, np.ndarray]
            Map of camera_name -> (H, W) float32 metric depth map.
        semantic_dict : Optional[Dict[str, np.ndarray]]
            Map of camera_name -> (H, W) uint8 semantic mask.
        camera_names : Optional[List[str]]
            List of cameras to process (defaults to front, rear, left, right).

        Returns
        -------
        Unified o3d.geometry.TriangleMesh with vertices, faces, colors, and normals.
        """
        cams = camera_names or ["front", "rear", "left", "right"]
        all_verts: List[np.ndarray] = []
        all_faces: List[np.ndarray] = []
        all_colors: List[np.ndarray] = []
        all_normals: List[np.ndarray] = []
        v_offset = 0

        for cam in cams:
            if cam not in rgb_dict or cam not in depth_dict:
                continue

            rgb = rgb_dict[cam]
            depth = depth_dict[cam]
            sem = semantic_dict.get(cam) if semantic_dict else None

            cm = self.triangulate_camera(cam, rgb, depth, sem)
            if len(cm.faces) == 0:
                continue

            all_verts.append(cm.vertices)
            all_faces.append(cm.faces + v_offset)
            all_colors.append(cm.colors)
            all_normals.append(cm.normals)
            v_offset += len(cm.vertices)

        if not all_verts:
            return o3d.geometry.TriangleMesh()

        merged_verts = np.vstack(all_verts).astype(np.float64)
        merged_faces = np.vstack(all_faces).astype(np.int32)
        merged_colors = np.vstack(all_colors).astype(np.float64)
        merged_normals = np.vstack(all_normals).astype(np.float64)

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(merged_verts)
        mesh.triangles = o3d.utility.Vector3iVector(merged_faces)
        mesh.vertex_colors = o3d.utility.Vector3dVector(merged_colors)
        mesh.vertex_normals = o3d.utility.Vector3dVector(merged_normals)

        return mesh

    @staticmethod
    def save_mesh_ply(mesh: o3d.geometry.TriangleMesh, out_path: Union[str, Path]) -> Path:
        """Save mesh to PLY format with vertex colors and normals."""
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_triangle_mesh(str(out), mesh, write_vertex_normals=True, write_vertex_colors=True)
        return out

    @staticmethod
    def load_mesh_ply(mesh_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load mesh from PLY file.
        Returns: (vertices (V, 3), faces (F, 3), colors (V, 4) RGBA, normals (V, 3))
        """
        p = Path(mesh_path)
        if not p.exists():
            raise FileNotFoundError(f"Mesh not found: {p}")

        mesh = o3d.io.read_triangle_mesh(str(p))
        verts = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.triangles, dtype=np.uint32)
        colors = np.asarray(mesh.vertex_colors, dtype=np.float32)

        if len(colors) == 0:
            colors = np.ones((len(verts), 4), dtype=np.float32) * 0.7
        elif colors.shape[1] == 3:
            alpha = np.ones((len(colors), 1), dtype=np.float32)
            colors = np.hstack([colors, alpha])

        if len(mesh.vertex_normals) > 0:
            normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
        else:
            mesh.compute_vertex_normals()
            normals = np.asarray(mesh.vertex_normals, dtype=np.float32)

        return verts, faces, colors, normals
