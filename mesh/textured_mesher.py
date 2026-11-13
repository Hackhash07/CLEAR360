"""
mesh/textured_mesher.py — Photorealistic Textured 3D Mesh Reconstruction Engine (M7.2)
====================================================================================

Reconstructs photorealistic, continuous 3D surface meshes from multi-camera RGB-D
data with true UV texture mapping and visibility-aware texture fusion.

Key capabilities:
  1. Single-camera organized-grid mesh generation with native UV texture coordinates.
  2. Depth discontinuity rejection with edge-length and silhouette preservation.
  3. Grazing-angle filtering to eliminate stretched edge textures.
  4. Multi-camera visibility-aware texture fusion resolving overlap seams and z-fighting.
  5. 2x2 Texture Atlas generation packing all surround cameras into a single composite map.
  6. Universal Wavefront OBJ + MTL + PNG textured export and PLY vertex-color fallback.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import time

import cv2
import numpy as np
import open3d as o3d

from pipeline.calibration import CalibrationStore, CameraModel


@dataclass
class TexturedCameraMesh:
    """Container for a single camera's textured 3D surface mesh."""
    camera_name: str
    vertices: np.ndarray        # (V, 3) float32 in world frame
    faces: np.ndarray           # (F, 3) uint32
    uvs: np.ndarray             # (V, 2) float32 in [0, 1]
    texture_image: np.ndarray   # (H, W, 3) uint8 RGB
    normals: np.ndarray         # (V, 3) float32
    vertex_colors: np.ndarray   # (V, 3) float32 in [0, 1]


@dataclass
class TexturedSceneMesh:
    """Container for complete fused 4-camera textured surround mesh."""
    vertices: np.ndarray                        # (V, 3) float32 in world frame
    faces: np.ndarray                           # (F, 3) uint32
    uvs: np.ndarray                             # (V, 2) float32 in atlas coordinates [0, 1]
    texture_atlas: np.ndarray                   # (H_atlas, W_atlas, 3) uint8 RGB
    normals: np.ndarray                         # (V, 3) float32
    vertex_colors: np.ndarray                   # (V, 3) float32 in [0, 1]
    camera_meshes: Dict[str, TexturedCameraMesh] = field(default_factory=dict)


class TexturedMesher:
    """
    High-fidelity textured RGB-D surface mesh reconstruction engine.
    """

    def __init__(
        self,
        calib_store: Optional[CalibrationStore] = None,
        stride: int = 1,
        min_depth: float = 0.20,
        max_depth: float = 35.0,
        max_edge_len: float = 0.35,
        max_depth_step_base: float = 0.15,
        depth_step_ratio: float = 0.06,
        min_grazing_cos: float = 0.15,
        exclude_ego: bool = True,
    ):
        """
        Parameters
        ----------
        calib_store : Optional[CalibrationStore]
            Authoritative camera calibration store.
        stride : int
            Pixel sampling stride (1=full 1280x720, 2=640x360). Default 1 for maximum texture fidelity.
        min_depth : float
            Near clipping plane in metres.
        max_depth : float
            Far clipping plane in metres.
        max_edge_len : float
            Maximum triangle 3D edge length in metres.
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

    def triangulate_camera_textured(
        self,
        camera_name: str,
        rgb: np.ndarray,
        depth: np.ndarray,
        semantic_mask: Optional[np.ndarray] = None,
    ) -> TexturedCameraMesh:
        """
        Reconstruct a textured 3D surface mesh from a single camera's RGB image and depth map.

        Parameters
        ----------
        camera_name : str ('front', 'rear', 'left', 'right')
        rgb : np.ndarray (H, W, 3) uint8 in RGB format
        depth : np.ndarray (H, W) float32 in metres
        semantic_mask : Optional[np.ndarray] (H, W) uint8

        Returns
        -------
        TexturedCameraMesh with world-space vertices, faces, UVs, texture, normals, and vertex colors.
        """
        cam_model: CameraModel = self.calib_store.get(camera_name)
        H, W = depth.shape
        fx = cam_model.intrinsics.fx
        fy = cam_model.intrinsics.fy
        cx = cam_model.intrinsics.cx
        cy = cam_model.intrinsics.cy
        T_world_cam = cam_model.T_world_cam
        cam_pos_world = T_world_cam[:3, 3]

        # 1. 2D Pixel Grid
        s = self.stride
        Hs = H // s
        Ws = W // s
        v_coords = np.arange(0, H, s)[:Hs]
        u_coords = np.arange(0, W, s)[:Ws]
        u_grid, v_grid = np.meshgrid(u_coords, v_coords)

        # Apply mild 3x3 median filter on depth to suppress single-pixel spikes
        if s == 1:
            depth_clean = cv2.medianBlur(depth.astype(np.float32), 3)
            d_grid = depth_clean[v_grid, u_grid]
        else:
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

        # 4. Accurate UV Texture Coordinates (OpenGL convention: [0, 1] with v=0 at bottom)
        u_norm = (u_grid.astype(np.float32)) / (W - 1.0)
        v_norm = 1.0 - (v_grid.astype(np.float32)) / (H - 1.0)
        uvs_grid = np.stack([u_norm, v_norm], axis=-1)

        # Colors from original RGB image
        if rgb.dtype == np.uint8:
            cols_grid = rgb[v_grid, u_grid].astype(np.float32) / 255.0
            tex_img = rgb.copy()
        else:
            cols_grid = rgb[v_grid, u_grid].astype(np.float32)
            tex_img = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)

        # 5. Vertex Indexing
        v_idx = np.full((Hs, Ws), -1, dtype=np.int32)
        v_idx[valid] = np.arange(np.sum(valid), dtype=np.int32)
        verts = pts_world[valid].astype(np.float32)
        uvs = uvs_grid[valid].astype(np.float32)
        cols = cols_grid[valid].astype(np.float32)

        if len(verts) < 3:
            return TexturedCameraMesh(
                camera_name=camera_name,
                vertices=np.zeros((0, 3), dtype=np.float32),
                faces=np.zeros((0, 3), dtype=np.uint32),
                uvs=np.zeros((0, 2), dtype=np.float32),
                texture_image=tex_img,
                normals=np.zeros((0, 3), dtype=np.float32),
                vertex_colors=np.zeros((0, 3), dtype=np.float32),
            )

        # 6. Organized Grid Triangulation
        # A = (i, j),   B = (i, j+1)
        # C = (i+1, j), D = (i+1, j+1)
        A = v_idx[:-1, :-1]
        B = v_idx[:-1, 1:]
        C = v_idx[1:, :-1]
        D = v_idx[1:, 1:]

        val_t1 = (A >= 0) & (B >= 0) & (C >= 0)
        val_t2 = (B >= 0) & (C >= 0) & (D >= 0)

        # 7. Adaptive Depth Discontinuity Rejection
        dA = d_grid[:-1, :-1]
        dB = d_grid[:-1, 1:]
        dC = d_grid[1:, :-1]
        dD = d_grid[1:, 1:]

        diff_t1 = np.maximum.reduce([np.abs(dA - dB), np.abs(dB - dC), np.abs(dC - dA)])
        thresh_t1 = self.max_depth_step_base + self.depth_step_ratio * np.minimum.reduce([dA, dB, dC])
        val_t1 = val_t1 & (diff_t1 < thresh_t1)

        diff_t2 = np.maximum.reduce([np.abs(dB - dC), np.abs(dC - dD), np.abs(dD - dB)])
        thresh_t2 = self.max_depth_step_base + self.depth_step_ratio * np.minimum.reduce([dB, dC, dD])
        val_t2 = val_t2 & (diff_t2 < thresh_t2)

        f1 = np.column_stack([A[val_t1], C[val_t1], B[val_t1]])
        f2 = np.column_stack([B[val_t2], C[val_t2], D[val_t2]])
        faces = np.vstack([f1, f2]).astype(np.uint32)

        if len(faces) == 0:
            return TexturedCameraMesh(
                camera_name=camera_name,
                vertices=verts,
                faces=np.zeros((0, 3), dtype=np.uint32),
                uvs=uvs,
                texture_image=tex_img,
                normals=np.zeros((len(verts), 3), dtype=np.float32),
                vertex_colors=cols,
            )

        # 8. 3D Edge Length Filtering (rejects stretched triangles bridging foreground to background)
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
            return TexturedCameraMesh(
                camera_name=camera_name,
                vertices=verts,
                faces=np.zeros((0, 3), dtype=np.uint32),
                uvs=uvs,
                texture_image=tex_img,
                normals=np.zeros((len(verts), 3), dtype=np.float32),
                vertex_colors=cols,
            )

        # 9. Grazing-Angle Rejection (rejects stretched edge textures)
        pA = verts[faces[:, 0]]
        pB = verts[faces[:, 1]]
        pC = verts[faces[:, 2]]
        face_normals = np.cross(pB - pA, pC - pA)
        fn_norm = np.linalg.norm(face_normals, axis=1, keepdims=True) + 1e-8
        face_normals /= fn_norm

        face_centers = (pA + pB + pC) / 3.0
        view_rays = cam_pos_world - face_centers
        vr_norm = np.linalg.norm(view_rays, axis=1, keepdims=True) + 1e-8
        view_rays /= vr_norm

        cos_angles = np.sum(face_normals * view_rays, axis=1)
        valid_cos = np.abs(cos_angles) >= self.min_grazing_cos
        faces = faces[valid_cos]

        # 10. Compute Smooth Vertex Normals
        mesh_o3d = o3d.geometry.TriangleMesh()
        mesh_o3d.vertices = o3d.utility.Vector3dVector(verts.astype(np.float64))
        mesh_o3d.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
        mesh_o3d.compute_vertex_normals()
        normals = np.asarray(mesh_o3d.vertex_normals, dtype=np.float32)

        return TexturedCameraMesh(
            camera_name=camera_name,
            vertices=verts,
            faces=faces,
            uvs=uvs,
            texture_image=tex_img,
            normals=normals,
            vertex_colors=cols,
        )

    def create_texture_atlas(
        self,
        rgb_dict: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, Dict[str, Tuple[float, float, float, float]]]:
        """
        Pack 4 camera RGB images (1280x720) into a unified 2x2 Texture Atlas (2560x1440).

        Layout:
          +---------------+---------------+
          | Front (0, 1)  | Rear (1, 1)   |  Y in [720, 1440) -> V in [0.5, 1.0]
          +---------------+---------------+
          | Left (0, 0)   | Right (1, 0)  |  Y in [0, 720)    -> V in [0.0, 0.5]
          +---------------+---------------+
            X in [0, 1280)  X in [1280, 2560)
            U in [0.0, 0.5] U in [0.5, 1.0]

        Returns
        -------
        atlas_img : (1440, 2560, 3) uint8 RGB
        uv_transforms : Dict[str, (u_min, v_min, u_scale, v_scale)]
        """
        sample_img = next(iter(rgb_dict.values()))
        H, W = sample_img.shape[:2]
        atlas = np.zeros((H * 2, W * 2, 3), dtype=np.uint8)

        # Image positions in atlas (top-left is (0, 0) in image rows)
        # Front: top-left (rows 0..H, cols 0..W)
        # Rear: top-right (rows 0..H, cols W..2W)
        # Left: bottom-left (rows H..2H, cols 0..W)
        # Right: bottom-right (rows H..2H, cols W..2W)
        cams_layout = {
            "front": (0, 0),        # row_idx, col_idx
            "rear": (0, 1),
            "left": (1, 0),
            "right": (1, 1),
        }

        uv_transforms = {}

        for cam, (r_idx, c_idx) in cams_layout.items():
            if cam in rgb_dict:
                img = rgb_dict[cam]
                if img.dtype != np.uint8:
                    img = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
                atlas[r_idx * H:(r_idx + 1) * H, c_idx * W:(c_idx + 1) * W] = img

            # In OpenGL texture coordinates (V=0 at bottom, V=1 at top):
            # Row 0 (top) corresponds to V in [0.5, 1.0]
            # Row 1 (bottom) corresponds to V in [0.0, 0.5]
            u_min = c_idx * 0.5
            v_min = (1 - r_idx) * 0.5
            u_scale = 0.5
            v_scale = 0.5
            uv_transforms[cam] = (u_min, v_min, u_scale, v_scale)

        return atlas, uv_transforms

    def prune_overlap_faces(
        self,
        camera_name: str,
        cm: TexturedCameraMesh,
        other_cams: Dict[str, CameraModel],
    ) -> TexturedCameraMesh:
        """
        Visibility-aware texture selection in multi-camera overlap zones.
        Prunes faces that have an inferior viewing angle compared to an adjacent camera,
        preventing duplicate overlapping geometry and z-fighting.
        """
        if len(cm.faces) == 0:
            return cm

        pA = cm.vertices[cm.faces[:, 0]]
        pB = cm.vertices[cm.faces[:, 1]]
        pC = cm.vertices[cm.faces[:, 2]]
        face_centers = (pA + pB + pC) / 3.0
        face_normals = np.cross(pB - pA, pC - pA)
        fn_norm = np.linalg.norm(face_normals, axis=1, keepdims=True) + 1e-8
        face_normals /= fn_norm

        this_cam = self.calib_store.get(camera_name)
        pos_this = this_cam.position_world
        rays_this = pos_this - face_centers
        rays_this /= (np.linalg.norm(rays_this, axis=1, keepdims=True) + 1e-8)
        cos_this = np.abs(np.sum(face_normals * rays_this, axis=1))

        # Check adjacent cameras (e.g. for front: left and right)
        adjacent_map = {
            "front": ["left", "right"],
            "rear": ["left", "right"],
            "left": ["front", "rear"],
            "right": ["front", "rear"],
        }
        neighbors = adjacent_map.get(camera_name, [])

        keep_mask = np.ones(len(cm.faces), dtype=bool)

        for n_name in neighbors:
            if n_name not in other_cams:
                continue
            n_cam = other_cams[n_name]
            pos_n = n_cam.position_world
            rays_n = pos_n - face_centers
            dist_n = np.linalg.norm(rays_n, axis=1)
            rays_n /= (dist_n[:, None] + 1e-8)
            cos_n = np.abs(np.sum(face_normals * rays_n, axis=1))

            # Project face centers into neighbor camera image
            # If face is well inside neighbor FoV and neighbor has a more frontal view (> +0.05 cos)
            # and is closer to neighbor camera, then prune from this camera
            dist_this = np.linalg.norm(pos_this - face_centers, axis=1)
            
            # Simple sector test: in overlap zone, camera with higher cos_angle wins
            better_in_neighbor = (cos_n > (cos_this + 0.05)) & (dist_n < dist_this * 1.2)
            keep_mask = keep_mask & (~better_in_neighbor)

        filtered_faces = cm.faces[keep_mask]
        return TexturedCameraMesh(
            camera_name=cm.camera_name,
            vertices=cm.vertices,
            faces=filtered_faces,
            uvs=cm.uvs,
            texture_image=cm.texture_image,
            normals=cm.normals,
            vertex_colors=cm.vertex_colors,
        )

    def reconstruct_textured_scene(
        self,
        rgb_dict: Dict[str, np.ndarray],
        depth_dict: Dict[str, np.ndarray],
        semantic_dict: Optional[Dict[str, np.ndarray]] = None,
        camera_names: Optional[List[str]] = None,
        enable_overlap_pruning: bool = True,
    ) -> TexturedSceneMesh:
        """
        Reconstruct complete 4-camera surround 3D mesh with visibility-aware texture fusion.

        Parameters
        ----------
        rgb_dict : Dict[str, np.ndarray]
            Map of camera_name -> (H, W, 3) RGB uint8 image.
        depth_dict : Dict[str, np.ndarray]
            Map of camera_name -> (H, W) float32 metric depth map.
        semantic_dict : Optional[Dict[str, np.ndarray]]
            Map of camera_name -> (H, W) uint8 semantic mask.
        camera_names : Optional[List[str]]
            List of cameras to fuse (defaults to ['front', 'rear', 'left', 'right']).
        enable_overlap_pruning : bool
            Whether to prune duplicate triangles in camera overlap zones.

        Returns
        -------
        TexturedSceneMesh with unified geometry, packed 2x2 texture atlas, and remapped UVs.
        """
        cams = camera_names or ["front", "rear", "left", "right"]

        # 1. Triangulate each camera mesh
        camera_meshes: Dict[str, TexturedCameraMesh] = {}
        for cam in cams:
            if cam not in rgb_dict or cam not in depth_dict:
                continue
            rgb = rgb_dict[cam]
            depth = depth_dict[cam]
            sem = semantic_dict.get(cam) if semantic_dict else None
            cm = self.triangulate_camera_textured(cam, rgb, depth, sem)
            if len(cm.faces) > 0:
                camera_meshes[cam] = cm

        if not camera_meshes:
            return TexturedSceneMesh(
                vertices=np.zeros((0, 3), dtype=np.float32),
                faces=np.zeros((0, 3), dtype=np.uint32),
                uvs=np.zeros((0, 2), dtype=np.float32),
                texture_atlas=np.zeros((10, 10, 3), dtype=np.uint8),
                normals=np.zeros((0, 3), dtype=np.float32),
                vertex_colors=np.zeros((0, 3), dtype=np.float32),
                camera_meshes={},
            )

        # 2. Build 2x2 Texture Atlas
        atlas, uv_transforms = self.create_texture_atlas(rgb_dict)

        # 3. Visibility-Aware Texture Selection & Overlap De-duplication
        all_cams = self.calib_store.all_cameras()
        if enable_overlap_pruning and len(camera_meshes) > 1:
            pruned_meshes = {}
            for cam_name, cm in camera_meshes.items():
                pruned_meshes[cam_name] = self.prune_overlap_faces(cam_name, cm, all_cams)
            camera_meshes = pruned_meshes

        all_verts: List[np.ndarray] = []
        all_faces: List[np.ndarray] = []
        all_uvs: List[np.ndarray] = []
        all_normals: List[np.ndarray] = []
        all_colors: List[np.ndarray] = []
        v_offset = 0

        for cam_name, cm in camera_meshes.items():
            u_min, v_min, u_scale, v_scale = uv_transforms[cam_name]

            # Remap UV coordinates into atlas quadrant
            uvs_atlas = cm.uvs.copy()
            uvs_atlas[:, 0] = u_min + cm.uvs[:, 0] * u_scale
            uvs_atlas[:, 1] = v_min + cm.uvs[:, 1] * v_scale

            all_verts.append(cm.vertices)
            all_faces.append(cm.faces + v_offset)
            all_uvs.append(uvs_atlas)
            all_normals.append(cm.normals)
            all_colors.append(cm.vertex_colors)
            v_offset += len(cm.vertices)

        merged_verts = np.vstack(all_verts).astype(np.float32)
        merged_faces = np.vstack(all_faces).astype(np.uint32)
        merged_uvs = np.vstack(all_uvs).astype(np.float32)
        merged_normals = np.vstack(all_normals).astype(np.float32)
        merged_colors = np.vstack(all_colors).astype(np.float32)

        return TexturedSceneMesh(
            vertices=merged_verts,
            faces=merged_faces,
            uvs=merged_uvs,
            texture_atlas=atlas,
            normals=merged_normals,
            vertex_colors=merged_colors,
            camera_meshes=camera_meshes,
        )

    # -------------------------------------------------------------------------
    # File I/O: Universal Wavefront OBJ + MTL + PNG Export
    # -------------------------------------------------------------------------
    @staticmethod
    def save_textured_mesh_obj(
        mesh: Union[TexturedCameraMesh, TexturedSceneMesh],
        out_dir: Union[str, Path],
        name_prefix: str = "scene_textured",
    ) -> Path:
        """
        Save textured mesh to universal Wavefront OBJ + MTL + PNG texture.

        Creates:
          - {out_dir}/{name_prefix}.obj
          - {out_dir}/{name_prefix}.mtl
          - {out_dir}/{name_prefix}_texture.png
        """
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        obj_file = out_path / f"{name_prefix}.obj"
        mtl_file = out_path / f"{name_prefix}.mtl"
        tex_file = out_path / f"{name_prefix}_texture.png"

        # Determine texture image
        if isinstance(mesh, TexturedSceneMesh):
            tex_img = mesh.texture_atlas
        else:
            tex_img = mesh.texture_image

        # Save texture image (convert RGB to BGR for OpenCV write)
        cv2.imwrite(str(tex_file), cv2.cvtColor(tex_img, cv2.COLOR_RGB2BGR))

        # Write MTL file
        with open(mtl_file, "w", encoding="utf-8") as fm:
            fm.write("# CLEAR360 Material Definition\n")
            fm.write(f"newmtl material_0\n")
            fm.write("Ka 1.000 1.000 1.000\n")
            fm.write("Kd 1.000 1.000 1.000\n")
            fm.write("Ks 0.000 0.000 0.000\n")
            fm.write("d 1.0\n")
            fm.write("illum 1\n")
            fm.write(f"map_Kd {tex_file.name}\n")

        # Write OBJ file
        verts = mesh.vertices
        uvs = mesh.uvs
        normals = mesh.normals
        faces = mesh.faces

        with open(obj_file, "w", encoding="utf-8") as fo:
            fo.write("# CLEAR360 Reconstructed 3D Textured Mesh\n")
            fo.write(f"mtllib {mtl_file.name}\n")

            # Vertices
            for v in verts:
                fo.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")

            # Texture coordinates (vt u v)
            for uv in uvs:
                fo.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")

            # Normals (vn nx ny nz)
            for n in normals:
                fo.write(f"vn {n[0]:.4f} {n[1]:.4f} {n[2]:.4f}\n")

            fo.write("usemtl material_0\n")
            fo.write("s 1\n")

            # Faces (1-indexed in OBJ: f v/vt/vn v/vt/vn v/vt/vn)
            for f in faces:
                i1, i2, i3 = f[0] + 1, f[1] + 1, f[2] + 1
                fo.write(f"f {i1}/{i1}/{i1} {i2}/{i2}/{i2} {i3}/{i3}/{i3}\n")

        return obj_file

    @staticmethod
    def save_textured_mesh_ply(
        mesh: Union[TexturedCameraMesh, TexturedSceneMesh],
        out_path: Union[str, Path],
    ) -> Path:
        """
        Save mesh to PLY format with vertex colors and normals as high-speed fallback.
        """
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        mesh_o3d = o3d.geometry.TriangleMesh()
        mesh_o3d.vertices = o3d.utility.Vector3dVector(mesh.vertices.astype(np.float64))
        mesh_o3d.triangles = o3d.utility.Vector3iVector(mesh.faces.astype(np.int32))
        mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(mesh.vertex_colors.astype(np.float64))
        mesh_o3d.vertex_normals = o3d.utility.Vector3dVector(mesh.normals.astype(np.float64))

        o3d.io.write_triangle_mesh(str(out), mesh_o3d, write_vertex_normals=True, write_vertex_colors=True)
        return out
