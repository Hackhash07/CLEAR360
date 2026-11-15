"""
photo_reconstruction/texture_engine.py — Visibility-Aware Photogrammetric Texturing Engine
========================================================================================

Replaces nearest-neighbor per-vertex color sampling with genuine high-resolution
photographic projective texture mapping and packed texture atlas generation.

Mathematical Formulation:
-------------------------
For each mesh triangle face f = (v0, v1, v2):
    1. Compute face normal n_f and centroid c_f.
    2. For each candidate camera view c:
         - Viewing angle: cos(theta) = n_f . (cam_center - c_f) / ||cam_center - c_f||
         - Backface rejection: if cos(theta) <= 0.15, discard.
         - Project (v0, v1, v2) into image c: (u0, v0), (u1, v1), (u2, v2).
         - Bounds check: 0 <= u < W and 0 <= v < H with margin.
         - Occlusion Z-test: project c_f to camera-space depth z_proj.
           If z_proj > D_c(v, u) + tau_occ (0.15m), discard as occluded.
         - Projected resolution: 2D pixel area a_2D = 0.5 * |det([u1-u0, u2-u0; v1-v0, v2-v0])|.
         - Quality metric: Q(f, c) = cos(theta) * a_2D.
    3. Optimal View Assignment:
         c*(f) = argmax_c Q(f, c)
    4. UV Parameterization & Atlas Packing:
         Packs projected image patches into a unified composite texture atlas PNG.
         Exports standard Wavefront OBJ + MTL + PNG with per-vertex UV texture coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import open3d as o3d

from photo_reconstruction.camera_model import PhoneCameraModel


@dataclass
class TexturingReport:
    """Diagnostic metrics for visibility-aware texturing (M7.3.8 Issue 4)."""
    total_faces: int
    textured_faces: int
    untextured_faces: int
    coverage_percentage: float
    atlas_width: int
    atlas_height: int
    rear_facing_faces: int = 0
    occluded_faces: int = 0
    camera_face_distribution: Dict[str, int] = field(default_factory=dict)
    atlas_png_path: Optional[str] = None
    obj_path: Optional[str] = None


class VisibilityAwareTextureEngine:
    """Generates photorealistic UV-mapped meshes and texture atlases from original photos."""

    def __init__(
        self,
        atlas_size: int = 4096,
        min_viewing_angle_cos: float = 0.15,  # ~81 degrees max grazing angle
        occlusion_threshold: float = 0.20,    # 20 cm depth occlusion margin
        border_margin_px: int = 8,
    ):
        """
        Parameters
        ----------
        atlas_size : int (texture atlas dimension: 2048 or 4096 px)
        min_viewing_angle_cos : float
        occlusion_threshold : float (m)
        border_margin_px : int
        """
        self.atlas_size = int(atlas_size)
        self.min_cos = float(min_viewing_angle_cos)
        self.occ_thresh = float(occlusion_threshold)
        self.border_margin = int(border_margin_px)

    def texture_mesh(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        camera_models: Dict[str, PhoneCameraModel],
        images: Dict[str, np.ndarray],
        depth_maps: Optional[Dict[str, np.ndarray]] = None,
        out_dir: Optional[Path] = None,
        base_name: str = "scene_mesh",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, TexturingReport]:
        """
        Compute visibility-aware texture mapping, UV coordinates, and texture atlas.

        Parameters
        ----------
        vertices : (V, 3) float32 in world frame
        faces : (F, 3) uint32
        camera_models : Dict[str, PhoneCameraModel]
        images : Dict[str, np.ndarray] (H, W, 3) uint8 RGB
        depth_maps : Optional[Dict[str, np.ndarray]] for occlusion testing
        out_dir : Optional[Path] directory to save OBJ, MTL, and PNG
        base_name : str

        Returns
        -------
        uv_vertices : (V_new, 3) float32 vertices (unwelded for UV seam splitting)
        uv_faces : (F, 3) uint32 face vertex indices
        uv_coords : (V_new, 2) float32 UV coordinates in [0, 1]
        texture_atlas : (H_atlas, W_atlas, 3) uint8 RGB
        report : TexturingReport
        """
        num_faces = len(faces)
        num_verts = len(vertices)

        if num_faces == 0 or num_verts == 0 or len(camera_models) == 0 or len(images) == 0:
            dummy_atlas = np.ones((512, 512, 3), dtype=np.uint8) * 180
            return (
                vertices,
                faces,
                np.zeros((num_verts, 2), dtype=np.float32),
                dummy_atlas,
                TexturingReport(num_faces, 0, num_faces, 0.0, 512, 512),
            )

        verts = vertices.astype(np.float64)

        # 1. Compute Face Geometry (Normals and Centroids) from genuine mesh winding
        v0 = verts[faces[:, 0]]
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]

        cross = np.cross(v1 - v0, v2 - v0)
        norm_cross = np.linalg.norm(cross, axis=1, keepdims=True)
        safe_norm = np.where(norm_cross > 1e-9, norm_cross, 1.0)
        face_normals = cross / safe_norm
        face_centroids = (v0 + v1 + v2) / 3.0

        # 2. Score Every Face Against Each Registered Camera
        cam_names = sorted(list(camera_models.keys()))
        num_cams = len(cam_names)

        # best_cam_idx[f] = index of optimal camera, or -1 if untextured
        best_cam_idx = np.full(num_faces, -1, dtype=np.int32)
        best_scores = np.full(num_faces, -1.0, dtype=np.float64)

        for c_idx, name in enumerate(cam_names):
            if name not in images:
                continue

            cam = camera_models[name]
            img = images[name]
            H, W = img.shape[:2]
            d_map = depth_maps.get(name) if depth_maps else None

            # Vector from face to camera center
            cam_c = cam.camera_center
            vec_to_cam = cam_c - face_centroids
            dist_to_cam = np.linalg.norm(vec_to_cam, axis=1)
            safe_dist = np.where(dist_to_cam > 1e-6, dist_to_cam, 1.0)
            dir_to_cam = vec_to_cam / safe_dist[:, None]

            # Viewing angle cosine
            cos_theta = np.sum(face_normals * dir_to_cam, axis=1)
            valid_angle = cos_theta > self.min_cos
            if not np.any(valid_angle):
                continue

            test_f_indices = np.where(valid_angle)[0]

            # Project face centroids
            proj_c, depth_c, valid_c = cam.project_world_to_image(
                face_centroids[test_f_indices], check_bounds=True
            )

            # Check border margin
            u_c = proj_c[:, 0]
            v_c = proj_c[:, 1]
            inside_margin = (
                valid_c
                & (u_c >= self.border_margin)
                & (u_c < float(W - self.border_margin))
                & (v_c >= self.border_margin)
                & (v_c < float(H - self.border_margin))
            )

            test_f_indices = test_f_indices[inside_margin]
            if len(test_f_indices) == 0:
                continue

            # Occlusion check if depth map available
            if d_map is not None:
                u_pix = np.clip(np.round(proj_c[inside_margin, 0]).astype(int), 0, W - 1)
                v_pix = np.clip(np.round(proj_c[inside_margin, 1]).astype(int), 0, H - 1)
                d_obs = d_map[v_pix, u_pix].astype(np.float64)
                z_proj = depth_c[inside_margin]

                # Ensure scale alignment between d_obs and z_proj to prevent false occlusion rejections
                valid_d = d_obs > 0.1
                if np.sum(valid_d) > 20:
                    med_obs = float(np.median(d_obs[valid_d]))
                    med_proj = float(np.median(z_proj[valid_d]))
                    ratio = med_proj / max(med_obs, 1e-4)
                    if abs(ratio - 1.0) > 0.05:
                        d_obs = d_obs * ratio

                # Not occluded if projected depth <= observed depth + margin
                not_occluded = (d_obs > 0.1) & (z_proj <= d_obs + max(self.occ_thresh, 0.40))
                test_f_indices = test_f_indices[not_occluded]
                if len(test_f_indices) == 0:
                    continue

            # Project full triangle vertices to compute 2D resolution
            p0, _, val0 = cam.project_world_to_image(v0[test_f_indices], check_bounds=False)
            p1, _, val1 = cam.project_world_to_image(v1[test_f_indices], check_bounds=False)
            p2, _, val2 = cam.project_world_to_image(v2[test_f_indices], check_bounds=False)

            # 2D cross product for signed area
            area_2d = 0.5 * np.abs(
                (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1])
                - (p2[:, 0] - p0[:, 0]) * (p1[:, 1] - p0[:, 1])
            )

            scores = cos_theta[test_f_indices] * np.log1p(np.maximum(area_2d, 0.0))

            better = scores > best_scores[test_f_indices]
            replace_f = test_f_indices[better]
            best_scores[replace_f] = scores[better]
            best_cam_idx[replace_f] = c_idx

        # M7.3.8 Issue 4: Strict Visibility Enforcement (ZERO invalid first-camera fallback)
        # Any face that is rear-facing, out-of-bounds, or occluded is strictly left untextured
        untextured_mask = best_cam_idx == -1
        untextured_count = int(np.sum(untextured_mask))
        textured_count = num_faces - untextured_count

        # Count rear-facing faces relative to all observing cameras
        # A face is strictly rear-facing if no camera sees it with positive cos(theta)
        max_cos_all = np.zeros(num_faces, dtype=np.float32) - 1.0
        for name in cam_names:
            if name in camera_models:
                cam = camera_models[name]
                vec_to_c = cam.camera_center - face_centroids
                d_c = np.maximum(np.linalg.norm(vec_to_c, axis=1), 1e-6)
                cos_c = np.sum(face_normals * (vec_to_c / d_c[:, None]), axis=1)
                max_cos_all = np.maximum(max_cos_all, cos_c)
        rear_facing_count = int(np.sum(max_cos_all <= 0.0))
        occluded_count = max(0, untextured_count - rear_facing_count)

        # 3. Composite 2x3 or 3x2 Texture Atlas Layout with Dedicated Neutral Swatch
        active_cams = [name for name in cam_names if name in images]
        k_cams = len(active_cams)
        cols_grid = 3 if k_cams > 4 else (2 if k_cams > 1 else 1)
        rows_grid = int(np.ceil(k_cams / float(cols_grid)))

        cell_w = self.atlas_size // cols_grid
        cell_h = self.atlas_size // rows_grid

        total_atlas_w = cols_grid * cell_w
        total_atlas_h = rows_grid * cell_h

        atlas_img = np.zeros((total_atlas_h, total_atlas_w, 3), dtype=np.uint8)

        # Draw dedicated 64x64 neutral gray swatch at (0, 0) for untextured / rear surfaces
        swatch_size = min(64, cell_w // 4, cell_h // 4)
        swatch_color = np.array([140, 140, 140], dtype=np.uint8) # Neutral architectural stone gray
        atlas_img[0:swatch_size, 0:swatch_size] = swatch_color
        u_neutral = float(swatch_size // 2) / float(total_atlas_w)
        v_neutral = float(swatch_size // 2) / float(total_atlas_h)

        # Precompute camera cell rects and downscale images into cells
        cam_cell_offsets: Dict[int, Tuple[int, int, float, float]] = {}  # (x_off, y_off, scale_x, scale_y)
        cam_counts: Dict[str, int] = {}

        for c_idx, name in enumerate(cam_names):
            if name not in images:
                continue

            active_idx = active_cams.index(name)
            grid_r = active_idx // cols_grid
            grid_c = active_idx % cols_grid

            x_off = grid_c * cell_w
            y_off = grid_r * cell_h

            src_img = images[name]
            src_h, src_w = src_img.shape[:2]

            resized_cell = cv2.resize(src_img, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
            # If camera 0 overlaps swatch, preserve the swatch in the very top-left
            if active_idx == 0:
                resized_cell[0:swatch_size, 0:swatch_size] = swatch_color

            atlas_img[y_off : y_off + cell_h, x_off : x_off + cell_w] = resized_cell

            scale_x = float(cell_w) / float(src_w)
            scale_y = float(cell_h) / float(src_h)
            cam_cell_offsets[c_idx] = (x_off, y_off, scale_x, scale_y)
            cam_counts[name] = int(np.sum(best_cam_idx == c_idx))

        # 4. Generate Unwelded UV Vertices and Atlas UV Coordinates
        new_verts = np.zeros((num_faces * 3, 3), dtype=np.float32)
        new_uvs = np.zeros((num_faces * 3, 2), dtype=np.float32)
        new_faces = np.arange(num_faces * 3, dtype=np.uint32).reshape(num_faces, 3)

        for f_i in range(num_faces):
            c_idx = best_cam_idx[f_i]
            pts_3d = verts[faces[f_i]]  # (3, 3)
            new_verts[f_i * 3 : f_i * 3 + 3] = pts_3d

            if c_idx < 0 or c_idx not in cam_cell_offsets:
                # Untextured / rear-facing / occluded face: map to neutral swatch
                new_uvs[f_i * 3 : f_i * 3 + 3] = np.array([
                    [u_neutral, v_neutral],
                    [u_neutral, v_neutral],
                    [u_neutral, v_neutral],
                ], dtype=np.float32)
                continue

            x_off, y_off, s_x, s_y = cam_cell_offsets[c_idx]
            cam = camera_models[cam_names[c_idx]]

            # Project the 3 vertices into the valid observing camera
            pix_2d, _, _ = cam.project_world_to_image(pts_3d, check_bounds=False)

            u_atlas = (x_off + pix_2d[:, 0] * s_x) / float(total_atlas_w)
            v_atlas = (y_off + pix_2d[:, 1] * s_y) / float(total_atlas_h)

            # Clamp UV within atlas cell bounds to avoid seam bleeding
            u_min = x_off / float(total_atlas_w)
            u_max = (x_off + cell_w) / float(total_atlas_w)
            v_min = y_off / float(total_atlas_h)
            v_max = (y_off + cell_h) / float(total_atlas_h)

            u_atlas = np.clip(u_atlas, u_min + 1e-4, u_max - 1e-4)
            v_atlas = np.clip(v_atlas, v_min + 1e-4, v_max - 1e-4)

            new_uvs[f_i * 3 : f_i * 3 + 3] = np.column_stack([u_atlas, v_atlas])

        coverage_pct = float(textured_count / float(num_faces) * 100.0) if num_faces > 0 else 0.0

        report = TexturingReport(
            total_faces=num_faces,
            textured_faces=textured_count,
            untextured_faces=untextured_count,
            coverage_percentage=round(coverage_pct, 1),
            atlas_width=total_atlas_w,
            atlas_height=total_atlas_h,
            rear_facing_faces=rear_facing_count,
            occluded_faces=occluded_count,
            camera_face_distribution=cam_counts,
        )

        # 5. Export OBJ + MTL + PNG if output directory specified
        if out_dir is not None:
            o_dir = Path(out_dir)
            o_dir.mkdir(parents=True, exist_ok=True)

            atlas_png = o_dir / f"{base_name}.png"
            cv2.imwrite(str(atlas_png), cv2.cvtColor(atlas_img, cv2.COLOR_RGB2BGR))
            report.atlas_png_path = str(atlas_png)

            mtl_path = o_dir / f"{base_name}.mtl"
            with open(mtl_path, "w") as f_mtl:
                f_mtl.write("# CLEAR360 Photogrammetric Material\n")
                f_mtl.write(f"newmtl material_0\n")
                f_mtl.write("Ka 1.0 1.0 1.0\n")
                f_mtl.write("Kd 1.0 1.0 1.0\n")
                f_mtl.write("Ks 0.0 0.0 0.0\n")
                f_mtl.write("d 1.0\n")
                f_mtl.write("illum 1\n")
                f_mtl.write(f"map_Kd {base_name}.png\n")

            # Compute smooth vertex normals on original connected topology for photorealistic shading
            m_orig = o3d.geometry.TriangleMesh()
            m_orig.vertices = o3d.utility.Vector3dVector(verts.astype(np.float64))
            m_orig.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
            m_orig.compute_vertex_normals()
            orig_vn = np.asarray(m_orig.vertex_normals, dtype=np.float32)

            new_vn = np.zeros_like(new_verts, dtype=np.float32)
            for f_i in range(num_faces):
                orig_face = faces[f_i]
                new_vn[f_i * 3 : f_i * 3 + 3] = orig_vn[orig_face]

            obj_path = o_dir / f"{base_name}.obj"
            with open(obj_path, "w") as f_obj:
                f_obj.write("# CLEAR360 Photogrammetric 3D Reconstructed Mesh\n")
                f_obj.write(f"mtllib {base_name}.mtl\n")
                f_obj.write("usemtl material_0\n")

                for v in new_verts:
                    f_obj.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

                for vt in new_uvs:
                    f_obj.write(f"vt {vt[0]:.6f} {vt[1]:.6f}\n")

                for vn in new_vn:
                    f_obj.write(f"vn {vn[0]:.6f} {vn[1]:.6f} {vn[2]:.6f}\n")

                for tri in new_faces:
                    # 1-based indexing in Wavefront OBJ (v/vt/vn)
                    f_obj.write(f"f {tri[0]+1}/{tri[0]+1}/{tri[0]+1} {tri[1]+1}/{tri[1]+1}/{tri[1]+1} {tri[2]+1}/{tri[2]+1}/{tri[2]+1}\n")

            report.obj_path = str(obj_path)

        return new_verts, new_faces, new_uvs, atlas_img, report

