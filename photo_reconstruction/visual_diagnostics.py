"""
photo_reconstruction/visual_diagnostics.py — Offscreen 3D Visual Diagnostics Engine
===================================================================================

Renders and exports the 9 required diagnostic views for CLEAR360 M7.3.3 Part 10:
  1. Brick wall alone.
  2. Ornamental fountain frame alone.
  3. Central sculpture alone.
  4. Basin alone.
  5. Ground pavement alone.
  6. All regions displayed separately in one scene (region-coded colors).
  7. Final assembled textured reconstruction.
  8. Camera-source-coloured fused point cloud.
  9. Before-and-after side-by-side comparison.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import cv2
import numpy as np
import open3d as o3d


class VisualDiagnosticsRenderer:
    """Offscreen headless renderer for photogrammetry diagnostics."""

    REGION_PALETTE = {
        "pavement": [0.40, 0.45, 0.50],        # Slate Gray
        "wall": [0.75, 0.28, 0.22],            # Brick Terracotta
        "fountain_frame": [0.88, 0.78, 0.68],  # Stone Limestone
        "basin": [0.28, 0.68, 0.72],           # Aqua Basin Stone
        "sculpture": [1.00, 0.84, 0.00],       # Vivid Gold
        "adjacent": [0.65, 0.55, 0.48],        # Neutral Earth
    }

    def __init__(self, session_dir: Path, width: int = 1024, height: int = 768):
        self.session_dir = Path(session_dir)
        self.out_dir = self.session_dir / "diagnostics"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.width = width
        self.height = height

    def _render_geometries(
        self,
        geoms: List[Any],
        out_path: Path,
        cam_pos: Optional[np.ndarray] = None,
        look_at: Optional[np.ndarray] = None,
        up: Optional[np.ndarray] = None,
        zoom: float = 0.35,
    ) -> bool:
        """Render geometries offscreen and save to PNG."""
        try:
            vis = o3d.visualization.Visualizer()
            vis.create_window(visible=False, width=self.width, height=self.height)
            for g in geoms:
                vis.add_geometry(g)

            ctr = vis.get_view_control()
            # If default viewpoints not specified, center on bounding box
            if cam_pos is not None and look_at is not None and up is not None:
                ctr.set_lookat(look_at)
                ctr.set_front((cam_pos - look_at) / np.linalg.norm(cam_pos - look_at))
                ctr.set_up(up)
            ctr.set_zoom(zoom)

            vis.poll_events()
            vis.update_renderer()
            img = vis.capture_screen_float_buffer(do_render=True)
            vis.destroy_window()

            img_np = (np.asarray(img) * 255).astype(np.uint8)
            cv2.imwrite(str(out_path), cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
            return True
        except Exception as e:
            print(f"Warning: Offscreen rendering error for {out_path.name}: {e}")
            return False

    def render_all_diagnostics(
        self,
        region_mesh_paths: Dict[str, Path],
        final_mesh_path: Path,
        camera_source_cloud_path: Optional[Path] = None,
        baseline_mesh_path: Optional[Path] = None,
    ) -> Dict[str, str]:
        """
        Generate all 9 required screenshots.
        """
        renders: Dict[str, str] = {}

        # Default camera setup for fountain front-oblique view
        # Scene center is around X=0.8, Y=8.0, Z=2.2
        look_at = np.array([0.8, 8.0, 2.2], dtype=np.float64)
        cam_pos = np.array([0.8, -1.5, 2.2], dtype=np.float64)
        cam_up = np.array([0.0, 0.2, 0.98], dtype=np.float64)

        # 1. Brick Wall alone
        if "wall" in region_mesh_paths and region_mesh_paths["wall"].exists():
            mesh_w = o3d.io.read_triangle_mesh(str(region_mesh_paths["wall"]))
            mesh_w.paint_uniform_color(self.REGION_PALETTE["wall"])
            mesh_w.compute_vertex_normals()
            p1 = self.out_dir / "1_brick_wall_alone.png"
            if self._render_geometries([mesh_w], p1, cam_pos, look_at, cam_up, zoom=0.40):
                renders["1_brick_wall"] = str(p1)

        # 2. Ornamental Fountain Frame alone
        if "fountain_frame" in region_mesh_paths and region_mesh_paths["fountain_frame"].exists():
            mesh_f = o3d.io.read_triangle_mesh(str(region_mesh_paths["fountain_frame"]))
            mesh_f.paint_uniform_color(self.REGION_PALETTE["fountain_frame"])
            mesh_f.compute_vertex_normals()
            p2 = self.out_dir / "2_fountain_frame_alone.png"
            if self._render_geometries([mesh_f], p2, cam_pos, look_at, cam_up, zoom=0.32):
                renders["2_fountain_frame"] = str(p2)

        # 3. Central Sculpture alone
        if "sculpture" in region_mesh_paths and region_mesh_paths["sculpture"].exists():
            mesh_s = o3d.io.read_triangle_mesh(str(region_mesh_paths["sculpture"]))
            mesh_s.paint_uniform_color(self.REGION_PALETTE["sculpture"])
            mesh_s.compute_vertex_normals()
            look_s = np.array([0.85, 8.05, 2.45], dtype=np.float64)
            cam_s = np.array([0.85, 4.0, 2.45], dtype=np.float64)
            p3 = self.out_dir / "3_central_sculpture_alone.png"
            if self._render_geometries([mesh_s], p3, cam_s, look_s, cam_up, zoom=0.15):
                renders["3_central_sculpture"] = str(p3)

        # 4. Basin alone
        if "basin" in region_mesh_paths and region_mesh_paths["basin"].exists():
            mesh_b = o3d.io.read_triangle_mesh(str(region_mesh_paths["basin"]))
            mesh_b.paint_uniform_color(self.REGION_PALETTE["basin"])
            mesh_b.compute_vertex_normals()
            look_b = np.array([0.0, 7.2, 1.2], dtype=np.float64)
            cam_b = np.array([0.0, 2.0, 2.0], dtype=np.float64)
            p4 = self.out_dir / "4_basin_alone.png"
            if self._render_geometries([mesh_b], p4, cam_b, look_b, cam_up, zoom=0.25):
                renders["4_basin"] = str(p4)

        # 5. Ground Pavement alone
        if "pavement" in region_mesh_paths and region_mesh_paths["pavement"].exists():
            mesh_p = o3d.io.read_triangle_mesh(str(region_mesh_paths["pavement"]))
            mesh_p.paint_uniform_color(self.REGION_PALETTE["pavement"])
            mesh_p.compute_vertex_normals()
            look_p = np.array([0.5, 6.5, 0.2], dtype=np.float64)
            cam_p = np.array([0.5, 0.0, 3.5], dtype=np.float64)
            p5 = self.out_dir / "5_pavement_alone.png"
            if self._render_geometries([mesh_p], p5, cam_p, look_p, cam_up, zoom=0.35):
                renders["5_pavement"] = str(p5)

        # 6. All regions displayed separately in one scene (color-coded)
        geoms_all = []
        for r_id, p in region_mesh_paths.items():
            if p.exists():
                m = o3d.io.read_triangle_mesh(str(p))
                m.paint_uniform_color(self.REGION_PALETTE.get(r_id, [0.7, 0.7, 0.7]))
                m.compute_vertex_normals()
                geoms_all.append(m)

        if geoms_all:
            p6 = self.out_dir / "6_all_regions_segmented.png"
            if self._render_geometries(geoms_all, p6, cam_pos, look_at, cam_up, zoom=0.35):
                renders["6_all_regions"] = str(p6)

        # 7. Final assembled reconstruction
        if final_mesh_path.exists():
            final_m = o3d.io.read_triangle_mesh(str(final_mesh_path))
            final_m.compute_vertex_normals()
            p7 = self.out_dir / "7_final_assembled_reconstruction.png"
            if self._render_geometries([final_m], p7, cam_pos, look_at, cam_up, zoom=0.35):
                renders["7_final_assembled"] = str(p7)

        # 8. Camera-source-coloured fused point cloud
        if camera_source_cloud_path and camera_source_cloud_path.exists():
            src_pcd = o3d.io.read_point_cloud(str(camera_source_cloud_path))
            p8 = self.out_dir / "8_camera_source_cloud.png"
            if self._render_geometries([src_pcd], p8, cam_pos, look_at, cam_up, zoom=0.35):
                renders["8_camera_source_cloud"] = str(p8)

        # 9. Before-and-after comparison
        if baseline_mesh_path and baseline_mesh_path.exists() and "7_final_assembled" in renders:
            p_base_render = self.out_dir / "baseline_render.png"
            m_base = o3d.io.read_triangle_mesh(str(baseline_mesh_path))
            m_base.compute_vertex_normals()
            if self._render_geometries([m_base], p_base_render, cam_pos, look_at, cam_up, zoom=0.35):
                # Stitch side by side
                im_base = cv2.imread(str(p_base_render))
                im_final = cv2.imread(renders["7_final_assembled"])
                if im_base is not None and im_final is not None:
                    h = min(im_base.shape[0], im_final.shape[0])
                    im_base = cv2.resize(im_base, (int(im_base.shape[1] * h / im_base.shape[0]), h))
                    im_final = cv2.resize(im_final, (int(im_final.shape[1] * h / im_final.shape[0]), h))

                    # Add text banners
                    cv2.putText(im_base, "BASELINE M7.3.2 (Fragmented BPA)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.putText(im_final, "M7.3.3 ASSEMBLED SCENE", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                    comparison = np.hstack([im_base, im_final])
                    p9 = self.out_dir / "9_before_after_comparison.png"
                    cv2.imwrite(str(p9), comparison)
                    renders["9_before_after"] = str(p9)

        return renders
