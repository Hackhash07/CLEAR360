"""
photo_reconstruction/stage_auditor.py — Comprehensive Pipeline Stage & Coordinate Auditor
========================================================================================

Inspects, validates, and exports every stage of the photogrammetry reconstruction pipeline
(Stages A through H) in accordance with CLEAR360 M7.3.3 requirements:
  A. Individual camera unprojected point clouds.
  B. Individual camera depth maps statistics.
  C. Sparse COLMAP reconstruction.
  D. Per-view SfM-scale aligned point clouds.
  E. Multi-view consistency-filtered point clouds.
  F. Individual region point clouds.
  G. Individual region meshes.
  H. Final assembled scene mesh.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
import open3d as o3d
import pycolmap

from photo_reconstruction.camera_model import PhoneCameraModel


class ReconstructionStageAuditor:
    """Audits coordinate systems, geometric consistency, and stage-by-stage data."""

    CAMERA_PALETTE = [
        [1.0, 0.15, 0.15],  # Cam 3: Bright Red
        [1.0, 0.55, 0.00],  # Cam 4: Vibrant Orange
        [0.95, 0.90, 0.10], # Cam 5: Yellow
        [0.15, 0.85, 0.25], # Cam 6: Green
        [0.10, 0.75, 0.95], # Cam 7: Cyan
        [0.25, 0.35, 0.95], # Cam 8: Royal Blue
    ]

    def __init__(self, session_dir: Path):
        self.session_dir = Path(session_dir)
        self.stages_dir = self.session_dir / "stages"
        self.stages_dir.mkdir(parents=True, exist_ok=True)

    def audit_camera_transforms(
        self,
        camera_models: Dict[str, PhoneCameraModel],
    ) -> Dict[str, Any]:
        """
        Verify every camera uses the same reference coordinate frame:
        - Orthonormality: det(R) == +1.0
        - Transpose inversion: R @ R.T == I
        - Physical camera optical center and optical axis
        """
        cam_audit = {}
        for name, cam in sorted(camera_models.items()):
            R = cam.T_world_cam[:3, :3]
            t = cam.T_world_cam[:3, 3]
            det_r = float(np.linalg.det(R))
            ortho_err = float(np.max(np.abs(R @ R.T - np.eye(3))))
            cam_center = cam.camera_center.tolist()
            optical_axis = cam.optical_axis.tolist()
            up_vector = cam.up_vector.tolist()

            cam_audit[name] = {
                "sensor_width": cam.width,
                "sensor_height": cam.height,
                "fx": float(cam.fx),
                "fy": float(cam.fy),
                "cx": float(cam.cx),
                "cy": float(cam.cy),
                "distortion": cam.distortion.tolist(),
                "camera_center": [round(x, 4) for x in cam_center],
                "optical_axis": [round(x, 4) for x in optical_axis],
                "up_vector": [round(x, 4) for x in up_vector],
                "det_R": round(det_r, 6),
                "orthonormality_error": round(ortho_err, 8),
                "is_valid_so3": abs(det_r - 1.0) < 1e-5 and ortho_err < 1e-5,
            }
        return cam_audit

    def audit_all_stages(
        self,
        camera_models: Dict[str, PhoneCameraModel],
        raw_depth_maps: Dict[str, np.ndarray],
        aligned_depth_maps: Dict[str, np.ndarray],
        sparse_recon_dir: Optional[Path] = None,
        consistency_cloud: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        region_clouds: Optional[Dict[str, np.ndarray]] = None,
        region_meshes: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
        final_mesh_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Execute full audit across Stages A through H."""
        audit_results: Dict[str, Any] = {
            "session_id": self.session_dir.name,
            "camera_transform_audit": self.audit_camera_transforms(camera_models),
            "stages": {},
        }

        # Stage A: Individual unprojected camera clouds
        stage_a: Dict[str, Any] = {}
        all_cam_pts = []
        all_cam_cols = []

        names = sorted(list(camera_models.keys()))
        for idx, name in enumerate(names):
            if name not in raw_depth_maps:
                continue
            cam = camera_models[name]
            d_raw = raw_depth_maps[name]
            H, W = d_raw.shape[:2]
            if cam.width != W or cam.height != H:
                cam = cam.rescale(W, H)

            s = 8
            v, u = np.mgrid[0:H:s, 0:W:s]
            d_sub = d_raw[v, u]
            valid = (d_sub > 0.3) & (d_sub < 20.0)
            u_val, v_val, d_val = u[valid], v[valid], d_sub[valid]
            pts_cam = cam.unproject_image_to_world(np.column_stack([u_val, v_val]), d_val)

            # Export per-camera cloud
            col = self.CAMERA_PALETTE[idx % len(self.CAMERA_PALETTE)]
            cols_cam = np.tile(col, (len(pts_cam), 1)).astype(np.float32)

            ply_path = self.stages_dir / f"stage_A_{Path(name).stem}_raw_cloud.ply"
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_cam))
            pcd.colors = o3d.utility.Vector3dVector(cols_cam)
            o3d.io.write_point_cloud(str(ply_path), pcd)

            all_cam_pts.append(pts_cam)
            all_cam_cols.append(cols_cam)

            stage_a[name] = {
                "point_count": len(pts_cam),
                "bbox_min": [round(x, 3) for x in pts_cam.min(axis=0).tolist()] if len(pts_cam) else [],
                "bbox_max": [round(x, 3) for x in pts_cam.max(axis=0).tolist()] if len(pts_cam) else [],
                "ply_path": str(ply_path),
            }

        # Save fused camera-source-colored cloud (Stages A & D diagnostic)
        if all_cam_pts:
            src_pts = np.vstack(all_cam_pts)
            src_cols = np.vstack(all_cam_cols)
            pcd_all = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(src_pts))
            pcd_all.colors = o3d.utility.Vector3dVector(src_cols)
            src_ply = self.stages_dir / "camera_source_colored_cloud.ply"
            o3d.io.write_point_cloud(str(src_ply), pcd_all)
            stage_a["camera_source_colored_cloud"] = str(src_ply)

        audit_results["stages"]["Stage_A_Individual_Camera_Clouds"] = stage_a

        # Stage B: Depth Maps
        stage_b: Dict[str, Any] = {}
        for name, d in raw_depth_maps.items():
            valid_d = d[(d > 0.1) & (d < 50.0)]
            stage_b[name] = {
                "shape": list(d.shape),
                "min_depth": round(float(valid_d.min()), 3) if len(valid_d) else 0.0,
                "max_depth": round(float(valid_d.max()), 3) if len(valid_d) else 0.0,
                "median_depth": round(float(np.median(valid_d)), 3) if len(valid_d) else 0.0,
            }
        audit_results["stages"]["Stage_B_Camera_Depth_Maps"] = stage_b

        # Stage C: Sparse COLMAP
        if sparse_recon_dir and Path(sparse_recon_dir).exists():
            try:
                recon = pycolmap.Reconstruction(str(sparse_recon_dir))
                audit_results["stages"]["Stage_C_Sparse_COLMAP"] = {
                    "num_cameras": len(recon.cameras),
                    "num_images": len(recon.images),
                    "num_points3D": len(recon.points3D),
                    "mean_reprojection_error": round(float(np.mean([p.error for p in recon.points3D.values()])), 3) if recon.points3D else 0.0,
                }
            except Exception as e:
                audit_results["stages"]["Stage_C_Sparse_COLMAP"] = {"error": str(e)}

        # Stage D: Aligned point clouds
        stage_d: Dict[str, Any] = {}
        aligned_pts_list = []
        for name in names:
            if name not in aligned_depth_maps:
                continue
            cam = camera_models[name]
            d_al = aligned_depth_maps[name]
            H, W = d_al.shape[:2]
            if cam.width != W or cam.height != H:
                cam = cam.rescale(W, H)
            s = 8
            v, u = np.mgrid[0:H:s, 0:W:s]
            d_sub = d_al[v, u]
            valid = (d_sub > 0.3) & (d_sub < 20.0)
            u_val, v_val, d_val = u[valid], v[valid], d_sub[valid]
            pts = cam.unproject_image_to_world(np.column_stack([u_val, v_val]), d_val)
            aligned_pts_list.append(pts)
            stage_d[name] = {
                "aligned_points": len(pts),
                "bbox_min": [round(x, 3) for x in pts.min(axis=0).tolist()] if len(pts) else [],
                "bbox_max": [round(x, 3) for x in pts.max(axis=0).tolist()] if len(pts) else [],
            }
        audit_results["stages"]["Stage_D_SfM_Scale_Aligned_Clouds"] = stage_d

        # Stage E: Multi-view consistency cloud
        if consistency_cloud is not None:
            c_pts, c_cols = consistency_cloud
            e_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(c_pts))
            if len(c_cols) == len(c_pts):
                e_pcd.colors = o3d.utility.Vector3dVector(c_cols)
            ply_e = self.stages_dir / "stage_E_mvs_consistency_cloud.ply"
            o3d.io.write_point_cloud(str(ply_e), e_pcd)
            audit_results["stages"]["Stage_E_MVS_Consistency_Cloud"] = {
                "points": len(c_pts),
                "bbox_min": [round(x, 3) for x in c_pts.min(axis=0).tolist()] if len(c_pts) else [],
                "bbox_max": [round(x, 3) for x in c_pts.max(axis=0).tolist()] if len(c_pts) else [],
                "ply_path": str(ply_e),
            }

        # Stage F: Region point clouds
        if region_clouds:
            stage_f: Dict[str, Any] = {}
            for r_id, r_pts in region_clouds.items():
                pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(r_pts))
                ply_path = self.stages_dir / f"stage_F_region_{r_id}.ply"
                o3d.io.write_point_cloud(str(ply_path), pcd)
                stage_f[r_id] = {
                    "points": len(r_pts),
                    "bbox_min": [round(x, 3) for x in r_pts.min(axis=0).tolist()] if len(r_pts) else [],
                    "bbox_max": [round(x, 3) for x in r_pts.max(axis=0).tolist()] if len(r_pts) else [],
                    "ply_path": str(ply_path),
                }
            audit_results["stages"]["Stage_F_Region_Point_Clouds"] = stage_f

        # Stage G: Region meshes
        if region_meshes:
            stage_g: Dict[str, Any] = {}
            for r_id, (m_verts, m_faces) in region_meshes.items():
                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(m_verts.astype(np.float64))
                mesh.triangles = o3d.utility.Vector3iVector(m_faces.astype(np.int32))
                obj_path = self.stages_dir / f"stage_G_region_{r_id}.obj"
                o3d.io.write_triangle_mesh(str(obj_path), mesh)

                # Count connected components
                cluster_tri, cluster_sizes, _ = mesh.cluster_connected_triangles()
                stage_g[r_id] = {
                    "vertices": len(m_verts),
                    "triangles": len(m_faces),
                    "connected_components": int(len(cluster_sizes)),
                    "is_edge_manifold": bool(mesh.is_edge_manifold()),
                    "obj_path": str(obj_path),
                }
            audit_results["stages"]["Stage_G_Region_Meshes"] = stage_g

        # Stage H: Final merged mesh
        if final_mesh_path and Path(final_mesh_path).exists():
            f_mesh = o3d.io.read_triangle_mesh(str(final_mesh_path))
            cluster_tri, cluster_sizes, _ = f_mesh.cluster_connected_triangles()
            audit_results["stages"]["Stage_H_Final_Merged_Mesh"] = {
                "vertices": len(f_mesh.vertices),
                "triangles": len(f_mesh.triangles),
                "connected_components": int(len(cluster_sizes)),
                "is_edge_manifold": bool(f_mesh.is_edge_manifold()),
                "is_vertex_manifold": bool(f_mesh.is_vertex_manifold()),
                "path": str(final_mesh_path),
            }

        # Save audit report JSON
        out_json = self.session_dir / "reconstruction_stage_audit.json"
        with open(out_json, "w") as f_out:
            json.dump(audit_results, f_out, indent=2)

        return audit_results
