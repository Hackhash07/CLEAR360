"""
photo_reconstruction/pipeline.py — End-to-End Smartphone Photogrammetry Pipeline
================================================================================

Orchestrates the entire photogrammetry workflow:
  Ingest -> Quality Assessment -> EXIF Analysis -> SfM Registration -> Dense MVS ->
  Surface Meshing -> Metric Scale Calibration -> Quality Report Generation.
"""

from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional, Callable
import shutil
import time
import json
import cv2
import numpy as np
import open3d as o3d

from photo_reconstruction.image_loader import ImageLoader
from photo_reconstruction.exif_reader import ExifReader
from photo_reconstruction.image_quality import ImageQualityAssessor
from photo_reconstruction.sfm_engine import SfMEngine
from photo_reconstruction.dense_reconstruction import DenseReconstructor
from photo_reconstruction.mesh_builder import MeshBuilder
from photo_reconstruction.scale_calibrator import ScaleCalibrator
from photo_reconstruction.quality_report import QualityReportGenerator
from photo_reconstruction.session_manager import SessionManager


QUALITY_PRESETS: Dict[str, Dict[str, Any]] = {
    "draft": {
        "max_image_size": 800,
        "sift_max_features": 1000,
        "voxel_size": 0.06,
        "poisson_depth": 7,
        "cv_sift_features": 1000,
        "use_mvs_verification": False,
    },
    "quick": {  # Backwards compatibility alias for draft
        "max_image_size": 1000,
        "sift_max_features": 1500,
        "voxel_size": 0.05,
        "poisson_depth": 7,
        "cv_sift_features": 1500,
        "use_mvs_verification": False,
    },
    "standard": {
        "max_image_size": 1600,
        "sift_max_features": 4000,
        "voxel_size": 0.03,
        "poisson_depth": 8,
        "cv_sift_features": 2500,
        "use_mvs_verification": True,
    },
    "high": {
        "max_image_size": 2400,
        "sift_max_features": 8000,
        "voxel_size": 0.02,
        "poisson_depth": 9,
        "cv_sift_features": 4000,
        "use_mvs_verification": True,
    },
    "maximum": {
        "max_image_size": 3200,
        "sift_max_features": 12000,
        "voxel_size": 0.015,
        "poisson_depth": 10,
        "cv_sift_features": 6000,
        "use_mvs_verification": True,
    },
}


def validate_quality_preset(preset: str) -> str:
    """Validate and normalize quality preset name."""
    if not isinstance(preset, str):
        raise TypeError(f"quality_preset must be a string, got {type(preset).__name__}")
    p = preset.strip().lower()
    if p not in QUALITY_PRESETS:
        allowed = ", ".join(sorted(QUALITY_PRESETS.keys()))
        raise ValueError(f"Unknown quality preset '{preset}'. Supported presets: {allowed}")
    return p


def resolve_scene_profile(session_dir: Path) -> str:
    """Return an explicitly selected scene profile; arbitrary phone photos default to generic."""
    profile_path = Path(session_dir) / "scene_profile.json"
    try:
        with profile_path.open("r", encoding="utf-8") as profile_file:
            metadata = json.load(profile_file)
    except FileNotFoundError:
        return "generic"
    except (OSError, json.JSONDecodeError, TypeError):
        return "generic"

    if not isinstance(metadata, dict):
        return "generic"
    profile = str(metadata.get("profile", "generic")).strip().lower()
    if profile == "architectural_fountain":
        return profile
    # Unknown or unsupported profile names must never activate a specialized assembler.
    return "generic"


class PhoneReconstructionPipeline:
    """End-to-end smartphone 3D photogrammetry reconstruction pipeline."""

    def __init__(self, data_root: Optional[Path] = None, quality_preset: str = "standard"):
        self.session_manager = SessionManager(data_root)
        self.quality_preset = validate_quality_preset(quality_preset)
        self.scale_calibrator = ScaleCalibrator()

    def run_pipeline(
        self,
        image_paths: List[Path],
        session_id: Optional[str] = None,
        reference_distance_m: Optional[float] = None,
        progress_cb: Optional[Callable[[int, str], None]] = None,
        quality_preset: Optional[str] = None,
        mesh_method: Optional[str] = None,
        use_mvs: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Execute full photogrammetry pipeline on a collection of smartphone images.

        Parameters
        ----------
        image_paths : List[Path]
            Paths to uploaded images.
        session_id : Optional[str]
            Session folder identifier (auto-generated if None).
        reference_distance_m : Optional[float]
            Optional user-specified physical distance in metres for scale recovery.
            If None or omitted, the reconstruction remains in uncalibrated photogrammetry scale.
        progress_cb : Optional[Callable[[int, str], None]]
            Callback for real-time progress updates.
        quality_preset : Optional[str]
            Quality preset override ('draft', 'standard', 'high', 'maximum', or alias 'quick').
            If None, uses the pipeline's configured quality_preset.

        Returns
        -------
        Dict with reconstruction results, paths to assets, and quality report.
        """
        active_preset = validate_quality_preset(quality_preset) if quality_preset is not None else self.quality_preset
        self.quality_preset = active_preset
        preset_cfg = QUALITY_PRESETS[active_preset]
        t_start = time.perf_counter()

        def update_progress(pct: int, msg: str):
            if progress_cb:
                progress_cb(pct, msg)

        # 1. Initialize Session Directory
        update_progress(5, "Creating photogrammetry reconstruction session...")
        session_dir = self.session_manager.create_session(session_id)
        current_session_id = session_dir.name

        img_target_dir = session_dir / "original_images"
        exif_dir = session_dir / "exif_metadata"
        poses_dir = session_dir / "camera_poses"
        sparse_dir = session_dir / "sparse_cloud"
        dense_dir = session_dir / "dense_cloud"
        mesh_dir = session_dir / "mesh"

        # 2. Ingest & Validate Source Images
        update_progress(10, f"Validating and ingesting {len(image_paths)} images...")
        copied_paths = []
        image_arrays = []
        camera_infos = []
        quality_evals = []

        for idx, p in enumerate(image_paths):
            src = Path(p)
            if not src.is_file():
                continue
            dest = img_target_dir / src.name
            if dest.exists() and dest.resolve() != src.resolve():
                dest = img_target_dir / f"{src.stem}_{idx:03d}{src.suffix}"
            if not dest.exists() or dest.resolve() != src.resolve():
                shutil.copy2(src, dest)
            copied_paths.append(dest)

            # Load and analyze
            rgb, meta = ImageLoader.load_image(dest)
            image_arrays.append(rgb)

            # Quality Assessment
            q_res = ImageQualityAssessor.evaluate(rgb)
            quality_evals.append({"file": dest.name, **q_res})

            # EXIF Extraction
            c_info = ExifReader.get_camera_info(dest, meta["width"], meta["height"])
            camera_infos.append({"file": dest.name, **c_info})

        # Save metadata JSON
        with open(exif_dir / "camera_infos.json", "w") as f:
            json.dump(camera_infos, f, indent=2)
        with open(exif_dir / "quality_assessments.json", "w") as f:
            json.dump(quality_evals, f, indent=2)

        # Check for mixed lenses
        is_mixed, mixed_warnings = ExifReader.check_mixed_lenses(camera_infos)

        if len(copied_paths) < 2:
            update_progress(100, "Reconstruction failed: fewer than 2 valid images.")
            report = QualityReportGenerator.generate_report(
                session_id=current_session_id,
                uploaded_count=len(image_paths),
                registered_count=0,
                failed_images=[Path(p).name for p in image_paths],
                sparse_point_count=0,
                dense_point_count=0,
                mesh_triangle_count=0,
                reprojection_error_px=0.0,
                processing_time_s=time.perf_counter() - t_start,
                scale_info=self.scale_calibrator.get_calibration_info(),
                notes="Failed: At least 2 valid photographs required.",
                quality_preset=active_preset,
            )
            QualityReportGenerator.save_report_json(report, session_dir / "report.json")
            return {"success": False, "report": report, "session_dir": session_dir, "quality_preset": active_preset}

        # 3. Structure from Motion (SfM)
        update_progress(25, f"Running camera pose estimation & feature matching ({active_preset} preset)...")
        sfm = SfMEngine(work_dir=session_dir / "sfm_work", quality_preset=active_preset)
        sfm_res = sfm.run_reconstruction(img_target_dir, progress_cb=update_progress)

        if not sfm_res["success"] or sfm_res["registered_count"] < 2:
            update_progress(100, "Reconstruction incomplete: could not register camera poses.")
            report = QualityReportGenerator.generate_report(
                session_id=current_session_id,
                uploaded_count=len(copied_paths),
                registered_count=sfm_res["registered_count"],
                failed_images=sfm_res["failed_images"],
                sparse_point_count=len(sfm_res["sparse_points"]),
                dense_point_count=0,
                mesh_triangle_count=0,
                reprojection_error_px=sfm_res["mean_reprojection_error"],
                processing_time_s=time.perf_counter() - t_start,
                scale_info=self.scale_calibrator.get_calibration_info(),
                notes=sfm_res["message"],
                quality_preset=active_preset,
            )
            QualityReportGenerator.save_report_json(report, session_dir / "report.json")
            return {"success": False, "report": report, "session_dir": session_dir, "quality_preset": active_preset}

        # Save camera poses
        camera_poses = sfm_res["camera_poses"]
        for img_name, pose_mat in camera_poses.items():
            np.save(poses_dir / f"{Path(img_name).stem}_pose.npy", pose_mat)

        # Save sparse cloud
        sparse_pts = sfm_res["sparse_points"]
        sparse_cols = sfm_res["sparse_colors"]
        DenseReconstructor.save_point_cloud_ply(sparse_pts, sparse_cols, sparse_dir / "sparse_cloud.ply")

        # 4. Dense Point Cloud Reconstruction
        update_progress(60, f"Generating dense 3D point cloud ({preset_cfg['voxel_size']}m voxel)...")
        from photo_reconstruction.depth_helper import IsolatedDepthEstimator
        reg_paths = [img_target_dir / name for name in camera_poses.keys()]
        raw_depth_maps = IsolatedDepthEstimator.estimate_depths_for_images(
            reg_paths, session_dir / "depth_cache"
        )
        # Strict MVS accepts only views whose affine depth calibration passed a
        # genuine held-out check. Draft mode may still use raw monocular depth as
        # a clearly unverified preview, but it is never used for strict fusion.
        mvs_flag = use_mvs if use_mvs is not None else preset_cfg.get("use_mvs_verification", False)
        depth_maps = dict(raw_depth_maps)
        texture_depth_maps = None
        camera_models = sfm_res.get("camera_models", {})
        sparse_tracks = sfm_res.get("sparse_tracks", {})
        trusted_camera_models = dict(camera_models)

        if raw_depth_maps and camera_models and sparse_tracks:
            update_progress(65, "Calibrating per-view depth against held-out SfM anchors...")
            from photo_reconstruction.depth_aligner import SfMDepthAligner
            aligner = SfMDepthAligner(mode="regularized")
            align_res = aligner.align_depth_maps(
                depth_maps=raw_depth_maps,
                camera_models=camera_models,
                sparse_tracks=sparse_tracks,
            )
            accepted_depth_maps = dict(align_res.aligned_depth_maps)
            accepted_names = sorted(accepted_depth_maps.keys())
            texture_depth_maps = accepted_depth_maps or None

            # In strict MVS mode, unvalidated monocular views are excluded from
            # both cross-view fusion and visibility/occlusion decisions.
            if mvs_flag:
                depth_maps = accepted_depth_maps
                trusted_camera_models = {
                    name: camera_models[name]
                    for name in accepted_names
                    if name in camera_models
                }
            else:
                # Draft is an explicitly approximate preview path. Use accepted
                # maps where available; if none pass, retain raw maps for preview.
                depth_maps = accepted_depth_maps or dict(raw_depth_maps)

            align_diag = {
                "mode": align_res.mode,
                "global_scale": float(align_res.global_scale),
                "total_training_anchors": int(align_res.total_anchors),
                "accepted_views": accepted_names,
                "rejected_views": sorted(
                    name for name, stats in align_res.view_stats.items() if not stats.success
                ),
                "limits": {
                    "median_relative_error": aligner.max_val_median_rel,
                    "p90_relative_error": aligner.max_val_p90_rel,
                },
                "views": {
                    name: {
                        "anchors": int(s.num_anchors),
                        "scale": float(s.median_scale),
                        "offset": float(s.depth_offset),
                        "mad_scale": float(s.mad_scale),
                        "raw_residual_median": float(s.raw_residual_median),
                        "aligned_residual_median": float(s.aligned_residual_median),
                        "validation_median_relative_error": float(s.validation_median_relative_error),
                        "validation_p90_relative_error": float(s.validation_p90_relative_error),
                        "residual_reduction_pct": float(s.residual_reduction_pct),
                        "success": bool(s.success),
                        "status": s.status_message,
                    }
                    for name, s in align_res.view_stats.items()
                },
            }
            with open(session_dir / "depth_alignment_report.json", "w", encoding="utf-8") as f_diag:
                json.dump(align_diag, f_diag, indent=2)

            if mvs_flag and len(depth_maps) < 2:
                rejected = {
                    name: stats.status_message
                    for name, stats in align_res.view_stats.items()
                    if not stats.success
                }
                raise RuntimeError(
                    "Strict multi-view reconstruction stopped: fewer than two camera views "
                    "passed held-out depth calibration. The per-view diagnostics were saved. "
                    "No unverified depth sheets were fused. "
                    f"Accepted views: {accepted_names}. Rejected views: {rejected}"
                )

        if mvs_flag and not (raw_depth_maps and camera_models and sparse_tracks):
            reason = (
                f"depth maps={len(raw_depth_maps)}, camera models={len(camera_models)}, "
                f"tracked views={len(sparse_tracks)}"
            )
            with open(session_dir / "depth_alignment_report.json", "w", encoding="utf-8") as f_diag:
                json.dump({
                    "status": "not_run",
                    "reason": "Strict MVS requires depth maps, SfM camera models, and sparse feature tracks.",
                    "available_inputs": reason,
                }, f_diag, indent=2)
            raise RuntimeError(
                "Strict multi-view reconstruction stopped: depth calibration could not run. "
                f"Available inputs: {reason}. No unverified monocular depth was fused."
            )

        if mvs_flag and len(depth_maps) < 2:
            raise RuntimeError(
                "Strict multi-view reconstruction stopped: fewer than two depth maps are "
                "available for validated geometric fusion. See depth_alignment_report.json; "
                "no unverified monocular depth was substituted."
            )

        dense_engine = DenseReconstructor(voxel_size=preset_cfg["voxel_size"])
        dense_camera_models = trusted_camera_models if mvs_flag else camera_models
        dense_pts, dense_cols, dense_norms = dense_engine.reconstruct_dense(
            image_dir=img_target_dir,
            camera_poses=camera_poses,
            depth_maps=depth_maps if depth_maps else None,
            sparse_points=sparse_pts,
            sparse_colors=sparse_cols,
            use_mvs_verification=mvs_flag,
            camera_models=dense_camera_models,
            session_dir=session_dir,
        )

        # Auto-Upright Ground Plane Alignment (Gravity Alignment)
        update_progress(75, "Auto-aligning ground plane and upright orientation...")
        dense_pts, dense_norms, camera_poses, sparse_pts, T_align = DenseReconstructor.align_ground_upright(
            points=dense_pts,
            normals=dense_norms,
            camera_poses=camera_poses,
            sparse_points=sparse_pts,
        )
        if camera_models:
            for c_name in camera_models:
                if c_name in camera_poses:
                    camera_models[c_name].T_world_cam = camera_poses[c_name]
                    camera_models[c_name].T_cam_world = np.linalg.inv(camera_poses[c_name])

        # Re-save aligned camera poses and sparse cloud in unified upright frame
        for img_name, pose_mat in camera_poses.items():
            np.save(poses_dir / f"{Path(img_name).stem}_pose.npy", pose_mat)
        if sparse_pts is not None and len(sparse_pts) > 0:
            DenseReconstructor.save_point_cloud_ply(sparse_pts, sparse_cols, sparse_dir / "sparse_cloud.ply")

        dense_ply = dense_dir / "dense_cloud.ply"
        DenseReconstructor.save_point_cloud_ply(dense_pts, dense_cols, dense_ply, normals=dense_norms)

        # 5. 3D Surface Meshing (M7.3.3 / M7.3.4 Modular Scene Assembly)
        img_dict = {}
        for p in copied_paths:
            if p.name in camera_models:
                bgr = cv2.imread(str(p))
                if bgr is not None:
                    img_dict[p.name] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        mesh_ply = mesh_dir / "scene_mesh.ply"
        mesh_obj = mesh_dir / "scene_mesh.obj"
        textured_ok = False

        # Generic smartphone sessions must not use the legacy fountain-specific
        # RegionSceneAssembler. That assembler segments fixed fountain coordinates
        # (wall, basin, sculpture, pavement), which corrupts arbitrary rooms/objects.
        # Keep it available only behind an explicit opt-in for a future, separately
        # selected architectural-fountain profile. The default phone workflow uses
        # the generic geometry produced by SfM/MVS and MeshBuilder.
        scene_profile = resolve_scene_profile(session_dir)

        if scene_profile == "architectural_fountain" and camera_models and img_dict:
            try:
                from photo_reconstruction.scene_assembly import RegionSceneAssembler
                update_progress(80, "Building explicitly requested fountain-specific scene assembly...")
                assembler = RegionSceneAssembler(session_dir)
                scene_regions, final_obj, final_ply, _ = assembler.assemble_scene_graph(
                    points=dense_pts,
                    colors=dense_cols,
                    normals=dense_norms,
                    camera_models=camera_models,
                    images_rgb=img_dict,
                    depth_maps=depth_maps if depth_maps else None,
                    sparse_points=sparse_pts,
                )
                mesh_unified = o3d.io.read_triangle_mesh(str(final_ply))
                mesh_v = np.asarray(mesh_unified.vertices, dtype=np.float32)
                mesh_f = np.asarray(mesh_unified.triangles, dtype=np.uint32)
                mesh_c = (
                    np.asarray(mesh_unified.vertex_colors, dtype=np.float32)
                    if len(mesh_unified.vertex_colors) > 0
                    else np.ones((len(mesh_v), 3), dtype=np.float32) * 0.7
                )
                mesh_n = np.asarray(mesh_unified.vertex_normals, dtype=np.float32)
                if len(mesh_v) == 0 or len(mesh_f) == 0:
                    raise RuntimeError("Fountain assembler returned an empty mesh")
                textured_ok = True
            except Exception as e:
                # Specialized mode failures are surfaced; do not silently pretend
                # that a generic mesh is an equivalent fountain reconstruction.
                raise RuntimeError(f"Architectural fountain assembly failed: {e}") from e

        if not textured_ok:
            chosen_method = mesh_method if mesh_method is not None else preset_cfg.get("mesh_method", "poisson")
            update_progress(80, f"Building 3D surface mesh using {chosen_method.upper()} meshing algorithm...")
            mesh_builder = MeshBuilder(method=chosen_method, poisson_depth=preset_cfg["poisson_depth"])
            registered_camera_centers = (
                np.asarray([cam.camera_center for cam in camera_models.values()], dtype=np.float64)
                if camera_models else None
            )
            mesh_v, mesh_f, mesh_c, mesh_n = mesh_builder.build_mesh(
                dense_pts,
                dense_cols,
                dense_norms,
                camera_centers=registered_camera_centers,
            )

        # 6. Metric Scale Recovery (if reference distance provided)
        if reference_distance_m is not None and reference_distance_m > 0 and len(mesh_v) >= 2:
            update_progress(90, f"Applying metric scale calibration ({reference_distance_m:.2f} m)...")
            pt_min = mesh_v.min(axis=0)
            pt_max = mesh_v.max(axis=0)
            self.scale_calibrator.compute_scale(pt_min, pt_max, reference_distance_m, reference_label="Bounding Box Extent")
            mesh_v, camera_poses = self.scale_calibrator.apply_scale(mesh_v, camera_poses)
            dense_pts, _ = self.scale_calibrator.apply_scale(dense_pts)
            if camera_models:
                for c_name in camera_models:
                    if c_name in camera_poses:
                        camera_models[c_name].T_world_cam = camera_poses[c_name]
                        camera_models[c_name].T_cam_world = np.linalg.inv(camera_poses[c_name])

        # 7. Visibility-Aware Photogrammetric Texturing (if not already textured by modular assembler)
        if not textured_ok and img_dict and len(mesh_f) > 0 and camera_models:
            update_progress(92, "Synthesizing visibility-aware UV texture mapping and packed texture atlas...")
            from photo_reconstruction.texture_engine import VisibilityAwareTextureEngine
            tex_engine = VisibilityAwareTextureEngine(atlas_size=4096)
            try:
                texture_camera_models = (
                    {name: camera_models[name] for name in texture_depth_maps if name in camera_models}
                    if texture_depth_maps else (trusted_camera_models if mvs_flag else camera_models)
                )
                texture_images = {
                    name: img_dict[name]
                    for name in texture_camera_models
                    if name in img_dict
                }
                tex_v, tex_f, tex_uv, tex_atlas, tex_rep = tex_engine.texture_mesh(
                    vertices=mesh_v,
                    faces=mesh_f,
                    camera_models=texture_camera_models,
                    images=texture_images,
                    depth_maps=texture_depth_maps,
                    out_dir=mesh_dir,
                    base_name="scene_mesh",
                )
                with open(session_dir / "texturing_report.json", "w") as f_tex:
                    json.dump({
                        "total_faces": tex_rep.total_faces,
                        "textured_faces": tex_rep.textured_faces,
                        "coverage_percentage": tex_rep.coverage_percentage,
                        "atlas_width": tex_rep.atlas_width,
                        "atlas_height": tex_rep.atlas_height,
                        "camera_face_distribution": tex_rep.camera_face_distribution,
                    }, f_tex, indent=2)
                textured_ok = True
            except Exception as e:
                print(f"Warning: Visibility texturing fallback: {e}")

        if not textured_ok:
            MeshBuilder.save_mesh_obj(mesh_v, mesh_f, mesh_c, mesh_obj)

        # Always provide the documented vertex-color PLY alongside the OBJ.
        # The textured OBJ carries UVs/atlas data, while PLY remains a useful
        # geometry + vertex-color artifact for viewers and regression checks.
        if not mesh_ply.exists():
            MeshBuilder.save_mesh_ply(mesh_v, mesh_f, mesh_c, mesh_ply, normals=mesh_n)

        # 7. Quality Report Compilation
        update_progress(95, "Compiling reconstruction quality report...")
        t_total = time.perf_counter() - t_start

        notes_text = (
            f"Successfully reconstructed {len(camera_poses)}/{len(copied_paths)} photographs. "
            f"Sparse: {len(sparse_pts):,} pts, Dense: {len(dense_pts):,} pts, Mesh: {len(mesh_f):,} triangles."
        )
        if is_mixed:
            notes_text += " Note: mixed camera focal lengths detected."

        report = QualityReportGenerator.generate_report(
            session_id=current_session_id,
            uploaded_count=len(copied_paths),
            registered_count=len(camera_poses),
            failed_images=sfm_res["failed_images"],
            sparse_point_count=len(sparse_pts),
            dense_point_count=len(dense_pts),
            mesh_triangle_count=len(mesh_f),
            reprojection_error_px=sfm_res["mean_reprojection_error"],
            processing_time_s=t_total,
            scale_info=self.scale_calibrator.get_calibration_info(),
            image_quality_summary={
                "mean_quality_score": round(float(np.mean([q["quality_score"] for q in quality_evals])), 1) if quality_evals else 0.0,
                "evaluations": quality_evals,
                "mixed_lenses_detected": is_mixed,
                "warnings": mixed_warnings,
            },
            notes=notes_text,
            quality_preset=active_preset,
        )
        QualityReportGenerator.save_report_json(report, session_dir / "report.json")

        # 8. M7.3.7 Global Camera Pose Graph, Reprojection Validation & Scene Manifest
        try:
            from photo_reconstruction.m7_3_7_pose_graph import (
                GlobalCameraPoseGraphAuditor,
                CameraReprojectionValidator,
                GlobalSceneManifestAuthor,
            )
            auditor = GlobalCameraPoseGraphAuditor(session_dir)
            cam_audit = auditor.audit_cameras()
            pose_graph = auditor.build_pose_graph()
            validator = CameraReprojectionValidator(session_dir)
            reproj_stats = validator.validate_and_generate_overlays(session_dir / "reprojection_overlays")

            scene_graph_file = session_dir / "scene_graph.json"
            regions_dict = {}
            if scene_graph_file.exists():
                with open(scene_graph_file) as f_sg:
                    sg_data = json.load(f_sg)
                    regions_dict = sg_data.get("regions", {})

            GlobalSceneManifestAuthor.create_manifest(
                session_id=current_session_id,
                camera_audit=cam_audit,
                pose_graph=pose_graph,
                reprojection_stats=reproj_stats,
                scene_regions=regions_dict,
                output_file=session_dir / "global_scene_manifest.json",
            )
        except Exception as e:
            print(f"Notice: M7.3.7 manifest generation: {e}")

        update_progress(100, f"Photogrammetry complete ({t_total:.1f} s). Model saved to {session_dir.name}.")

        return {
            "success": True,
            "session_id": current_session_id,
            "session_dir": session_dir,
            "mesh_ply": mesh_ply,
            "mesh_obj": mesh_obj,
            "dense_ply": dense_ply,
            "sparse_pts_count": len(sparse_pts),
            "dense_pts_count": len(dense_pts),
            "mesh_triangles_count": len(mesh_f),
            "camera_poses": camera_poses,
            "quality_preset": active_preset,
            "report": report,
        }


def main():
    """Terminal CLI entrypoint for CLEAR360 Phone Photogrammetry Pipeline."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="CLEAR360 CLI: Multi-View Smartphone Photogrammetry & 3D Reconstruction Pipeline"
    )
    parser.add_argument(
        "images",
        nargs="+",
        help="Directory containing images or individual image file paths to reconstruct.",
    )
    parser.add_argument(
        "--preset",
        "-p",
        choices=["draft", "quick", "standard", "high", "maximum"],
        default="standard",
        help="Quality preset (default: standard).",
    )
    parser.add_argument(
        "--session",
        "-s",
        default=None,
        help="Custom session identifier for output folder.",
    )
    parser.add_argument(
        "--distance",
        "-d",
        type=float,
        default=None,
        help="Optional known physical reference distance in metres for metric scale calibration.",
    )
    parser.add_argument(
        "--mesh-method",
        "-m",
        choices=["bpa", "poisson", "hybrid"],
        default=None,
        help="Surface meshing method: 'bpa' (multi-radius ball-pivoting), 'poisson', or 'hybrid'.",
    )
    parser.add_argument(
        "--mvs",
        action="store_true",
        help="Enable multi-view forward-backward geometric consistency verification.",
    )

    args = parser.parse_args()

    # Collect image paths
    raw_inputs = [Path(p).expanduser().resolve() for p in args.images]
    image_paths = []

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".heic"}
    for item in raw_inputs:
        if item.is_dir():
            for f in sorted(item.iterdir()):
                if f.suffix.lower() in valid_exts:
                    image_paths.append(f)
        elif item.is_file() and item.suffix.lower() in valid_exts:
            image_paths.append(item)

    if len(image_paths) < 2:
        print(f"❌ Error: At least 2 photographic views are required for 3D SfM. Found {len(image_paths)}.", file=sys.stderr)
        sys.exit(1)

    print(f"================================================================")
    print(f" CLEAR360 PHOTOGRAMMETRY & 3D RECONSTRUCTION ENGINE (CLI)")
    print(f"================================================================")
    print(f"  Images Found    : {len(image_paths)}")
    print(f"  Quality Preset  : {args.preset.upper()}")
    print(f"  Session ID      : {args.session or '(auto-generated)'}")
    print(f"  Known Distance  : {f'{args.distance:.2f} m' if args.distance else 'Uncalibrated photogrammetry scale'}")
    print(f"================================================================\n")

    def cli_progress(pct: int, msg: str):
        bar_len = 30
        filled = int(bar_len * pct / 100)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"\r[{bar}] {pct:3d}% — {msg:<50}", end="", flush=True)
        if pct >= 100:
            print()

    pipeline = PhoneReconstructionPipeline(quality_preset=args.preset)
    result = pipeline.run_pipeline(
        image_paths=image_paths,
        session_id=args.session,
        reference_distance_m=args.distance,
        progress_cb=cli_progress,
        mesh_method=args.mesh_method,
        use_mvs=args.mvs if args.mvs else None,
    )

    print("\n----------------------------------------------------------------")
    if result["success"]:
        rep = result["report"]
        print("✅ RECONSTRUCTION SUCCESSFUL")
        print(f"  Session Directory   : {result['session_dir']}")
        print(f"  Reconstruction Grade: {rep.get('reconstruction_grade', 'N/A')}")
        print(f"  Registered Cameras  : {rep['image_counts']['registered']} / {rep['image_counts']['uploaded']}")
        print(f"  Sparse Points       : {rep['geometric_metrics']['sparse_points']:,}")
        print(f"  Dense Points        : {rep['geometric_metrics']['dense_points']:,}")
        print(f"  Mesh Triangles      : {rep['geometric_metrics']['mesh_triangles']:,}")
        print(f"  Reprojection Error  : {rep['geometric_metrics']['mean_reprojection_error_px']:.3f} px")
        print(f"  Mesh OBJ Path       : {result['mesh_obj']}")
        print(f"  Mesh PLY Path       : {result['mesh_ply']}")
    else:
        print("❌ RECONSTRUCTION FAILED")
        print(f"  Reason: {result.get('error', 'Unknown failure')}")
        sys.exit(1)
    print("----------------------------------------------------------------\n")


if __name__ == "__main__":
    main()

