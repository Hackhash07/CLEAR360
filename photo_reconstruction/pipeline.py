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
import numpy as np

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
    },
    "quick": {  # Backwards compatibility alias for draft
        "max_image_size": 1000,
        "sift_max_features": 1500,
        "voxel_size": 0.05,
        "poisson_depth": 7,
        "cv_sift_features": 1500,
    },
    "standard": {
        "max_image_size": 1600,
        "sift_max_features": 4000,
        "voxel_size": 0.03,
        "poisson_depth": 8,
        "cv_sift_features": 2500,
    },
    "high": {
        "max_image_size": 2400,
        "sift_max_features": 8000,
        "voxel_size": 0.02,
        "poisson_depth": 9,
        "cv_sift_features": 4000,
    },
    "maximum": {
        "max_image_size": 3200,
        "sift_max_features": 12000,
        "voxel_size": 0.015,
        "poisson_depth": 10,
        "cv_sift_features": 6000,
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
        update_progress(65, f"Generating dense 3D point cloud with surface normals ({preset_cfg['voxel_size']}m voxel)...")
        from photo_reconstruction.depth_helper import IsolatedDepthEstimator
        reg_paths = [img_target_dir / name for name in camera_poses.keys()]
        depth_maps = IsolatedDepthEstimator.estimate_depths_for_images(reg_paths, session_dir / "depth_cache")

        dense_engine = DenseReconstructor(voxel_size=preset_cfg["voxel_size"])
        dense_pts, dense_cols, dense_norms = dense_engine.reconstruct_dense(
            image_dir=img_target_dir,
            camera_poses=camera_poses,
            depth_maps=depth_maps if depth_maps else None,
            sparse_points=sparse_pts,
            sparse_colors=sparse_cols,
        )
        dense_ply = dense_dir / "dense_cloud.ply"
        DenseReconstructor.save_point_cloud_ply(dense_pts, dense_cols, dense_ply, normals=dense_norms)

        # 5. 3D Surface Meshing
        update_progress(80, f"Building continuous 3D surface mesh (Poisson depth {preset_cfg['poisson_depth']})...")
        mesh_builder = MeshBuilder(method="poisson", poisson_depth=preset_cfg["poisson_depth"])
        mesh_v, mesh_f, mesh_c, mesh_n = mesh_builder.build_mesh(dense_pts, dense_cols, dense_norms)

        # 6. Metric Scale Recovery (if reference distance provided)
        if reference_distance_m is not None and reference_distance_m > 0 and len(mesh_v) >= 2:
            update_progress(90, f"Applying metric scale calibration ({reference_distance_m:.2f} m)...")
            # Use extreme bounding box extent as baseline reference if specific points not specified
            pt_min = mesh_v.min(axis=0)
            pt_max = mesh_v.max(axis=0)
            self.scale_calibrator.compute_scale(pt_min, pt_max, reference_distance_m, reference_label="Bounding Box Extent")
            mesh_v, camera_poses = self.scale_calibrator.apply_scale(mesh_v, camera_poses)
            dense_pts, _ = self.scale_calibrator.apply_scale(dense_pts)

        # Save Meshes (both PLY and OBJ)
        mesh_ply = mesh_dir / "scene_mesh.ply"
        mesh_obj = mesh_dir / "scene_mesh.obj"
        MeshBuilder.save_mesh_ply(mesh_v, mesh_f, mesh_c, mesh_ply, normals=mesh_n)
        MeshBuilder.save_mesh_obj(mesh_v, mesh_f, mesh_c, mesh_obj)

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
