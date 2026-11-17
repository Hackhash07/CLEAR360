"""OpenMVS command-line backend for photo-to-3D reconstruction.

This backend is optional. The existing CLEAR360 Python reconstruction pipeline
remains the default; OpenMVS can be selected when its binaries are installed.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from photo_reconstruction.openmvs_assets import copy_mtl_diffuse_maps
from photo_reconstruction.openmvs_quality import (
    densify_common_options,
    densify_fusion_options,
    reconstruct_mesh_options,
    refine_mesh_options,
    texture_mesh_options,
)
from photo_reconstruction.camera_calibration import resolve_colmap_camera_mode
from photo_reconstruction.openmvs_tools import (
    configured_openmvs_root,
    find_openmvs_executable,
)
from photo_reconstruction.scene_orientation import (
    align_glb_file,
    align_obj_file,
    align_open3d_geometry_file,
    estimate_world_up,
    obj_median_center,
    rotation_from_to,
)


ProgressCallback = Optional[Callable[[int, str], None]]


def _colmap_registration_summary(
    images_txt: Path,
    image_paths: list[Path],
    points3d_txt: Path,
) -> dict:
    """Summarize registered uploads and sparse reconstruction residuals."""
    from photo_reconstruction.coverage_diagnostics import (
        parse_images_txt,
        parse_points3d_tracks,
    )

    registered = parse_images_txt(images_txt)
    registered_names = {Path(image.name).name for image in registered}
    uploaded_names = {Path(path).name for path in image_paths}
    registered_upload_names = registered_names.intersection(uploaded_names)
    failed_names = sorted(uploaded_names - registered_upload_names)
    image_id_to_name = {
        image.image_id: Path(image.name).name for image in registered
        if Path(image.name).name in uploaded_names
    }
    registered_ids = set(image_id_to_name)

    reprojection_errors: list[float] = []
    sparse_point_count = 0
    per_image_tracks: dict[int, int] = {}
    pairwise_tracks: dict[tuple[int, int], int] = {}
    if points3d_txt.is_file():
        try:
            sparse_point_count, per_image_tracks, pairwise_tracks = parse_points3d_tracks(
                points3d_txt
            )
        except (OSError, ValueError):
            sparse_point_count = 0

        for raw_line in points3d_txt.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            try:
                float(parts[1])
                float(parts[2])
                float(parts[3])
                error = float(parts[7])
            except ValueError:
                continue
            if np.isfinite(error):
                reprojection_errors.append(error)

    from photo_reconstruction.reconstruction_qa import summarize_camera_overlap

    pairwise_named = {
        (image_id_to_name[image_id_a], image_id_to_name[image_id_b]): int(shared_count)
        for (image_id_a, image_id_b), shared_count in pairwise_tracks.items()
        if image_id_a in registered_ids and image_id_b in registered_ids
    }
    per_camera_named = {
        image_id_to_name[image_id]: int(per_image_tracks.get(image_id, 0))
        for image_id in registered_ids
    }
    camera_overlap = summarize_camera_overlap(
        sorted(image_id_to_name.values()),
        pairwise_named,
        per_camera_named,
        minimum_shared_tracks=15,
    )
    camera_overlap_warnings = camera_overlap["warnings"]

    mean_error = (
        float(np.mean(reprojection_errors)) if reprojection_errors else None
    )
    return {
        "uploaded_count": len(uploaded_names),
        "registered_count": len(registered_upload_names),
        "failed_images": failed_names,
        "sparse_point_count": sparse_point_count,
        "mean_reprojection_error_px": round(mean_error, 4) if mean_error is not None else None,
        "registration_completeness_percentage": round(
            100.0 * len(registered_upload_names) / max(1, len(uploaded_names)), 1
        ),
        "camera_overlap": camera_overlap,
        "warnings": camera_overlap_warnings,
    }


class OpenMVSBackend:
    """Run COLMAP + OpenMVS on a set of photographs and export a textured GLB."""

    def __init__(self, openmvs_dir: Optional[Path] = None, colmap_bin: Optional[str] = None):
        self.openmvs_dir = (
            Path(openmvs_dir).expanduser() if openmvs_dir else configured_openmvs_root()
        )
        self.colmap_bin = colmap_bin or shutil.which("colmap") or "colmap"

    def _binary(self, name: str) -> str:
        found = find_openmvs_executable(name, openmvs_dir=self.openmvs_dir)
        if found:
            return found
        root_hint = str(self.openmvs_dir) if self.openmvs_dir else (
            "~/Applications/OpenMVS-2.4.0"
        )
        raise FileNotFoundError(
            f"Cannot find executable {name}. Checked {root_hint}, its bin/build subfolders, "
            "and PATH. Set CLEAR360_OPENMVS_DIR to the OpenMVS install folder."
        )

    @staticmethod
    def _run(command: list[str], cwd: Path, log_file: Path,
             progress: ProgressCallback, percent: int, message: str) -> None:
        if progress:
            progress(percent, message)
        with log_file.open("a", encoding="utf-8") as log:
            log.write("\n$ " + " ".join(command) + "\n")
            log.flush()
            proc = subprocess.Popen(
                command, cwd=str(cwd), stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                log.write(line)
                log.flush()
            code = proc.wait()
        if code != 0:
            raise RuntimeError(
                f"{Path(command[0]).name} exited with code {code}. See {log_file}."
            )

    def run(self, image_paths: list[Path], session_dir: Path,
            quality_preset: str = "standard",
            progress_cb: ProgressCallback = None,
            photo_set_report: Optional[dict] = None) -> dict:
        """Reconstruct uploaded images and return session output paths."""
        if len(image_paths) < 6:
            raise ValueError(
                "OpenMVS reconstruction requires at least 6 overlapping photographs. "
                "Use a walk-around photo sequence with substantial shared scene content."
            )

        session_dir = Path(session_dir)
        images_dir = session_dir / "original_images"
        images_dir.mkdir(parents=True, exist_ok=True)

        colmap_candidate = Path(self.colmap_bin).expanduser()
        if colmap_candidate.is_file() and os.access(colmap_candidate, os.X_OK):
            colmap = str(colmap_candidate)
        else:
            colmap = shutil.which(self.colmap_bin)
        if not colmap:
            raise FileNotFoundError("COLMAP executable not found. Activate its environment or set CLEAR360_COLMAP_BIN.")

        densify = self._binary("DensifyPointCloud")
        reconstruct = self._binary("ReconstructMesh")
        refine = self._binary("RefineMesh")
        texture = self._binary("TextureMesh")
        interface_colmap = self._binary("InterfaceCOLMAP")

        # Audit only after executable checks so dependency errors remain actionable;
        # no OpenMVS work/stage folders are created before this gate.
        from photo_reconstruction.photo_set_auditor import PhotoSetQualityAuditor

        upload_preflight_snapshot = photo_set_report
        photo_set_report = PhotoSetQualityAuditor.audit_paths(image_paths)
        if upload_preflight_snapshot is not None:
            photo_set_report = PhotoSetQualityAuditor.merge_source_camera_metadata(
                photo_set_report,
                upload_preflight_snapshot,
            )
            photo_set_report["upload_preflight_snapshot"] = {
                "status": upload_preflight_snapshot.get("status"),
                "warnings": upload_preflight_snapshot.get("warnings", []),
                "camera": upload_preflight_snapshot.get("camera", {}),
            }
        if not photo_set_report.get("can_reconstruct", False):
            raise ValueError(
                "Photo-set preflight blocked OpenMVS reconstruction: "
                + "; ".join(photo_set_report.get("warnings", []))
            )

        work = session_dir / "openmvs"
        sparse = work / "sparse"
        dense = work / "dense"
        sparse.mkdir(parents=True, exist_ok=True)
        dense.mkdir(parents=True, exist_ok=True)
        log_file = session_dir / "openmvs_pipeline.log"
        log_file.write_text("CLEAR360 OpenMVS pipeline\n", encoding="utf-8")

        camera_mode = resolve_colmap_camera_mode(photo_set_report)

        preset = quality_preset.lower()
        if preset in {"draft", "quick"}:
            feature_size, max_image_size, max_resolution, resolution_level = 1000, 1000, 800, 1
        elif preset == "high":
            feature_size, max_image_size, max_resolution, resolution_level = 6000, 2400, 2400, 0
        elif preset == "maximum":
            feature_size, max_image_size, max_resolution, resolution_level = 8000, 3200, 3200, 0
        else:
            # Standard keeps the full 1600px undistorted images for dense stereo.
            # The previous level=1 reduced them to ~800px and discarded fine detail.
            feature_size, max_image_size, max_resolution, resolution_level = 2500, 1600, 1600, 0
        threads = 1  # Keep laptop memory usage predictable.

        db = work / "database.db"
        sparse_out = sparse / "0"
        self._run(
            [colmap, "feature_extractor", "--database_path", str(db),
             "--image_path", str(images_dir),
             "--ImageReader.single_camera", camera_mode["single_camera"],
             "--ImageReader.single_camera_per_image", camera_mode["single_camera_per_image"],
             "--ImageReader.camera_model", "SIMPLE_RADIAL",
             "--FeatureExtraction.max_image_size", str(max_image_size),
             "--SiftExtraction.max_num_features", str(feature_size),
             "--FeatureExtraction.use_gpu", "0"],
            work, log_file, progress_cb, 8,
             f"Extracting image features (COLMAP; {camera_mode['strategy']})...",
        )
        self._run(
            [colmap, "exhaustive_matcher", "--database_path", str(db),
             "--FeatureMatching.use_gpu", "0"],
            work, log_file, progress_cb, 20, "Matching overlapping photographs...",
        )
        self._run(
            [colmap, "mapper", "--database_path", str(db), "--image_path", str(images_dir),
             "--output_path", str(sparse)],
            work, log_file, progress_cb, 34, "Estimating camera poses and sparse 3D structure...",
        )
        if not sparse_out.exists() or not any(sparse_out.iterdir()):
            raise RuntimeError("COLMAP produced no registered model. Add more overlapping photos.")
        # COLMAP's world axes are arbitrary. Export registered camera poses so
        # a consistent camera-up direction can align the final 3D exports.
        pose_text_dir = work / "sparse_text"
        pose_text_dir.mkdir(parents=True, exist_ok=True)
        self._run(
            [colmap, "model_converter", "--input_path", str(sparse_out),
             "--output_path", str(pose_text_dir), "--output_type", "TXT"],
            work, log_file, progress_cb, 38, "Estimating the scene's upright direction...",
        )
        images_txt = pose_text_dir / "images.txt"
        if not images_txt.is_file():
            raise RuntimeError("COLMAP model conversion did not create sparse_text/images.txt.")
        registration_summary = _colmap_registration_summary(
            images_txt,
            image_paths,
            pose_text_dir / "points3D.txt",
        )
        registration_summary["camera_calibration"] = camera_mode
        if registration_summary["registered_count"] < 2:
            raise RuntimeError(
                "COLMAP registered fewer than two uploaded photographs; "
                "the camera solution is insufficient for 3D reconstruction."
            )

        self._run(
            [colmap, "image_undistorter", "--image_path", str(images_dir),
             "--input_path", str(sparse_out), "--output_path", str(dense),
             "--output_type", "COLMAP", "--max_image_size", str(max_image_size)],
            work, log_file, progress_cb, 46, "Undistorting images for dense reconstruction...",
        )
        scene_mvs = work / "scene.mvs"
        self._run(
            [interface_colmap, "-i", str(dense), "-o", str(scene_mvs)],
            work, log_file, progress_cb, 54, "Preparing calibrated OpenMVS scene...",
        )
        if not scene_mvs.exists():
            raise RuntimeError("COLMAP-to-OpenMVS conversion did not create scene.mvs.")

        scene_fuse = work / "scene_fuse.mvs"
        shutil.copy2(scene_mvs, scene_fuse)
        common = densify_common_options(
            resolution_level=resolution_level,
            max_resolution=max_resolution,
            max_threads=threads,
            number_views=5,
            postprocess_dmaps=2,
        )
        self._run(
            [densify, "-i", str(scene_mvs), "-w", str(work), *common, "--fusion-mode", "1"],
            work, log_file, progress_cb, 62, "Estimating multi-view depth maps...",
        )
        # The second pass uses the independent copy so first-pass depth maps remain reusable.
        self._run(
            [densify, "-i", str(scene_fuse), "-w", str(work), *common,
             *densify_fusion_options(minimum_views=2)],
            work, log_file, progress_cb, 72, "Fusing depth maps into a dense point cloud...",
        )

        dense_scene = work / "scene_fuse_dense.mvs"
        dense_ply = work / "scene_fuse_dense.ply"
        if not dense_scene.exists() or not dense_ply.exists():
            raise RuntimeError("OpenMVS dense fusion did not produce its MVS and PLY outputs.")

        mesh_scene = work / "scene_mesh.mvs"
        self._run(
            [reconstruct, "-i", str(dense_scene), "-o", str(mesh_scene),
             *reconstruct_mesh_options(max_threads=threads, remove_spurious=20)],
            work, log_file, progress_cb, 80, "Reconstructing triangular surface mesh...",
        )
        mesh_ply = work / "scene_mesh.ply"
        if not mesh_ply.exists():
            raise RuntimeError("OpenMVS mesh reconstruction did not create scene_mesh.ply.")

        refined_scene = work / "scene_mesh_refined.mvs"
        self._run(
            [refine, "-i", str(mesh_scene), "-m", str(mesh_ply), "-o", str(refined_scene),
             *refine_mesh_options(max_threads=threads, max_views=6)],
            work, log_file, progress_cb, 86, "Refining mesh geometry...",
        )
        texture_options = texture_mesh_options(max_threads=threads, max_texture_size=2048)
        glb_work = work / "scene_textured.glb"
        self._run(
            [texture, "-i", str(refined_scene), "-o", str(glb_work),
             "--export-type", "glb", *texture_options],
            work, log_file, progress_cb, 92, "Generating textured GLB export...",
        )

        # Export a second copy as OBJ so the native VisPy viewer can consume
        # its UV coordinates and atlas through TextureFilter.
        obj_work = work / "scene_textured.obj"
        self._run(
            [texture, "-i", str(refined_scene), "-o", str(obj_work),
             "--export-type", "obj", *texture_options],
            work, log_file, progress_cb, 96, "Preparing textured OBJ and atlas for the desktop viewer...",
        )

        mesh_dir = session_dir / "mesh"
        mesh_dir.mkdir(parents=True, exist_ok=True)
        final_ply = mesh_dir / "scene_mesh.ply"
        final_glb = mesh_dir / "scene_textured.glb"
        final_obj = mesh_dir / "scene_mesh.obj"
        refined_ply = work / "scene_mesh_refined.ply"
        if not refined_ply.exists():
            refined_ply = mesh_ply
        shutil.copy2(refined_ply, final_ply)
        shutil.copy2(glb_work, final_glb)

        # The dedicated OBJ texturing pass exports the viewer's UV atlas.
        texture_obj = obj_work
        texture_mtl = work / "scene_textured.mtl"
        atlas_files = sorted(work.glob("scene_textured_*.png"))
        if not texture_obj.exists():
            raise RuntimeError("OpenMVS OBJ export did not create scene_textured.obj.")
        if not atlas_files:
            raise RuntimeError("OpenMVS OBJ export did not create its texture atlas PNG.")
        shutil.copy2(texture_obj, final_obj)

        for index, atlas in enumerate(atlas_files):
            target_name = "scene_mesh.png" if index == 0 else f"scene_mesh_{index}.png"
            shutil.copy2(atlas, mesh_dir / target_name)

        if not texture_mtl.exists():
            raise RuntimeError("OpenMVS OBJ export did not create scene_textured.mtl.")

        # OBJ export writes its own JPEG diffuse maps. The PNGs above belong to
        # the separate GLB export and are not necessarily the maps referenced by
        # the OBJ's MTL. Copy every map_Kd asset and rewrite it relative to MTL.
        copy_mtl_diffuse_maps(
            texture_mtl,
            mesh_dir,
            output_name="scene_mesh.mtl",
        )

        # Keep the copied OBJ linked to the material file copied beside it.
        obj_lines = []
        for line in final_obj.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines():
            if line.lstrip().lower().startswith("mtllib "):
                indent = line[: len(line) - len(line.lstrip())]
                line = f"{indent}mtllib scene_mesh.mtl"
            obj_lines.append(line)
        final_obj.write_text("\n".join(obj_lines) + "\n", encoding="utf-8")

        orientation = None
        estimated_up = estimate_world_up(images_txt)
        if estimated_up is not None:
            up_vector, consistency, pose_count = estimated_up
            alignment_rotation = rotation_from_to(
                up_vector, np.array([0.0, 0.0, 1.0], dtype=np.float64)
            )
            center = obj_median_center(final_obj)
            obj_ok = align_obj_file(final_obj, alignment_rotation, center=center)
            ply_ok = align_open3d_geometry_file(
                final_ply, alignment_rotation, center, point_cloud=False
            )
            cloud_ok = align_open3d_geometry_file(
                dense_ply, alignment_rotation, center, point_cloud=True
            )
            glb_ok = align_glb_file(final_glb, alignment_rotation, center)
            orientation = {
                "method": "COLMAP camera-up consensus",
                "registered_camera_poses": pose_count,
                "up_consistency": round(consistency, 4),
                "estimated_up_before_alignment": [float(x) for x in up_vector],
                "aligned_to_world_axis": "+Z",
                "exports_aligned": {
                    "obj": obj_ok, "ply": ply_ok,
                    "dense_cloud": cloud_ok, "glb": glb_ok,
                },
            }
            with log_file.open("a", encoding="utf-8") as log:
                log.write("\nScene orientation alignment: " + str(orientation) + "\n")
        else:
            orientation = {
                "method": "not applied",
                "reason": (
                    "COLMAP camera-up directions were unavailable or inconsistent; "
                    "the model retains the original arbitrary SfM orientation."
                ),
            }
            with log_file.open("a", encoding="utf-8") as log:
                log.write("\nScene orientation alignment skipped: " + orientation["reason"] + "\n")

        # Run the same geometry/material QA used by the native backend.
        from photo_reconstruction.reconstruction_qa import audit_reconstruction_outputs

        mesh_quality = audit_reconstruction_outputs(
            final_ply,
            obj_path=final_obj,
            glb_path=final_glb,
        )
        qa_path = session_dir / "reconstruction_quality.json"
        qa_path.write_text(json.dumps(mesh_quality, indent=2), encoding="utf-8")

        if progress_cb:
            progress_cb(100, "OpenMVS reconstruction and quality audit complete.")
        return {
            "success": True, "mesh_ply": final_ply, "mesh_obj": final_obj,
            "mesh_glb": final_glb, "dense_ply": dense_ply, "backend": "openmvs",
            "log_file": log_file, "quality_preset": preset,
            "orientation": orientation,
            "registration": registration_summary,
            "registered_count": registration_summary["registered_count"],
            "failed_images": registration_summary["failed_images"],
            "sparse_point_count": registration_summary["sparse_point_count"],
            "reprojection_error_px": registration_summary["mean_reprojection_error_px"],
            "mesh_quality": mesh_quality,
            "mesh_quality_report": qa_path,
            "photo_set": photo_set_report,
        }
