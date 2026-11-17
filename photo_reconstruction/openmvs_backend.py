"""OpenMVS command-line backend for photo-to-3D reconstruction.

This backend is optional. The existing CLEAR360 Python reconstruction pipeline
remains the default; OpenMVS can be selected when its binaries are installed.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from photo_reconstruction.openmvs_assets import (
    copy_mtl_diffuse_maps,
    export_obj_materials_to_glb,
)
from photo_reconstruction.openmvs_quality import (
    densify_common_options,
    densify_fusion_options,
    reconstruct_mesh_options,
    refine_mesh_options,
    texture_mesh_options,
    apple_silicon_high_profile,
    resolve_openmvs_thread_count,
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


def _ply_vertex_count(path: Path) -> int | None:
    """Read the vertex count from a PLY header without loading the point data."""
    try:
        with Path(path).open("rb") as handle:
            for line in handle:
                if line.startswith(b"element vertex "):
                    return int(line.split()[2])
                if line.strip() == b"end_header":
                    break
    except (OSError, ValueError, IndexError):
        return None
    return None



def _physical_memory_bytes() -> int | None:
    """Return installed physical RAM where sysconf exposes it."""
    try:
        pages = int(os.sysconf("SC_PHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, TypeError, ValueError):
        return None
    total = pages * page_size
    return total if total > 0 else None


def _openmvs_thread_count() -> int:
    """Resolve a memory-aware OpenMVS thread cap, honoring explicit user override."""
    return resolve_openmvs_thread_count(
        platform_system=platform.system(),
        platform_machine=platform.machine(),
        cpu_count=os.cpu_count() or 1,
        total_memory_bytes=_physical_memory_bytes(),
        override=os.environ.get("CLEAR360_OPENMVS_THREADS"),
    )


def _is_apple_silicon() -> bool:
    """Detect native Apple Silicon execution (not Rosetta's x86_64 process)."""
    return (
        platform.system() == "Darwin"
        and platform.machine().lower() in {"arm64", "aarch64"}
    )


def _colmap_feature_extraction_options(
    max_image_size: int,
    max_num_features: int,
    *,
    robust_sift: bool = True,
    num_threads: int | None = None,
) -> list[str]:
    """Return current COLMAP CPU SIFT options, with optional preview thread limits."""
    options = [
        "--FeatureExtraction.max_image_size", str(max_image_size),
        "--SiftExtraction.max_num_features", str(max_num_features),
        "--FeatureExtraction.use_gpu", "0",
    ]
    if num_threads is not None:
        options.extend(["--FeatureExtraction.num_threads", str(num_threads)])
    if robust_sift:
        options.extend([
            "--SiftExtraction.estimate_affine_shape", "1",
            "--SiftExtraction.domain_size_pooling", "1",
        ])
    return options


def _colmap_feature_matching_options(
    *, guided_matching: bool = True, num_threads: int | None = None
) -> list[str]:
    """Return CPU matching options with optional thread limits for preview runs."""
    options = ["--FeatureMatching.use_gpu", "0"]
    if num_threads is not None:
        options.extend(["--FeatureMatching.num_threads", str(num_threads)])
    if guided_matching:
        options.extend(["--FeatureMatching.guided_matching", "1"])
    return options


def _openmvs_fusion_minimum_views(quality_preset: str) -> int:
    """Use strict evidence for full runs and two-view support for previews."""
    return 2 if quality_preset.lower() in {"fast", "draft", "quick"} else 3


def _openmvs_point_cloud_filter_threshold(quality_preset: str) -> int | None:
    """Apply camera-visibility cleanup only to full-quality reconstructions."""
    return None if quality_preset.lower() in {"fast", "draft", "quick"} else -20

def _openmvs_standard_profile() -> dict[str, int]:
    """Balanced Standard settings: reduce dense-MVS cost, retain final asset stages."""
    return {
        "feature_size": 6000,
        "max_image_size": 1440,
        "max_resolution": 1400,
        "resolution_level": 0,
        "dense_views": 4,
        "dense_iterations": 2,
        "geometric_iterations": 2,
        "min_dense_resolution": 560,
        "sub_resolution_levels": 1,
        "postprocess_dmaps": 2,
    }


def _validate_sparse_model_for_dense(
    registration_summary: dict[str, Any],
    uploaded_count: int,
) -> None:
    """Stop dense MVS when sparse cameras do not form a sufficiently supported block."""
    if uploaded_count < 1:
        raise ValueError("At least one uploaded image is required for sparse-model validation.")
    registered_count = int(registration_summary.get("registered_count", 0) or 0)
    minimum_registered = max(3, int(np.ceil(0.60 * uploaded_count)))
    if registered_count < minimum_registered:
        failed = registration_summary.get("failed_images", [])
        failed_text = ", ".join(str(name) for name in failed[:8]) or "not reported"
        raise RuntimeError(
            "COLMAP sparse reconstruction is too incomplete for dense MVS: "
            f"{registered_count}/{uploaded_count} images registered; "
            f"at least {minimum_registered} are required. "
            f"Unregistered images: {failed_text}. Capture more overlapping viewpoints."
        )

    overlap = registration_summary.get("camera_overlap") or {}
    if not overlap.get("available") or overlap.get("status") != "connected":
        components = overlap.get("components") or []
        component_summary = "; ".join(
            ", ".join(str(name) for name in component)
            for component in components[:6]
        ) or "no connected camera-track evidence"
        raise RuntimeError(
            "COLMAP registered cameras do not form one connected co-visibility graph. "
            f"Status: {overlap.get('status', 'unavailable')}; groups: {component_summary}. "
            "Add overlapping photographs between viewpoints before dense reconstruction."
        )


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
        stage_started = time.perf_counter()
        tool_name = Path(command[0]).name
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
            elapsed = time.perf_counter() - stage_started
            log.write(
                f"\n[CLEAR360 stage timing] {tool_name}: {elapsed:.1f} seconds; "
                f"exit_code={code}\n"
            )
            log.flush()
        if code != 0:
            raise RuntimeError(
                f"{tool_name} exited with code {code}. See {log_file}."
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

        preset = quality_preset.lower()
        fast_preview = preset == "fast"
        densify = self._binary("DensifyPointCloud")
        interface_colmap = self._binary("InterfaceCOLMAP")
        if fast_preview:
            # The dense point-cloud preview does not require mesh or texture tools.
            reconstruct = refine = texture = None
        else:
            reconstruct = self._binary("ReconstructMesh")
            refine = self._binary("RefineMesh")
            texture = self._binary("TextureMesh")

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

        standard_profile = None
        apple_silicon_profile = None
        if fast_preview:
            # Low-resolution geometry-only preview. Runtime target is a benchmark
            # objective, not a guaranteed deadline for every photo set.
            feature_size, max_image_size, max_resolution, resolution_level = 1500, 800, 640, 1
        elif preset in {"draft", "quick"}:
            feature_size, max_image_size, max_resolution, resolution_level = 2500, 1000, 800, 1
        elif preset == "high" and _is_apple_silicon():
            # Keep COLMAP feature extraction unchanged; trim only dense MVS work.
            apple_silicon_profile = apple_silicon_high_profile()
            feature_size = apple_silicon_profile["feature_size"]
            max_image_size = apple_silicon_profile["max_image_size"]
            max_resolution = apple_silicon_profile["max_resolution"]
            resolution_level = apple_silicon_profile["resolution_level"]
        elif preset == "high":
            feature_size, max_image_size, max_resolution, resolution_level = 8192, 2400, 2400, 0
        elif preset == "maximum":
            feature_size, max_image_size, max_resolution, resolution_level = 12000, 3200, 3200, 0
        else:
            # Standard uses a balanced depth solve while retaining full mesh and texture stages.
            standard_profile = _openmvs_standard_profile()
            feature_size = standard_profile["feature_size"]
            max_image_size = standard_profile["max_image_size"]
            max_resolution = standard_profile["max_resolution"]
            resolution_level = standard_profile["resolution_level"]

        threads = 2 if fast_preview else _openmvs_thread_count()
        if fast_preview:
            dense_views = 3
            dense_iterations = 1
            geometric_iterations = 1
            min_dense_resolution = 320
            sub_resolution_levels = 1
        elif standard_profile is not None:
            dense_views = standard_profile["dense_views"]
            dense_iterations = standard_profile["dense_iterations"]
            geometric_iterations = standard_profile["geometric_iterations"]
            min_dense_resolution = standard_profile["min_dense_resolution"]
            sub_resolution_levels = standard_profile["sub_resolution_levels"]
        elif apple_silicon_profile is not None:
            dense_views = apple_silicon_profile["dense_views"]
            dense_iterations = apple_silicon_profile["dense_iterations"]
            geometric_iterations = apple_silicon_profile["geometric_iterations"]
            min_dense_resolution = apple_silicon_profile["min_dense_resolution"]
            sub_resolution_levels = apple_silicon_profile["sub_resolution_levels"]
        else:
            # Draft/Quick keeps the established conservative full-depth solve;
            # High/Maximum retain the original maximum-quality settings.
            dense_views = 5
            dense_iterations = 3
            geometric_iterations = 2
            min_dense_resolution = 640
            sub_resolution_levels = 2

        db = work / "database.db"
        sparse_out = sparse / "0"
        self._run(
            [colmap, "feature_extractor", "--database_path", str(db),
             "--image_path", str(images_dir),
             "--ImageReader.single_camera", camera_mode["single_camera"],
             "--ImageReader.single_camera_per_image", camera_mode["single_camera_per_image"],
             "--ImageReader.camera_model", "SIMPLE_RADIAL",
             *_colmap_feature_extraction_options(
                 max_image_size,
                 feature_size,
                 robust_sift=not fast_preview,
                 num_threads=2 if fast_preview else None,
             )],
            work, log_file, progress_cb, 8,
             f"Extracting image features (COLMAP; {camera_mode['strategy']})...",
        )
        self._run(
            [colmap, "exhaustive_matcher", "--database_path", str(db),
             *_colmap_feature_matching_options(
                 guided_matching=not fast_preview,
                 num_threads=2 if fast_preview else None,
             )],
            work, log_file, progress_cb, 20, "Matching overlapping photographs...",
        )
        mapper_command = [
            colmap, "mapper", "--database_path", str(db),
            "--image_path", str(images_dir), "--output_path", str(sparse),
        ]
        if fast_preview:
            mapper_command.extend(["--Mapper.num_threads", "2"])
        self._run(
            mapper_command,
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
        with log_file.open("a", encoding="utf-8") as log:
            log.write(
                "\nCOLMAP registration summary: "
                + json.dumps(registration_summary, sort_keys=True)
                + "\n"
            )
        _validate_sparse_model_for_dense(registration_summary, len(image_paths))

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
        # Standard and Apple-Silicon High use short-gap interpolation only,
        # avoiding the additional full-resolution CPU confidence pass.
        if standard_profile is not None:
            postprocess_dmaps = standard_profile["postprocess_dmaps"]
        elif apple_silicon_profile is not None:
            postprocess_dmaps = apple_silicon_profile["postprocess_dmaps"]
        else:
            postprocess_dmaps = 2 if preset in {"fast", "draft", "quick"} else 10
        pipeline_configuration = {
            "quality_preset": preset,
            "performance_profile": (
                "apple_silicon_high_balanced"
                if apple_silicon_profile is not None
                else "default"
            ),
            "colmap": {
                "feature_type": "SIFT",
                "max_image_size": max_image_size,
                "max_num_features": feature_size,
                "affine_shape": not fast_preview,
                "domain_size_pooling": not fast_preview,
                "guided_matching": not fast_preview,
                "gpu_enabled": False,
                "camera_calibration": camera_mode,
            },
            "openmvs": {
                "max_resolution": max_resolution,
                "resolution_level": resolution_level,
                "number_views": dense_views,
                "minimum_views_fuse": _openmvs_fusion_minimum_views(preset),
                "iterations": dense_iterations,
                "geometric_iterations": geometric_iterations,
                "postprocess_dmaps": postprocess_dmaps,
                "point_cloud_visibility_filter_threshold": _openmvs_point_cloud_filter_threshold(preset),
                "cpu_confidence_recalibration": bool(postprocess_dmaps & 8),
                "max_threads": threads,
                "texture_threads": threads if not fast_preview else 0,
                "texture_mesh_passes": 1 if not fast_preview else 0,
                "glb_export": "converted from the same OpenMVS-textured OBJ/MTL package",
                "refine_max_views": 6,
                "preview_only": fast_preview,
                "meshing_skipped": fast_preview,
                "mesh_qa_skipped": fast_preview,
                "refinement_skipped": fast_preview,
                "texturing_skipped": fast_preview,
            },
            "input_photo_count": len(image_paths),
        }
        with log_file.open("a", encoding="utf-8") as log:
            log.write("\nPipeline configuration: " + json.dumps(pipeline_configuration, sort_keys=True) + "\n")
        common = densify_common_options(
            resolution_level=resolution_level,
            max_resolution=max_resolution,
            max_threads=threads,
            number_views=dense_views,
            postprocess_dmaps=postprocess_dmaps,
            iterations=dense_iterations,
            geometric_iterations=geometric_iterations,
            min_resolution=min_dense_resolution,
            sub_resolution_levels=sub_resolution_levels,
        )
        self._run(
            [densify, "-i", str(scene_mvs), "-w", str(work), *common, "--fusion-mode", "1"],
            work, log_file, progress_cb, 62, "Estimating multi-view depth maps...",
        )
        # The second pass uses the independent copy so first-pass depth maps remain reusable.
        self._run(
            [densify, "-i", str(scene_fuse), "-w", str(work), *common,
             *densify_fusion_options(
                 minimum_views=_openmvs_fusion_minimum_views(preset)
             )],
            work, log_file, progress_cb, 72, "Fusing depth maps into a dense point cloud...",
        )

        dense_scene = work / "scene_fuse_dense.mvs"
        dense_ply_unfiltered = work / "scene_fuse_dense.ply"
        if not dense_scene.exists() or not dense_ply_unfiltered.exists():
            raise RuntimeError("OpenMVS dense fusion did not produce its MVS and PLY outputs.")

        # Visibility filtering uses the registered camera rays to remove points
        # contradicted by camera-point intersections. Preserve the original cloud
        # and fall back to it if this optional cleanup stage fails.
        dense_scene_for_mesh = dense_scene
        dense_ply = dense_ply_unfiltered
        visibility_filter_threshold = _openmvs_point_cloud_filter_threshold(preset)
        dense_point_count_unfiltered = _ply_vertex_count(dense_ply_unfiltered)
        if fast_preview and not dense_point_count_unfiltered:
            raise RuntimeError(
                "Fast Preview dense fusion produced no readable PLY vertices. "
                "The point-cloud preview was not loaded; inspect OpenMVS depth and fusion logs."
            )
        dense_point_count = dense_point_count_unfiltered
        visibility_filter_report = {
            "requested": visibility_filter_threshold is not None,
            "threshold": visibility_filter_threshold,
            "applied": False,
            "unfiltered_scene": str(dense_scene),
            "unfiltered_ply": str(dense_ply_unfiltered),
            "unfiltered_point_count": dense_point_count_unfiltered,
        }
        if visibility_filter_threshold is not None:
            filtered_output_base = work / "scene_visibility_filter.mvs"
            filtered_scene = work / "scene_visibility_filter_filtered.mvs"
            filtered_ply = work / "scene_visibility_filter_filtered.ply"
            try:
                self._run(
                    [
                        densify, "-i", str(dense_scene), "-o", str(filtered_output_base),
                        "-w", str(work), "--max-threads", str(threads),
                        "--filter-point-cloud", str(visibility_filter_threshold),
                    ],
                    work, log_file, progress_cb, 76,
                    "Removing dense points contradicted by camera visibility...",
                )
                if not filtered_scene.is_file() or not filtered_ply.is_file():
                    raise RuntimeError(
                        "OpenMVS visibility filtering did not create the expected filtered MVS and PLY files."
                    )
                filtered_point_count = _ply_vertex_count(filtered_ply)
                if filtered_point_count == 0:
                    raise RuntimeError("OpenMVS visibility filtering produced an empty point cloud.")
                dense_scene_for_mesh = filtered_scene
                dense_ply = filtered_ply
                dense_point_count = filtered_point_count
                visibility_filter_report["applied"] = True
                visibility_filter_report["filtered_point_count"] = filtered_point_count
                visibility_filter_report["remaining_point_ratio"] = (
                    round(filtered_point_count / dense_point_count_unfiltered, 4)
                    if dense_point_count_unfiltered else None
                )
                visibility_filter_report["filtered_scene"] = str(filtered_scene)
                visibility_filter_report["filtered_ply"] = str(filtered_ply)
            except Exception as exc:
                visibility_filter_report["error"] = str(exc)
                with log_file.open("a", encoding="utf-8") as log:
                    log.write(
                        "\nVisibility-filter fallback: " + str(exc)
                        + "\nUsing the preserved unfiltered dense point cloud.\n"
                    )

        pipeline_configuration["openmvs"]["visibility_filter_applied"] = (
            visibility_filter_report["applied"]
        )
        pipeline_configuration["openmvs"]["visibility_filter_error"] = (
            visibility_filter_report.get("error")
        )
        with log_file.open("a", encoding="utf-8") as log:
            log.write(
                "\nVisibility-filter result: "
                + json.dumps(visibility_filter_report, sort_keys=True)
                + "\n"
            )

        mesh_ply: Path | None = None
        if not fast_preview:
            mesh_scene = work / "scene_mesh.mvs"
            self._run(
                [reconstruct, "-i", str(dense_scene_for_mesh), "-o", str(mesh_scene),
                 *reconstruct_mesh_options(max_threads=threads, remove_spurious=20)],
                work, log_file, progress_cb, 80, "Reconstructing triangular surface mesh...",
            )
            mesh_ply = work / "scene_mesh.ply"
            if not mesh_ply.exists():
                raise RuntimeError("OpenMVS mesh reconstruction did not create scene_mesh.ply.")

        mesh_dir = session_dir / "mesh"
        mesh_dir.mkdir(parents=True, exist_ok=True)
        final_ply = mesh_dir / "scene_mesh.ply"
        final_glb: Path | None = None
        final_obj: Path | None = None

        if fast_preview:
            # Stop at the dense point cloud. This avoids triangular meshing,
            # full mesh QA, refinement, orientation and texture exports.
            final_ply = mesh_dir / "scene_geometry_preview.ply"
            shutil.copy2(dense_ply, final_ply)
            orientation = {
                "method": "not applied",
                "reason": "Fast point-cloud preview skips meshing, refinement, orientation and texturing.",
            }
            with log_file.open("a", encoding="utf-8") as log:
                log.write(
                    "\nFast preview: point-cloud output only; triangular meshing, "
                    "mesh QA, refinement and texturing intentionally skipped.\n"
                )
        else:
            refined_scene = work / "scene_mesh_refined.mvs"
            self._run(
                [refine, "-i", str(mesh_scene), "-m", str(mesh_ply), "-o", str(refined_scene),
                 *refine_mesh_options(max_threads=threads, max_views=6)],
                work, log_file, progress_cb, 86, "Refining mesh geometry...",
            )
            # OpenMVS TextureMesh calculates view selection, seams and texture
            # patches before exporting. Run that expensive calculation ONCE for
            # OBJ, then derive GLB from the exact same UV/material/image package.
            # Use the same memory-aware thread cap for the texture stage.
            texture_options = texture_mesh_options(
                max_threads=threads, max_texture_size=2048
            )
            obj_work = work / "scene_textured.obj"
            self._run(
                [texture, "-i", str(refined_scene), "-o", str(obj_work),
                 "--export-type", "obj", *texture_options],
                work, log_file, progress_cb, 92,
                "Generating the texture atlas once (OpenMVS OBJ)...",
            )

            final_glb = mesh_dir / "scene_textured.glb"
            final_obj = mesh_dir / "scene_mesh.obj"
            refined_ply = work / "scene_mesh_refined.ply"
            if not refined_ply.exists():
                refined_ply = mesh_ply
            shutil.copy2(refined_ply, final_ply)

            texture_mtl = work / "scene_textured.mtl"
            if not obj_work.is_file():
                raise RuntimeError("OpenMVS OBJ texturing did not create scene_textured.obj.")
            if not texture_mtl.is_file():
                raise RuntimeError("OpenMVS OBJ texturing did not create scene_textured.mtl.")
            shutil.copy2(obj_work, final_obj)

            # Copy every map_Kd image and make MTL paths self-contained beside
            # the final OBJ. The native viewer consumes this exact package.
            copy_mtl_diffuse_maps(
                texture_mtl,
                mesh_dir,
                output_name="scene_mesh.mtl",
            )

            obj_lines = []
            for line in final_obj.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines():
                if line.lstrip().lower().startswith("mtllib "):
                    indent = line[: len(line) - len(line.lstrip())]
                    line = f"{indent}mtllib scene_mesh.mtl"
                obj_lines.append(line)
            final_obj.write_text("\n".join(obj_lines) + "\n", encoding="utf-8")

            conversion_started = time.perf_counter()
            conversion_report = export_obj_materials_to_glb(final_obj, final_glb)
            conversion_elapsed = time.perf_counter() - conversion_started
            pipeline_configuration["openmvs"]["texture_mesh_passes"] = 1
            pipeline_configuration["openmvs"]["texture_threads"] = threads
            pipeline_configuration["exports"] = {
                "glb_conversion": "trimesh from OpenMVS OBJ/MTL",
                "glb_texture_source": "same diffuse maps used by the OBJ",
                **conversion_report,
            }
            with log_file.open("a", encoding="utf-8") as log:
                log.write(
                    "\n[CLEAR360 stage timing] OBJ-to-GLB conversion: "
                    f"{conversion_elapsed:.1f} seconds\n"
                    "Single OpenMVS TextureMesh pass reused for OBJ and GLB.\n"
                    "OBJ-to-GLB validation: "
                    + json.dumps(conversion_report, sort_keys=True)
                    + "\n"
                )

            orientation = None
        estimated_up = None if fast_preview else estimate_world_up(images_txt)
        if fast_preview:
            orientation = {
                "method": "not applied",
                "reason": "Fast geometry preview skips orientation and texture finishing.",
            }
        elif estimated_up is not None:
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

        if fast_preview:
            mesh_quality = {
                "status": "preview_only",
                "score": None,
                "geometry_status": "not_evaluated_point_cloud",
                "geometry": {},
                "warnings": [
                    "Fast Preview is a dense point cloud, not a triangulated mesh.",
                    "Mesh topology and texture QA are not evaluated.",
                    "Run Standard or High for a final textured mesh.",
                ],
            }
        else:
            from photo_reconstruction.reconstruction_qa import audit_reconstruction_outputs

            mesh_quality = audit_reconstruction_outputs(
                final_ply,
                obj_path=final_obj,
                glb_path=final_glb,
            )
        qa_path = session_dir / "reconstruction_quality.json"
        qa_path.write_text(json.dumps(mesh_quality, indent=2), encoding="utf-8")

        if progress_cb:
            progress_cb(
                100,
                "Fast point-cloud preview complete."
                if fast_preview else "OpenMVS reconstruction and quality audit complete.",
            )
        return {
            "success": True, "mesh_ply": final_ply, "mesh_obj": final_obj,
            "mesh_glb": final_glb, "dense_ply": dense_ply, "backend": "openmvs",
            "backend_label": "COLMAP + OpenMVS Fast Preview" if fast_preview else "COLMAP + OpenMVS",
            "preview_only": fast_preview,
            "log_file": log_file, "quality_preset": preset,
            "pipeline_configuration": pipeline_configuration,
            "visibility_filter": visibility_filter_report,
            "dense_ply_unfiltered": dense_ply_unfiltered,
            "dense_point_count": dense_point_count,
            "dense_point_count_unfiltered": dense_point_count_unfiltered,
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
