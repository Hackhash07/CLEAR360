"""
Run M7.3.6 consensus fusion using every view that passes strict depth calibration.

This is the next controlled reconstruction stage after the calibration strategy
comparison. It:
  * loads the saved COLMAP model/tracks from diagnose_image_set.py;
  * recalibrates ALL cached registered depth maps together using regularized
    held-out calibration (same pooled prior and 4% / 12% limits);
  * selects all accepted views (or a user-specified accepted subset);
  * runs M7.3.6 multi-view depth/3D consensus, normal estimation, and
    normal-compatible voxel fusion;
  * exports a colored+normal PLY and JSON report.

It does NOT run surface meshing, texture generation, canonical scene assembly,
or the full smartphone reconstruction pipeline. The result remains in COLMAP's
arbitrary scene gauge unless independently metric-scale calibrated.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import open3d as o3d

from photo_reconstruction.depth_aligner import SfMDepthAligner
from photo_reconstruction.diagnose_cached_depth import (
    _build_camera_models_and_tracks,
    _load_best_colmap_model,
)
from photo_reconstruction.m7_3_6_fusion import M736GeometricConsensusEngine


def _load_depth_maps(cache_dir: Path, names: list[str]) -> dict[str, np.ndarray]:
    maps: dict[str, np.ndarray] = {}
    missing = []
    for name in names:
        path = cache_dir / f"{Path(name).stem}_depth.npy"
        if not path.is_file():
            missing.append(str(path))
            continue
        depth = np.load(path, allow_pickle=False)
        if depth.ndim != 2:
            raise RuntimeError(f"Expected a 2D depth map at {path}; got {depth.shape}.")
        if not np.isfinite(depth).any():
            raise RuntimeError(f"Depth map has no finite values: {path}")
        maps[name] = np.asarray(depth, dtype=np.float32)
    if missing:
        raise FileNotFoundError(
            "Missing cached depth maps for registered views:\n  "
            + "\n  ".join(missing)
        )
    if not maps:
        raise RuntimeError(f"No cached depth maps found in {cache_dir}.")
    return maps


def _load_rgb_images(images_dir: Path, names: list[str]) -> dict[str, np.ndarray]:
    images = {}
    for name in names:
        path = images_dir / name
        image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"Could not read source RGB image: {path}")
        images[name] = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return images


def _cloud_summary(points: np.ndarray) -> dict[str, Any]:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(pts) == 0:
        return {
            "point_count": 0,
            "bounds_min_sfm_units": None,
            "bounds_max_sfm_units": None,
            "bounds_extent_sfm_units": None,
        }
    lower = pts.min(axis=0)
    upper = pts.max(axis=0)
    return {
        "point_count": int(len(pts)),
        "bounds_min_sfm_units": lower.tolist(),
        "bounds_max_sfm_units": upper.tolist(),
        "bounds_extent_sfm_units": (upper - lower).tolist(),
    }


def run_multiview_fusion(
    images_dir: Path,
    output_dir: Path,
    cache_dir: Path,
    requested_views: list[str] | None = None,
    stride: int = 6,
    voxel_size: float = 0.025,
    normal_compatibility: float = 0.707,
    minimum_verified_samples: int = 1000,
    minimum_support_count: int = 1,
) -> dict[str, Any]:
    if stride < 1:
        raise ValueError("--stride must be >= 1.")
    if voxel_size <= 0:
        raise ValueError("--voxel-size must be > 0.")
    if not (-1.0 <= normal_compatibility <= 1.0):
        raise ValueError("--normal-compatibility must be between -1 and 1.")
    if minimum_support_count not in (1, 2):
        raise ValueError("--minimum-support-count must be 1 or 2.")

    sparse_root = output_dir / "sfm_work" / "sparse"
    model_dir, reconstruction = _load_best_colmap_model(sparse_root)

    pose_dir = output_dir / "camera_poses"
    saved_pose_stems = {
        path.stem[:-5] for path in pose_dir.glob("*_pose.npy")
    } if pose_dir.is_dir() else set()

    camera_models, sparse_tracks = _build_camera_models_and_tracks(
        reconstruction, saved_pose_stems
    )
    if not camera_models:
        raise RuntimeError("No saved COLMAP camera models or sparse tracks were recovered.")

    # Load every registered view's cached prediction before selecting accepted
    # views. This is essential: the regularized/global prior must be based on the
    # same complete eight-view set used by the depth-calibration diagnostic.
    all_names = sorted(camera_models)
    all_depth_maps = _load_depth_maps(cache_dir, all_names)
    camera_models = {name: camera_models[name] for name in all_depth_maps}
    sparse_tracks = {
        name: sparse_tracks.get(name, []) for name in all_depth_maps
    }

    aligner = SfMDepthAligner(mode="regularized")
    alignment = aligner.align_depth_maps(
        depth_maps=all_depth_maps,
        camera_models=camera_models,
        sparse_tracks=sparse_tracks,
    )

    accepted_views = sorted(alignment.aligned_depth_maps)
    rejected_views = sorted(
        name for name, stats in alignment.view_stats.items() if not stats.success
    )
    if requested_views:
        selected_views = sorted(set(requested_views))
        unknown = sorted(set(selected_views) - set(camera_models))
        if unknown:
            raise ValueError(f"Requested views are not registered/cached: {unknown}")
        rejected_requested = sorted(set(selected_views) - set(accepted_views))
        if rejected_requested:
            raise RuntimeError(
                "Requested views did not pass held-out calibration: "
                + ", ".join(rejected_requested)
            )
    else:
        selected_views = accepted_views

    per_view_calibration = {}
    for name, stats in sorted(alignment.view_stats.items()):
        per_view_calibration[name] = {
            "usable_anchors": int(stats.num_anchors),
            "scale": float(stats.median_scale),
            "offset": float(stats.depth_offset),
            "validation_median_relative_error": float(
                stats.validation_median_relative_error
            ),
            "validation_p90_relative_error": float(
                stats.validation_p90_relative_error
            ),
            "accepted": bool(stats.success),
            "status": str(stats.status_message),
        }

    report: dict[str, Any] = {
        "status": "not_run",
        "images_dir": str(images_dir),
        "output_dir": str(output_dir),
        "depth_cache": str(cache_dir),
        "colmap_model": str(model_dir),
        "depth_strategy": "regularized",
        "calibration_limits": {
            "median_relative_error": float(aligner.max_val_median_rel),
            "p90_relative_error": float(aligner.max_val_p90_rel),
        },
        "registered_depth_views": len(all_depth_maps),
        "calibration_accepted_views": accepted_views,
        "calibration_rejected_views": rejected_views,
        "calibration_training_anchors": int(alignment.total_anchors),
        "per_view_calibration": per_view_calibration,
        "selected_views": selected_views,
        "stride_px": int(stride),
        "voxel_size_sfm_units": float(voxel_size),
        "normal_compatibility_dot_threshold": float(normal_compatibility),
        "minimum_verified_source_samples": int(minimum_verified_samples),
        "minimum_support_count_from_other_views": int(minimum_support_count),
        "note": (
            "This stage runs actual M7.3.6 multi-view geometric consensus, normal "
            "estimation, and normal-compatible voxel fusion. It does not run MVS "
            "stereo, meshing, texturing, or canonical scene assembly. Coordinates "
            "and voxel size are in arbitrary COLMAP scene units unless an external "
            "metric scale is independently established."
        ),
    }

    if len(selected_views) < 2:
        report["status"] = "not_enough_calibrated_views"
        return report

    for name in selected_views:
        if not (images_dir / name).is_file():
            raise FileNotFoundError(f"Source image not found: {images_dir / name}")

    selected_depth_maps = {
        name: alignment.aligned_depth_maps[name] for name in selected_views
    }
    selected_cameras = {
        name: camera_models[name] for name in selected_views
    }
    selected_images = _load_rgb_images(images_dir, selected_views)

    engine = M736GeometricConsensusEngine(
        normal_compat_thresh=normal_compatibility,
        voxel_fusion_size=voxel_size,
        minimum_support_count=minimum_support_count,
    )
    points, colors, normals, weights, engine_report = (
        engine.evaluate_multi_view_consensus(
            camera_models=selected_cameras,
            depth_maps=selected_depth_maps,
            images_rgb=selected_images,
            stride=stride,
        )
    )

    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    colors = np.asarray(colors, dtype=np.float32).reshape(-1, 3)
    normals = np.asarray(normals, dtype=np.float32).reshape(-1, 3)
    weights = np.asarray(weights, dtype=np.float32).reshape(-1)

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    cloud.colors = o3d.utility.Vector3dVector(
        np.clip(colors, 0.0, 1.0).astype(np.float64)
    )
    if len(normals) == len(points):
        cloud.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))

    run_dir = output_dir / (
        "multiview_consensus_fusion"
        if args.minimum_support_count == 1
        else f"multiview_consensus_fusion_support{args.minimum_support_count}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    ply_path = run_dir / "verified_multiview_consensus_cloud.ply"
    if not o3d.io.write_point_cloud(str(ply_path), cloud, write_ascii=False):
        raise RuntimeError(f"Could not write fused point cloud: {ply_path}")

    engine_counts = {
        "total_candidate_samples": int(engine_report.total_candidates),
        "multi_view_consensus_samples": int(engine_report.consensus_pts_count),
        "single_other_view_validated_samples": int(engine_report.validated_pts_count),
        "single_view_unverified_samples": int(engine_report.single_view_pts_count),
        "rejected_samples": int(engine_report.rejected_outliers_count),
    }
    verified_samples = (
        engine_counts["multi_view_consensus_samples"]
        + engine_counts["single_other_view_validated_samples"]
        if minimum_support_count == 1
        else engine_counts["multi_view_consensus_samples"]
    )

    report.update({
        "status": (
            "multiview_fusion_completed_point_gate_pass"
            if verified_samples >= minimum_verified_samples
            else "multiview_fusion_completed_point_gate_below_minimum"
        ),
        "engine_counts": engine_counts,
        "verified_source_samples": int(verified_samples),
        "verified_source_sample_rate_pct": round(
            100.0 * verified_samples
            / max(engine_counts["total_candidate_samples"], 1),
            2,
        ),
        "fused_point_cloud": _cloud_summary(points),
        "normal_count": int(len(normals)),
        "weight_count": int(len(weights)),
        "point_cloud_file": str(ply_path),
        "engine_report_note": (
            "Only observed counts were copied from the engine report. Fixed placeholder "
            "layer-thickness diagnostic values are deliberately not presented as measurements."
        ),
    })

    if hasattr(engine, "uncertain_pts") and len(engine.uncertain_pts) > 0:
        uncertain = o3d.geometry.PointCloud()
        uncertain.points = o3d.utility.Vector3dVector(
            np.asarray(engine.uncertain_pts, dtype=np.float64)
        )
        if hasattr(engine, "uncertain_cols") and len(engine.uncertain_cols) == len(engine.uncertain_pts):
            uncertain.colors = o3d.utility.Vector3dVector(
                np.asarray(engine.uncertain_cols, dtype=np.float64)
            )
        uncertain_path = run_dir / "uncertain_diagnostic_cloud.ply"
        o3d.io.write_point_cloud(str(uncertain_path), uncertain, write_ascii=False)
        report["uncertain_point_cloud_file"] = str(uncertain_path)
        report["uncertain_point_count"] = int(len(engine.uncertain_pts))

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, help="Synthetic/source image directory.")
    parser.add_argument("--output", required=True, help="Existing diagnostic output folder.")
    parser.add_argument("--depth-cache", default=None, help="Depth cache folder.")
    parser.add_argument(
        "--views",
        nargs="+",
        default=None,
        help="Optional subset of accepted image filenames. Default: every calibrated view.",
    )
    parser.add_argument("--stride", type=int, default=6)
    parser.add_argument("--voxel-size", type=float, default=0.025)
    parser.add_argument("--normal-compatibility", type=float, default=0.707)
    parser.add_argument("--minimum-verified-samples", type=int, default=1000)
    parser.add_argument(
        "--minimum-support-count",
        type=int,
        choices=(1, 2),
        default=1,
        help="Require each source sample to agree with at least this many other views.",
    )
    args = parser.parse_args()

    images_dir = Path(args.images).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    cache_dir = (
        Path(args.depth_cache).expanduser().resolve()
        if args.depth_cache
        else output_dir / "depth_cache_hypersim_depth_anything_v2_metric_hypersim_vitb"
    )
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {images_dir}")
    if not output_dir.is_dir():
        raise FileNotFoundError(f"Diagnostic output directory not found: {output_dir}")
    if not cache_dir.is_dir():
        raise FileNotFoundError(f"Depth cache directory not found: {cache_dir}")

    report = run_multiview_fusion(
        images_dir=images_dir,
        output_dir=output_dir,
        cache_dir=cache_dir,
        requested_views=args.views,
        stride=args.stride,
        voxel_size=args.voxel_size,
        normal_compatibility=args.normal_compatibility,
        minimum_verified_samples=args.minimum_verified_samples,
        minimum_support_count=args.minimum_support_count,
    )

    run_dir = output_dir / "multiview_consensus_fusion"
    run_dir.mkdir(parents=True, exist_ok=True)
    report_path = run_dir / "fusion_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("CLEAR360 MULTI-VIEW CONSENSUS FUSION")
    print(f"Calibration accepted: {len(report['calibration_accepted_views'])}/"
          f"{report['registered_depth_views']}")
    print("Accepted views:", ", ".join(report["calibration_accepted_views"]))
    print("Selected views:", ", ".join(report["selected_views"]))
    print(f"Minimum support from other views: {args.minimum_support_count}")
    print(f"Status: {report['status']}")

    if "engine_counts" in report:
        counts = report["engine_counts"]
        print(f"Candidate source samples: {counts['total_candidate_samples']:,}")
        print(f"Verified source samples (at selected support): {report['verified_source_samples']:,}")
        print(
            f"Multi-view consensus samples: {counts['multi_view_consensus_samples']:,}"
        )
        print(
            "Single-other-view validated samples: "
            f"{counts['single_other_view_validated_samples']:,}"
        )
        print(f"Unverified single-view samples: {counts['single_view_unverified_samples']:,}")
        print(f"Voxel-fused output points: {report['fused_point_cloud']['point_count']:,}")
        print(
            "Fused cloud extent (COLMAP scene units): "
            f"{report['fused_point_cloud']['bounds_extent_sfm_units']}"
        )
        print(f"Saved PLY: {report['point_cloud_file']}")
        if report.get("uncertain_point_cloud_file"):
            print(f"Saved uncertain diagnostic PLY: {report['uncertain_point_cloud_file']}")
    else:
        print("Fusion was skipped; fewer than two accepted views were available.")

    print(f"Saved report: {report_path}")
    return 0 if report["status"] == "multiview_fusion_completed_point_gate_pass" else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Multi-view fusion failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
