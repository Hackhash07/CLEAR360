"""
Run only the accepted two-view geometric-consensus fusion stage.

This consumes the existing COLMAP model and Hypersim depth cache, repeats the
held-out depth calibration, then calls M7.3.6's geometric consensus + normal
estimation + normal-compatible voxel fusion for exactly two requested views.

It exports a fused point cloud and diagnostics. It does NOT run dense MVS,
surface reconstruction/meshing, texture generation, or the full phone pipeline.
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
    result = {}
    missing = []
    for name in names:
        path = cache_dir / f"{Path(name).stem}_depth.npy"
        if not path.is_file():
            missing.append(str(path))
            continue
        depth = np.load(path, allow_pickle=False)
        if depth.ndim != 2:
            raise RuntimeError(f"Depth map must be 2D: {path} has shape {depth.shape}")
        if not np.isfinite(depth).any():
            raise RuntimeError(f"Depth map contains no finite values: {path}")
        result[name] = np.asarray(depth, dtype=np.float32)
    if missing:
        raise FileNotFoundError("Missing cached depth maps:\n  " + "\n  ".join(missing))
    return result


def _load_rgb_images(images_dir: Path, names: list[str]) -> dict[str, np.ndarray]:
    result = {}
    for name in names:
        path = images_dir / name
        image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"Could not read source image: {path}")
        result[name] = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return result


def _point_cloud_summary(points: np.ndarray) -> dict[str, Any]:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(pts) == 0:
        return {
            "point_count": 0,
            "bounds_min_sfm_units": None,
            "bounds_max_sfm_units": None,
            "bounds_extent_sfm_units": None,
        }
    lower = np.min(pts, axis=0)
    upper = np.max(pts, axis=0)
    return {
        "point_count": int(len(pts)),
        "bounds_min_sfm_units": lower.tolist(),
        "bounds_max_sfm_units": upper.tolist(),
        "bounds_extent_sfm_units": (upper - lower).tolist(),
    }


def run_fusion(
    images_dir: Path,
    output_dir: Path,
    cache_dir: Path,
    view_names: list[str],
    stride: int = 6,
    voxel_size: float = 0.025,
    normal_compatibility: float = 0.707,
    minimum_verified_samples: int = 1000,
) -> dict[str, Any]:
    if stride < 1:
        raise ValueError("--stride must be >= 1")
    if voxel_size <= 0:
        raise ValueError("--voxel-size must be > 0")
    if not (-1.0 <= normal_compatibility <= 1.0):
        raise ValueError("--normal-compatibility must be between -1 and 1")
    if len(view_names) != 2 or view_names[0] == view_names[1]:
        raise ValueError("Exactly two distinct --views must be supplied.")

    for name in view_names:
        if not (images_dir / name).is_file():
            raise FileNotFoundError(f"Source image not found: {images_dir / name}")

    sparse_root = output_dir / "sfm_work" / "sparse"
    model_dir, reconstruction = _load_best_colmap_model(sparse_root)

    pose_dir = output_dir / "camera_poses"
    saved_pose_stems = {
        path.stem[:-5] for path in pose_dir.glob("*_pose.npy")
    } if pose_dir.is_dir() else set()
    camera_models_all, sparse_tracks_all = _build_camera_models_and_tracks(
        reconstruction, saved_pose_stems
    )

    if any(name not in camera_models_all for name in view_names):
        absent = [name for name in view_names if name not in camera_models_all]
        raise RuntimeError(f"Requested views missing from saved COLMAP model: {absent}")

    depth_maps_all = _load_depth_maps(cache_dir, view_names)
    camera_models = {name: camera_models_all[name] for name in view_names}
    sparse_tracks = {
        name: sparse_tracks_all.get(name, []) for name in view_names
    }

    aligner = SfMDepthAligner(mode="regularized")
    alignment = aligner.align_depth_maps(
        depth_maps=depth_maps_all,
        camera_models=camera_models,
        sparse_tracks=sparse_tracks,
    )

    not_accepted = [
        name for name in view_names
        if name not in alignment.aligned_depth_maps
    ]
    calibration = {}
    for name in view_names:
        stats = alignment.view_stats[name]
        calibration[name] = {
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
            "status": stats.status_message,
        }

    base_report: dict[str, Any] = {
        "status": "not_run",
        "images_dir": str(images_dir),
        "output_dir": str(output_dir),
        "depth_cache": str(cache_dir),
        "colmap_model": str(model_dir),
        "views": view_names,
        "stride_px": int(stride),
        "voxel_size_sfm_units": float(voxel_size),
        "normal_compatibility_dot_threshold": float(normal_compatibility),
        "calibration": calibration,
        "training_anchors_total": int(alignment.total_anchors),
        "minimum_verified_samples": int(minimum_verified_samples),
        "note": (
            "This stage runs geometric consistency, Open3D normal estimation, and "
            "normal-compatible voxel fusion only. It does not run MVS, meshing, "
            "texturing, or the full phone reconstruction pipeline. Coordinates and "
            "voxel size are in the arbitrary COLMAP scene gauge unless separately "
            "converted using an external metric scale."
        ),
    }

    if not_accepted:
        base_report["status"] = "requested_views_failed_depth_calibration"
        base_report["unaccepted_views"] = not_accepted
        return base_report

    aligned_depth_maps = {
        name: alignment.aligned_depth_maps[name] for name in view_names
    }
    images_rgb = _load_rgb_images(images_dir, view_names)

    engine = M736GeometricConsensusEngine(
        normal_compat_thresh=normal_compatibility,
        voxel_fusion_size=voxel_size,
    )
    points, colors, normals, weights, engine_report = (
        engine.evaluate_multi_view_consensus(
            camera_models=camera_models,
            depth_maps=aligned_depth_maps,
            images_rgb=images_rgb,
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

    run_dir = output_dir / "two_view_consensus_fusion"
    run_dir.mkdir(parents=True, exist_ok=True)
    ply_path = run_dir / "verified_consensus_cloud.ply"
    if not o3d.io.write_point_cloud(str(ply_path), cloud, write_ascii=False):
        raise RuntimeError(f"Open3D could not write point cloud: {ply_path}")

    # These are actual engine counts. The engine's layer-thickness fields are
    # intentionally omitted because the current implementation reports fixed
    # placeholder values for those diagnostics.
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
    )
    point_summary = _point_cloud_summary(points)

    base_report.update({
        "status": (
            "fusion_completed_point_gate_pass"
            if verified_samples >= minimum_verified_samples
            else "fusion_completed_point_gate_below_minimum"
        ),
        "engine_counts": engine_counts,
        "verified_source_samples": int(verified_samples),
        "verified_source_sample_rate_pct": round(
            100.0 * verified_samples
            / max(engine_counts["total_candidate_samples"], 1),
            2,
        ),
        "point_cloud": point_summary,
        "normal_count": int(len(normals)),
        "weight_count": int(len(weights)),
        "point_cloud_file": str(ply_path),
        "fusion_parameters": {
            "stride_px": int(stride),
            "voxel_size_sfm_units": float(voxel_size),
            "normal_compatibility_dot_threshold": float(normal_compatibility),
        },
    })
    return base_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, help="Directory containing synthetic source JPGs.")
    parser.add_argument("--output", required=True, help="Existing diagnostic output directory.")
    parser.add_argument("--depth-cache", default=None, help="Default: OUTPUT/depth_cache_hypersim.")
    parser.add_argument("--views", nargs=2, default=["5.jpg", "8.jpg"])
    parser.add_argument("--stride", type=int, default=6)
    parser.add_argument("--voxel-size", type=float, default=0.025)
    parser.add_argument("--normal-compatibility", type=float, default=0.707)
    parser.add_argument("--minimum-verified-samples", type=int, default=1000)
    args = parser.parse_args()

    images_dir = Path(args.images).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    cache_dir = (
        Path(args.depth_cache).expanduser().resolve()
        if args.depth_cache else output_dir / "depth_cache_hypersim"
    )
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {images_dir}")
    if not output_dir.is_dir():
        raise FileNotFoundError(f"Diagnostic output directory not found: {output_dir}")
    if not cache_dir.is_dir():
        raise FileNotFoundError(f"Depth cache directory not found: {cache_dir}")

    report = run_fusion(
        images_dir=images_dir,
        output_dir=output_dir,
        cache_dir=cache_dir,
        view_names=list(args.views),
        stride=args.stride,
        voxel_size=args.voxel_size,
        normal_compatibility=args.normal_compatibility,
        minimum_verified_samples=args.minimum_verified_samples,
    )

    run_dir = output_dir / "two_view_consensus_fusion"
    run_dir.mkdir(parents=True, exist_ok=True)
    report_path = run_dir / "fusion_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("CLEAR360 TWO-VIEW CONSENSUS FUSION")
    print(f"Views: {', '.join(args.views)}")
    print(f"Status: {report['status']}")
    for name, stats in report["calibration"].items():
        print(
            f"  {name}: median={stats['validation_median_relative_error']:.2%}, "
            f"p90={stats['validation_p90_relative_error']:.2%}, "
            f"accepted={stats['accepted']}"
        )

    if "engine_counts" in report:
        counts = report["engine_counts"]
        print(f"Candidate samples: {counts['total_candidate_samples']:,}")
        print(f"Verified samples: {report['verified_source_samples']:,}")
        print(f"Voxel-fused output points: {report['point_cloud']['point_count']:,}")
        print(
            "Fused point-cloud extent (COLMAP scene units): "
            f"{report['point_cloud']['bounds_extent_sfm_units']}"
        )
        print(f"Saved PLY: {report['point_cloud_file']}")
    else:
        print("Fusion was skipped because requested views failed depth calibration.")

    print(f"Saved report: {report_path}")
    return 0 if report["status"] == "fusion_completed_point_gate_pass" else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Two-view consensus fusion failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
