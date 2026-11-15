"""
Run a low-memory validation of a photo set without building a dense cloud or mesh.

Examples:
    python -m photo_reconstruction.diagnose_image_set \
        --images /path/to/CLEAR360_Synthetic_Bedroom_Test/images \
        --output data/phone_reconstruction/synthetic_bedroom_validation \
        --ground-truth /path/to/CLEAR360_Synthetic_Bedroom_Test/ground_truth.json

    python -m photo_reconstruction.diagnose_image_set \
        --images /path/to/images \
        --output data/phone_reconstruction/my_photo_set_sfm \
        --depth-source hypersim

The default mode runs only SfM. Optional depth calibration runs isolated depth
inference and held-out calibration, but NEVER runs MVS fusion, dense reconstruction,
surface meshing, or texturing.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

from photo_reconstruction.sfm_engine import SfMEngine


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _load_ground_truth(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        item["image"]: item
        for item in payload.get("camera_poses", [])
        if item.get("image")
    }


def _rotation_error_deg(R_est: np.ndarray, R_gt: np.ndarray) -> float:
    relative = R_est.T @ R_gt
    value = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(value)))


def _compare_camera_trajectory(
    camera_models: dict[str, Any],
    ground_truth: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    names = sorted(set(camera_models) & set(ground_truth))
    if len(names) < 3:
        return {
            "status": "not_enough_common_cameras",
            "common_cameras": names,
            "minimum_required": 3,
        }

    estimated = np.asarray(
        [camera_models[name].camera_center for name in names],
        dtype=np.float64,
    )
    truth = np.asarray(
        [ground_truth[name]["camera_center_world"] for name in names],
        dtype=np.float64,
    )

    x_mean = estimated.mean(axis=0)
    y_mean = truth.mean(axis=0)
    x_centered = estimated - x_mean
    y_centered = truth - y_mean

    covariance = (y_centered.T @ x_centered) / len(names)
    U, singular_values, Vt = np.linalg.svd(covariance)
    correction = np.eye(3, dtype=np.float64)
    correction[2, 2] = 1.0 if np.linalg.det(U @ Vt) >= 0.0 else -1.0
    rotation = U @ correction @ Vt

    variance = float(np.sum(x_centered * x_centered) / len(names))
    if variance <= 1e-12:
        return {
            "status": "degenerate_estimated_camera_trajectory",
            "common_cameras": names,
        }

    scale = float(np.sum(singular_values * np.diag(correction)) / variance)
    translation = y_mean - scale * (rotation @ x_mean)
    estimated_aligned = (scale * (rotation @ estimated.T)).T + translation

    position_errors = np.linalg.norm(estimated_aligned - truth, axis=1)
    rotation_errors = []
    per_camera = {}

    for index, name in enumerate(names):
        R_est = np.asarray(camera_models[name].T_world_cam[:3, :3], dtype=np.float64)
        R_gt = np.asarray(ground_truth[name]["T_world_cam"], dtype=np.float64)[:3, :3]
        R_aligned = rotation @ R_est
        angle_error = _rotation_error_deg(R_aligned, R_gt)
        rotation_errors.append(angle_error)
        per_camera[name] = {
            "camera_center_error_m": float(position_errors[index]),
            "orientation_error_deg": angle_error,
        }

    return {
        "status": "compared_after_similarity_alignment",
        "common_cameras": names,
        "similarity_scale": scale,
        "camera_center_rmse_m": float(np.sqrt(np.mean(position_errors ** 2))),
        "camera_center_median_error_m": float(np.median(position_errors)),
        "orientation_median_error_deg": float(np.median(rotation_errors)),
        "orientation_p90_error_deg": float(np.percentile(rotation_errors, 90)),
        "per_camera": per_camera,
        "note": (
            "COLMAP has a free similarity gauge. Camera centres were aligned to "
            "the supplied ground truth by one global similarity transform before "
            "position/orientation comparison."
        ),
    }


def _run_depth_calibration(
    image_paths: list[Path],
    output_dir: Path,
    sfm_result: dict[str, Any],
    depth_source: str,
    weights_arg: str | None,
) -> dict[str, Any]:
    from photo_reconstruction.depth_aligner import SfMDepthAligner
    from photo_reconstruction.depth_helper import IsolatedDepthEstimator

    camera_models = sfm_result.get("camera_models", {})
    sparse_tracks = sfm_result.get("sparse_tracks", {})
    if len(camera_models) < 2 or not any(sparse_tracks.values()):
        return {
            "status": "skipped",
            "reason": (
                "SfM did not provide at least two calibrated cameras and sparse "
                "2D-3D tracks. The OpenCV fallback cannot supply these tracks."
            ),
        }

    project_root = Path(__file__).resolve().parent.parent
    max_depth = 20.0 if depth_source == "hypersim" else 80.0
    default_name = (
        "depth_anything_v2_metric_hypersim_vits.pth"
        if depth_source == "hypersim"
        else "depth_anything_v2_metric_vkitti_vits.pth"
    )
    weights_path = (
        Path(weights_arg).expanduser()
        if weights_arg
        else project_root / "weights" / default_name
    )
    if not weights_path.is_file():
        raise FileNotFoundError(f"Depth checkpoint not found: {weights_path}")

    registered_paths = [
        path for path in image_paths if path.name in camera_models
    ]
    cache_dir = output_dir / f"depth_cache_{depth_source}"
    depth_maps = IsolatedDepthEstimator.estimate_depths_for_images(
        image_paths=registered_paths,
        cache_dir=cache_dir,
        project_root=project_root,
        weights_path=weights_path,
        max_depth=max_depth,
        timeout_s=900,
        raise_on_error=True,
    )
    camera_models = {
        name: camera_models[name]
        for name in depth_maps
        if name in camera_models
    }
    sparse_tracks = {
        name: sparse_tracks.get(name, [])
        for name in depth_maps
    }

    aligner = SfMDepthAligner(mode="regularized")
    alignment = aligner.align_depth_maps(
        depth_maps=depth_maps,
        camera_models=camera_models,
        sparse_tracks=sparse_tracks,
    )

    views = {}
    for name, stats in sorted(alignment.view_stats.items()):
        views[name] = {
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

    accepted = sorted(alignment.aligned_depth_maps)
    return {
        "status": "mvs_ready" if len(accepted) >= 2 else "not_ready",
        "depth_source": depth_source,
        "cache_dir": str(cache_dir),
        "checkpoint": str(weights_path),
        "views_checked": len(depth_maps),
        "accepted_views": accepted,
        "rejected_views": sorted(
            name for name, stats in alignment.view_stats.items()
            if not stats.success
        ),
        "training_anchors": int(alignment.total_anchors),
        "global_scale": float(alignment.global_scale),
        "validation_limits": {
            "median_relative_error": float(aligner.max_val_median_rel),
            "p90_relative_error": float(aligner.max_val_p90_rel),
        },
        "views": views,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, help="Folder containing the input JPG/PNG images.")
    parser.add_argument("--output", required=True, help="Output folder for this diagnostic run.")
    parser.add_argument("--ground-truth", default=None, help="Optional ground_truth.json from the synthetic dataset.")
    parser.add_argument(
        "--depth-source",
        choices=("none", "hypersim", "vkitti"),
        default="none",
        help="Default 'none' runs SfM only. Other modes additionally run depth inference and held-out calibration.",
    )
    parser.add_argument("--weights", default=None, help="Optional override for the selected metric checkpoint.")
    args = parser.parse_args()

    image_dir = Path(args.images).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    image_paths = sorted(
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    if len(image_paths) < 2:
        raise RuntimeError(f"Need at least two JPG/PNG images; found {len(image_paths)}.")

    output_dir.mkdir(parents=True, exist_ok=True)
    print("CLEAR360 PHOTO-SET DIAGNOSTIC")
    print(f"Input images: {len(image_paths)}")
    print(f"Output: {output_dir}")
    print("Quality preset: draft (SfM only; no dense reconstruction or mesh)")
    print()

    engine = SfMEngine(
        work_dir=output_dir / "sfm_work",
        quality_preset="draft",
    )
    sfm_result = engine.run_reconstruction(image_dir)

    camera_models = sfm_result.get("camera_models", {})
    sparse_tracks = sfm_result.get("sparse_tracks", {})
    per_image_tracks = {
        name: len(sparse_tracks.get(name, []))
        for name in sorted(camera_models)
    }
    sfm_summary = {
        "success": bool(sfm_result.get("success", False)),
        "uploaded_images": len(image_paths),
        "registered_images": int(sfm_result.get("registered_count", 0)),
        "failed_images": list(sfm_result.get("failed_images", [])),
        "sparse_points": int(len(sfm_result.get("sparse_points", []))),
        "mean_reprojection_error_px": float(
            sfm_result.get("mean_reprojection_error", 0.0)
        ),
        "camera_names": sorted(camera_models),
        "tracks_per_image": per_image_tracks,
        "message": str(sfm_result.get("message", "")),
    }

    if camera_models:
        pose_dir = output_dir / "camera_poses"
        pose_dir.mkdir(parents=True, exist_ok=True)
        for name, camera in camera_models.items():
            np.save(pose_dir / f"{Path(name).stem}_pose.npy", camera.T_world_cam)

    gt_path = Path(args.ground_truth).expanduser() if args.ground_truth else None
    ground_truth = _load_ground_truth(gt_path)
    trajectory_report = (
        _compare_camera_trajectory(camera_models, ground_truth)
        if ground_truth
        else {"status": "not_requested"}
    )

    report = {
        "input_dir": str(image_dir),
        "sfm": sfm_summary,
        "ground_truth_comparison": trajectory_report,
        "depth_calibration": {"status": "not_requested"},
        "note": (
            "This diagnostic ends after sparse SfM unless --depth-source is selected. "
            "It never runs dense MVS, dense point-cloud generation, meshing, or texturing."
        ),
    }

    print(f"SfM status: {'SUCCESS' if sfm_summary['success'] else 'FAILED'}")
    print(
        f"Registered: {sfm_summary['registered_images']}/{sfm_summary['uploaded_images']}"
    )
    print(f"Sparse points: {sfm_summary['sparse_points']:,}")
    print(f"Mean reprojection error: {sfm_summary['mean_reprojection_error_px']:.3f} px")
    if per_image_tracks:
        print("2D-3D tracks per image:")
        for name, count in per_image_tracks.items():
            print(f"  {name}: {count}")

    if args.depth_source != "none" and sfm_summary["success"]:
        print()
        print(f"Running isolated {args.depth_source} depth + held-out calibration only...")
        report["depth_calibration"] = _run_depth_calibration(
            image_paths,
            output_dir,
            sfm_result,
            args.depth_source,
            args.weights,
        )
        depth_report = report["depth_calibration"]
        print(
            f"Depth views accepted: {len(depth_report.get('accepted_views', []))}/"
            f"{depth_report.get('views_checked', 0)}"
        )

    report_path = output_dir / "image_set_diagnostic.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print()
    print(f"Saved report: {report_path}")
    if ground_truth:
        print("Ground-truth camera comparison:")
        print(json.dumps(trajectory_report, indent=2))
    return 0 if sfm_summary["success"] else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Image-set diagnostic failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
