"""
Low-memory two-view geometric-consensus preflight for cached CLEAR360 depth maps.

This diagnostic:
  * loads an existing COLMAP sparse model and its 2D-3D tracks;
  * recalibrates cached monocular depth maps using the same held-out aligner;
  * keeps only the requested views that pass calibration (default: 5.jpg and 8.jpg);
  * performs the production M7.3.6 pairwise depth + 3D consistency test in both directions;
  * reports whether the production >=1000 verified-source-sample gate is reached.

It does NOT run dense MVS, Open3D normal estimation, voxel fusion, meshing, or texturing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from photo_reconstruction.depth_aligner import SfMDepthAligner
from photo_reconstruction.diagnose_cached_depth import (
    _build_camera_models_and_tracks,
    _load_best_colmap_model,
)


def _bilinear_sample(depth: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Bilinearly sample a 2D map using the same border convention as M7.3.6."""
    height, width = depth.shape
    if height < 2 or width < 2:
        raise ValueError("Depth maps must be at least 2x2 for bilinear sampling.")

    u0 = np.clip(np.floor(u).astype(np.int32), 0, width - 2)
    v0 = np.clip(np.floor(v).astype(np.int32), 0, height - 2)
    du = u - u0
    dv = v - v0

    d00 = depth[v0, u0]
    d01 = depth[v0, u0 + 1]
    d10 = depth[v0 + 1, u0]
    d11 = depth[v0 + 1, u0 + 1]
    return (
        (1.0 - du) * (1.0 - dv) * d00
        + du * (1.0 - dv) * d01
        + (1.0 - du) * dv * d10
        + du * dv * d11
    )


def _adaptive_depth_threshold(depth_z: np.ndarray) -> np.ndarray:
    """Mirror M736GeometricConsensusEngine.adaptive_depth_threshold."""
    z = np.clip(np.asarray(depth_z, dtype=np.float64), 1.0, 30.0)
    return np.clip(0.005 * z + 0.001 * (z ** 2), 0.015, 0.050)


def _evaluate_direction(
    source_name: str,
    target_name: str,
    source_camera: Any,
    target_camera: Any,
    source_depth: np.ndarray,
    target_depth: np.ndarray,
    stride: int,
) -> dict[str, Any]:
    """
    Test source-view depth samples against the target view.

    A source sample is verified only when both axial depth residual <= tau(z)
    and 3D back-projection distance <= 2*tau(z), matching the M7.3.6 gate.
    Counts are directional source samples, not deduplicated physical landmarks.
    """
    height, width = source_depth.shape
    v_grid, u_grid = np.mgrid[0:height:stride, 0:width:stride]
    sampled_source_depth = source_depth[v_grid, u_grid]

    source_valid = (
        np.isfinite(sampled_source_depth)
        & (sampled_source_depth > 0.5)
        & (sampled_source_depth < 30.0)
    )
    source_pixels = np.column_stack(
        [u_grid[source_valid], v_grid[source_valid]]
    ).astype(np.float64)
    source_z = sampled_source_depth[source_valid].astype(np.float64)

    candidate_count = int(len(source_z))
    if candidate_count == 0:
        return {
            "source": source_name,
            "target": target_name,
            "candidate_samples": 0,
            "projected_samples": 0,
            "target_depth_valid_samples": 0,
            "depth_agreement_samples": 0,
            "spatial_agreement_samples": 0,
            "verified_samples": 0,
            "not_projected_samples": 0,
            "no_target_depth_samples": 0,
            "inconsistent_samples": 0,
            "verified_pct_of_candidates": 0.0,
            "verified_pct_of_observed": 0.0,
        }

    source_points_world = source_camera.unproject_image_to_world(
        source_pixels, source_z
    )
    projected_pixels, projected_z, valid_geometry = (
        target_camera.project_world_to_image(
            source_points_world, check_bounds=True
        )
    )
    valid_geometry = (
        np.asarray(valid_geometry, dtype=bool)
        & np.isfinite(projected_pixels).all(axis=1)
        & np.isfinite(projected_z)
        & (projected_z > 0.5)
    )

    projected_indices = np.flatnonzero(valid_geometry)
    projected_count = int(len(projected_indices))
    if projected_count == 0:
        return {
            "source": source_name,
            "target": target_name,
            "candidate_samples": candidate_count,
            "projected_samples": 0,
            "target_depth_valid_samples": 0,
            "depth_agreement_samples": 0,
            "spatial_agreement_samples": 0,
            "verified_samples": 0,
            "not_projected_samples": candidate_count,
            "no_target_depth_samples": 0,
            "inconsistent_samples": 0,
            "verified_pct_of_candidates": 0.0,
            "verified_pct_of_observed": 0.0,
        }

    uv = projected_pixels[projected_indices]
    u = uv[:, 0]
    v = uv[:, 1]
    target_sampled_depth = _bilinear_sample(target_depth, u, v)
    target_depth_valid = (
        np.isfinite(target_sampled_depth)
        & (target_sampled_depth > 0.5)
        & (target_sampled_depth < 30.0)
    )

    valid_indices_in_projected = np.flatnonzero(target_depth_valid)
    valid_source_indices = projected_indices[valid_indices_in_projected]
    valid_u = u[valid_indices_in_projected]
    valid_v = v[valid_indices_in_projected]
    valid_projected_z = projected_z[valid_source_indices]
    valid_target_z = target_sampled_depth[valid_indices_in_projected]

    if len(valid_target_z) == 0:
        return {
            "source": source_name,
            "target": target_name,
            "candidate_samples": candidate_count,
            "projected_samples": projected_count,
            "target_depth_valid_samples": 0,
            "depth_agreement_samples": 0,
            "spatial_agreement_samples": 0,
            "verified_samples": 0,
            "not_projected_samples": candidate_count - projected_count,
            "no_target_depth_samples": projected_count,
            "inconsistent_samples": 0,
            "verified_pct_of_candidates": 0.0,
            "verified_pct_of_observed": 0.0,
        }

    target_pixels = np.column_stack([valid_u, valid_v])
    target_backprojected_world = target_camera.unproject_image_to_world(
        target_pixels, valid_target_z
    )
    spatial_distance = np.linalg.norm(
        source_points_world[valid_source_indices] - target_backprojected_world,
        axis=1,
    )
    depth_residual = np.abs(valid_projected_z - valid_target_z)
    tau = _adaptive_depth_threshold(valid_projected_z)

    depth_agrees = depth_residual <= tau
    spatial_agrees = spatial_distance <= (2.0 * tau)
    verified = depth_agrees & spatial_agrees

    verified_count = int(np.count_nonzero(verified))
    observed_count = int(len(valid_target_z))
    return {
        "source": source_name,
        "target": target_name,
        "candidate_samples": candidate_count,
        "projected_samples": projected_count,
        "target_depth_valid_samples": observed_count,
        "depth_agreement_samples": int(np.count_nonzero(depth_agrees)),
        "spatial_agreement_samples": int(np.count_nonzero(spatial_agrees)),
        "verified_samples": verified_count,
        "not_projected_samples": candidate_count - projected_count,
        "no_target_depth_samples": projected_count - observed_count,
        "inconsistent_samples": observed_count - verified_count,
        "verified_pct_of_candidates": round(
            100.0 * verified_count / max(candidate_count, 1), 2
        ),
        "verified_pct_of_observed": round(
            100.0 * verified_count / max(observed_count, 1), 2
        ),
        "depth_residual_median_m": float(np.median(depth_residual)),
        "depth_residual_p90_m": float(np.percentile(depth_residual, 90)),
        "spatial_residual_median_m": float(np.median(spatial_distance)),
        "spatial_residual_p90_m": float(np.percentile(spatial_distance, 90)),
    }


def _load_depth_maps(cache_dir: Path, camera_names: list[str]) -> dict[str, np.ndarray]:
    depth_maps: dict[str, np.ndarray] = {}
    missing = []
    for name in camera_names:
        depth_path = cache_dir / f"{Path(name).stem}_depth.npy"
        if not depth_path.is_file():
            missing.append(str(depth_path))
            continue
        depth = np.load(depth_path, allow_pickle=False)
        if depth.ndim != 2:
            raise RuntimeError(f"Expected a 2D depth map in {depth_path}; got {depth.shape}.")
        depth_maps[name] = np.asarray(depth, dtype=np.float32)

    if missing:
        raise FileNotFoundError(
            "Missing cached depth maps:\n  " + "\n  ".join(missing)
        )
    return depth_maps


def run_preflight(
    images_dir: Path,
    output_dir: Path,
    cache_dir: Path,
    view_names: list[str],
    stride: int = 6,
    min_verified_points: int = 1000,
) -> dict[str, Any]:
    if stride < 1:
        raise ValueError("--stride must be >= 1.")
    if len(view_names) != 2 or view_names[0] == view_names[1]:
        raise ValueError("Exactly two distinct --views must be supplied.")

    for name in view_names:
        if not (images_dir / name).is_file():
            raise FileNotFoundError(f"Requested source image not found: {images_dir / name}")

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
        raise RuntimeError("No saved COLMAP camera models or 2D-3D tracks were recovered.")

    depth_maps = _load_depth_maps(cache_dir, sorted(camera_models))
    camera_models = {name: camera_models[name] for name in depth_maps}
    sparse_tracks = {
        name: sparse_tracks.get(name, []) for name in depth_maps
    }

    aligner = SfMDepthAligner(mode="regularized")
    alignment = aligner.align_depth_maps(
        depth_maps=depth_maps,
        camera_models=camera_models,
        sparse_tracks=sparse_tracks,
    )

    requested = set(view_names)
    accepted = set(alignment.aligned_depth_maps)
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
            "status": stats.status_message,
        }

    result: dict[str, Any] = {
        "status": "not_run",
        "images_dir": str(images_dir),
        "output_dir": str(output_dir),
        "depth_cache": str(cache_dir),
        "colmap_model": str(model_dir),
        "views_requested": view_names,
        "sample_stride_px": int(stride),
        "depth_calibration": {
            "views_checked": len(depth_maps),
            "accepted_views": sorted(accepted),
            "training_anchors": int(alignment.total_anchors),
            "validation_limits": {
                "median_relative_error": float(aligner.max_val_median_rel),
                "p90_relative_error": float(aligner.max_val_p90_rel),
            },
            "views": per_view_calibration,
        },
        "pairwise_checks": [],
        "minimum_verified_source_samples": int(min_verified_points),
        "verified_source_samples_total": 0,
        "candidate_source_samples_total": 0,
        "note": (
            "Verified counts are directional source-depth samples before voxel deduplication. "
            "This mirrors the production pre-fusion verified-point count; it does not "
            "run normal estimation, voxel fusion, MVS, meshing, or texturing."
        ),
    }

    not_calibrated = sorted(requested - accepted)
    if not_calibrated:
        result["status"] = "depth_pair_not_calibrated"
        result["uncalibrated_requested_views"] = not_calibrated
        return result

    # Match production MVS: scale camera intrinsics to each depth-map grid.
    pair_cameras = {}
    for name in view_names:
        depth = alignment.aligned_depth_maps[name]
        height, width = depth.shape
        pair_cameras[name] = camera_models[name].rescale(width, height)

    checks = [
        _evaluate_direction(
            view_names[0], view_names[1],
            pair_cameras[view_names[0]], pair_cameras[view_names[1]],
            alignment.aligned_depth_maps[view_names[0]],
            alignment.aligned_depth_maps[view_names[1]],
            stride,
        ),
        _evaluate_direction(
            view_names[1], view_names[0],
            pair_cameras[view_names[1]], pair_cameras[view_names[0]],
            alignment.aligned_depth_maps[view_names[1]],
            alignment.aligned_depth_maps[view_names[0]],
            stride,
        ),
    ]
    candidates_total = sum(item["candidate_samples"] for item in checks)
    verified_total = sum(item["verified_samples"] for item in checks)
    result["pairwise_checks"] = checks
    result["candidate_source_samples_total"] = int(candidates_total)
    result["verified_source_samples_total"] = int(verified_total)
    result["verified_source_sample_rate_pct"] = round(
        100.0 * verified_total / max(candidates_total, 1), 2
    )
    result["camera_baseline_m"] = float(
        np.linalg.norm(
            pair_cameras[view_names[0]].camera_center
            - pair_cameras[view_names[1]].camera_center
        )
    )
    result["status"] = (
        "pre_fusion_point_gate_pass"
        if verified_total >= min_verified_points
        else "pre_fusion_point_gate_fail"
    )
    result["production_point_gate_pass"] = bool(
        verified_total >= min_verified_points
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, help="Directory containing the synthetic source JPGs.")
    parser.add_argument("--output", required=True, help="Existing output directory from diagnose_image_set.")
    parser.add_argument(
        "--depth-cache",
        default=None,
        help="Cached aligned-source depth directory (default: OUTPUT/depth_cache_hypersim).",
    )
    parser.add_argument(
        "--views",
        nargs=2,
        default=["5.jpg", "8.jpg"],
        metavar=("SOURCE_A.jpg", "SOURCE_B.jpg"),
        help="Exactly two calibrated views to compare (default: 5.jpg 8.jpg).",
    )
    parser.add_argument("--stride", type=int, default=6, help="Pixel sampling stride; 6 matches M7.3.6.")
    parser.add_argument(
        "--min-verified-points",
        type=int,
        default=1000,
        help="Production pre-fusion verified-point minimum (default: 1000).",
    )
    args = parser.parse_args()

    images_dir = Path(args.images).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    cache_dir = (
        Path(args.depth_cache).expanduser().resolve()
        if args.depth_cache
        else output_dir / "depth_cache_hypersim"
    )
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {images_dir}")
    if not output_dir.is_dir():
        raise FileNotFoundError(f"Diagnostic output directory not found: {output_dir}")
    if not cache_dir.is_dir():
        raise FileNotFoundError(f"Depth cache directory not found: {cache_dir}")

    report = run_preflight(
        images_dir=images_dir,
        output_dir=output_dir,
        cache_dir=cache_dir,
        view_names=list(args.views),
        stride=args.stride,
        min_verified_points=args.min_verified_points,
    )

    report_path = output_dir / "two_view_consensus_preflight.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("CLEAR360 TWO-VIEW GEOMETRIC-CONSENSUS PREFLIGHT")
    print(f"Views: {', '.join(args.views)}")
    print(f"Depth calibration accepted: {', '.join(report['depth_calibration']['accepted_views'])}")
    print(f"Status: {report['status']}")
    if report["pairwise_checks"]:
        print()
        print(
            f"{'DIRECTION':18} {'CANDIDATES':>11} {'PROJECTED':>10} "
            f"{'DEPTH OK':>9} {'VERIFIED':>10} {'VERIFIED %':>11}"
        )
        print("-" * 76)
        for item in report["pairwise_checks"]:
            direction = f"{item['source']} -> {item['target']}"
            print(
                f"{direction:18} {item['candidate_samples']:11d} "
                f"{item['projected_samples']:10d} "
                f"{item['target_depth_valid_samples']:9d} "
                f"{item['verified_samples']:10d} "
                f"{item['verified_pct_of_candidates']:10.2f}%"
            )
        print()
        print(
            f"Verified source samples total: "
            f"{report['verified_source_samples_total']:,} / "
            f"{report['minimum_verified_source_samples']:,}"
        )
        print(f"Camera baseline: {report['camera_baseline_m']:.4f} m")
    else:
        print("Requested pair did not both pass held-out depth calibration.")
        print(
            "Requested view statuses:",
            json.dumps(
                {
                    name: report["depth_calibration"]["views"].get(name)
                    for name in args.views
                },
                indent=2,
            ),
        )
    print(f"Saved report: {report_path}")
    return 0 if report["status"] == "pre_fusion_point_gate_pass" else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Two-view preflight failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
