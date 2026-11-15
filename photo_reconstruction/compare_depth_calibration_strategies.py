"""
Compare strict held-out metric-depth calibration strategies on one cached photo set.

Runs GLOBAL, PER_VIEW, and REGULARIZED fits against the same saved COLMAP
reconstruction, depth maps, sparse tracks, deterministic train/validation splits,
and unchanged validation limits. No depth inference or reconstruction is run.

The purpose is to diagnose whether pooled/global regularization is the main reason
views are rejected. A passing result is a calibration diagnostic, not a guarantee
that multi-view fusion or a final mesh will be good.
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


def summarize_view_stats(view_stats: dict[str, Any]) -> dict[str, Any]:
    """Return stable per-strategy acceptance totals and per-view held-out errors."""
    accepted = sorted(
        name for name, stats in view_stats.items() if bool(stats.success)
    )
    rejected = sorted(
        name for name, stats in view_stats.items() if not bool(stats.success)
    )
    views = {}
    for name, stats in sorted(view_stats.items()):
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
            "status": str(stats.status_message),
        }

    return {
        "accepted_count": len(accepted),
        "views_checked": len(view_stats),
        "accepted_views": accepted,
        "rejected_views": rejected,
        "views": views,
    }


def compare_strategies(
    depth_maps: dict[str, np.ndarray],
    camera_models: dict[str, Any],
    sparse_tracks: dict[str, list[dict[str, Any]]],
    modes: tuple[str, ...] = ("global", "per_view", "regularized"),
    median_limit: float = 0.04,
    p90_limit: float = 0.12,
    min_anchors: int = 15,
) -> dict[str, Any]:
    if not depth_maps:
        raise ValueError("No cached depth maps were supplied.")
    if median_limit <= 0 or p90_limit <= 0:
        raise ValueError("Validation limits must be positive.")
    if not modes or any(mode not in {"global", "per_view", "regularized"} for mode in modes):
        raise ValueError("Modes must be selected from global, per_view, regularized.")

    summaries = {}
    for mode in modes:
        aligner = SfMDepthAligner(
            mode=mode,
            min_anchors=min_anchors,
            max_validation_median_relative_error=median_limit,
            max_validation_p90_relative_error=p90_limit,
        )
        alignment = aligner.align_depth_maps(
            depth_maps=depth_maps,
            camera_models=camera_models,
            sparse_tracks=sparse_tracks,
        )
        summary = summarize_view_stats(alignment.view_stats)
        summary.update({
            "mode": mode,
            "pooled_global_scale": float(alignment.global_scale),
            "training_anchor_count": int(alignment.total_anchors),
            "validation_limits": {
                "median_relative_error": float(aligner.max_val_median_rel),
                "p90_relative_error": float(aligner.max_val_p90_rel),
            },
        })
        summaries[mode] = summary

    per_view_comparison = {}
    all_names = sorted({
        name
        for summary in summaries.values()
        for name in summary["views"]
    })
    for name in all_names:
        per_view_comparison[name] = {}
        for mode, summary in summaries.items():
            view = summary["views"].get(name)
            if view is not None:
                per_view_comparison[name][mode] = {
                    "median_pct": 100.0 * view["validation_median_relative_error"],
                    "p90_pct": 100.0 * view["validation_p90_relative_error"],
                    "accepted": view["accepted"],
                    "scale": view["scale"],
                    "offset": view["offset"],
                    "usable_anchors": view["usable_anchors"],
                }

    best_mode = None
    if summaries:
        # Rank by accepted count first; tie-break by lower total normalized
        # validation error (median + P90), not by visual appearance.
        def rank(item: tuple[str, dict[str, Any]]) -> tuple[float, float, str]:
            mode, summary = item
            total_error = 0.0
            for view in summary["views"].values():
                total_error += (
                    view["validation_median_relative_error"] / median_limit
                    + view["validation_p90_relative_error"] / p90_limit
                )
            return (
                -float(summary["accepted_count"]),
                total_error,
                mode,
            )

        best_mode = sorted(summaries.items(), key=rank)[0][0]

    return {
        "status": "comparison_completed",
        "depth_map_count": len(depth_maps),
        "camera_count": len(camera_models),
        "validation_method": (
            "Same SfM tracks, image-wise deterministic held-out split, anchor filters, "
            "and identical median/P90 acceptance thresholds for every strategy."
        ),
        "strategies": summaries,
        "per_view_comparison": per_view_comparison,
        "highest_acceptance_strategy": best_mode,
        "interpretation_warning": (
            "A strategy with more accepted views is not automatically more physically correct. "
            "Inspect per-view scale/offset stability and downstream geometric consistency. "
            "Do not relax the held-out error limits based on this comparison."
        ),
    }


def _load_depth_maps(cache_dir: Path, names: list[str]) -> dict[str, np.ndarray]:
    maps = {}
    missing = []
    for name in names:
        path = cache_dir / f"{Path(name).stem}_depth.npy"
        if not path.is_file():
            missing.append(str(path))
            continue
        depth = np.load(path, allow_pickle=False)
        if depth.ndim != 2:
            raise RuntimeError(f"Expected 2D depth map at {path}, got {depth.shape}.")
        maps[name] = np.asarray(depth, dtype=np.float32)
    if missing:
        raise FileNotFoundError("Missing cached depth maps:\n  " + "\n  ".join(missing))
    if not maps:
        raise RuntimeError(f"No depth maps found in {cache_dir}.")
    return maps


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        required=True,
        help="Existing diagnostic output folder containing sfm_work/ and camera_poses/.",
    )
    parser.add_argument(
        "--depth-cache",
        default=None,
        help="Depth cache folder (default: OUTPUT/depth_cache_hypersim).",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("global", "per_view", "regularized"),
        default=["global", "per_view", "regularized"],
    )
    parser.add_argument("--median-limit", type=float, default=0.04)
    parser.add_argument("--p90-limit", type=float, default=0.12)
    parser.add_argument("--min-anchors", type=int, default=15)
    args = parser.parse_args()

    output_dir = Path(args.output).expanduser().resolve()
    cache_dir = (
        Path(args.depth_cache).expanduser().resolve()
        if args.depth_cache
        else output_dir / "depth_cache_hypersim"
    )
    if not output_dir.is_dir():
        raise FileNotFoundError(f"Diagnostic output directory not found: {output_dir}")
    if not cache_dir.is_dir():
        raise FileNotFoundError(f"Depth cache directory not found: {cache_dir}")

    model_dir, reconstruction = _load_best_colmap_model(
        output_dir / "sfm_work" / "sparse"
    )
    pose_dir = output_dir / "camera_poses"
    saved_pose_stems = {
        path.stem[:-5] for path in pose_dir.glob("*_pose.npy")
    } if pose_dir.is_dir() else set()
    camera_models, sparse_tracks = _build_camera_models_and_tracks(
        reconstruction, saved_pose_stems
    )
    if not camera_models:
        raise RuntimeError("No camera models/tracks matched the saved COLMAP model.")

    depth_maps = _load_depth_maps(cache_dir, sorted(camera_models))
    camera_models = {name: camera_models[name] for name in depth_maps}
    sparse_tracks = {
        name: sparse_tracks.get(name, []) for name in depth_maps
    }

    report = compare_strategies(
        depth_maps=depth_maps,
        camera_models=camera_models,
        sparse_tracks=sparse_tracks,
        modes=tuple(args.modes),
        median_limit=args.median_limit,
        p90_limit=args.p90_limit,
        min_anchors=args.min_anchors,
    )
    report.update({
        "output_dir": str(output_dir),
        "depth_cache": str(cache_dir),
        "colmap_model": str(model_dir),
        "limits": {
            "median_relative_error": float(args.median_limit),
            "p90_relative_error": float(args.p90_limit),
        },
    })

    output_path = output_dir / "calibration_strategy_comparison.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("CLEAR360 DEPTH CALIBRATION STRATEGY COMPARISON")
    print(f"Depth views: {report['depth_map_count']}")
    print(f"Fixed limits: median <= {args.median_limit:.1%}, p90 <= {args.p90_limit:.1%}")
    print()
    print(
        f"{'STRATEGY':14} {'ACCEPTED':>10} {'GLOBAL SCALE':>13} "
        f"{'TRAIN ANCHORS':>14}"
    )
    print("-" * 56)
    for mode in args.modes:
        row = report["strategies"][mode]
        print(
            f"{mode:14} {row['accepted_count']:>4}/{row['views_checked']:<5} "
            f"{row['pooled_global_scale']:13.6f} "
            f"{row['training_anchor_count']:14d}"
        )
    print()
    for name, modes in report["per_view_comparison"].items():
        print(name)
        for mode in args.modes:
            item = modes.get(mode)
            if item is None:
                continue
            status = "PASS" if item["accepted"] else "FAIL"
            print(
                f"  {mode:12} median={item['median_pct']:6.2f}% "
                f"p90={item['p90_pct']:6.2f}% "
                f"scale={item['scale']:.5f} offset={item['offset']:.5f} {status}"
            )
    print()
    print(f"Highest acceptance (diagnostic ranking only): {report['highest_acceptance_strategy']}")
    print(f"Saved report: {output_path}")
    print("No depth inference, MVS, point-cloud fusion, meshing, or texturing was run.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Calibration strategy comparison failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
