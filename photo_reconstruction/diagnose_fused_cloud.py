"""
Diagnose spatial fragmentation in a fused CLEAR360 point cloud.

Runs a small DBSCAN scale sweep using Open3D and exports a cluster-coloured PLY
at the selected epsilon factor. This is a point-cloud diagnostic only: it does
not alter the source cloud or run normals, reconstruction, meshing, or texturing.

DBSCAN components are density-connected groups, not guaranteed mesh components.
Interpret them together with the rendered cloud and the full epsilon sweep.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d


def _nearest_neighbor_summary(distances: np.ndarray) -> dict[str, float]:
    values = np.asarray(distances, dtype=np.float64).ravel()
    values = values[np.isfinite(values) & (values > 0.0)]
    if len(values) == 0:
        raise ValueError("Could not estimate positive nearest-neighbour spacing.")
    return {
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def _summarize_labels(labels: np.ndarray, total_points: int) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64).ravel()
    if len(labels) != total_points:
        raise ValueError("Cluster-label count must equal point count.")

    valid_labels = labels[labels >= 0]
    cluster_ids, counts = np.unique(valid_labels, return_counts=True)
    sizes = sorted((int(x) for x in counts), reverse=True)
    noise_count = int(np.count_nonzero(labels < 0))
    largest = sizes[0] if sizes else 0
    return {
        "cluster_count": int(len(cluster_ids)),
        "noise_points": noise_count,
        "noise_fraction": float(noise_count / max(total_points, 1)),
        "largest_cluster_points": largest,
        "largest_cluster_fraction_of_all_points": float(
            largest / max(total_points, 1)
        ),
        "top_10_cluster_sizes": sizes[:10],
    }


def analyze_cloud(
    input_ply: Path,
    output_dir: Path,
    eps_factors: tuple[float, ...] = (1.5, 2.0, 3.0, 4.0, 6.0),
    selected_eps_factor: float = 3.0,
    min_points: int = 8,
) -> dict[str, Any]:
    if not input_ply.is_file():
        raise FileNotFoundError(f"Point-cloud file not found: {input_ply}")
    if not eps_factors or any(x <= 0 for x in eps_factors):
        raise ValueError("All --eps-factors values must be positive.")
    if selected_eps_factor not in eps_factors:
        raise ValueError("--selected-eps-factor must appear in --eps-factors.")
    if min_points < 2:
        raise ValueError("--min-points must be at least 2.")

    cloud = o3d.io.read_point_cloud(str(input_ply))
    points = np.asarray(cloud.points, dtype=np.float64)
    if len(points) < min_points:
        raise RuntimeError(
            f"Point cloud has only {len(points)} points; at least {min_points} required."
        )

    finite_mask = np.isfinite(points).all(axis=1)
    removed_nonfinite = int(np.count_nonzero(~finite_mask))
    if removed_nonfinite:
        cloud = cloud.select_by_index(np.flatnonzero(finite_mask).tolist())
        points = np.asarray(cloud.points, dtype=np.float64)
    if len(points) < min_points:
        raise RuntimeError("Too few finite points remain after filtering.")

    nearest = np.asarray(cloud.compute_nearest_neighbor_distance(), dtype=np.float64)
    nn = _nearest_neighbor_summary(nearest)
    base_spacing = nn["median"]
    if not np.isfinite(base_spacing) or base_spacing <= 0:
        raise RuntimeError("Median nearest-neighbour spacing is invalid.")

    sweep = []
    labels_by_factor: dict[float, np.ndarray] = {}
    for factor in eps_factors:
        eps = float(base_spacing * factor)
        labels = np.asarray(
            cloud.cluster_dbscan(
                eps=eps,
                min_points=int(min_points),
                print_progress=False,
            ),
            dtype=np.int64,
        )
        labels_by_factor[float(factor)] = labels
        summary = _summarize_labels(labels, len(points))
        sweep.append({
            "eps_factor": float(factor),
            "eps_sfm_units": eps,
            **summary,
        })

    chosen_labels = labels_by_factor[float(selected_eps_factor)]
    chosen_summary = _summarize_labels(chosen_labels, len(points))

    # Give each density cluster a deterministic colour; noise is light grey.
    cluster_ids = sorted(int(x) for x in np.unique(chosen_labels) if x >= 0)
    rng = np.random.default_rng(360)
    palette = rng.uniform(0.15, 0.95, size=(len(cluster_ids), 3))
    colour_by_id = {cluster_id: palette[i] for i, cluster_id in enumerate(cluster_ids)}
    colours = np.empty((len(points), 3), dtype=np.float64)
    for i, label in enumerate(chosen_labels):
        colours[i] = (
            colour_by_id[int(label)]
            if label >= 0
            else np.array([0.55, 0.55, 0.55], dtype=np.float64)
        )

    coloured = o3d.geometry.PointCloud()
    coloured.points = o3d.utility.Vector3dVector(points)
    coloured.colors = o3d.utility.Vector3dVector(colours)
    if cloud.has_normals():
        normals = np.asarray(cloud.normals, dtype=np.float64)
        if len(normals) == len(points):
            coloured.normals = o3d.utility.Vector3dVector(normals)

    lower = np.min(points, axis=0)
    upper = np.max(points, axis=0)
    output_dir.mkdir(parents=True, exist_ok=True)
    cluster_ply = output_dir / "cluster_colored_cloud.ply"
    if not o3d.io.write_point_cloud(str(cluster_ply), coloured, write_ascii=False):
        raise RuntimeError(f"Could not write cluster-coloured PLY: {cluster_ply}")

    report = {
        "input_ply": str(input_ply),
        "input_points": int(len(points)),
        "removed_nonfinite_points": removed_nonfinite,
        "bounds_min_sfm_units": lower.tolist(),
        "bounds_max_sfm_units": upper.tolist(),
        "bounds_extent_sfm_units": (upper - lower).tolist(),
        "nearest_neighbor_spacing_sfm_units": nn,
        "dbscan_min_points": int(min_points),
        "selected_eps_factor": float(selected_eps_factor),
        "selected_eps_sfm_units": float(base_spacing * selected_eps_factor),
        "selected_cluster_summary": chosen_summary,
        "epsilon_sweep": sweep,
        "cluster_colored_ply": str(cluster_ply),
        "interpretation_note": (
            "DBSCAN measures density-connected point groups at selected radii. "
            "A component can split because of sparse sampling or join because eps "
            "is large; use the epsilon sweep and cluster-coloured PLY together. "
            "These are not mesh connected components and do not certify surface quality."
        ),
    }
    report_path = output_dir / "point_cloud_fragmentation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Fused point-cloud PLY.")
    parser.add_argument("--output", required=True, help="Directory for diagnostic outputs.")
    parser.add_argument(
        "--eps-factors",
        nargs="+",
        type=float,
        default=[1.5, 2.0, 3.0, 4.0, 6.0],
        help="DBSCAN radii as multiples of median nearest-neighbour spacing.",
    )
    parser.add_argument("--selected-eps-factor", type=float, default=3.0)
    parser.add_argument("--min-points", type=int, default=8)
    args = parser.parse_args()

    report = analyze_cloud(
        input_ply=Path(args.input).expanduser().resolve(),
        output_dir=Path(args.output).expanduser().resolve(),
        eps_factors=tuple(float(x) for x in args.eps_factors),
        selected_eps_factor=float(args.selected_eps_factor),
        min_points=int(args.min_points),
    )

    print("CLEAR360 POINT-CLOUD FRAGMENTATION DIAGNOSTIC")
    print(f"Points: {report['input_points']:,}")
    print(
        "Median nearest-neighbour spacing: "
        f"{report['nearest_neighbor_spacing_sfm_units']['median']:.6f} scene units"
    )
    print()
    print(
        f"{'EPS FACTOR':>10} {'EPS':>12} {'CLUSTERS':>10} "
        f"{'LARGEST %':>11} {'NOISE %':>9} {'LARGEST N':>10}"
    )
    print("-" * 70)
    for row in report["epsilon_sweep"]:
        print(
            f"{row['eps_factor']:10.2f} {row['eps_sfm_units']:12.6f} "
            f"{row['cluster_count']:10d} "
            f"{100*row['largest_cluster_fraction_of_all_points']:10.2f}% "
            f"{100*row['noise_fraction']:8.2f}% "
            f"{row['largest_cluster_points']:10d}"
        )
    print()
    print(f"Cluster-coloured PLY: {report['cluster_colored_ply']}")
    print(f"JSON report: {report['report_path']}")
    print("No reconstruction, meshing, or texturing was run.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Point-cloud fragmentation diagnostic failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
