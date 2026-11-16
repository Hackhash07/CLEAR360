"""Create per-photo overlays showing where the saved dense cloud projects."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from photo_reconstruction.coverage_diagnostics import (
    parse_cameras_txt,
    parse_images_txt,
    parse_points3d_tracks,
    project_points,
)


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Project the saved dense cloud into every original source photo."
    )
    parser.add_argument("session", help="Session folder name, e.g. session_022")
    parser.add_argument(
        "--data-root", type=Path,
        default=project_root / "data" / "phone_reconstruction",
    )
    parser.add_argument("--max-points", type=int, default=1_500_000)
    args = parser.parse_args()

    session = args.data_root.expanduser().resolve() / args.session
    work = session / "openmvs"
    images_dir = session / "original_images"
    cameras_path = work / "sparse_text" / "cameras.txt"
    images_path = work / "sparse_text" / "images.txt"
    points_path = work / "sparse_text" / "points3D.txt"

    if not images_dir.is_dir():
        raise FileNotFoundError(f"Original session photos not found: {images_dir}")
    for path in (cameras_path, images_path, points_path):
        if not path.is_file():
            raise FileNotFoundError(f"COLMAP text model file not found: {path}")

    # The alignment script backed up the original-world-frame dense cloud.
    # Use it because COLMAP camera poses are still expressed in that frame.
    dense_ply = work / "scene_fuse_dense.ply.before_orientation"
    if not dense_ply.is_file():
        dense_ply = work / "scene_fuse_dense.ply"
        marker = session / "orientation_alignment.json"
        if marker.is_file():
            raise FileNotFoundError(
                "The dense cloud is already oriented but the original-frame "
                "backup is missing. Restore scene_fuse_dense.ply.before_orientation "
                "or rerun the diagnostic on an unaligned dense cloud."
            )
    if not dense_ply.is_file():
        raise FileNotFoundError(f"Dense point cloud not found: {dense_ply}")

    cameras = parse_cameras_txt(cameras_path)
    images = parse_images_txt(images_path)
    sparse_count, observations, pairwise = parse_points3d_tracks(points_path)

    cloud = o3d.io.read_point_cloud(str(dense_ply))
    if len(cloud.points) == 0:
        raise ValueError(f"Dense point cloud is empty: {dense_ply}")
    points = np.asarray(cloud.points, dtype=np.float64)
    if len(points) > args.max_points:
        stride = int(np.ceil(len(points) / args.max_points))
        points = points[::stride]

    output_dir = session / "diagnostics" / "dense_projection"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    missing_images = []
    for image in images:
        camera = cameras.get(image.camera_id)
        if camera is None:
            continue
        source = images_dir / Path(image.name).name
        if not source.is_file():
            missing_images.append(image.name)
            continue
        frame = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if frame is None:
            missing_images.append(image.name)
            continue
        if frame.shape[1] != camera.width or frame.shape[0] != camera.height:
            scale_x = frame.shape[1] / camera.width
            scale_y = frame.shape[0] / camera.height
        else:
            scale_x = scale_y = 1.0

        u, v, camera_z, valid = project_points(points, camera, image)
        px = np.rint(u[valid] * scale_x).astype(np.int32)
        py = np.rint(v[valid] * scale_y).astype(np.int32)
        depth = camera_z[valid]
        inside = (
            (px >= 0) & (px < frame.shape[1]) &
            (py >= 0) & (py < frame.shape[0])
        )
        px, py, depth = px[inside], py[inside], depth[inside]

        # Z-buffer: only the nearest projected dense point at each pixel counts.
        flat_index = py * frame.shape[1] + px
        z_buffer = np.full(frame.shape[0] * frame.shape[1], np.inf, dtype=np.float32)
        np.minimum.at(z_buffer, flat_index, depth.astype(np.float32))
        visible = np.isfinite(z_buffer)
        mask = visible.reshape(frame.shape[:2]).astype(np.uint8) * 255
        # A 3x3 dilated footprint is a visualization aid, not a reconstruction.
        dilated = cv2.dilate(mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
        covered_pixels = int(np.count_nonzero(dilated))
        coverage_pct = 100.0 * covered_pixels / float(frame.shape[0] * frame.shape[1])
        overlay = frame.copy()
        overlay[dilated > 0] = (30, 235, 90)
        blended = cv2.addWeighted(frame, 0.78, overlay, 0.22, 0.0)
        cv2.imwrite(str(output_dir / f"{Path(image.name).stem}_dense_coverage.jpg"), blended)

        rows.append({
            "image": image.name,
            "registered_image_id": image.image_id,
            "sparse_track_observations": observations.get(image.image_id, 0),
            "visible_dense_pixels": int(np.count_nonzero(visible)),
            "projected_coverage_percent_3x3": round(coverage_pct, 3),
            "overlay": str(output_dir / f"{Path(image.name).stem}_dense_coverage.jpg"),
        })

    pair_rows = []
    for (first, second), shared in sorted(pairwise.items()):
        pair_rows.append({
            "image_a": next((item.name for item in images if item.image_id == first), str(first)),
            "image_b": next((item.name for item in images if item.image_id == second), str(second)),
            "shared_sparse_3d_tracks": shared,
        })

    report = {
        "session": args.session,
        "registered_image_count": len(images),
        "dense_cloud_source": str(dense_ply),
        "dense_cloud_points_sampled": int(len(points)),
        "sparse_3d_point_count": sparse_count,
        "per_image": rows,
        "pairwise_sparse_overlap": pair_rows,
        "missing_source_images": missing_images,
        "interpretation": (
            "Green overlays mark projected dense-cloud samples, not ground-truth visible pixels. "
            "Large photo regions without green samples are candidates for weak or missing MVS depth; "
            "they do not prove that a surface is physically absent. Pairwise track counts measure "
            "sparse feature co-visibility, not dense-surface completeness."
        ),
    }
    report_path = output_dir / "coverage_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Registered photos: {len(images)}")
    print(f"Sparse 3D points: {sparse_count:,}")
    print(f"Dense cloud points sampled: {len(points):,}")
    for row in rows:
        print(
            f"{row['image']}: sparse observations={row['sparse_track_observations']:,}, "
            f"dense projection={row['projected_coverage_percent_3x3']:.2f}%"
        )
    print(f"Coverage overlays: {output_dir}")
    print(f"Report: {report_path}")
    if missing_images:
        print("Missing source images:", ", ".join(missing_images))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
