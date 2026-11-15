"""
Recheck saved metric depth maps against the saved COLMAP sparse reconstruction.

This command does NOT run depth inference, dense fusion, or meshing. It rebuilds
SfM camera models and 2D-3D tracks from sfm_work/sparse, reads cached *_depth.npy
files, runs held-out depth calibration, and writes depth_alignment_recheck.json.

Example:
    python -m photo_reconstruction.diagnose_cached_depth \
        --session bedroom_8photos_m7_3_13_check
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from photo_reconstruction.camera_model import PhoneCameraModel
from photo_reconstruction.depth_aligner import SfMDepthAligner


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _load_best_colmap_model(sparse_root: Path):
    try:
        import pycolmap
    except ImportError as exc:
        raise RuntimeError(
            "pycolmap is required to rebuild the saved 2D-3D tracks. "
            "Activate the same Python environment used for reconstruction."
        ) from exc

    candidates = []
    if sparse_root.is_dir():
        candidates.append(sparse_root)
        candidates.extend(p for p in sparse_root.iterdir() if p.is_dir())

    reconstructions = []
    failures = []
    for model_dir in candidates:
        has_model_file = any(
            (model_dir / filename).exists()
            for filename in ("cameras.bin", "cameras.txt", "images.bin", "images.txt")
        )
        if not has_model_file:
            continue
        try:
            reconstruction = pycolmap.Reconstruction(str(model_dir))
            point_count = len(reconstruction.points3D)
            image_count = len(reconstruction.images)
            if image_count:
                reconstructions.append((point_count, image_count, model_dir, reconstruction))
        except Exception as exc:
            failures.append(f"{model_dir}: {exc}")

    if not reconstructions:
        details = "\n".join(failures) if failures else "No COLMAP model files found."
        raise RuntimeError(
            f"Could not load a reconstruction from {sparse_root}.\n{details}"
        )

    reconstructions.sort(key=lambda item: (item[0], item[1]), reverse=True)
    _, _, model_dir, reconstruction = reconstructions[0]
    return model_dir, reconstruction


def _camera_pose_from_colmap(image: Any) -> np.ndarray:
    if hasattr(image, "cam_from_world"):
        cfw = image.cam_from_world() if callable(image.cam_from_world) else image.cam_from_world
        rotation = cfw.rotation.matrix()
        translation = np.asarray(cfw.translation, dtype=np.float64)
    else:
        import pycolmap
        rotation = pycolmap.qvec_to_rotmat(image.qvec)
        translation = np.asarray(image.tvec, dtype=np.float64)

    T_cam_world = np.eye(4, dtype=np.float64)
    T_cam_world[:3, :3] = rotation
    T_cam_world[:3, 3] = translation
    return np.linalg.inv(T_cam_world)


def _build_camera_models_and_tracks(reconstruction, saved_pose_stems: set[str]):
    camera_models = {}
    sparse_tracks = {}

    for image in reconstruction.images.values():
        image_name = str(image.name)
        if saved_pose_stems and Path(image_name).stem not in saved_pose_stems:
            # The production pipeline removes camera-pose outliers before writing
            # camera_poses/*.npy. Keep the diagnostic to those retained views.
            continue

        T_world_cam = _camera_pose_from_colmap(image)
        camera = reconstruction.cameras[image.camera_id]
        camera_models[image_name] = PhoneCameraModel.from_colmap_camera(
            cam=camera,
            image_name=image_name,
            T_world_cam=T_world_cam,
        )

        tracks = []
        for point2D in image.points2D:
            has_point = point2D.has_point3D() if callable(point2D.has_point3D) else bool(point2D.has_point3D)
            if not has_point:
                continue
            point3D_id = point2D.point3D_id
            if point3D_id not in reconstruction.points3D:
                continue
            point3D = reconstruction.points3D[point3D_id]
            tracks.append({
                "xy": (float(point2D.xy[0]), float(point2D.xy[1])),
                "xyz": (
                    float(point3D.xyz[0]),
                    float(point3D.xyz[1]),
                    float(point3D.xyz[2]),
                ),
                "error": float(point3D.error),
                "point3D_id": int(point3D_id),
            })
        sparse_tracks[image_name] = tracks

    return camera_models, sparse_tracks


def _load_cached_depth_maps(session_dir: Path, camera_models):
    depth_dir = session_dir / "depth_cache"
    image_dir = session_dir / "original_images"
    if not depth_dir.is_dir() or not image_dir.is_dir():
        raise RuntimeError("Session must contain original_images/ and depth_cache/.")

    images_by_stem = {}
    for image_path in image_dir.iterdir():
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES:
            stem = image_path.stem
            if stem in images_by_stem:
                raise RuntimeError(
                    f"Duplicate image filename stem '{stem}' prevents unambiguous depth-cache lookup."
                )
            images_by_stem[stem] = image_path.name

    depth_maps = {}
    missing = []
    for name in camera_models:
        image_name = images_by_stem.get(Path(name).stem, name)
        depth_path = depth_dir / f"{Path(image_name).stem}_depth.npy"
        if not depth_path.exists():
            missing.append(image_name)
            continue
        depth = np.load(depth_path, allow_pickle=False)
        if depth.ndim != 2:
            raise RuntimeError(f"Expected a 2D depth map at {depth_path}, got shape {depth.shape}.")
        depth_maps[name] = np.asarray(depth, dtype=np.float32)

    if not depth_maps:
        raise RuntimeError(f"No cached depth maps matched the registered views in {depth_dir}.")
    if missing:
        print("Missing cached maps for: " + ", ".join(missing))
    return depth_maps


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True, help="Session folder name under data/phone_reconstruction/")
    parser.add_argument(
        "--data-root",
        default=str(Path(__file__).resolve().parent.parent / "data"),
        help="Data directory containing phone_reconstruction/ (default: project data/)",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser()
    session_root = data_root if data_root.name == "phone_reconstruction" else data_root / "phone_reconstruction"
    session_dir = session_root / args.session
    if not session_dir.is_dir():
        raise RuntimeError(f"Session folder not found: {session_dir}")

    sparse_root = session_dir / "sfm_work" / "sparse"
    model_dir, reconstruction = _load_best_colmap_model(sparse_root)

    pose_dir = session_dir / "camera_poses"
    saved_pose_stems = {
        path.stem[:-5]
        for path in pose_dir.glob("*_pose.npy")
    } if pose_dir.is_dir() else set()

    camera_models, sparse_tracks = _build_camera_models_and_tracks(
        reconstruction,
        saved_pose_stems,
    )
    if not camera_models:
        raise RuntimeError(
            "No saved camera pose names matched the selected COLMAP model. "
            "Check that this session contains camera_poses/*_pose.npy."
        )

    depth_maps = _load_cached_depth_maps(session_dir, camera_models)
    # Keep only views with both a camera model, a track list, and a cached map.
    camera_models = {name: camera_models[name] for name in depth_maps}
    sparse_tracks = {name: sparse_tracks.get(name, []) for name in depth_maps}

    aligner = SfMDepthAligner(mode="regularized")
    report = aligner.align_depth_maps(
        depth_maps=depth_maps,
        camera_models=camera_models,
        sparse_tracks=sparse_tracks,
    )

    accepted = sorted(report.aligned_depth_maps)
    rejected = sorted(
        name for name, stats in report.view_stats.items() if not stats.success
    )
    views = {}
    print("\nCACHED DEPTH CALIBRATION (no inference / no meshing)")
    print(f"Session: {session_dir.name}")
    print(f"COLMAP model: {model_dir}")
    print(f"Views checked: {len(depth_maps)}")
    print(f"Training anchors: {report.total_anchors}")
    print()
    print(f"{'VIEW':24} {'ANCHORS':>8} {'MEDIAN':>10} {'P90':>10} {'STATUS':>10}")
    print("-" * 68)

    for name, stats in sorted(report.view_stats.items()):
        camera = camera_models[name]
        depth = depth_maps[name]
        depth_height, depth_width = depth.shape
        status = "ACCEPTED" if stats.success else "REJECTED"
        print(
            f"{name[:24]:24} {stats.num_anchors:8d} "
            f"{stats.validation_median_relative_error:9.2%} "
            f"{stats.validation_p90_relative_error:9.2%} {status:>10}"
        )
        views[name] = {
            "camera_resolution": [int(camera.width), int(camera.height)],
            "depth_resolution": [int(depth_width), int(depth_height)],
            "pixel_scale_x": float(depth_width / max(camera.width, 1)),
            "pixel_scale_y": float(depth_height / max(camera.height, 1)),
            "tracks_available": int(len(sparse_tracks.get(name, []))),
            "usable_anchors": int(stats.num_anchors),
            "scale": float(stats.median_scale),
            "offset": float(stats.depth_offset),
            "validation_median_relative_error": float(stats.validation_median_relative_error),
            "validation_p90_relative_error": float(stats.validation_p90_relative_error),
            "aligned_residual_median": float(stats.aligned_residual_median),
            "accepted": bool(stats.success),
            "status": stats.status_message,
        }

    result = {
        "status": "mvs_ready" if len(accepted) >= 2 else "not_ready",
        "session": session_dir.name,
        "colmap_model": str(model_dir),
        "views_checked": len(depth_maps),
        "accepted_views": accepted,
        "rejected_views": rejected,
        "training_anchors": int(report.total_anchors),
        "global_scale": float(report.global_scale),
        "validation_limits": {
            "median_relative_error": float(aligner.max_val_median_rel),
            "p90_relative_error": float(aligner.max_val_p90_rel),
        },
        "views": views,
        "note": (
            "This is a calibration-only diagnostic. It does not run dense MVS, "
            "point-cloud fusion, surface meshing, or texture generation."
        ),
    }
    output_path = session_dir / "depth_alignment_recheck.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print()
    print(f"Accepted views: {len(accepted)} / {len(depth_maps)}")
    print("MVS input gate: " + ("READY (at least two validated views)" if len(accepted) >= 2 else "NOT READY"))
    print(f"Saved: {output_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Depth diagnostic failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
