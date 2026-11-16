"""Align an existing OpenMVS session upright without rerunning dense reconstruction."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

from photo_reconstruction.scene_orientation import (
    align_glb_file,
    align_obj_file,
    align_open3d_geometry_file,
    estimate_world_up,
    obj_median_center,
    rotation_from_to,
)


def latest_openmvs_session(data_root: Path) -> Path:
    candidates = [
        path for path in data_root.glob("session_*")
        if (path / "openmvs" / "sparse" / "0").is_dir()
        and (path / "mesh" / "scene_mesh.obj").is_file()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No OpenMVS session with a sparse model and OBJ mesh found under {data_root}"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    default_root = project_root / "data" / "phone_reconstruction"
    parser = argparse.ArgumentParser(
        description="Align a saved COLMAP + OpenMVS model to the phone-camera up direction."
    )
    parser.add_argument(
        "session",
        nargs="?",
        help="Session folder name (for example session_090). Defaults to the newest OpenMVS session.",
    )
    parser.add_argument("--data-root", type=Path, default=default_root)
    args = parser.parse_args()

    data_root = args.data_root.expanduser().resolve()
    if args.session:
        session_dir = data_root / args.session
    else:
        session_dir = latest_openmvs_session(data_root)
    if not session_dir.is_dir():
        raise FileNotFoundError(f"Session folder not found: {session_dir}")

    marker = session_dir / "orientation_alignment.json"
    if marker.exists():
        raise RuntimeError(
            f"Orientation was already applied to this session ({marker}). "
            "Refusing to rotate the model twice."
        )

    openmvs_dir = session_dir / "openmvs"
    sparse_dir = openmvs_dir / "sparse" / "0"
    text_dir = openmvs_dir / "sparse_text"
    images_txt = text_dir / "images.txt"
    if not images_txt.exists():
        colmap = os.environ.get("CLEAR360_COLMAP_BIN") or shutil.which("colmap")
        if not colmap:
            raise FileNotFoundError(
                "COLMAP is not on PATH. Set CLEAR360_COLMAP_BIN to its executable."
            )
        result = subprocess.run(
            [
                colmap,
                "model_converter",
                "--input_path", str(sparse_dir),
                "--output_path", str(text_dir),
                "--output_type", "TXT",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "COLMAP model_converter failed:\n"
                + (result.stdout or "")
                + (result.stderr or "")
            )

    estimate = estimate_world_up(images_txt)
    if estimate is None:
        raise RuntimeError(
            "The saved camera poses do not provide a stable up direction. "
            "No output files were changed."
        )
    up_vector, consistency, pose_count = estimate
    rotation = rotation_from_to(up_vector, __import__("numpy").array([0.0, 0.0, 1.0]))

    mesh_dir = session_dir / "mesh"
    obj_file = mesh_dir / "scene_mesh.obj"
    ply_file = mesh_dir / "scene_mesh.ply"
    glb_file = mesh_dir / "scene_textured.glb"
    dense_file = openmvs_dir / "scene_fuse_dense.ply"
    required = [obj_file, ply_file, glb_file]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing model exports: " + ", ".join(missing))

    center = obj_median_center(obj_file)
    files_to_backup = [obj_file, ply_file, glb_file]
    if dense_file.is_file():
        files_to_backup.append(dense_file)
    backups = []
    for source in files_to_backup:
        backup = source.with_name(source.name + ".before_orientation")
        if backup.exists():
            raise FileExistsError(f"Backup already exists; refusing to overwrite: {backup}")
        shutil.copy2(source, backup)
        backups.append(str(backup))

    try:
        results = {
            "obj": align_obj_file(obj_file, rotation, center=center),
            "ply": align_open3d_geometry_file(
                ply_file, rotation, center, point_cloud=False
            ),
            "glb": align_glb_file(glb_file, rotation, center),
        }
        if dense_file.is_file():
            results["dense_cloud"] = align_open3d_geometry_file(
                dense_file, rotation, center, point_cloud=True
            )
        report = {
            "method": "COLMAP camera-up consensus",
            "session_id": session_dir.name,
            "registered_camera_poses": pose_count,
            "up_consistency": round(consistency, 4),
            "estimated_up_before_alignment": [float(x) for x in up_vector],
            "aligned_to_world_axis": "+Z",
            "exports_aligned": results,
            "backups": backups,
        }
        if not all(results.values()):
            raise RuntimeError("One or more export formats could not be aligned: " + str(results))
        marker.write_text(json.dumps(report, indent=2), encoding="utf-8")
        report_path = session_dir / "report.json"
        if report_path.exists():
            try:
                existing = json.loads(report_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                existing = {}
            existing["orientation"] = report
            report_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    except Exception:
        # Restore every file that may have been modified if any format fails.
        for backup in backups:
            backup_path = Path(backup)
            original = backup_path.with_name(backup_path.name.removesuffix(".before_orientation"))
            shutil.copy2(backup_path, original)
        raise

    print(f"Aligned existing session: {session_dir}")
    print(f"Estimated camera up: {up_vector.tolist()}")
    print(f"Pose consistency: {consistency:.3f} across {pose_count} registered images")
    print("Updated exports: " + ", ".join(name for name, ok in results.items() if ok))
    print("Original files backed up with suffix: .before_orientation")
    print("No COLMAP/OpenMVS dense reconstruction was rerun.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
