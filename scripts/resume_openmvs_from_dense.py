"""Resume OpenMVS mesh generation from an already fused dense scene.

This script does not run COLMAP, InterfaceCOLMAP, depth estimation or fusion.
It only runs ReconstructMesh -> RefineMesh -> TextureMesh against the saved
scene_fuse_dense.mvs / scene_fuse_dense.ply pair.

All outputs are staged separately. The installed mesh folder is replaced only
after geometry, GLB, OBJ, MTL and every map_Kd reference have been validated.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path

import numpy as np

from photo_reconstruction.openmvs_assets import copy_mtl_diffuse_maps
from photo_reconstruction.openmvs_quality import (
    reconstruct_mesh_options,
    refine_mesh_options,
    texture_mesh_options,
)
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


def _binary(name: str) -> str:
    root = configured_openmvs_root()
    found = find_openmvs_executable(name, openmvs_dir=root)
    if found:
        return found
    resolved = shutil.which(name)
    if resolved:
        return resolved
    raise FileNotFoundError(
        f"Cannot find {name}; set CLEAR360_OPENMVS_DIR to your OpenMVS 2.4.0 folder."
    )


def _run(command: list[str], cwd: Path, log_file: Path, stage_name: str) -> None:
    print(f"\n[{stage_name}] {' '.join(command)}", flush=True)
    with log_file.open("a", encoding="utf-8") as log:
        log.write(f"\n[{stage_name}] {' '.join(command)}\n")
        log.flush()
        proc = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        code = proc.wait()
        if code != 0:
            raise RuntimeError(
                f"{stage_name} failed with exit code {code}. Inspect {log_file}."
            )


def _rewrite_obj_mtllib(obj_path: Path, library_name: str) -> None:
    lines: list[str] = []
    replaced = False
    for line in obj_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.lstrip().lower().startswith("mtllib "):
            indent = line[: len(line) - len(line.lstrip())]
            line = f"{indent}mtllib {library_name}"
            replaced = True
        lines.append(line)
    if not replaced:
        lines.insert(0, f"mtllib {library_name}")
    obj_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def resume_from_dense(
    session: Path,
    *,
    output_tag: str = "mesh_from_existing_dense_v2",
    backup_name: str = "mesh_before_dense_resume_v2",
    free_space_support: bool = False,
    remove_spurious: float = 20.0,
    max_texture_size: int = 2048,
) -> dict:
    session = session.expanduser().resolve()
    work = session / "openmvs"
    dense_scene = work / "scene_fuse_dense.mvs"
    dense_ply = work / "scene_fuse_dense.ply"
    dense_images = work / "dense" / "images"
    images_txt = work / "sparse_text" / "images.txt"
    mesh_dir = session / "mesh"
    backup_dir = session / backup_name
    stage = work / output_tag
    log_file = session / "openmvs_dense_resume.log"

    if not dense_scene.is_file() or not dense_ply.is_file():
        raise FileNotFoundError(
            "A complete saved dense pair is required: scene_fuse_dense.mvs and .ply"
        )
    if not images_txt.is_file():
        raise FileNotFoundError(f"COLMAP camera poses missing: {images_txt}")
    if not dense_images.is_dir():
        raise FileNotFoundError(
            f"Undistorted COLMAP images folder missing: {dense_images}"
        )
    if not (session / "original_images").is_dir():
        raise FileNotFoundError(f"Original photographs missing: {session / 'original_images'}")
    if mesh_dir.exists() and not mesh_dir.is_dir():
        raise NotADirectoryError(f"Session mesh path is not a directory: {mesh_dir}")
    if backup_dir.exists():
        raise FileExistsError(
            f"Backup already exists: {backup_dir}. Refusing to overwrite it."
        )
    if stage.exists():
        if any(stage.iterdir()):
            raise FileExistsError(
                f"Resume staging directory contains previous outputs: {stage}. "
                "Keep it for inspection and use a new --output-tag."
            )
        stage.rmdir()

    pose_estimate = estimate_world_up(images_txt)
    if pose_estimate is None:
        raise RuntimeError("COLMAP poses do not provide a stable camera-up direction.")
    up_vector, consistency, pose_count = pose_estimate

    stage.mkdir(parents=True)
    log_file.write_text(
        "CLEAR360 resume from existing dense point cloud; no densification or fusion.\n",
        encoding="utf-8",
    )

    reconstruct = _binary("ReconstructMesh")
    refine = _binary("RefineMesh")
    texture = _binary("TextureMesh")

    mesh_scene = stage / "scene_mesh.mvs"
    mesh_ply = stage / "scene_mesh.ply"
    reconstruct_options = reconstruct_mesh_options(
        max_threads=1,
        remove_spurious=remove_spurious,
        free_space_support=free_space_support,
    )
    _run(
        [reconstruct, "-i", str(dense_scene), "-o", str(mesh_scene),
         *reconstruct_options],
        work,
        log_file,
        "Reconstruct mesh from saved dense scene",
    )
    if not mesh_scene.is_file() or not mesh_ply.is_file():
        raise RuntimeError("ReconstructMesh did not produce scene_mesh.mvs and scene_mesh.ply.")

    refined_scene = stage / "scene_mesh_refined.mvs"
    refined_ply = stage / "scene_mesh_refined.ply"
    _run(
        [refine, "-i", str(mesh_scene), "-m", str(mesh_ply), "-o", str(refined_scene),
         *refine_mesh_options(max_threads=1, max_views=6)],
        work,
        log_file,
        "Refine mesh at native image resolution",
    )
    if not refined_scene.is_file() or not refined_ply.is_file():
        raise RuntimeError("RefineMesh did not produce the expected refined mesh.")

    texture_options = texture_mesh_options(
        max_threads=1, max_texture_size=max_texture_size
    )
    glb_work = stage / "scene_textured.glb"
    obj_work = stage / "scene_textured.obj"
    _run(
        [texture, "-i", str(refined_scene), "-m", str(refined_ply),
         "-o", str(glb_work), "--export-type", "glb", *texture_options],
        work,
        log_file,
        "Export textured GLB",
    )
    _run(
        [texture, "-i", str(refined_scene), "-m", str(refined_ply),
         "-o", str(obj_work), "--export-type", "obj", *texture_options],
        work,
        log_file,
        "Export textured OBJ and MTL",
    )

    mtl_work = stage / "scene_textured.mtl"
    if not all(path.is_file() for path in (glb_work, obj_work, mtl_work)):
        raise RuntimeError("TextureMesh did not create the expected GLB/OBJ/MTL files.")

    export_dir = stage / "export"
    export_dir.mkdir()
    new_obj = export_dir / "scene_mesh.obj"
    new_ply = export_dir / "scene_mesh.ply"
    new_glb = export_dir / "scene_textured.glb"
    new_mtl = export_dir / "scene_mesh.mtl"
    shutil.copy2(obj_work, new_obj)
    shutil.copy2(refined_ply, new_ply)
    shutil.copy2(glb_work, new_glb)
    copy_mtl_diffuse_maps(mtl_work, export_dir, output_name=new_mtl.name)
    _rewrite_obj_mtllib(new_obj, new_mtl.name)

    for index, atlas in enumerate(sorted(stage.glob("scene_textured_*.png"))):
        target_name = "scene_mesh.png" if index == 0 else f"scene_mesh_{index}.png"
        shutil.copy2(atlas, export_dir / target_name)

    rotation = rotation_from_to(
        np.asarray(up_vector, dtype=np.float64),
        np.array([0.0, 0.0, 1.0], dtype=np.float64),
    )
    center = obj_median_center(new_obj)
    if not align_obj_file(new_obj, rotation, center=center):
        raise RuntimeError("Could not align OBJ.")
    if not align_open3d_geometry_file(new_ply, rotation, center, point_cloud=False):
        raise RuntimeError("Could not align refined PLY.")
    if not align_glb_file(new_glb, rotation, center):
        raise RuntimeError("Could not align textured GLB.")

    # The saved dense PLY/MVS pair remains in the original COLMAP coordinate
    # system so it can be reused for future mesh comparisons.
    aligned_dense = stage / "scene_fuse_dense_aligned.ply"
    shutil.copy2(dense_ply, aligned_dense)
    if not align_open3d_geometry_file(aligned_dense, rotation, center, point_cloud=True):
        raise RuntimeError("Could not create aligned dense-cloud diagnostic copy.")

    map_paths: list[str] = []
    for line in new_mtl.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.lstrip().lower().startswith("map_kd "):
            tokens = shlex.split(line, comments=True)
            if len(tokens) < 2 or not (export_dir / tokens[-1]).is_file():
                raise RuntimeError(f"Missing MTL diffuse map: {line}")
            map_paths.append(tokens[-1])
    if not map_paths:
        raise RuntimeError("Exported MTL has no diffuse texture maps.")

    # Validate minimum output integrity before touching the installed mesh.
    if new_obj.stat().st_size < 1000 or new_ply.stat().st_size < 1000 or new_glb.stat().st_size < 1000:
        raise RuntimeError("One or more new mesh exports are unexpectedly small.")

    report = {
        "session": session.name,
        "method": "resume from existing dense scene; no COLMAP or DensifyPointCloud",
        "dense_input": str(dense_scene),
        "dense_point_cloud": str(dense_ply),
        "registered_camera_poses": pose_count,
        "camera_up_consistency": round(float(consistency), 4),
        "mesh_options": {
            "remove_spurious": remove_spurious,
            "close_holes": 30,
            "smooth_iterations": 2,
            "free_space_support": free_space_support,
        },
        "refine_options": {
            "resolution_level": 0,
            "minimum_resolution": 640,
            "max_views": 6,
            "scales": 2,
        },
        "texture_options": {
            "max_texture_size": max_texture_size,
            "local_seam_leveling": True,
            "global_seam_leveling": True,
        },
        "texture_maps": map_paths,
        "backup_directory": str(backup_dir) if mesh_dir.exists() else None,
        "installed_without_prior_mesh": not mesh_dir.exists(),
        "staging_directory": str(stage),
        "log_file": str(log_file),
    }

    had_existing_mesh = mesh_dir.exists()
    if had_existing_mesh:
        os.replace(mesh_dir, backup_dir)
    try:
        os.replace(export_dir, mesh_dir)
    except Exception:
        if had_existing_mesh and backup_dir.exists():
            os.replace(backup_dir, mesh_dir)
        raise

    (session / "dense_resume_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"Mesh installed: {mesh_dir}")
    if had_existing_mesh:
        print(f"Previous mesh preserved: {backup_dir}")
    else:
        print("No previous mesh folder existed; the validated mesh was installed as new.")
    print(f"Saved dense cloud reused: {dense_ply}")
    print(f"Registered camera poses: {pose_count}")
    print(f"Camera-up consistency: {consistency:.3f}")
    print(f"Log: {log_file}")
    print("COLMAP, DensifyPointCloud and fusion were not rerun.")
    return report


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Resume mesh/refine/texture from a saved dense OpenMVS scene."
    )
    parser.add_argument("session", help="Session folder, e.g. session_027")
    parser.add_argument(
        "--data-root", type=Path,
        default=project_root / "data" / "phone_reconstruction",
    )
    parser.add_argument("--output-tag", default="mesh_from_existing_dense_v2")
    parser.add_argument("--backup-name", default="mesh_before_dense_resume_v2")
    parser.add_argument("--free-space-support", action="store_true")
    parser.add_argument("--remove-spurious", type=float, default=20.0)
    parser.add_argument("--max-texture-size", type=int, default=2048)
    args = parser.parse_args()
    resume_from_dense(
        args.data_root.expanduser().resolve() / args.session,
        output_tag=args.output_tag,
        backup_name=args.backup_name,
        free_space_support=args.free_space_support,
        remove_spurious=args.remove_spurious,
        max_texture_size=args.max_texture_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
