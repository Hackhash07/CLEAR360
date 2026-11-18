"""Re-densify and rebuild an OpenMVS session while reusing its COLMAP workspace.

This avoids repeating feature extraction, image matching, mapping and image
undistortion. It recomputes depth maps at native undistorted resolution, then
reconstructs/refines/textures a candidate mesh in a separate workspace. The
existing mesh is preserved until all outputs and MTL texture references pass
validation.
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
    densify_common_options,
    densify_fusion_options,
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
    lines = []
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


def rebuild_dense_session(
    session: Path,
    *,
    output_tag: str = "dense_quality_rebuild",
    backup_name: str = "mesh_before_dense_quality_rebuild",
    max_resolution: int = 1600,
    number_views: int = 5,
    remove_spurious: float = 20.0,
) -> dict:
    session = session.expanduser().resolve()
    openmvs_dir = session / "openmvs"
    dense_input = openmvs_dir / "dense"
    poses_file = openmvs_dir / "sparse_text" / "images.txt"
    cameras_file = openmvs_dir / "sparse_text" / "cameras.txt"
    images_file = openmvs_dir / "sparse_text" / "images.txt"
    source_photos = session / "original_images"
    mesh_dir = session / "mesh"
    backup_dir = session / backup_name
    stage = openmvs_dir / output_tag

    if not dense_input.is_dir():
        raise FileNotFoundError(f"Saved undistorted COLMAP workspace missing: {dense_input}")
    if not (dense_input / "images").is_dir():
        raise FileNotFoundError(f"Undistorted COLMAP images folder missing: {dense_input / 'images'}")
    if not poses_file.is_file() or not cameras_file.is_file():
        raise FileNotFoundError("Saved COLMAP text camera model is missing.")
    if not source_photos.is_dir():
        raise FileNotFoundError(f"Original uploaded photos missing: {source_photos}")
    photo_count = sum(
        path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        for path in source_photos.iterdir()
    )
    if photo_count < 6:
        raise ValueError(f"OpenMVS expects at least 6 photos; found {photo_count}.")
    if not mesh_dir.is_dir():
        raise FileNotFoundError(f"Current mesh folder missing: {mesh_dir}")
    if backup_dir.exists():
        raise FileExistsError(
            f"Backup already exists: {backup_dir}. Choose a new --backup-name."
        )
    if stage.exists():
        raise FileExistsError(
            f"Rebuild workspace already exists: {stage}. Choose a new --output-tag."
        )
    if max_resolution < 640 or number_views < 2 or remove_spurious < 0:
        raise ValueError("Invalid dense or mesh quality options.")

    pose_estimate = estimate_world_up(poses_file)
    if pose_estimate is None:
        raise RuntimeError("COLMAP camera poses do not provide a stable up direction.")
    up_vector, consistency, pose_count = pose_estimate

    stage.mkdir(parents=True)
    log_file = session / "openmvs_dense_rebuild.log"
    log_file.write_text(
        "CLEAR360 OpenMVS 2.4.0 re-densification; COLMAP SfM and undistortion reused.\n",
        encoding="utf-8",
    )

    interface = _binary("InterfaceCOLMAP")
    densify = _binary("DensifyPointCloud")
    reconstruct = _binary("ReconstructMesh")
    refine = _binary("RefineMesh")
    texture = _binary("TextureMesh")

    scene_mvs = stage / "scene.mvs"
    _run(
        [interface, "-i", str(dense_input), "-o", str(scene_mvs)],
        stage,
        log_file,
        "Re-import saved COLMAP dense workspace",
    )
    if not scene_mvs.is_file():
        raise RuntimeError("InterfaceCOLMAP did not create the staged scene.mvs.")

    scene_fuse = stage / "scene_fuse.mvs"
    shutil.copy2(scene_mvs, scene_fuse)
    dense_options = densify_common_options(
        resolution_level=0,
        max_resolution=max_resolution,
        max_threads=1,
        number_views=number_views,
        postprocess_dmaps=2,
    )
    _run(
        [densify, "-i", str(scene_mvs), "-w", str(stage),
         *dense_options, "--fusion-mode", "1"],
        stage,
        log_file,
        "Full-resolution depth maps",
    )
    _run(
        [densify, "-i", str(scene_fuse), "-w", str(stage),
         *dense_options, *densify_fusion_options(minimum_views=2)],
        stage,
        log_file,
        "Geometric consistency and dense fusion",
    )

    dense_scene = stage / "scene_fuse_dense.mvs"
    dense_ply = stage / "scene_fuse_dense.ply"
    if not dense_scene.is_file() or not dense_ply.is_file():
        raise RuntimeError("DensifyPointCloud did not create scene_fuse_dense.mvs and .ply.")

    mesh_scene = stage / "scene_mesh.mvs"
    mesh_ply = stage / "scene_mesh.ply"
    _run(
        [reconstruct, "-i", str(dense_scene), "-o", str(mesh_scene),
         *reconstruct_mesh_options(max_threads=1, remove_spurious=remove_spurious)],
        stage,
        log_file,
        "Reconstruct surface mesh",
    )
    if not mesh_scene.is_file() or not mesh_ply.is_file():
        raise RuntimeError("ReconstructMesh did not create the expected mesh files.")

    refined_scene = stage / "scene_mesh_refined.mvs"
    refined_ply = stage / "scene_mesh_refined.ply"
    _run(
        [refine, "-i", str(mesh_scene), "-m", str(mesh_ply), "-o", str(refined_scene),
         *refine_mesh_options(max_threads=1, max_views=6)],
        stage,
        log_file,
        "Refine mesh at native image resolution",
    )
    if not refined_scene.is_file() or not refined_ply.is_file():
        raise RuntimeError("RefineMesh did not create the expected refined mesh files.")

    texture_options = texture_mesh_options(max_threads=1, max_texture_size=2048)
    glb_work = stage / "scene_textured.glb"
    obj_work = stage / "scene_textured.obj"
    _run(
        [texture, "-i", str(refined_scene), "-m", str(refined_ply),
         "-o", str(glb_work), "--export-type", "glb", *texture_options],
        stage,
        log_file,
        "Export textured GLB",
    )
    _run(
        [texture, "-i", str(refined_scene), "-m", str(refined_ply),
         "-o", str(obj_work), "--export-type", "obj", *texture_options],
        stage,
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
        raise RuntimeError("Could not align rebuilt OBJ.")
    if not align_open3d_geometry_file(new_ply, rotation, center, point_cloud=False):
        raise RuntimeError("Could not align rebuilt mesh PLY.")
    if not align_glb_file(new_glb, rotation, center):
        raise RuntimeError("Could not align rebuilt GLB.")
    # Keep the staged MVS project and its original-frame dense PLY consistent.
    # Export a separate Z-up copy for inspection instead of mutating the source.
    aligned_dense_ply = stage / "scene_fuse_dense_aligned.ply"
    shutil.copy2(dense_ply, aligned_dense_ply)
    if not align_open3d_geometry_file(
        aligned_dense_ply, rotation, center, point_cloud=True
    ):
        raise RuntimeError("Could not align rebuilt dense point cloud.")

    # Validate all map_Kd paths before swapping the current mesh directory.
    for line in new_mtl.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.lstrip().lower().startswith("map_kd "):
            tokens = shlex.split(line, comments=True)
            if len(tokens) < 2 or not (export_dir / tokens[-1]).is_file():
                raise RuntimeError(f"Missing MTL diffuse map: {line}")

    report = {
        "session": session.name,
        "reconstruction": "reused COLMAP sparse model and undistorted image workspace",
        "registered_camera_poses": pose_count,
        "camera_up_consistency": round(float(consistency), 4),
        "dense_options": {
            "resolution_level": 0,
            "max_resolution": max_resolution,
            "min_resolution": 640,
            "number_views": number_views,
            "number_views_fuse": 2,
            "patch_match_iterations": 3,
            "geometric_iterations": 2,
            "postprocess_dmaps": 2,
            "threads": 1,
        },
        "mesh_options": {
            "remove_spurious": remove_spurious,
            "close_holes": 30,
            "smooth_iterations": 2,
            "refine_resolution_level": 0,
            "refine_min_resolution": 640,
            "refine_max_views": 6,
            "refine_scales": 2,
        },
        "note": (
            "postprocess-dmaps=2 interpolates only short depth gaps with similar boundary depths; "
            "it does not synthesize large unobserved surfaces. Lower remove-spurious values are "
            "more aggressive at deleting long-edge faces."
        ),
        "workspace": str(stage),
        "aligned_dense_cloud": str(aligned_dense_ply),
        "log_file": str(log_file),
        "backup_directory": str(backup_dir),
    }

    os.replace(mesh_dir, backup_dir)
    try:
        os.replace(export_dir, mesh_dir)
    except Exception:
        os.replace(backup_dir, mesh_dir)
        raise
    (session / "dense_quality_rebuild_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print(f"New mesh installed: {mesh_dir}")
    print(f"Previous mesh preserved: {backup_dir}")
    print(f"Staged dense workspace: {stage}")
    print(f"Registered camera poses: {pose_count}")
    print(f"Camera-up consistency: {consistency:.3f}")
    print(f"Log: {log_file}")
    print("COLMAP matching/mapping and image undistortion were not rerun.")
    print("Large surfaces with no multi-view depth support will still remain missing.")
    return report


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Re-densify one session using saved COLMAP poses without rerunning SfM."
    )
    parser.add_argument("session", help="Session folder name, e.g. session_022")
    parser.add_argument(
        "--data-root", type=Path,
        default=project_root / "data" / "phone_reconstruction",
    )
    parser.add_argument("--output-tag", default="dense_quality_rebuild")
    parser.add_argument("--backup-name", default="mesh_before_dense_quality_rebuild")
    parser.add_argument("--max-resolution", type=int, default=1600)
    parser.add_argument("--number-views", type=int, default=5)
    parser.add_argument(
        "--remove-spurious", type=float, default=20.0,
        help="20 is the OpenMVS 2.4.0 default; lower values remove more long-edge faces",
    )
    args = parser.parse_args()
    session = args.data_root.expanduser().resolve() / args.session
    rebuild_dense_session(
        session,
        output_tag=args.output_tag,
        backup_name=args.backup_name,
        max_resolution=args.max_resolution,
        number_views=args.number_views,
        remove_spurious=args.remove_spurious,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
