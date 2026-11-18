"""Rebuild an existing OpenMVS mesh without rerunning COLMAP or depth estimation.

The command reuses session_<id>/openmvs/scene_fuse_dense.mvs, reconstructs and
refines the surface with conservative cleanup settings, retextures it, aligns
the exports upright, and only then swaps the new exports into session/mesh.
The previous mesh directory is preserved as mesh_before_quality_rebuild.
"""

from __future__ import annotations

import argparse
import json
import os
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


def _run(command: list[str], cwd: Path, log_file: Path) -> None:
    with log_file.open("a", encoding="utf-8") as log:
        log.write("\n$ " + " ".join(command) + "\n")
        log.flush()
        result = subprocess.run(
            command,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        if result.stdout:
            log.write(result.stdout)
            log.flush()
        if result.returncode != 0:
            raise RuntimeError(
                f"{Path(command[0]).name} failed with exit code "
                f"{result.returncode}. Inspect {log_file}."
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
        f"Cannot find {name}. Set CLEAR360_OPENMVS_DIR to your OpenMVS installation."
    )


def _rewrite_obj_mtllib(obj_path: Path, library_name: str) -> None:
    lines = []
    found = False
    for line in obj_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.lstrip().lower().startswith("mtllib "):
            indent = line[: len(line) - len(line.lstrip())]
            line = f"{indent}mtllib {library_name}"
            found = True
        lines.append(line)
    if not found:
        lines.insert(0, f"mtllib {library_name}")
    obj_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def rebuild_session(
    session_dir: Path,
    *,
    output_tag: str = "mesh_quality_rebuild_v2",
    backup_name: str = "mesh_before_quality_rebuild_v2",
    remove_spurious: float = 20.0,
    free_space_support: bool = True,
) -> dict:
    session_dir = session_dir.expanduser().resolve()
    work = session_dir / "openmvs"
    dense_scene = work / "scene_fuse_dense.mvs"
    images_txt = work / "sparse_text" / "images.txt"
    mesh_dir = session_dir / "mesh"
    backup_dir = session_dir / backup_name
    stage = work / output_tag

    if not dense_scene.is_file():
        raise FileNotFoundError(f"Saved dense OpenMVS scene not found: {dense_scene}")
    if not images_txt.is_file():
        raise FileNotFoundError(f"COLMAP camera poses not found: {images_txt}")
    if not mesh_dir.is_dir():
        raise FileNotFoundError(f"Current mesh folder not found: {mesh_dir}")
    if backup_dir.exists():
        raise FileExistsError(
            f"Backup already exists: {backup_dir}. Refusing to overwrite it."
        )
    if not (session_dir / "original_images").is_dir():
        raise FileNotFoundError(
            f"Original source images missing: {session_dir / 'original_images'}"
        )
    if stage.exists():
        # A previous failed launch can leave an empty directory before the
        # first OpenMVS executable starts. Reuse only that harmless empty dir;
        # never delete partial reconstruction outputs.
        if any(stage.iterdir()):
            raise FileExistsError(
                f"Rebuild workspace already contains files: {stage}. "
                "Keep it for inspection or choose a different --output-tag."
            )
        stage.rmdir()

    pose_estimate = estimate_world_up(images_txt)
    if pose_estimate is None:
        raise RuntimeError(
            "COLMAP poses do not provide a stable camera-up direction. "
            "No existing mesh files were changed."
        )
    up_vector, consistency, pose_count = pose_estimate

    stage.mkdir(parents=True)
    log_file = session_dir / "openmvs_mesh_rebuild.log"
    log_file.write_text(
        "CLEAR360 mesh-only OpenMVS rebuild; existing dense reconstruction reused.\n",
        encoding="utf-8",
    )

    reconstruct = _binary("ReconstructMesh")
    refine = _binary("RefineMesh")
    texture = _binary("TextureMesh")

    mesh_scene = stage / "scene_mesh.mvs"
    mesh_ply = stage / "scene_mesh.ply"
    _run(
        [reconstruct, "-i", str(dense_scene), "-o", str(mesh_scene),
         *reconstruct_mesh_options(
             max_threads=1,
             remove_spurious=remove_spurious,
             free_space_support=free_space_support,
         )],
        work,
        log_file,
    )
    if not mesh_scene.is_file() or not mesh_ply.is_file():
        raise RuntimeError(
            "ReconstructMesh did not create both scene_mesh.mvs and scene_mesh.ply."
        )

    refined_scene = stage / "scene_mesh_refined.mvs"
    refined_ply = stage / "scene_mesh_refined.ply"
    _run(
        [refine, "-i", str(mesh_scene), "-m", str(mesh_ply),
         "-o", str(refined_scene),
         *refine_mesh_options(max_threads=1, max_views=6)],
        work,
        log_file,
    )
    if not refined_scene.is_file() or not refined_ply.is_file():
        raise RuntimeError(
            "RefineMesh did not create both scene_mesh_refined.mvs and .ply."
        )

    texture_options = texture_mesh_options(max_threads=1, max_texture_size=2048)
    glb_work = stage / "scene_textured.glb"
    obj_work = stage / "scene_textured.obj"
    _run(
        [texture, "-i", str(refined_scene), "-m", str(refined_ply),
         "-o", str(glb_work), "--export-type", "glb", *texture_options],
        work,
        log_file,
    )
    _run(
        [texture, "-i", str(refined_scene), "-m", str(refined_ply),
         "-o", str(obj_work), "--export-type", "obj", *texture_options],
        work,
        log_file,
    )

    if not glb_work.is_file() or not obj_work.is_file():
        raise RuntimeError("TextureMesh did not create both GLB and OBJ exports.")
    mtl_work = stage / "scene_textured.mtl"
    if not mtl_work.is_file():
        raise RuntimeError("TextureMesh OBJ export did not create its MTL file.")

    export_dir = stage / "export"
    export_dir.mkdir()
    new_obj = export_dir / "scene_mesh.obj"
    new_mtl = export_dir / "scene_mesh.mtl"
    new_ply = export_dir / "scene_mesh.ply"
    new_glb = export_dir / "scene_textured.glb"
    shutil.copy2(obj_work, new_obj)
    shutil.copy2(refined_ply, new_ply)
    shutil.copy2(glb_work, new_glb)
    copy_mtl_diffuse_maps(mtl_work, export_dir, output_name=new_mtl.name)
    _rewrite_obj_mtllib(new_obj, new_mtl.name)

    # Keep the GLB texture sidecars for inspection/export; the OBJ uses the
    # JPEG maps referenced by its own MTL and copied above.
    for index, atlas in enumerate(sorted(stage.glob("scene_textured_*.png"))):
        target_name = "scene_mesh.png" if index == 0 else f"scene_mesh_{index}.png"
        shutil.copy2(atlas, export_dir / target_name)

    rotation = rotation_from_to(
        np.asarray(up_vector, dtype=np.float64),
        np.array([0.0, 0.0, 1.0], dtype=np.float64),
    )
    center = obj_median_center(new_obj)
    if not align_obj_file(new_obj, rotation, center=center):
        raise RuntimeError("Could not align the rebuilt OBJ.")
    if not align_open3d_geometry_file(new_ply, rotation, center, point_cloud=False):
        raise RuntimeError("Could not align the rebuilt PLY.")
    if not align_glb_file(new_glb, rotation, center):
        raise RuntimeError("Could not align the rebuilt GLB.")

    # Validate every material map before replacing the currently working mesh.
    for line in new_mtl.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.lstrip().lower().startswith("map_kd "):
            continue
        import shlex
        tokens = shlex.split(line, comments=True)
        if len(tokens) < 2 or not (export_dir / tokens[-1]).is_file():
            raise RuntimeError(f"Rebuilt MTL has a missing diffuse map: {line}")

    report = {
        "session": session_dir.name,
        "method": "reuse saved dense OpenMVS scene; rebuild mesh/refine/texture only",
        "registered_camera_poses": pose_count,
        "camera_up_consistency": round(float(consistency), 4),
        "estimated_up_before_alignment": [float(value) for value in up_vector],
        "reconstruct_mesh": {
            "remove_spurious": remove_spurious, "close_holes": 30, "smooth_iterations": 2,
            "free_space_support": free_space_support,
            "note": "20 is the OpenMVS 2.4.0 default; smaller values remove more long-edge faces",
        },
        "refine_mesh": {
            "resolution_level": 0, "min_resolution": 640, "max_views": 6,
            "close_holes": 30, "scales": 2, "reduce_memory": 1,
        },
        "texture_mesh": {
            "resolution_level": 0, "max_texture_size": 2048,
            "seam_leveling": True,
        },
        "unseen_surface_note": (
            "Faces not visible in any source photograph can still be marked with "
            "OpenMVS empty-color. More overlapping viewpoints are required to texture them."
        ),
        "backup_directory": str(backup_dir),
        "log_file": str(log_file),
    }

    # Keep the old session mesh untouched until every new export has passed validation.
    os.replace(mesh_dir, backup_dir)
    try:
        os.replace(export_dir, mesh_dir)
    except Exception:
        os.replace(backup_dir, mesh_dir)
        raise

    (session_dir / "mesh_quality_rebuild_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"Rebuilt mesh installed in: {mesh_dir}")
    print(f"Original mesh preserved in: {backup_dir}")
    print(f"Registered camera poses: {pose_count}")
    print(f"Camera-up consistency: {consistency:.3f}")
    print(f"Log: {log_file}")
    print(
        "Note: unseen surfaces are still limited by photo coverage; "
        "this rebuild cannot invent views that were never photographed."
    )
    return report


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Rebuild one saved OpenMVS mesh without rerunning COLMAP or DensifyPointCloud."
    )
    parser.add_argument("session", help="Saved session folder name, e.g. session_022")
    parser.add_argument(
        "--data-root", type=Path,
        default=project_root / "data" / "phone_reconstruction",
    )
    parser.add_argument("--output-tag", default="mesh_quality_rebuild_v2")
    parser.add_argument("--backup-name", default="mesh_before_quality_rebuild_v2")
    parser.add_argument(
        "--remove-spurious", type=float, default=20.0,
        help="OpenMVS long-edge / isolated-component filter multiplier (20 is the 2.4.0 default; smaller is more aggressive)",
    )
    parser.add_argument(
        "--no-free-space-support", action="store_true",
        help="Disable OpenMVS free-space support (enabled by default for this comparison rebuild)",
    )
    args = parser.parse_args()
    session_dir = args.data_root.expanduser().resolve() / args.session

    rebuild_session(
        session_dir,
        output_tag=args.output_tag,
        backup_name=args.backup_name,
        remove_spurious=args.remove_spurious,
        free_space_support=not args.no_free_space_support,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
