#!/usr/bin/env python3
"""Safely run OpenMVS TextureMesh only against a saved reconstruction mesh.

Dry-run is the default. Execution requires --run and writes to a new texture_ab_*
folder. COLMAP, DensifyPointCloud, ReconstructMesh and RefineMesh are never run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

FIX_SHA = "eeedab7acceae48a92ddf44eaf4eb6ffe2fef14d"
FIX_URL = f"https://github.com/cdcseacave/openMVS/commit/{FIX_SHA}"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


class TextureRetextureError(RuntimeError):
    """A safe texture-only experiment could not be prepared or verified."""


@dataclass(frozen=True)
class SessionInputs:
    session_dir: Path
    work_dir: Path
    scene_file: Path
    mesh_ply: Path
    image_dir: Path
    image_files: tuple[Path, ...]
    baseline_mesh_files: tuple[Path, ...]
    texture_threads: int
    texture_threads_source: str


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def directory_hashes(directory: Path) -> dict[str, str]:
    """Hash every file currently present, detecting additions and removals too."""
    return {
        path.relative_to(directory).as_posix(): sha256_file(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def build_info_for(executable: Path) -> dict[str, object]:
    """Verify that the executable is the separately built, pinned upstream fix."""
    metadata_path = executable.with_name("TextureMesh.build.json")
    if not metadata_path.is_file():
        raise TextureRetextureError(
            f"Missing provenance file {metadata_path}; build with "
            "scripts/build_openmvs_texture_fix_macos.sh, not the old v2.4.0 binary."
        )
    try:
        info = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TextureRetextureError(f"Cannot read TextureMesh build metadata: {exc}") from exc
    if info.get("source_commit") != FIX_SHA:
        raise TextureRetextureError(
            f"TextureMesh was not built from the pinned upstream fix {FIX_SHA}."
        )
    if info.get("binary_sha256") != sha256_file(executable):
        raise TextureRetextureError("TextureMesh binary hash does not match its build metadata.")
    return info


def resolve_executable(value: str) -> Path:
    candidate = Path(value).expanduser()
    if candidate.parent == Path(".") and not candidate.is_absolute():
        candidate = Path(shutil.which(value) or value)
    candidate = candidate.resolve()
    if not candidate.is_file() or not candidate.stat().st_mode & 0o111:
        raise TextureRetextureError(f"TextureMesh is missing or not executable: {candidate}")
    build_info_for(candidate)
    return candidate


def recorded_texture_threads(work_dir: Path) -> tuple[int, str]:
    log_path = work_dir.parent / "openmvs_pipeline.log"
    if log_path.is_file():
        marker = "Pipeline configuration: "
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.startswith(marker):
                continue
            try:
                cfg = json.loads(line[len(marker):]).get("openmvs", {})
                count = int(cfg.get("texture_threads") or cfg.get("max_threads"))
                if 1 <= count <= 64:
                    return count, "session_049 pipeline configuration"
            except (TypeError, ValueError, json.JSONDecodeError):
                break
    return 2, "conservative fallback (original thread count not recorded)"


def inspect_session(session_path: str | Path) -> SessionInputs:
    session = Path(session_path).expanduser().resolve()
    if not session.is_dir():
        raise TextureRetextureError(f"Session directory not found: {session}")
    work = session / "openmvs"
    scene, mesh = work / "scene_mesh.mvs", work / "scene_mesh.ply"
    images = work / "dense" / "images"
    final_mesh_dir = session / "mesh"
    missing = [p for p in (scene, mesh) if not p.is_file()]
    if missing:
        raise TextureRetextureError(
            "Saved pre-texture scene is incomplete; missing: "
            + ", ".join(str(p) for p in missing)
            + ". Stop; do not run a new reconstruction as a fallback."
        )
    if not images.is_dir():
        raise TextureRetextureError(f"Saved undistorted image folder not found: {images}")
    if not final_mesh_dir.is_dir():
        raise TextureRetextureError(f"Baseline final mesh folder not found: {final_mesh_dir}")
    baseline_mesh_files = tuple(sorted(p for p in final_mesh_dir.rglob("*") if p.is_file()))
    if not baseline_mesh_files:
        raise TextureRetextureError(f"Baseline final mesh folder is empty: {final_mesh_dir}")
    image_files = tuple(sorted(
        p for p in images.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    ))
    if not image_files:
        raise TextureRetextureError(f"No undistorted source photos found in {images}")
    threads, source = recorded_texture_threads(work)
    return SessionInputs(session, work.resolve(), scene.resolve(), mesh.resolve(),
                         images.resolve(), image_files, baseline_mesh_files, threads, source)


def resolve_output_dir(session: Path, requested: str | Path | None) -> Path:
    candidate = session / "texture_ab_fixed" if requested is None else Path(requested).expanduser()
    if not candidate.is_absolute():
        candidate = session / candidate
    candidate = candidate.resolve()
    if candidate.parent != session.resolve():
        raise TextureRetextureError("Output must be a direct child of the selected session.")
    if not candidate.name.startswith("texture_ab_"):
        raise TextureRetextureError("Output folder name must start with 'texture_ab_'.")
    if candidate.exists():
        raise TextureRetextureError(f"Output already exists; refusing to overwrite: {candidate}")
    return candidate


def build_command(
    executable: Path,
    inputs: SessionInputs,
    output: Path,
    *,
    max_threads: int | None = None,
    max_texture_size: int = 2048,
    virtual_face_images: int = 1,
    local_seam_leveling: int = 1,
    global_seam_leveling: int = 0,
) -> list[str]:
    """Session-049 TextureMesh options; first A/B variable is only the binary."""
    threads = inputs.texture_threads if max_threads is None else max_threads
    if threads < 1 or max_texture_size < 256 or virtual_face_images < 1:
        raise ValueError("Invalid threads, texture size, or virtual-face count.")
    if local_seam_leveling not in (0, 1) or global_seam_leveling not in (0, 1):
        raise ValueError("Seam-leveling values must be 0 or 1.")
    return [
        str(executable), "-i", str(inputs.scene_file),
        "-o", str(output / "scene_textured.obj"), "--export-type", "obj",
        "--max-threads", str(threads), "--resolution-level", "0",
        "--min-resolution", "640", "--max-texture-size", str(max_texture_size),
        "--virtual-face-images", str(virtual_face_images),
        "--cost-smoothness-ratio", "0.7", "--close-holes", "0",
        "--sharpness-weight", "0.5", "--ignore-mask-label=-1",
        "--local-seam-leveling", str(local_seam_leveling),
        "--global-seam-leveling", str(global_seam_leveling),
    ]


def diffuse_maps(mtl: Path) -> list[Path]:
    result: list[Path] = []
    if not mtl.is_file():
        return result
    for line in mtl.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            parts = shlex.split(line, comments=True)
        except ValueError:
            continue
        if parts and parts[0].lower() == "map_kd" and len(parts) >= 2:
            result.append((mtl.parent / parts[-1]).resolve())
    return result


def run_texture_only(
    inputs: SessionInputs,
    executable: Path,
    output: Path,
    *,
    max_threads: int | None = None,
    max_texture_size: int = 2048,
    virtual_face_images: int = 1,
    local_seam_leveling: int = 1,
    global_seam_leveling: int = 0,
) -> dict[str, object]:
    """Run TextureMesh only, preserving all input files and baseline exports."""
    output = resolve_output_dir(inputs.session_dir, output)
    executable = executable.resolve()
    build_info = build_info_for(executable)
    command = build_command(executable, inputs, output, max_threads=max_threads,
                            max_texture_size=max_texture_size,
                            virtual_face_images=virtual_face_images,
                            local_seam_leveling=local_seam_leveling,
                            global_seam_leveling=global_seam_leveling)

    def input_hashes() -> dict[str, object]:
        return {
            "scene_mesh_mvs": sha256_file(inputs.scene_file),
            "scene_mesh_ply": sha256_file(inputs.mesh_ply),
            "undistorted_images": directory_hashes(inputs.image_dir),
            "baseline_final_mesh_package": directory_hashes(inputs.session_dir / "mesh"),
        }

    before = input_hashes()
    executable_hash = sha256_file(executable)
    output.mkdir(parents=False, exist_ok=False)
    started_at, started = datetime.now(timezone.utc).isoformat(), time.perf_counter()
    launch_error = None
    try:
        process = subprocess.run(command, cwd=inputs.work_dir, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                 errors="replace", check=False)
        exit_code, stdout = process.returncode, process.stdout or ""
    except OSError as exc:
        launch_error, exit_code = str(exc), -1
        stdout = f"Could not launch TextureMesh: {exc}\n"
    elapsed = time.perf_counter() - started
    log_path = output / "TextureMesh.log"
    log_path.write_text(stdout, encoding="utf-8")
    after = input_hashes()
    unchanged = before == after
    obj, mtl = output / "scene_textured.obj", output / "scene_textured.mtl"
    maps = diffuse_maps(mtl)
    missing = [p for p in maps if not p.is_file()]
    output_files = [p for p in (obj, mtl, *maps) if p.is_file()]
    empty_outputs = [str(p) for p in output_files if p.stat().st_size == 0]
    valid_outputs = (
        exit_code == 0 and obj.is_file() and mtl.is_file()
        and bool(maps) and not missing and not empty_outputs
    )
    manifest: dict[str, object] = {
        "schema_version": 1, "created_at": started_at,
        "status": "success" if valid_outputs and unchanged else "failed",
        "upstream_fix": {"commit": FIX_SHA, "url": FIX_URL},
        "executable": str(executable), "executable_sha256": executable_hash,
        "executable_build_info": build_info,
        "session_dir": str(inputs.session_dir), "input_scene": str(inputs.scene_file),
        "input_mesh_ply": str(inputs.mesh_ply), "input_image_dir": str(inputs.image_dir),
        "input_image_count": len(inputs.image_files),
        "input_image_names": [p.name for p in inputs.image_files],
        "baseline_mesh_files": [str(p) for p in inputs.baseline_mesh_files],
        "baseline_final_mesh_unchanged": (
            before["baseline_final_mesh_package"] == after["baseline_final_mesh_package"]
        ),
        "texture_threads_source": inputs.texture_threads_source,
        "input_sha256_before": before, "input_sha256_after": after,
        "input_files_unchanged": unchanged, "command": command,
        "command_display": shlex.join(command), "working_directory": str(inputs.work_dir),
        "exit_code": exit_code, "launch_error": launch_error,
        "elapsed_seconds": round(elapsed, 3), "log_file": str(log_path),
        "output_obj": str(obj) if obj.is_file() else None,
        "output_mtl": str(mtl) if mtl.is_file() else None,
        "diffuse_maps": [str(p) for p in maps],
        "missing_diffuse_maps": [str(p) for p in missing],
        "empty_output_files": empty_outputs,
        "output_sha256": {p.name: sha256_file(p) for p in [obj, mtl, *maps] if p.is_file()},
        "geometry_stages_run": [],
    }
    (output / "texture_reprocess_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if exit_code != 0:
        raise TextureRetextureError(f"TextureMesh failed with exit code {exit_code}. See {log_path}.")
    if not unchanged:
        raise TextureRetextureError("Input hashes changed during texture-only processing.")
    if not valid_outputs:
        raise TextureRetextureError(f"Incomplete or empty OBJ/MTL/diffuse-map output. Inspect {log_path}.")
    return manifest


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Texture an existing OpenMVS mesh only; dry-run by default.")
    p.add_argument("--session", required=True)
    p.add_argument("--texture-mesh", required=True, help="Pinned fixed TextureMesh executable")
    p.add_argument("--output-dir", default=None, help="New texture_ab_* child (default texture_ab_fixed)")
    p.add_argument("--max-threads", type=int, default=None, help="Default: thread count from session log")
    p.add_argument("--max-texture-size", type=int, default=2048)
    p.add_argument("--virtual-face-images", type=int, default=1)
    p.add_argument("--local-seam-leveling", type=int, choices=(0, 1), default=1)
    p.add_argument("--global-seam-leveling", type=int, choices=(0, 1), default=0)
    p.add_argument("--run", action="store_true", help="Execute; without this flag only dry-run")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        inputs = inspect_session(args.session)
        executable = resolve_executable(args.texture_mesh)
        output = resolve_output_dir(inputs.session_dir, args.output_dir)
        threads = inputs.texture_threads if args.max_threads is None else args.max_threads
        command = build_command(executable, inputs, output, max_threads=threads,
                                max_texture_size=args.max_texture_size,
                                virtual_face_images=args.virtual_face_images,
                                local_seam_leveling=args.local_seam_leveling,
                                global_seam_leveling=args.global_seam_leveling)
    except (TextureRetextureError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print("Texture-only preflight passed")
    print(f"Input scene: {inputs.scene_file}")
    print(f"Input mesh:  {inputs.mesh_ply}")
    print(f"Image views: {len(inputs.image_files)}")
    print(f"Threads:     {threads} ({inputs.texture_threads_source})")
    print(f"Output:      {output}")
    print(f"Command:     {shlex.join(command)}")
    print("Geometry stages: NONE")
    if not args.run:
        print("DRY RUN ONLY: no files created; pass --run after reviewing the command.")
        return 0
    try:
        result = run_texture_only(inputs, executable, output, max_threads=threads,
                                  max_texture_size=args.max_texture_size,
                                  virtual_face_images=args.virtual_face_images,
                                  local_seam_leveling=args.local_seam_leveling,
                                  global_seam_leveling=args.global_seam_leveling)
    except (TextureRetextureError, OSError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Texture-only run complete: {result['elapsed_seconds']} seconds")
    print(f"Manifest: {output / 'texture_reprocess_manifest.json'}")
    print("Diffuse maps:\n" + "\n".join(result["diffuse_maps"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
