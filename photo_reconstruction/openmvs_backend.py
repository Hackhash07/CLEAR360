"""OpenMVS command-line backend for photo-to-3D reconstruction.

This backend is optional. The existing CLEAR360 Python reconstruction pipeline
remains the default; OpenMVS can be selected when its binaries are installed.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Optional

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


ProgressCallback = Optional[Callable[[int, str], None]]


class OpenMVSBackend:
    """Run COLMAP + OpenMVS on a set of photographs and export a textured GLB."""

    def __init__(self, openmvs_dir: Optional[Path] = None, colmap_bin: Optional[str] = None):
        self.openmvs_dir = (
            Path(openmvs_dir).expanduser() if openmvs_dir else configured_openmvs_root()
        )
        self.colmap_bin = colmap_bin or shutil.which("colmap") or "colmap"

    def _binary(self, name: str) -> str:
        found = find_openmvs_executable(name, openmvs_dir=self.openmvs_dir)
        if found:
            return found
        root_hint = str(self.openmvs_dir) if self.openmvs_dir else (
            "~/Applications/OpenMVS-2.4.0"
        )
        raise FileNotFoundError(
            f"Cannot find executable {name}. Checked {root_hint}, its bin/build subfolders, "
            "and PATH. Set CLEAR360_OPENMVS_DIR to the OpenMVS install folder."
        )

    @staticmethod
    def _run(command: list[str], cwd: Path, log_file: Path,
             progress: ProgressCallback, percent: int, message: str) -> None:
        if progress:
            progress(percent, message)
        with log_file.open("a", encoding="utf-8") as log:
            log.write("\n$ " + " ".join(command) + "\n")
            log.flush()
            proc = subprocess.Popen(
                command, cwd=str(cwd), stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                log.write(line)
                log.flush()
            code = proc.wait()
        if code != 0:
            raise RuntimeError(
                f"{Path(command[0]).name} exited with code {code}. See {log_file}."
            )

    def run(self, image_paths: list[Path], session_dir: Path,
            quality_preset: str = "standard",
            progress_cb: ProgressCallback = None) -> dict:
        """Reconstruct uploaded images and return session output paths."""
        session_dir = Path(session_dir)
        images_dir = session_dir / "original_images"
        images_dir.mkdir(parents=True, exist_ok=True)
        if len(image_paths) < 6:
            raise ValueError(
                "OpenMVS reconstruction requires at least 6 overlapping photographs. "
                "Use a walk-around photo sequence with substantial shared scene content."
            )

        colmap_candidate = Path(self.colmap_bin).expanduser()
        if colmap_candidate.is_file() and os.access(colmap_candidate, os.X_OK):
            colmap = str(colmap_candidate)
        else:
            colmap = shutil.which(self.colmap_bin)
        if not colmap:
            raise FileNotFoundError("COLMAP executable not found. Activate its environment or set CLEAR360_COLMAP_BIN.")

        densify = self._binary("DensifyPointCloud")
        reconstruct = self._binary("ReconstructMesh")
        refine = self._binary("RefineMesh")
        texture = self._binary("TextureMesh")
        interface_colmap = self._binary("InterfaceCOLMAP")
        work = session_dir / "openmvs"
        sparse = work / "sparse"
        dense = work / "dense"
        sparse.mkdir(parents=True, exist_ok=True)
        dense.mkdir(parents=True, exist_ok=True)
        log_file = session_dir / "openmvs_pipeline.log"
        log_file.write_text("CLEAR360 OpenMVS pipeline\n", encoding="utf-8")

        preset = quality_preset.lower()
        if preset in {"draft", "quick"}:
            feature_size, max_image_size, max_resolution, resolution_level = 1000, 1000, 800, 1
        elif preset == "high":
            feature_size, max_image_size, max_resolution, resolution_level = 6000, 2400, 1600, 1
        elif preset == "maximum":
            feature_size, max_image_size, max_resolution, resolution_level = 8000, 3200, 2000, 0
        else:
            feature_size, max_image_size, max_resolution, resolution_level = 2500, 1600, 1200, 1
        threads = 1  # Keep laptop memory usage predictable.

        db = work / "database.db"
        sparse_out = sparse / "0"
        self._run(
            [colmap, "feature_extractor", "--database_path", str(db),
             "--image_path", str(images_dir), "--ImageReader.single_camera", "1",
             "--ImageReader.camera_model", "SIMPLE_RADIAL",
             "--SiftExtraction.max_image_size", str(max_image_size),
             "--SiftExtraction.max_num_features", str(feature_size),
             "--SiftExtraction.use_gpu", "0"],
            work, log_file, progress_cb, 8, "Extracting image features (COLMAP)...",
        )
        self._run(
            [colmap, "exhaustive_matcher", "--database_path", str(db),
             "--SiftMatching.use_gpu", "0"],
            work, log_file, progress_cb, 20, "Matching overlapping photographs...",
        )
        self._run(
            [colmap, "mapper", "--database_path", str(db), "--image_path", str(images_dir),
             "--output_path", str(sparse)],
            work, log_file, progress_cb, 34, "Estimating camera poses and sparse 3D structure...",
        )
        if not sparse_out.exists() or not any(sparse_out.iterdir()):
            raise RuntimeError("COLMAP produced no registered model. Add more overlapping photos.")
        # COLMAP's world axes are arbitrary. Export registered camera poses so
        # a consistent camera-up direction can align the final 3D exports.
        pose_text_dir = work / "sparse_text"
        self._run(
            [colmap, "model_converter", "--input_path", str(sparse_out),
             "--output_path", str(pose_text_dir), "--output_type", "TXT"],
            work, log_file, progress_cb, 38, "Estimating the scene's upright direction...",
        )
        images_txt = pose_text_dir / "images.txt"

        self._run(
            [colmap, "image_undistorter", "--image_path", str(images_dir),
             "--input_path", str(sparse_out), "--output_path", str(dense),
             "--output_type", "COLMAP", "--max_image_size", str(max_image_size)],
            work, log_file, progress_cb, 46, "Undistorting images for dense reconstruction...",
        )
        scene_mvs = work / "scene.mvs"
        self._run(
            [interface_colmap, "-i", str(dense), "-o", str(scene_mvs)],
            work, log_file, progress_cb, 54, "Preparing calibrated OpenMVS scene...",
        )
        if not scene_mvs.exists():
            raise RuntimeError("COLMAP-to-OpenMVS conversion did not create scene.mvs.")

        scene_fuse = work / "scene_fuse.mvs"
        shutil.copy2(scene_mvs, scene_fuse)
        common = [
            "--archive-type", "2", "--max-threads", str(threads),
            "--resolution-level", str(resolution_level),
            "--max-resolution", str(max_resolution), "--min-resolution", "320",
            "--number-views", "4", "--iters", "2", "--geometric-iters", "1",
            "--tower-mode", "0",
        ]
        self._run(
            [densify, "-i", str(scene_mvs), "-w", str(work), *common, "--fusion-mode", "1"],
            work, log_file, progress_cb, 62, "Estimating multi-view depth maps...",
        )
        # The second pass uses the independent copy so first-pass depth maps remain reusable.
        self._run(
            [densify, "-i", str(scene_fuse), "-w", str(work), *common,
             "--number-views-fuse", "2", "--fusion-mode", "0"],
            work, log_file, progress_cb, 72, "Fusing depth maps into a dense point cloud...",
        )

        dense_scene = work / "scene_fuse_dense.mvs"
        dense_ply = work / "scene_fuse_dense.ply"
        if not dense_scene.exists() or not dense_ply.exists():
            raise RuntimeError("OpenMVS dense fusion did not produce its MVS and PLY outputs.")

        mesh_scene = work / "scene_mesh.mvs"
        self._run(
            [reconstruct, "-i", str(dense_scene), "-o", str(mesh_scene),
             "--archive-type", "2", "--max-threads", str(threads),
             "--close-holes", "0", "--smooth", "0"],
            work, log_file, progress_cb, 80, "Reconstructing triangular surface mesh...",
        )
        mesh_ply = work / "scene_mesh.ply"
        if not mesh_ply.exists():
            raise RuntimeError("OpenMVS mesh reconstruction did not create scene_mesh.ply.")

        refined_scene = work / "scene_mesh_refined.mvs"
        self._run(
            [refine, "-i", str(mesh_scene), "-m", str(mesh_ply), "-o", str(refined_scene),
             "--archive-type", "2", "--max-threads", str(threads),
             "--resolution-level", "1", "--min-resolution", "320",
             "--max-views", "4", "--decimate", "1", "--close-holes", "0",
             "--scales", "1", "--reduce-memory", "1"],
            work, log_file, progress_cb, 86, "Refining mesh geometry...",
        )
        texture_options = [
            "--max-threads", str(threads),
            "--resolution-level", "0", "--min-resolution", "320",
            "--max-texture-size", "2048", "--virtual-face-images", "3",
            "--cost-smoothness-ratio", "0.7", "--close-holes", "0",
            "--sharpness-weight", "0.5", "--ignore-mask-label=-1",
            "--local-seam-leveling", "0", "--global-seam-leveling", "0",
        ]
        glb_work = work / "scene_textured.glb"
        self._run(
            [texture, "-i", str(refined_scene), "-o", str(glb_work),
             "--export-type", "glb", *texture_options],
            work, log_file, progress_cb, 92, "Generating textured GLB export...",
        )

        # Export a second copy as OBJ so the native VisPy viewer can consume
        # its UV coordinates and atlas through TextureFilter.
        obj_work = work / "scene_textured.obj"
        self._run(
            [texture, "-i", str(refined_scene), "-o", str(obj_work),
             "--export-type", "obj", *texture_options],
            work, log_file, progress_cb, 96, "Preparing textured OBJ and atlas for the desktop viewer...",
        )

        mesh_dir = session_dir / "mesh"
        mesh_dir.mkdir(parents=True, exist_ok=True)
        final_ply = mesh_dir / "scene_mesh.ply"
        final_glb = mesh_dir / "scene_textured.glb"
        final_obj = mesh_dir / "scene_mesh.obj"
        refined_ply = work / "scene_mesh_refined.ply"
        if not refined_ply.exists():
            refined_ply = mesh_ply
        shutil.copy2(refined_ply, final_ply)
        shutil.copy2(glb_work, final_glb)

        # The dedicated OBJ texturing pass exports the viewer's UV atlas.
        texture_obj = obj_work
        texture_mtl = work / "scene_textured.mtl"
        atlas_files = sorted(work.glob("scene_textured_*.png"))
        if not texture_obj.exists():
            raise RuntimeError("OpenMVS OBJ export did not create scene_textured.obj.")
        if not atlas_files:
            raise RuntimeError("OpenMVS OBJ export did not create its texture atlas PNG.")
        shutil.copy2(texture_obj, final_obj)

        atlas_name_map = {}
        for index, atlas in enumerate(atlas_files):
            target_name = "scene_mesh.png" if index == 0 else f"scene_mesh_{index}.png"
            shutil.copy2(atlas, mesh_dir / target_name)
            atlas_name_map[atlas.name] = target_name

        if texture_mtl.exists():
            # Keep every MTL map_Kd reference valid after renaming/copying atlases.
            rewritten_lines = []
            for line in texture_mtl.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines():
                stripped = line.lstrip()
                if stripped.lower().startswith("map_kd "):
                    tokens = stripped.split()
                    original_name = Path(tokens[-1]).name
                    target_name = atlas_name_map.get(original_name)
                    if target_name:
                        indent = line[: len(line) - len(stripped)]
                        line = f"{indent}map_Kd {target_name}"
                rewritten_lines.append(line)
            (mesh_dir / "scene_mesh.mtl").write_text(
                "\n".join(rewritten_lines) + "\n", encoding="utf-8"
            )

            # Keep the copied OBJ linked to the material file copied beside it.
            obj_lines = []
            for line in final_obj.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines():
                if line.lstrip().lower().startswith("mtllib "):
                    indent = line[: len(line) - len(line.lstrip())]
                    line = f"{indent}mtllib scene_mesh.mtl"
                obj_lines.append(line)
            final_obj.write_text("\n".join(obj_lines) + "\n", encoding="utf-8")

        orientation = None
        estimated_up = estimate_world_up(images_txt)
        if estimated_up is not None:
            up_vector, consistency, pose_count = estimated_up
            alignment_rotation = rotation_from_to(
                up_vector, np.array([0.0, 0.0, 1.0], dtype=np.float64)
            )
            center = obj_median_center(final_obj)
            obj_ok = align_obj_file(final_obj, alignment_rotation, center=center)
            ply_ok = align_open3d_geometry_file(
                final_ply, alignment_rotation, center, point_cloud=False
            )
            cloud_ok = align_open3d_geometry_file(
                dense_ply, alignment_rotation, center, point_cloud=True
            )
            glb_ok = align_glb_file(final_glb, alignment_rotation, center)
            orientation = {
                "method": "COLMAP camera-up consensus",
                "registered_camera_poses": pose_count,
                "up_consistency": round(consistency, 4),
                "estimated_up_before_alignment": [float(x) for x in up_vector],
                "aligned_to_world_axis": "+Z",
                "exports_aligned": {
                    "obj": obj_ok, "ply": ply_ok,
                    "dense_cloud": cloud_ok, "glb": glb_ok,
                },
            }
            with log_file.open("a", encoding="utf-8") as log:
                log.write("\nScene orientation alignment: " + str(orientation) + "\n")
        else:
            orientation = {
                "method": "not applied",
                "reason": (
                    "COLMAP camera-up directions were unavailable or inconsistent; "
                    "the model retains the original arbitrary SfM orientation."
                ),
            }
            with log_file.open("a", encoding="utf-8") as log:
                log.write("\nScene orientation alignment skipped: " + orientation["reason"] + "\n")

        if progress_cb:
            progress_cb(100, "OpenMVS reconstruction complete. Textured GLB and mesh are ready.")
        return {
            "success": True, "mesh_ply": final_ply, "mesh_obj": final_obj,
            "mesh_glb": final_glb, "dense_ply": dense_ply, "backend": "openmvs",
            "log_file": log_file, "quality_preset": preset,
            "orientation": orientation,
        }
