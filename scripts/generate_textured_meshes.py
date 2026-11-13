"""
scripts/generate_textured_meshes.py — Pre-generate 4-Camera Textured Meshes (M7.2)
===================================================================================

Generates 4-camera fused textured meshes with 2x2 texture atlases, visibility-aware
overlap pruning, and OBJ+MTL+PNG and PLY exports for both Ground Truth and AI pipelines.
"""

from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from mesh.textured_mesher import TexturedMesher
from validation.gt_loader import load_depth_exr

DATA_ROOT = PROJECT_ROOT / "data"
TEX_OUT_DIR = DATA_ROOT / "textured_meshes"
TEX_OUT_DIR.mkdir(parents=True, exist_ok=True)
CAMERA_NAMES = ["front", "rear", "left", "right"]
FRAMES = [0, 5, 10]


def generate_all_textured_meshes():
    print("=================================================================")
    print("CLEAR360 — M7.2 Phase C: Multi-Camera Textured Mesh Generation")
    print("=================================================================")

    # Use stride=2 for efficient viewer loading and real-time interaction
    mesher = TexturedMesher(stride=2, min_depth=0.25, max_depth=35.0, max_edge_len=0.35)

    for frame_idx in FRAMES:
        frame_str = f"frame_{frame_idx:04d}"
        print(f"\nProcessing {frame_str}...")

        # 1. Load RGB images and semantic masks
        rgb_dict = {}
        sem_dict = {}
        for cam in CAMERA_NAMES:
            rgb_path = DATA_ROOT / "photoreal_frames" / cam / f"{frame_str}.png"
            if not rgb_path.exists():
                print(f"Warning: RGB missing: {rgb_path}")
                continue
            img_bgr = cv2.imread(str(rgb_path))
            rgb_dict[cam] = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            sem_path = DATA_ROOT / "photoreal_gt_semantic" / cam / f"{frame_str}.png"
            if sem_path.exists():
                sem_dict[cam] = cv2.imread(str(sem_path), cv2.IMREAD_GRAYSCALE)

        # 2. Ground-Truth Pipeline (EXR + GT semantics)
        gt_depth_dict = {}
        for cam in CAMERA_NAMES:
            gt_path = DATA_ROOT / "photoreal_gt_depth" / cam / f"{frame_str}.exr"
            if gt_path.exists():
                gt_depth_dict[cam] = load_depth_exr(gt_path)

        if len(gt_depth_dict) == 4:
            t0 = time.perf_counter()
            mesh_gt = mesher.reconstruct_textured_scene(
                rgb_dict=rgb_dict,
                depth_dict=gt_depth_dict,
                semantic_dict=sem_dict,
                enable_overlap_pruning=True,
            )
            dt = (time.perf_counter() - t0) * 1000.0

            # Save OBJ + MTL + PNG
            prefix_gt = f"{frame_str}_ground_truth"
            mesher.save_textured_mesh_obj(mesh_gt, TEX_OUT_DIR, name_prefix=prefix_gt)

            # Save PLY fallback
            mesher.save_textured_mesh_ply(mesh_gt, TEX_OUT_DIR / f"{prefix_gt}.ply")

            print(f"  [GT] Reconstructed in {dt:.1f} ms: {len(mesh_gt.vertices):,} verts, "
                  f"{len(mesh_gt.faces):,} faces -> {prefix_gt}.obj/.ply")

        # 3. Calibrated AI Pipeline (NPY, no GT semantics)
        ai_depth_dict = {}
        for cam in CAMERA_NAMES:
            ai_path = DATA_ROOT / "photoreal_calib_depth" / cam / f"{frame_str}.npy"
            if ai_path.exists():
                ai_depth_dict[cam] = np.load(str(ai_path)).astype(np.float32)

        if len(ai_depth_dict) == 4:
            t0 = time.perf_counter()
            mesh_ai = mesher.reconstruct_textured_scene(
                rgb_dict=rgb_dict,
                depth_dict=ai_depth_dict,
                semantic_dict=None,  # STRICT: No GT semantics in AI pipeline
                enable_overlap_pruning=True,
            )
            dt = (time.perf_counter() - t0) * 1000.0

            prefix_ai = f"{frame_str}_ai_calibrated"
            mesher.save_textured_mesh_obj(mesh_ai, TEX_OUT_DIR, name_prefix=prefix_ai)
            mesher.save_textured_mesh_ply(mesh_ai, TEX_OUT_DIR / f"{prefix_ai}.ply")

            print(f"  [AI] Reconstructed in {dt:.1f} ms: {len(mesh_ai.vertices):,} verts, "
                  f"{len(mesh_ai.faces):,} faces -> {prefix_ai}.obj/.ply")

    print("\nTextured mesh generation complete!")


if __name__ == "__main__":
    generate_all_textured_meshes()

