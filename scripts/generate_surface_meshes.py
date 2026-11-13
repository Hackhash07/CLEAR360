"""
scripts/generate_surface_meshes.py — Pre-generate 3D Surface Meshes for CLEAR360 M7.1
====================================================================================

Generates continuous, watertight 3D surface meshes from multi-camera RGB-D data
for both Ground Truth (Cycles simulation) and Calibrated AI (Depth Anything V2)
pipelines across key benchmark frames (0, 5, 10) and test scenarios.

Output directory: data/meshes/
Output format: PLY (with 32-bit float vertices, faces, vertex RGB colors, and normals).
"""

from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import open3d as o3d

from mesh.rgbd_mesher import RGBDSurfaceMesher
from validation.gt_loader import load_depth_exr
DATA_ROOT = PROJECT_ROOT / "data"
MESH_OUT_DIR = DATA_ROOT / "meshes"
CAMERA_NAMES = ["front", "rear", "left", "right"]


def generate_frame_meshes(frame_idx: int, mesher: RGBDSurfaceMesher) -> None:
    """Generate and save GT and AI surface meshes for a benchmark frame."""
    frame_str = f"frame_{frame_idx:04d}"
    print(f"\n==========================================")
    print(f"Processing {frame_str}...")
    print(f"==========================================")

    # 1. Load RGB images and semantic masks
    rgb_dict = {}
    sem_dict = {}
    for cam in CAMERA_NAMES:
        rgb_path = DATA_ROOT / "photoreal_frames" / cam / f"{frame_str}.png"
        if not rgb_path.exists():
            print(f"Warning: RGB not found: {rgb_path}")
            continue
        # cv2 loads BGR, convert to RGB
        img_bgr = cv2.imread(str(rgb_path))
        rgb_dict[cam] = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        sem_path = DATA_ROOT / "photoreal_gt_semantic" / cam / f"{frame_str}.png"
        if sem_path.exists():
            sem_dict[cam] = cv2.imread(str(sem_path), cv2.IMREAD_GRAYSCALE)

    # 2. Ground-Truth Pipeline
    gt_depth_dict = {}
    for cam in CAMERA_NAMES:
        gt_path = DATA_ROOT / "photoreal_gt_depth" / cam / f"{frame_str}.exr"
        if gt_path.exists():
            gt_depth_dict[cam] = load_depth_exr(gt_path)

    if len(gt_depth_dict) == 4:
        t0 = time.perf_counter()
        mesh_gt = mesher.reconstruct_scene_mesh(
            rgb_dict=rgb_dict,
            depth_dict=gt_depth_dict,
            semantic_dict=sem_dict,
        )
        dt = (time.perf_counter() - t0) * 1000.0
        out_gt = MESH_OUT_DIR / f"{frame_str}_ground_truth_mesh.ply"
        mesher.save_mesh_ply(mesh_gt, out_gt)
        print(f"  [GT] Mesh generated in {dt:.1f} ms: {len(mesh_gt.vertices):,} vertices, "
              f"{len(mesh_gt.triangles):,} faces -> {out_gt.name}")

    # 3. Calibrated AI Pipeline (Note: strict AI separation — no GT semantic mask used)
    ai_depth_dict = {}
    for cam in CAMERA_NAMES:
        ai_path = DATA_ROOT / "photoreal_calib_depth" / cam / f"{frame_str}.npy"
        if ai_path.exists():
            ai_depth_dict[cam] = np.load(str(ai_path)).astype(np.float32)

    if len(ai_depth_dict) == 4:
        t0 = time.perf_counter()
        mesh_ai = mesher.reconstruct_scene_mesh(
            rgb_dict=rgb_dict,
            depth_dict=ai_depth_dict,
            semantic_dict=None,  # AI pipeline does not use GT semantics
        )
        dt = (time.perf_counter() - t0) * 1000.0
        out_ai = MESH_OUT_DIR / f"{frame_str}_ai_calibrated_mesh.ply"
        mesher.save_mesh_ply(mesh_ai, out_ai)
        print(f"  [AI] Mesh generated in {dt:.1f} ms: {len(mesh_ai.vertices):,} vertices, "
              f"{len(mesh_ai.triangles):,} faces -> {out_ai.name}")


def generate_scenario_meshes(scenario_name: str, mesher: RGBDSurfaceMesher) -> None:
    """Generate and save GT and AI surface meshes for an M8 scenario."""
    scen_dir = DATA_ROOT / "m8_scenarios" / scenario_name
    if not scen_dir.exists():
        return

    print(f"\nProcessing scenario: {scenario_name}...")
    rgb_dict = {}
    sem_dict = {}
    gt_depth_dict = {}
    ai_depth_dict = {}

    for cam in CAMERA_NAMES:
        rgb_p = scen_dir / "rgb" / f"{cam}.png"
        if rgb_p.exists():
            img_bgr = cv2.imread(str(rgb_p))
            rgb_dict[cam] = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        sem_p = scen_dir / "gt_semantic" / f"{cam}.png"
        if sem_p.exists():
            sem_dict[cam] = cv2.imread(str(sem_p), cv2.IMREAD_GRAYSCALE)

        gt_p = scen_dir / "gt_depth" / f"{cam}.exr"
        if gt_p.exists():
            gt_depth_dict[cam] = load_depth_exr(gt_p)

        ai_p = scen_dir / "calib_depth" / f"{cam}.npy"
        if ai_p.exists():
            ai_depth_dict[cam] = np.load(str(ai_p)).astype(np.float32)

    # Reconstruct GT
    if len(gt_depth_dict) == 4 and len(rgb_dict) == 4:
        mesh_gt = mesher.reconstruct_scene_mesh(
            rgb_dict=rgb_dict,
            depth_dict=gt_depth_dict,
            semantic_dict=sem_dict,
        )
        out_gt = MESH_OUT_DIR / f"{scenario_name}_ground_truth_mesh.ply"
        mesher.save_mesh_ply(mesh_gt, out_gt)
        print(f"  [GT] Scenario mesh: {len(mesh_gt.vertices):,} vertices, {len(mesh_gt.triangles):,} faces")

    # Reconstruct AI
    if len(ai_depth_dict) == 4 and len(rgb_dict) == 4:
        mesh_ai = mesher.reconstruct_scene_mesh(
            rgb_dict=rgb_dict,
            depth_dict=ai_depth_dict,
            semantic_dict=None,
        )
        out_ai = MESH_OUT_DIR / f"{scenario_name}_ai_calibrated_mesh.ply"
        mesher.save_mesh_ply(mesh_ai, out_ai)
        print(f"  [AI] Scenario mesh: {len(mesh_ai.vertices):,} vertices, {len(mesh_ai.triangles):,} faces")


def main():
    MESH_OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Configure mesher with stride=2 (640x360 grid) for crisp photorealism and >100 FPS performance
    mesher = RGBDSurfaceMesher(
        stride=2,
        min_depth=0.25,
        max_depth=35.0,
        max_edge_len=0.40,
        max_depth_step_base=0.25,
        depth_step_ratio=0.08,
        min_grazing_cos=0.10,
        exclude_ego=True,
    )

    # Process all benchmark frames
    for f in [0, 5, 10]:
        generate_frame_meshes(f, mesher)

    # Process M8 scenarios
    scenarios = [
        "scenario_01_narrow_street",
        "scenario_02_double_parked",
        "scenario_03_close_pillars_walls",
        "scenario_04_tight_parking_gap",
        "scenario_05_speed_breakers_kerbs",
        "scenario_06_approaching_pedestrians",
        "scenario_07_harsh_lighting_shadows",
        "scenario_08_partially_occluded",
        "scenario_09_t_junction_turn",
        "scenario_10_construction_zone",
    ]
    for sc in scenarios:
        generate_scenario_meshes(sc, mesher)

    print("\n[Done] All surface meshes generated successfully in data/meshes/.")


if __name__ == "__main__":
    main()
