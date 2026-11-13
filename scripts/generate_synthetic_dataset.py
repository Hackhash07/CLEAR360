#!/usr/bin/env python3
"""
scripts/generate_synthetic_dataset.py — Analytical Synthetic Surround Dataset Generator
========================================================================================

Generates the complete 20-frame CLEAR360 four-camera dataset analytically using
exact ray-casting against the Indian parking environment geometry and pinhole camera models.

This produces:
  - 1280×720 8-bit RGB PNG images in data/sim_frames/{camera}/frame_####.png
  - 1280×720 32-bit float OpenEXR Z-depth maps in data/gt_depth/{camera}/frame_####.exr
  - 1280×720 32-bit float OpenEXR world position maps in data/gt_position/{camera}/frame_####.exr
  - 1280×720 8-bit grayscale PNG semantic masks in data/gt_semantic/{camera}/frame_####.png

Geometry & Coordinate Conventions (NEVER DEVIATE):
  - World / Vehicle frame: X = RIGHT, Y = FORWARD, Z = UP. Vehicle at origin.
  - Camera frame: OpenCV convention (x = right, y = down, z = forward into scene).
  - Depth: Camera-space Z-depth in metres (float32).
  - Pass indices:
      0: sky/unknown, 1: road, 2: vehicle, 3: parked car, 4: wall,
      5: pillar, 6: pedestrian obstacle, 7: kerb/speed-breaker.

Usage:
  python scripts/generate_synthetic_dataset.py [--frames 20] [--output-root .]
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import cv2
import Imath
import numpy as np
import OpenEXR

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.calibration import CalibrationStore


def get_obstacle_position(frame_idx: int, total_frames: int = 20) -> np.ndarray:
    """Compute interpolated world position (X, Y, Z) of the dynamic obstacle."""
    # Keyframe specifications:
    # Frame 0: (3.0, 7.0, 0.25)
    # Frame 9 (halfway): (-3.0, 5.5, 0.25)
    # Frame 19: (-5.0, 3.0, 0.25)
    p0 = np.array([3.0, 7.0, 0.25])
    p1 = np.array([-3.0, 5.5, 0.25])
    p2 = np.array([-5.0, 3.0, 0.25])

    mid = (total_frames - 1) / 2.0
    if frame_idx <= mid:
        alpha = frame_idx / mid
        return (1.0 - alpha) * p0 + alpha * p1
    else:
        alpha = (frame_idx - mid) / ((total_frames - 1) - mid)
        return (1.0 - alpha) * p1 + alpha * p2


def intersect_plane_z(ray_origin: np.ndarray, ray_dirs: np.ndarray, z_plane: float,
                      x_min: float, x_max: float, y_min: float, y_max: float):
    """Vectorised ray-plane intersection for Z = z_plane."""
    dz = ray_dirs[..., 2]
    # Guard against rays parallel to plane
    valid_dz = np.abs(dz) > 1e-6
    t = np.full(dz.shape, np.inf, dtype=np.float32)

    t_calc = (z_plane - ray_origin[2]) / np.where(valid_dz, dz, 1.0)
    hit_mask = valid_dz & (t_calc > 0.1)

    px = ray_origin[0] + t_calc * ray_dirs[..., 0]
    py = ray_origin[1] + t_calc * ray_dirs[..., 1]
    bounds_mask = (px >= x_min) & (px <= x_max) & (py >= y_min) & (py <= y_max)

    valid_hits = hit_mask & bounds_mask
    t[valid_hits] = t_calc[valid_hits]
    return t, np.array([0.0, 0.0, 1.0 if ray_origin[2] > z_plane else -1.0])


def intersect_aabb(ray_origin: np.ndarray, ray_dirs: np.ndarray,
                   min_pt: np.ndarray, max_pt: np.ndarray):
    """Vectorised ray-AABB intersection using slab method."""
    # min_pt, max_pt: (3,)
    inv_dir = 1.0 / np.where(np.abs(ray_dirs) < 1e-6, 1e-6 * np.sign(ray_dirs + 1e-9), ray_dirs)

    t1 = (min_pt[0] - ray_origin[0]) * inv_dir[..., 0]
    t2 = (max_pt[0] - ray_origin[0]) * inv_dir[..., 0]
    t3 = (min_pt[1] - ray_origin[1]) * inv_dir[..., 1]
    t4 = (max_pt[1] - ray_origin[1]) * inv_dir[..., 1]
    t5 = (min_pt[2] - ray_origin[2]) * inv_dir[..., 2]
    t6 = (max_pt[2] - ray_origin[2]) * inv_dir[..., 2]

    tmin = np.maximum(np.maximum(np.minimum(t1, t2), np.minimum(t3, t4)), np.minimum(t5, t6))
    tmax = np.minimum(np.minimum(np.maximum(t1, t2), np.maximum(t3, t4)), np.maximum(t5, t6))

    hit_mask = (tmax >= np.maximum(0.1, tmin))
    t = np.full(tmin.shape, np.inf, dtype=np.float32)
    t[hit_mask] = np.where(tmin[hit_mask] > 0.1, tmin[hit_mask], tmax[hit_mask])
    return t


def intersect_cylinder_z(ray_origin: np.ndarray, ray_dirs: np.ndarray,
                         center_xy: np.ndarray, radius: float, z_min: float, z_max: float):
    """Vectorised ray-cylinder intersection with infinite cylinder capped at z_min, z_max."""
    ox = ray_origin[0] - center_xy[0]
    oy = ray_origin[1] - center_xy[1]
    dx = ray_dirs[..., 0]
    dy = ray_dirs[..., 1]

    a = dx * dx + dy * dy
    b = 2.0 * (ox * dx + oy * dy)
    c = ox * ox + oy * oy - radius * radius

    disc = b * b - 4.0 * a * c
    valid = (disc >= 0) & (a > 1e-6)

    t = np.full(dx.shape, np.inf, dtype=np.float32)
    sqrt_disc = np.sqrt(np.maximum(0.0, disc))
    t0 = (-b - sqrt_disc) / (2.0 * np.where(valid, a, 1.0))

    pz0 = ray_origin[2] + t0 * ray_dirs[..., 2]
    hit0 = valid & (t0 > 0.1) & (pz0 >= z_min) & (pz0 <= z_max)
    t[hit0] = t0[hit0]
    return t


def save_exr_depth(path: Path, depth_map: np.ndarray) -> None:
    """Save (H, W) float32 depth map into EXR with single channel 'Z'."""
    path.parent.mkdir(parents=True, exist_ok=True)
    H, W = depth_map.shape
    header = OpenEXR.Header(W, H)
    header['channels'] = {'Z': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))}
    out = OpenEXR.OutputFile(str(path), header)
    out.writePixels({'Z': depth_map.astype(np.float32).tobytes()})
    out.close()


def save_exr_position(path: Path, position_map: np.ndarray) -> None:
    """Save (H, W, 3) float32 world XYZ into EXR with channels 'X', 'Y', 'Z'."""
    path.parent.mkdir(parents=True, exist_ok=True)
    H, W, _ = position_map.shape
    header = OpenEXR.Header(W, H)
    header['channels'] = {
        'X': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
        'Y': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
        'Z': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
    }
    out = OpenEXR.OutputFile(str(path), header)
    out.writePixels({
        'X': position_map[..., 0].astype(np.float32).tobytes(),
        'Y': position_map[..., 1].astype(np.float32).tobytes(),
        'Z': position_map[..., 2].astype(np.float32).tobytes(),
    })
    out.close()


def render_scene(frame_idx: int, cam_name: str, cam_model, total_frames: int = 20):
    """Raycast the entire Indian parking scene for one camera and frame."""
    W = cam_model.intrinsics.width
    H = cam_model.intrinsics.height
    fx = cam_model.intrinsics.fx
    fy = cam_model.intrinsics.fy
    cx = cam_model.intrinsics.cx
    cy = cam_model.intrinsics.cy

    # Camera optical center in world frame
    T_world_cam = cam_model.T_world_cam
    ray_origin = T_world_cam[:3, 3]
    R_world_cam = T_world_cam[:3, :3]

    # Compute normalized camera-space rays
    u_grid, v_grid = np.meshgrid(np.arange(W), np.arange(H))
    x_cam = (u_grid - cx) / fx
    y_cam = (v_grid - cy) / fy
    z_cam = np.ones_like(x_cam, dtype=np.float32)

    rays_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)  # (H, W, 3)
    # Transform rays to world space: (R_world_cam @ ray_cam)
    rays_world = rays_cam @ R_world_cam.T
    ray_lens = np.linalg.norm(rays_world, axis=-1, keepdims=True)
    rays_world = rays_world / ray_lens  # Unit vectors

    # Buffers
    closest_t = np.full((H, W), np.inf, dtype=np.float32)
    semantic_mask = np.zeros((H, W), dtype=np.uint8)
    rgb = np.full((H, W, 3), [135, 175, 215], dtype=np.uint8)  # Indian sky blue

    # Sun lighting vector (elevation 55 deg, azimuth 160 deg)
    sun_az = math.radians(160)
    sun_el = math.radians(55)
    sun_dir = np.array([
        math.cos(sun_el) * math.sin(sun_az),
        math.cos(sun_el) * math.cos(sun_az),
        math.sin(sun_el)
    ], dtype=np.float32)
    sun_dir /= np.linalg.norm(sun_dir)

    # 1. Road plane (pass_index=1)
    t_road, n_road = intersect_plane_z(ray_origin, rays_world, 0.0, -10.0, 10.0, -10.0, 10.0)
    hit = t_road < closest_t
    closest_t[hit] = t_road[hit]
    semantic_mask[hit] = 1
    # Add subtle parking bay stripe texture on road
    px_road = ray_origin[0] + t_road * rays_world[..., 0]
    py_road = ray_origin[1] + t_road * rays_world[..., 1]
    is_stripe = (np.abs(px_road - 2.5) < 0.08) | (np.abs(px_road + 2.5) < 0.08)
    road_col = np.where(is_stripe[..., None], [240, 240, 240], [55, 55, 58])
    rgb[hit] = road_col[hit]

    # 2. Ego vehicle (pass_index=2)
    t_ego = intersect_aabb(ray_origin, rays_world,
                           np.array([-0.895, -2.15, 0.0]),
                           np.array([0.895, 2.15, 1.635]))
    hit = t_ego < closest_t
    closest_t[hit] = t_ego[hit]
    semantic_mask[hit] = 2
    rgb[hit] = [40, 110, 210]  # Blue SUV

    # 3. Parked cars (pass_index=3)
    t_pc1 = intersect_aabb(ray_origin, rays_world,
                           np.array([-5.875, -7.1, 0.0]),
                           np.array([-4.125, -2.9, 1.5]))
    hit = t_pc1 < closest_t
    closest_t[hit] = t_pc1[hit]
    semantic_mask[hit] = 3
    rgb[hit] = [215, 175, 35]  # Mustard Yellow

    t_pc2 = intersect_aabb(ray_origin, rays_world,
                           np.array([4.125, -7.1, 0.0]),
                           np.array([5.875, -2.9, 1.5]))
    hit = t_pc2 < closest_t
    closest_t[hit] = t_pc2[hit]
    semantic_mask[hit] = 3
    rgb[hit] = [190, 85, 45]   # Red/Copper

    # 4. Boundary walls (pass_index=4)
    # North wall: Y=+9.5, thickness 0.3, Z: 0..2.5
    t_wn = intersect_aabb(ray_origin, rays_world, np.array([-10.0, 9.35, 0.0]), np.array([10.0, 9.65, 2.5]))
    hit = t_wn < closest_t
    closest_t[hit] = t_wn[hit]
    semantic_mask[hit] = 4
    rgb[hit] = [175, 170, 165]

    # South wall: Y=-9.5
    t_ws = intersect_aabb(ray_origin, rays_world, np.array([-10.0, -9.65, 0.0]), np.array([10.0, -9.35, 2.5]))
    hit = t_ws < closest_t
    closest_t[hit] = t_ws[hit]
    semantic_mask[hit] = 4
    rgb[hit] = [170, 168, 160]

    # East wall: X=+9.5
    t_we = intersect_aabb(ray_origin, rays_world, np.array([9.35, -10.0, 0.0]), np.array([9.65, 10.0, 2.5]))
    hit = t_we < closest_t
    closest_t[hit] = t_we[hit]
    semantic_mask[hit] = 4
    rgb[hit] = [165, 165, 165]

    # West wall: X=-9.5
    t_ww = intersect_aabb(ray_origin, rays_world, np.array([-9.65, -10.0, 0.0]), np.array([-9.35, 10.0, 2.5]))
    hit = t_ww < closest_t
    closest_t[hit] = t_ww[hit]
    semantic_mask[hit] = 4
    rgb[hit] = [165, 165, 165]

    # 5. Pillar (pass_index=5)
    t_pil = intersect_cylinder_z(ray_origin, rays_world, np.array([-6.0, 4.0]), 0.20, 0.0, 2.5)
    hit = t_pil < closest_t
    closest_t[hit] = t_pil[hit]
    semantic_mask[hit] = 5
    rgb[hit] = [210, 60, 60]  # Concrete pillar red accent

    # 6. Dynamic Pedestrian Obstacle (pass_index=6)
    obs_pos = get_obstacle_position(frame_idx, total_frames)
    t_obs = intersect_aabb(ray_origin, rays_world,
                           obs_pos - np.array([0.25, 0.25, 0.25]),
                           obs_pos + np.array([0.25, 0.25, 0.25]))
    hit = t_obs < closest_t
    closest_t[hit] = t_obs[hit]
    semantic_mask[hit] = 6
    rgb[hit] = [40, 220, 95]   # Green active obstacle

    # 7. Kerbs and speed breaker (pass_index=7)
    # Kerb left: X=-4.5, Y: -7.5..7.5, Z: 0..0.15
    t_kl = intersect_aabb(ray_origin, rays_world, np.array([-4.6, -7.5, 0.0]), np.array([-4.4, 7.5, 0.15]))
    hit = t_kl < closest_t
    closest_t[hit] = t_kl[hit]
    semantic_mask[hit] = 7
    rgb[hit] = [140, 95, 35]

    # Kerb right: X=+4.5
    t_kr = intersect_aabb(ray_origin, rays_world, np.array([4.4, -7.5, 0.0]), np.array([4.6, 7.5, 0.15]))
    hit = t_kr < closest_t
    closest_t[hit] = t_kr[hit]
    semantic_mask[hit] = 7
    rgb[hit] = [140, 95, 35]

    # Speed breaker: Y=5.5, X: -1.75..1.75, Z: 0..0.10
    t_sb = intersect_aabb(ray_origin, rays_world, np.array([-1.75, 5.3, 0.0]), np.array([1.75, 5.7, 0.10]))
    hit = t_sb < closest_t
    closest_t[hit] = t_sb[hit]
    semantic_mask[hit] = 7
    rgb[hit] = [230, 210, 25]  # Indian yellow/black speed bump

    # Compute world position map
    hit_any = closest_t < np.inf
    position_map = np.zeros((H, W, 3), dtype=np.float32)
    position_map[hit_any] = ray_origin + closest_t[hit_any, None] * rays_world[hit_any]

    # Compute OpenCV camera-space Z-depth in metres:
    # Transform hit points to camera frame: p_cam = T_cam_world @ [P_world, 1]
    # In OpenCV camera coords, z_cam is positive depth into the scene.
    depth_map = np.zeros((H, W), dtype=np.float32)
    P_hom = np.concatenate([position_map[hit_any], np.ones((np.sum(hit_any), 1), dtype=np.float32)], axis=-1)
    p_cam = P_hom @ cam_model.T_cam_world.T
    depth_map[hit_any] = np.maximum(0.0, p_cam[:, 2])

    # Unhit sky pixels get background depth (80.0 m)
    depth_map[~hit_any] = 80.0
    position_map[~hit_any] = ray_origin + 80.0 * rays_world[~hit_any]

    return rgb, depth_map, position_map, semantic_mask


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic CLEAR360 dataset.")
    parser.add_argument("--frames", type=int, default=20, help="Number of frames to generate (default: 20)")
    parser.add_argument("--output-root", type=str, default=".", help="Root output directory")
    args = parser.parse_args()

    out_root = Path(args.output_root).resolve()
    num_frames = args.frames

    print("=" * 70)
    print("CLEAR360 SYNTHETIC SURROUND DATASET GENERATOR")
    print(f"Output directory : {out_root}")
    print(f"Frames to render : {num_frames} (0 to {num_frames - 1})")
    print("=" * 70)

    # Load camera calibration store
    config_yaml = out_root / "config" / "cameras.yaml"
    if not config_yaml.exists():
        print(f"ERROR: Calibration file not found at {config_yaml}")
        sys.exit(1)

    store = CalibrationStore(config_yaml)
    store.load()

    cameras = ["front", "rear", "left", "right"]
    t0_total = time.time()

    for frame_id in range(num_frames):
        t0_frame = time.time()
        for cam_name in cameras:
            cam_model = store.get(cam_name)
            rgb, depth, pos, sem = render_scene(frame_id, cam_name, cam_model, num_frames)

            # Paths
            rgb_path = out_root / "data" / "sim_frames" / cam_name / f"frame_{frame_id:04d}.png"
            depth_path = out_root / "data" / "gt_depth" / cam_name / f"frame_{frame_id:04d}.exr"
            pos_path = out_root / "data" / "gt_position" / cam_name / f"frame_{frame_id:04d}.exr"
            sem_path = out_root / "data" / "gt_semantic" / cam_name / f"frame_{frame_id:04d}.png"

            # Save PNGs (cv2 takes BGR for colour PNG)
            rgb_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

            sem_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(sem_path), sem)

            # Save EXRs
            save_exr_depth(depth_path, depth)
            save_exr_position(pos_path, pos)

        dt_frame = time.time() - t0_frame
        print(f"  Frame {frame_id:02d}/{num_frames - 1:02d} rendered across 4 cameras in {dt_frame:.2f}s")

    total_time = time.time() - t0_total
    print("=" * 70)
    print(f"SUCCESS: Generated {num_frames * len(cameras)} FrameSets in {total_time:.2f}s!")
    print(f"Data root: {out_root / 'data'}")
    print("=" * 70)


if __name__ == "__main__":
    main()

