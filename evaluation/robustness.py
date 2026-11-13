"""
evaluation/robustness.py — Sensor Perturbation & Robustness Simulation for CLEAR360 (M8)
========================================================================================

Simulates real-world automotive sensor degradation:
  1. Camera brightness changes (underexposure / overexposure)
  2. Harsh shadows and contrast shifts
  3. Motion blur (linear camera velocity blur)
  4. Partial lens obstruction (dirt, water droplets, mud splatter)
  5. Small camera orientation perturbations (mounting vibration, +/- 1.5 deg drift)
  6. Camera frame synchronization errors (inter-camera temporal jitter Delta t)
  7. Missing camera feeds (dropped frames / offline hardware)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np


def apply_brightness_change(image: np.ndarray, factor: float = 1.0) -> np.ndarray:
    """
    Simulates exposure / illumination variation.
    factor < 1.0 : underexposed / dark scene / tunnel entry
    factor > 1.0 : overexposed / direct glare
    """
    img_float = image.astype(np.float32) * factor
    return np.clip(img_float, 0, 255).astype(np.uint8)


def apply_harsh_shadow(
    image: np.ndarray,
    shadow_factor: float = 0.35,
    angle_deg: float = 30.0,
    coverage: float = 0.5,
) -> np.ndarray:
    """
    Simulates deep cast shadows cutting across the camera frame.
    """
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)

    # Create linear gradient / step shadow across specified angle
    rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    x_coords, y_coords = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    proj = x_coords * cos_a + y_coords * sin_a

    thresh = np.percentile(proj, coverage * 100.0)
    # Smooth step transition to mimic soft penumbra
    mask = 1.0 / (1.0 + np.exp(-(proj - thresh) * 15.0))
    # In shadow region, scale down brightness
    shadow_mult = shadow_factor + (1.0 - shadow_factor) * (1.0 - mask)
    if image.ndim == 3:
        shadow_mult = np.expand_dims(shadow_mult, axis=-1)

    result = image.astype(np.float32) * shadow_mult
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_motion_blur(
    image: np.ndarray,
    kernel_size: int = 15,
    angle_deg: float = 0.0,
) -> np.ndarray:
    """
    Simulates linear vehicle motion blur.
    kernel_size: blur magnitude in pixels (odd integer >= 3)
    angle_deg: blur direction (0 deg = horizontal, 90 deg = vertical)
    """
    if kernel_size <= 1:
        return image.copy()

    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    center = kernel_size // 2

    # Draw line along angle
    rad = math.radians(angle_deg)
    dx = math.cos(rad) * (kernel_size / 2.0)
    dy = math.sin(rad) * (kernel_size / 2.0)

    p1 = (int(round(center - dx)), int(round(center - dy)))
    p2 = (int(round(center + dx)), int(round(center + dy)))
    cv2.line(kernel, p1, p2, 1.0, thickness=1)

    kernel_sum = kernel.sum()
    if kernel_sum > 0:
        kernel /= kernel_sum

    return cv2.filter2D(image, -1, kernel)


def apply_lens_obstruction(
    image: np.ndarray,
    severity: float = 0.4,
    num_splatters: int = 8,
    seed: Optional[int] = 42,
) -> np.ndarray:
    """
    Simulates physical obstruction on camera lens (mud splatter, rain droplets, road grime).
    severity: fraction of obstruction intensity (0.0 to 1.0)
    num_splatters: number of distinct droplet / mud clusters
    """
    rng = np.random.RandomState(seed)
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)

    for _ in range(num_splatters):
        cx = rng.randint(int(w * 0.1), int(w * 0.9))
        cy = rng.randint(int(h * 0.1), int(h * 0.9))
        rx = rng.randint(int(w * 0.04), int(w * 0.15))
        ry = rng.randint(int(h * 0.04), int(h * 0.15))
        angle = rng.uniform(0, 360)

        # Draw soft ellipse on mask
        cv2.ellipse(mask, (cx, cy), (rx, ry), angle, 0, 360, 1.0, -1)

    # Blur mask to mimic out-of-focus near-lens optical blur
    mask = cv2.GaussianBlur(mask, (51, 51), 0)
    mask = np.clip(mask * severity, 0.0, 1.0)

    # Dirt color: dark brownish-grey
    dirt_color = np.array([45, 55, 65], dtype=np.float32)  # BGR / RGB
    if image.ndim == 3:
        mask_3d = np.expand_dims(mask, axis=-1)
        res = image.astype(np.float32) * (1.0 - mask_3d) + dirt_color * mask_3d
    else:
        res = image.astype(np.float32) * (1.0 - mask) + 50.0 * mask

    return np.clip(res, 0, 255).astype(np.uint8)


def apply_calibration_perturbation(
    extrinsic_matrix: np.ndarray,
    roll_deg: float = 0.0,
    pitch_deg: float = 0.0,
    yaw_deg: float = 0.0,
    trans_perturb_m: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """
    Simulates physical mounting vibration or mechanical misalignment drift (+/- 1.5 deg).
    Applies small rotational and translational perturbations to the 4x4 extrinsic matrix T_cam_world.
    """
    rx = math.radians(pitch_deg)
    ry = math.radians(yaw_deg)
    rz = math.radians(roll_deg)

    # Small rotation matrix (Euler XYZ)
    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(rx), -math.sin(rx)],
        [0, math.sin(rx), math.cos(rx)],
    ], dtype=np.float64)

    Ry = np.array([
        [math.cos(ry), 0, math.sin(ry)],
        [0, 1, 0],
        [-math.sin(ry), 0, math.cos(ry)],
    ], dtype=np.float64)

    Rz = np.array([
        [math.cos(rz), -math.sin(rz), 0],
        [math.sin(rz), math.cos(rz), 0],
        [0, 0, 1],
    ], dtype=np.float64)

    R_delta = Rz @ Ry @ Rx
    T_delta = np.eye(4, dtype=np.float64)
    T_delta[:3, :3] = R_delta
    T_delta[:3, 3] = np.array(trans_perturb_m, dtype=np.float64)

    return T_delta @ extrinsic_matrix


def apply_sync_jitter(
    frames: Dict[str, np.ndarray],
    jitter_ms: Dict[str, float],
) -> Dict[str, np.ndarray]:
    """
    Tags frames with simulated inter-camera hardware clock synchronization jitter.
    Returns dictionary with simulated metadata timestamp offsets.
    """
    # For synthetic frames, temporal jitter is represented as latency tags
    out_frames = {}
    for cam_name, frame in frames.items():
        out_frames[cam_name] = {
            "image": frame,
            "jitter_ms": jitter_ms.get(cam_name, 0.0),
        }
    return out_frames


def simulate_missing_cameras(
    camera_data: Dict[str, any],
    dropped_cameras: List[str],
) -> Dict[str, any]:
    """
    Simulates hardware bus disconnect or frame drop by removing specified camera feeds.
    """
    return {k: v for k, v in camera_data.items() if k not in dropped_cameras}

