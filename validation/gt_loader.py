"""
CLEAR360 Ground-Truth Data Loader
Loads Blender-exported EXR depth and position maps for validation.

Coordinate convention: World frame X=right, Y=forward, Z=up.
Depth convention: camera Z-depth in metres (not Euclidean).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def load_depth_exr(path: Path) -> np.ndarray:
    """Load a single-channel depth map from an OpenEXR file.

    Parameters
    ----------
    path : Path
        Path to the .exr file containing depth.

    Returns
    -------
    np.ndarray
        Array of shape (H, W) with dtype float32 containing camera Z-depth in metres.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Depth EXR not found at {path}")

    try:
        import Imath  # type: ignore
        import OpenEXR  # type: ignore

        exr = OpenEXR.InputFile(str(path))
        header = exr.header()
        dw = header["dataWindow"]
        w = dw.max.x - dw.min.x + 1
        h = dw.max.y - dw.min.y + 1

        channels = header["channels"].keys()
        pt = Imath.PixelType(Imath.PixelType.FLOAT)

        # Blender RenderLayers standard depth channel names
        if "Z" in channels:
            chan_name = "Z"
        elif "R" in channels:
            chan_name = "R"
        elif "Y" in channels:
            chan_name = "Y"
        elif "V" in channels:
            chan_name = "V"
        else:
            chan_name = next(iter(channels))

        raw = exr.channel(chan_name, pt)
        depth = np.frombuffer(raw, dtype=np.float32).reshape(h, w).copy()
        return depth
    except ImportError:
        # Fallback to OpenCV if OpenEXR bindings are not available
        import cv2  # type: ignore

        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"Failed to read EXR with OpenCV: {path}")
        if img.ndim == 3:
            img = img[:, :, 0]
        return img.astype(np.float32)


def load_position_exr(path: Path) -> np.ndarray:
    """Load a 3-channel world-space position map from an OpenEXR file.

    Parameters
    ----------
    path : Path
        Path to the .exr file containing position pass.

    Returns
    -------
    np.ndarray
        Array of shape (H, W, 3) with dtype float32 containing world XYZ coordinates in metres.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Position EXR not found at {path}")

    try:
        import Imath  # type: ignore
        import OpenEXR  # type: ignore

        exr = OpenEXR.InputFile(str(path))
        header = exr.header()
        dw = header["dataWindow"]
        w = dw.max.x - dw.min.x + 1
        h = dw.max.y - dw.min.y + 1

        channels = set(header["channels"].keys())
        pt = Imath.PixelType(Imath.PixelType.FLOAT)

        # Position pass channels may be named X, Y, Z or R, G, B
        if {"X", "Y", "Z"}.issubset(channels):
            chan_x, chan_y, chan_z = "X", "Y", "Z"
        elif {"R", "G", "B"}.issubset(channels):
            chan_x, chan_y, chan_z = "R", "G", "B"
        else:
            sorted_chans = sorted(list(channels))
            if len(sorted_chans) >= 3:
                chan_x, chan_y, chan_z = sorted_chans[:3]
            else:
                raise ValueError(
                    f"Position EXR {path} does not contain at least 3 channels: {channels}"
                )

        raw_x = np.frombuffer(exr.channel(chan_x, pt), dtype=np.float32).reshape(h, w)
        raw_y = np.frombuffer(exr.channel(chan_y, pt), dtype=np.float32).reshape(h, w)
        raw_z = np.frombuffer(exr.channel(chan_z, pt), dtype=np.float32).reshape(h, w)

        pos = np.stack([raw_x, raw_y, raw_z], axis=-1).copy()
        return pos
    except ImportError:
        import cv2  # type: ignore

        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"Failed to read EXR with OpenCV: {path}")
        if img.ndim == 2:
            img = np.repeat(img[:, :, np.newaxis], 3, axis=-1)
        elif img.shape[2] > 3:
            img = img[:, :, :3]
        # cv2 loads multi-channel EXR in BGR order
        img_rgb = img[:, :, ::-1].copy()
        return img_rgb.astype(np.float32)


def load_semantic_png(path: Path) -> np.ndarray:
    """Load an 8-bit single-channel semantic class mask from a PNG file.

    Parameters
    ----------
    path : Path
        Path to the .png semantic mask.

    Returns
    -------
    np.ndarray
        Array of shape (H, W) with dtype uint8 containing pass_index class labels.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Semantic PNG not found at {path}")

    with Image.open(path) as img:
        arr = np.array(img)
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        return arr.astype(np.uint8)


def depth_z_to_euclidean(
    depth_z: np.ndarray, fx: float, fy: float, cx: float, cy: float
) -> np.ndarray:
    """Convert camera-Z depth map to Euclidean distance map.

    Formula:
        d_e[u, v] = d_z[u, v] * sqrt(1 + ((u - cx) / fx)^2 + ((v - cy) / fy)^2)

    Context:
        - Camera Z-depth (d_z) is the orthogonal projection along the optical axis (+z_cam).
          This is used for pinhole inverse projection: (x, y, z) = ((u-cx)*z/fx, (v-cy)*z/fy, z).
        - Euclidean distance (d_e) is the true line-of-sight distance from the camera center
          to the 3D surface point: sqrt(x^2 + y^2 + z^2).

    Parameters
    ----------
    depth_z : np.ndarray
        Input depth map of shape (H, W) with camera Z distances in metres.
    fx, fy : float
        Focal lengths along x and y in pixel units.
    cx, cy : float
        Principal point coordinates in pixel units.

    Returns
    -------
    np.ndarray
        Euclidean range map of shape (H, W) with dtype float32.
    """
    h, w = depth_z.shape
    u = np.arange(w, dtype=np.float32)
    v = np.arange(h, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)

    scale = np.sqrt(1.0 + ((uu - cx) / fx) ** 2 + ((vv - cy) / fy) ** 2)
    return (depth_z * scale).astype(np.float32)


def verify_depth_vs_position(
    depth: np.ndarray,
    position_map: np.ndarray,
    T_cam_world: np.ndarray,
    intrinsics_dict: dict[str, Any],
    sample_stride: int = 16,
    threshold_m: float = 0.01,
) -> dict[str, Any]:
    """Verify geometric consistency between depth map and world position map.

    For a uniform grid sample of valid pixels, this unprojects the camera Z-depth
    into world coordinates using camera intrinsics and the camera-to-world transform,
    then computes point-to-point discrepancy against the position map.

    Parameters
    ----------
    depth : np.ndarray
        (H, W) camera Z-depth in metres.
    position_map : np.ndarray
        (H, W, 3) world XYZ positions in metres.
    T_cam_world : np.ndarray
        (4, 4) world-to-camera matrix (OpenCV convention).
    intrinsics_dict : dict
        Dictionary containing 'fx', 'fy', 'cx', 'cy' (or nested under 'K').
    sample_stride : int, optional
        Subsampling stride across rows and columns (default 16).
    threshold_m : float, optional
        Maximum allowable mean error in metres to consider the check passed (default 0.01 m).

    Returns
    -------
    dict
        Dictionary containing:
        - 'mean_error_m': float, mean Euclidean error in metres
        - 'max_error_m': float, maximum Euclidean error in metres
        - 'passed': bool, True if mean_error_m <= threshold_m
        - 'num_samples': int, count of evaluated pixels
    """
    if "K" in intrinsics_dict and isinstance(intrinsics_dict["K"], dict):
        k_dict = intrinsics_dict["K"]
        fx = float(k_dict["fx"])
        fy = float(k_dict["fy"])
        cx = float(k_dict["cx"])
        cy = float(k_dict["cy"])
    else:
        fx = float(intrinsics_dict["fx"])
        fy = float(intrinsics_dict["fy"])
        cx = float(intrinsics_dict["cx"])
        cy = float(intrinsics_dict["cy"])

    # Compute Camera-to-World matrix: T_c2w = inv(T_cam_world)
    T_c2w = np.linalg.inv(T_cam_world)
    R_c2w = T_c2w[:3, :3]
    t_c2w = T_c2w[:3, 3]

    h, w = depth.shape
    v_indices = np.arange(0, h, sample_stride)
    u_indices = np.arange(0, w, sample_stride)
    uu, vv = np.meshgrid(u_indices, v_indices)
    uu = uu.flatten()
    vv = vv.flatten()

    sampled_depths = depth[vv, uu]
    sampled_positions = position_map[vv, uu]  # (N, 3)

    # Filter out invalid, background, or zero pixels
    valid_mask = (
        np.isfinite(sampled_depths)
        & (sampled_depths > 0.05)
        & (sampled_depths < 100.0)
        & np.all(np.isfinite(sampled_positions), axis=1)
        & (np.linalg.norm(sampled_positions, axis=1) > 0.01)
    )

    if not np.any(valid_mask):
        return {
            "mean_error_m": 0.0,
            "max_error_m": 0.0,
            "passed": True,
            "num_samples": 0,
        }

    uu_valid = uu[valid_mask]
    vv_valid = vv[valid_mask]
    z_valid = sampled_depths[valid_mask]
    gt_valid = sampled_positions[valid_mask]

    # Unproject to camera frame
    x_c = (uu_valid - cx) * z_valid / fx
    y_c = (vv_valid - cy) * z_valid / fy
    pts_cam = np.stack([x_c, y_c, z_valid], axis=1)  # (M, 3)

    # Transform to world frame: P_w = pts_cam @ R_c2w.T + t_c2w
    unproj_world = pts_cam @ R_c2w.T + t_c2w

    # Euclidean error
    errors = np.linalg.norm(unproj_world - gt_valid, axis=1)
    mean_err = float(np.mean(errors))
    max_err = float(np.max(errors))
    passed = bool(mean_err <= threshold_m)

    return {
        "mean_error_m": mean_err,
        "max_error_m": max_err,
        "passed": passed,
        "num_samples": int(len(errors)),
    }
