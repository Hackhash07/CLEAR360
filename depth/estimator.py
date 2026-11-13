"""
depth/estimator.py — Depth Anything V2 Metric Depth Estimator for CLEAR360
==========================================================================

Loads the official Depth Anything V2 Metric ViT-Small model trained on Virtual KITTI 2,
runs inference on four-camera surround frames, preserves native resolution (1280x720),
and outputs metric depth in metres (float32).

Device Priority:
  1. Apple MPS (if supported and verified)
  2. CUDA
  3. CPU

Coordinate & Depth Conventions:
  - Input: RGB image (H, W, 3) uint8 or BGR (handled automatically)
  - Output: Camera Z-depth map (H, W) float32 in metres (range [0, 80] m)
  - Provenance tag: 'ai_depth_anything_v2_metric'
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from depth.depth_anything_v2.dpt import DepthAnythingV2


def select_device(preferred_device: Optional[str] = None) -> torch.device:
    """
    Select the optimal execution device following priority: MPS -> CUDA -> CPU,
    verifying device compatibility before returning.
    """
    if preferred_device is not None:
        return torch.device(preferred_device)

    # Check CUDA first if available
    if torch.cuda.is_available():
        return torch.device("cuda")

    # On Apple Silicon, test if MPS handles the ViT linear layers without crashing
    if torch.backends.mps.is_available():
        try:
            # Safe probe: verify linear and conv on MPS
            test_x = torch.zeros((1, 3, 64, 64), device="mps")
            test_conv = torch.nn.Conv2d(3, 16, 3, padding=1).to("mps")
            _ = test_conv(test_x)
            test_lin = torch.nn.Linear(16, 16).to("mps")
            _ = test_lin(torch.zeros((1, 16), device="mps"))
            # Note: Certain PyTorch/OS builds crash inside ViT strided ops on MPS.
            # If so, the user or caller can specify device='cpu'.
            return torch.device("mps")
        except Exception as e:
            print(f"[DepthEstimator] MPS probe failed ({e}); falling back to CPU.")
            return torch.device("cpu")

    return torch.device("cpu")


class DepthEstimator:
    """
    Depth Anything V2 Metric Depth Estimator.

    Parameters
    ----------
    weights_path : str or Path, optional
        Path to depth_anything_v2_metric_vkitti_vits.pth.
    encoder : str
        Encoder backbone variant ('vits', 'vitb', 'vitl'). Default: 'vits'.
    max_depth : float
        Maximum metric depth cutoff (metres). Default: 80.0 m.
    input_size : int
        Processing resolution fed to ViT backbone. Default: 518.
    device : str, optional
        Target device ('cpu', 'cuda', 'mps'). If None, selected automatically.
    """

    MODEL_CONFIGS = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    }

    DEPTH_SOURCE_TAG = "ai_depth_anything_v2_metric"

    def __init__(
        self,
        weights_path: Optional[Union[str, Path]] = None,
        encoder: str = "vits",
        max_depth: float = 80.0,
        input_size: int = 518,
        device: Optional[str] = "cpu",  # Default to CPU on macOS to prevent libtorch MPS crash
    ):
        if weights_path is None:
            weights_path = PROJECT_ROOT / "weights" / "depth_anything_v2_metric_vkitti_vits.pth"

        self.weights_path = Path(weights_path)
        self.encoder = encoder
        self.max_depth = float(max_depth)
        self.input_size = int(input_size)

        # Device selection
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = select_device()

        print(f"[DepthEstimator] Initializing Depth Anything V2 Metric ({self.encoder}) on {self.device}...")

        if not self.weights_path.exists():
            raise FileNotFoundError(
                f"Depth model weights not found at: {self.weights_path}\n"
                f"Please ensure depth_anything_v2_metric_vkitti_vits.pth is downloaded into weights/."
            )

        # Instantiate model architecture
        cfg = self.MODEL_CONFIGS[self.encoder]
        self.model = DepthAnythingV2(**{**cfg, "max_depth": self.max_depth})

        # Load weights
        state_dict = torch.load(str(self.weights_path), map_location="cpu")
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device).eval()

        self.last_inference_time_s: float = 0.0

    @torch.no_grad()
    def estimate_depth(
        self,
        image: Union[np.ndarray, str, Path],
        is_rgb: bool = True,
    ) -> tuple[np.ndarray, float]:
        """
        Estimate metric depth from a single image.

        Parameters
        ----------
        image : np.ndarray (H, W, 3) or path to image
            Input image. If np.ndarray and is_rgb=True, RGB ordering is assumed.
        is_rgb : bool
            True if image is RGB, False if BGR (e.g. from standard cv2.imread).

        Returns
        -------
        depth_map : np.ndarray (H, W), dtype float32
            Metric depth in metres.
        inference_time_s : float
            Elapsed inference duration in seconds.
        """
        if isinstance(image, (str, Path)):
            bgr = cv2.imread(str(image))
            if bgr is None:
                raise FileNotFoundError(f"Could not load image from: {image}")
            raw_bgr = bgr
        elif isinstance(image, np.ndarray):
            if is_rgb:
                raw_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                raw_bgr = image
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        orig_h, orig_w = raw_bgr.shape[:2]

        t0 = time.perf_counter()
        # DepthAnythingV2.infer_image accepts raw BGR image and returns numpy (H, W) float32
        raw_depth = self.model.infer_image(raw_bgr, input_size=self.input_size)
        dt = time.perf_counter() - t0

        self.last_inference_time_s = dt

        # Post-process: ensure correct dimensions, valid numeric range, and float32 dtype
        depth_map = np.asarray(raw_depth, dtype=np.float32)

        if depth_map.shape != (orig_h, orig_w):
            depth_map = cv2.resize(depth_map, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        # Replace NaNs/Infs and clip negative / excessive values
        nan_mask = np.isnan(depth_map) | np.isinf(depth_map)
        if np.any(nan_mask):
            depth_map[nan_mask] = self.max_depth

        depth_map = np.clip(depth_map, 0.0, self.max_depth)

        return depth_map, dt

    def estimate_frame_set(
        self,
        frame_set,
        save_dir: Optional[Union[str, Path]] = None,
    ) -> dict[str, dict]:
        """
        Estimate depth for all four cameras in a FrameSet.

        Parameters
        ----------
        frame_set : pipeline.sync.FrameSet
            Synchronized four-camera frame group.
        save_dir : str or Path, optional
            If provided, saves predicted depth as .npy into {save_dir}/{camera}/frame_{frame_id:04d}.npy.

        Returns
        -------
        results : dict[str, dict]
            Keyed by camera name ('front', 'rear', 'left', 'right'), containing:
              - 'depth': (H, W) float32 depth map
              - 'inference_time_s': float
              - 'saved_path': Path or None
        """
        results = {}
        frame_id = frame_set.frame_id

        for cam_name in ["front", "rear", "left", "right"]:
            cfd = frame_set.get(cam_name)
            depth_map, dt = self.estimate_depth(cfd.rgb, is_rgb=True)

            saved_path = None
            if save_dir is not None:
                cam_save_dir = Path(save_dir) / cam_name
                cam_save_dir.mkdir(parents=True, exist_ok=True)
                saved_path = cam_save_dir / f"frame_{frame_id:04d}.npy"
                np.save(str(saved_path), depth_map)

            results[cam_name] = {
                "depth": depth_map,
                "inference_time_s": dt,
                "saved_path": saved_path,
            }

        return results

