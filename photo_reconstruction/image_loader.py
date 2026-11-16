"""
photo_reconstruction/image_loader.py — Image Loading and Validation for Phone Photogrammetry
=============================================================================================

Handles loading, format conversion, and integrity validation for smartphone photographs.
Supports JPG, JPEG, PNG, and HEIC (where supported by system libraries).
Preserves original source photographs without destructive downsampling.
"""

from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import os
import cv2
import numpy as np
from PIL import Image


def ensure_image_format_support(file_path: str | Path) -> None:
    """Register optional Pillow decoders required by the source image format."""
    suffix = Path(file_path).suffix.lower()
    if suffix in {".heic", ".heif"}:
        try:
            from pillow_heif import register_heif_opener
        except ImportError as exc:
            raise RuntimeError(
                "HEIC/HEIF decoding requires the pillow-heif package. "
                "Install the CLEAR360 dependencies with: python -m pip install -e ."
            ) from exc
        register_heif_opener(thumbnails=False)


class ImageLoader:
    """Validates and loads smartphone photographs for 3D reconstruction."""

    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".tiff", ".bmp"}

    MIN_WIDTH = 320
    MIN_HEIGHT = 240

    @classmethod
    def is_supported(cls, file_path: Path) -> bool:
        """Check if file extension is supported."""
        return file_path.suffix.lower() in cls.SUPPORTED_EXTENSIONS

    @classmethod
    def load_image(cls, file_path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Load an image file and return (RGB numpy array, metadata dict).

        Parameters
        ----------
        file_path : Path
            Path to the image file.

        Returns
        -------
        rgb_img : np.ndarray (H, W, 3) uint8
        meta : Dict with filename, width, height, file_size_bytes, aspect_ratio
        """
        p = Path(file_path)
        if not p.is_file():
            raise FileNotFoundError(f"Image file does not exist: {p}")

        if not cls.is_supported(p):
            raise ValueError(f"Unsupported image format: {p.suffix}. Supported: {cls.SUPPORTED_EXTENSIONS}")

        file_size = os.path.getsize(p)
        ensure_image_format_support(p)

        # Use PIL for robust image decoding
        try:
            with Image.open(p) as img_pil:
                img_pil = img_pil.convert("RGB")
                rgb_arr = np.array(img_pil, dtype=np.uint8)
        except Exception as e:
            # Fallback to OpenCV
            bgr = cv2.imread(str(p))
            if bgr is None:
                raise IOError(f"Failed to decode image: {p}. Error: {e}")
            rgb_arr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        H, W, C = rgb_arr.shape
        if W < cls.MIN_WIDTH or H < cls.MIN_HEIGHT:
            raise ValueError(f"Image resolution {W}x{H} is below minimum requirement {cls.MIN_WIDTH}x{cls.MIN_HEIGHT}")

        meta = {
            "file_name": p.name,
            "file_path": str(p.resolve()),
            "width": W,
            "height": H,
            "channels": C,
            "file_size_bytes": file_size,
            "aspect_ratio": float(W) / float(H),
        }

        return rgb_arr, meta

    @classmethod
    def load_image_batch(cls, file_paths: List[Path]) -> Tuple[List[np.ndarray], List[Dict[str, Any]], List[str]]:
        """
        Load multiple images, returning loaded images, metadata, and error messages for any failures.
        """
        images = []
        metas = []
        errors = []

        for p in file_paths:
            try:
                img, meta = cls.load_image(p)
                images.append(img)
                metas.append(meta)
            except Exception as e:
                errors.append(f"{p.name}: {str(e)}")

        return images, metas, errors

