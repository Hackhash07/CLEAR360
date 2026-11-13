"""
photo_reconstruction/exif_reader.py — EXIF Metadata Extraction and Camera Intrinsics Initialization
===================================================================================================

Extracts smartphone camera hardware specifications, focal lengths, and exposure parameters
from image EXIF tags. Initializes camera intrinsic matrices and flags mixed-lens capture sessions.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from PIL import Image, ExifTags
import numpy as np


class ExifReader:
    """Extracts and parses camera metadata from photograph EXIF headers."""

    DEFAULT_SENSOR_WIDTH_MM = 6.17  # Standard 1/2.3" to 1/2.55" smartphone sensor width
    FULL_FRAME_35MM_WIDTH = 36.0

    @classmethod
    def read_exif(cls, image_path: Path) -> Dict[str, Any]:
        """
        Extract raw EXIF dictionary with human-readable keys.
        """
        p = Path(image_path)
        exif_data = {}
        if not p.is_file():
            return exif_data

        try:
            with Image.open(p) as img:
                raw_exif = img._getexif()
                if raw_exif:
                    for tag_id, value in raw_exif.items():
                        tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                        # Convert tuple/rational values to float/string
                        if hasattr(value, "numerator") and hasattr(value, "denominator"):
                            exif_data[tag_name] = float(value)
                        elif isinstance(value, (bytes, bytearray)):
                            exif_data[tag_name] = str(value)
                        else:
                            exif_data[tag_name] = value
        except Exception:
            pass

        return exif_data

    @classmethod
    def get_camera_info(cls, image_path: Path, width: int, height: int) -> Dict[str, Any]:
        """
        Extract structured camera model and estimated intrinsics.

        Returns
        -------
        Dict with keys:
          - make, model, lens_model
          - focal_length_mm, focal_length_35mm
          - estimated_fx, estimated_fy, cx, cy
          - has_exif (bool)
          - intrinsics_source ('exif_35mm', 'exif_mm', 'default_fallback')
        """
        raw_exif = cls.read_exif(image_path)

        make = raw_exif.get("Make", "Unknown")
        model = raw_exif.get("Model", "Smartphone")
        lens = raw_exif.get("LensModel", raw_exif.get("LensInfo", "Standard Lens"))

        cx = width / 2.0
        cy = height / 2.0
        max_dim = max(width, height)

        def _to_float(val):
            if val is None:
                return None
            if isinstance(val, (tuple, list)) and len(val) == 2:
                return float(val[0]) / float(val[1]) if val[1] != 0 else float(val[0])
            if hasattr(val, "numerator") and hasattr(val, "denominator"):
                return float(val.numerator) / float(val.denominator) if val.denominator != 0 else float(val.numerator)
            try:
                return float(val)
            except (ValueError, TypeError):
                return None

        f_35mm = _to_float(raw_exif.get("FocalLengthIn35mmFilm"))
        f_mm = _to_float(raw_exif.get("FocalLength"))

        # 1. Best: 35mm equivalent focal length
        if f_35mm and f_35mm > 0:
            fx = f_35mm * max_dim / cls.FULL_FRAME_35MM_WIDTH
            fy = fx
            source = "exif_35mm"
        # 2. Physical focal length + estimated smartphone sensor size
        elif f_mm and f_mm > 0:
            fx = f_mm * max_dim / cls.DEFAULT_SENSOR_WIDTH_MM
            fy = fx
            source = "exif_mm"
        # 3. Default fallback for standard smartphone main camera (~65° HFOV, fx ~ 1.2 * max_dim)
        else:
            fx = 1.20 * max_dim
            fy = fx
            source = "default_fallback"

        return {
            "make": str(make).strip(),
            "model": str(model).strip(),
            "lens_model": str(lens).strip(),
            "focal_length_mm": f_mm,
            "focal_length_35mm": f_35mm,
            "estimated_fx": float(fx),
            "estimated_fy": float(fy),
            "cx": float(cx),
            "cy": float(cy),
            "has_exif": bool(raw_exif),
            "intrinsics_source": source,
            "iso": raw_exif.get("ISOSpeedRatings"),
            "exposure_time": _to_float(raw_exif.get("ExposureTime")),
            "f_number": _to_float(raw_exif.get("FNumber")),
        }

    @classmethod
    def check_mixed_lenses(cls, camera_infos: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
        """
        Analyze a collection of images for mixed camera lenses (e.g. ultra-wide vs telephoto).

        Returns
        -------
        is_mixed : bool
        warnings : List[str]
        """
        warnings = []
        if len(camera_infos) < 2:
            return False, warnings

        fx_values = [info["estimated_fx"] for info in camera_infos]
        fx_min, fx_max = min(fx_values), max(fx_values)

        # If focal length varies by more than 20%, flag mixed lenses
        ratio = fx_max / (fx_min + 1e-6)
        if ratio > 1.25:
            warnings.append(
                f"Mixed camera lenses detected across session (Focal length variation: {ratio:.1f}x). "
                f"For optimal 3D photogrammetry, capture all images using the primary 1x lens."
            )
            return True, warnings

        # Check for multiple models
        models = set(info["model"] for info in camera_infos if info["model"] != "Smartphone")
        if len(models) > 1:
            warnings.append(f"Multiple camera models detected: {list(models)}. Per-camera bundle adjustment recommended.")
            return True, warnings

        return False, warnings
