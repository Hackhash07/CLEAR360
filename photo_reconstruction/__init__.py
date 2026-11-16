"""Smartphone photogrammetry and 3D reconstruction package.

Public classes are lazily imported so lightweight utilities (for example,
OpenMVS command builders and reconstruction QA) do not import the full desktop
application, Open3D stack or ML runtime as a side effect.
"""

from __future__ import annotations

import importlib
import os
from typing import Any

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

_LAZY_EXPORTS = {
    "ImageLoader": ("photo_reconstruction.image_loader", "ImageLoader"),
    "ExifReader": ("photo_reconstruction.exif_reader", "ExifReader"),
    "ImageQualityAssessor": (
        "photo_reconstruction.image_quality",
        "ImageQualityAssessor",
    ),
    "SfMEngine": ("photo_reconstruction.sfm_engine", "SfMEngine"),
    "DenseReconstructor": (
        "photo_reconstruction.dense_reconstruction",
        "DenseReconstructor",
    ),
    "MeshBuilder": ("photo_reconstruction.mesh_builder", "MeshBuilder"),
    "ScaleCalibrator": (
        "photo_reconstruction.scale_calibrator",
        "ScaleCalibrator",
    ),
    "QualityReportGenerator": (
        "photo_reconstruction.quality_report",
        "QualityReportGenerator",
    ),
    "SessionManager": (
        "photo_reconstruction.session_manager",
        "SessionManager",
    ),
    "PhoneReconstructionPipeline": (
        "photo_reconstruction.pipeline",
        "PhoneReconstructionPipeline",
    ),
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    """Load a public package symbol only when a caller requests it."""
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from exc

    module = importlib.import_module(module_name)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
