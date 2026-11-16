"""Shared camera-calibration policy for COLMAP-backed reconstruction paths."""

from __future__ import annotations

from typing import Any, Mapping


def resolve_colmap_camera_mode(photo_set_report: Mapping[str, Any] | None) -> dict[str, str]:
    """Choose shared or per-image COLMAP intrinsics from the upload audit.

    Shared calibration is only selected when the upload audit finds no mixed
    camera/lens metadata and every photograph has matching pixel dimensions.
    Different image dimensions cannot safely share one COLMAP camera record.
    """
    report = photo_set_report or {}
    camera_report = report.get("camera") or {}
    mixed_lenses = bool(camera_report.get("mixed_lenses_detected", False))
    warnings = list(camera_report.get("warnings") or [])

    dimensions = (report.get("image_dimensions") or {}).get("dimensions") or []
    dimension_pairs = {
        (int(item["width"]), int(item["height"]))
        for item in dimensions
        if item.get("width") and item.get("height")
    }
    mixed_resolutions = len(dimension_pairs) > 1

    per_image_calibration = mixed_lenses or mixed_resolutions
    reasons: list[str] = []
    if mixed_lenses:
        reasons.extend(str(item) for item in warnings)
    if mixed_resolutions:
        reasons.append(
            "Different image dimensions were detected; shared camera calibration "
            "requires matching image dimensions."
        )

    if per_image_calibration:
        return {
            "single_camera": "0",
            "single_camera_per_image": "1",
            "strategy": "per_image_camera",
            "reason": "; ".join(reasons) or "Independent camera calibration is required.",
        }

    return {
        "single_camera": "1",
        "single_camera_per_image": "0",
        "strategy": "shared_camera",
        "reason": "No mixed camera/lens metadata or image dimensions were detected.",
    }


def pycolmap_camera_mode(pycolmap_module: Any, strategy: str) -> Any:
    """Return the PyCOLMAP enum matching CLEAR360's camera-calibration strategy."""
    enum_name = "SINGLE" if strategy == "shared_camera" else "PER_IMAGE"
    camera_mode = getattr(pycolmap_module, "CameraMode", None)
    value = getattr(camera_mode, enum_name, None) if camera_mode is not None else None
    if value is None:
        raise RuntimeError(
            f"The installed PyCOLMAP build does not expose CameraMode.{enum_name}. "
            "Upgrade PyCOLMAP to a version that supports explicit camera modes."
        )
    return value
