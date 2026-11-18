"""Cross-image preflight diagnostics for uploaded photogrammetry photo sets."""

from __future__ import annotations

import hashlib
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from photo_reconstruction.exif_reader import ExifReader


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class PhotoSetQualityAuditor:
    """Audit a complete upload set before expensive camera reconstruction.

    Individual sharpness/exposure scores are not treated as proof of overlap.
    True pairwise overlap is reported as unmeasured until feature matching/SfM.
    """

    @classmethod
    def audit_paths(cls, image_paths: Sequence[str | Path]) -> dict[str, Any]:
        """Load, assess and audit raw image paths for non-GUI/CLI callers."""
        from photo_reconstruction.image_loader import ImageLoader
        from photo_reconstruction.image_quality import ImageQualityAssessor

        records: list[dict[str, Any]] = []
        for raw_path in image_paths:
            path = Path(raw_path).expanduser()
            try:
                rgb, meta = ImageLoader.load_image(path)
                exif = ExifReader.get_camera_info(path, meta["width"], meta["height"])
                quality = ImageQualityAssessor.evaluate(rgb)
                records.append({
                    "path": str(path.resolve()),
                    "meta": meta,
                    "exif": exif,
                    "quality": quality,
                })
            except Exception as exc:
                records.append({
                    "path": str(path.resolve()),
                    "error": str(exc),
                })
        return cls.audit(records)

    @staticmethod
    def merge_source_camera_metadata(
        actual_report: Mapping[str, Any],
        source_report: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        """Preserve original EXIF camera metadata after format conversion.

        The worker may convert HEIC/PNG images to JPEG for COLMAP. Image pixels
        remain usable, but that conversion can remove EXIF. Re-audit the exact
        working files for readability/quality while retaining stronger camera
        metadata already extracted from the original uploaded files.
        """
        merged = dict(actual_report)
        if not source_report:
            return merged

        actual_camera = dict(actual_report.get("camera") or {})
        source_camera = dict(source_report.get("camera") or {})
        if source_camera:
            camera = dict(actual_camera)
            camera.update(source_camera)
            camera["mixed_lenses_detected"] = bool(
                actual_camera.get("mixed_lenses_detected", False)
                or source_camera.get("mixed_lenses_detected", False)
            )
            camera["warnings"] = list(dict.fromkeys(
                list(actual_camera.get("warnings") or [])
                + list(source_camera.get("warnings") or [])
            ))
            camera["metadata_available_count"] = max(
                int(actual_camera.get("metadata_available_count", 0) or 0),
                int(source_camera.get("metadata_available_count", 0) or 0),
            )
            camera["metadata_coverage_percentage"] = max(
                float(actual_camera.get("metadata_coverage_percentage", 0.0) or 0.0),
                float(source_camera.get("metadata_coverage_percentage", 0.0) or 0.0),
            )
            merged["camera"] = camera

        merged["warnings"] = list(dict.fromkeys(
            list(actual_report.get("warnings") or [])
            + list(source_report.get("warnings") or [])
        ))
        merged["source_camera_metadata_preserved"] = bool(
            int(source_camera.get("metadata_available_count", 0) or 0) > 0
        )
        return merged

    @classmethod
    def audit(cls, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        warnings: list[str] = []
        invalid_images: list[dict[str, str]] = []
        valid_records: list[Mapping[str, Any]] = []
        hashes: dict[str, list[str]] = {}

        for index, record in enumerate(records):
            meta = record.get("meta") or {}
            raw_path = record.get("path") or meta.get("file_path")
            display_name = str(
                meta.get("file_name")
                or (Path(str(raw_path)).name if raw_path else f"image_{index + 1}")
            )
            error = record.get("error")
            if error:
                invalid_images.append({"image": display_name, "reason": str(error)})
                continue

            if not meta or not record.get("quality"):
                invalid_images.append({
                    "image": display_name,
                    "reason": "Image metadata or quality assessment is missing.",
                })
                continue

            if raw_path:
                path = Path(str(raw_path)).expanduser()
                if not path.is_file():
                    invalid_images.append({
                        "image": display_name,
                        "reason": "File is missing or is not readable.",
                    })
                    continue
                try:
                    digest = _sha256(path)
                    hashes.setdefault(digest, []).append(display_name)
                except OSError as exc:
                    invalid_images.append({"image": display_name, "reason": str(exc)})
                    continue

            valid_records.append(record)

        duplicate_groups = [
            sorted(names) for names in hashes.values() if len(names) > 1
        ]
        duplicate_images = sum(len(names) - 1 for names in duplicate_groups)
        valid_count = len(valid_records)
        unique_image_count = max(0, valid_count - duplicate_images)

        if invalid_images:
            warnings.append(
                f"{len(invalid_images)} uploaded image(s) could not be validated."
            )
        if duplicate_groups:
            warnings.append(
                f"{duplicate_images} exact duplicate image(s) were detected. "
                "Remove duplicate copies or capture additional viewpoints."
            )

        quality_records = [record["quality"] for record in valid_records]
        scores = [
            float(item["quality_score"])
            for item in quality_records
            if item.get("quality_score") is not None
        ]
        blurry_images = [
            str((record.get("meta") or {}).get("file_name", "image"))
            for record in valid_records
            if (record.get("quality") or {}).get("is_blurry")
        ]
        rejected_images = [
            str((record.get("meta") or {}).get("file_name", "image"))
            for record in valid_records
            if str((record.get("quality") or {}).get("status", "")).upper() == "REJECT"
        ]
        warning_images = [
            str((record.get("meta") or {}).get("file_name", "image"))
            for record in valid_records
            if str((record.get("quality") or {}).get("status", "")).upper() == "WARNING"
        ]
        low_feature_images = [
            str((record.get("meta") or {}).get("file_name", "image"))
            for record in valid_records
            if int((record.get("quality") or {}).get("keypoint_count", 999999)) < 300
        ]

        if rejected_images:
            warnings.append(
                f"{len(rejected_images)} image(s) received a REJECT quality rating. "
                "Consider retaking or removing them."
            )
        if blurry_images and not rejected_images:
            warnings.append(
                f"{len(blurry_images)} image(s) appear blurry. Inspect them before reconstruction."
            )
        if warning_images:
            warnings.append(
                f"{len(warning_images)} image(s) received a WARNING quality rating. "
                "Inspect exposure, sharpness and feature density."
            )
        if low_feature_images:
            warnings.append(
                f"{len(low_feature_images)} image(s) have low feature density; "
                "plain surfaces can make camera matching unreliable."
            )

        camera_infos = [
            record.get("exif") or {}
            for record in valid_records
            if record.get("exif")
        ]
        mixed_lenses, camera_warnings = ExifReader.check_mixed_lenses(camera_infos)
        warnings.extend(camera_warnings)

        exif_count = sum(bool(info.get("has_exif")) for info in camera_infos)
        image_dimensions = []
        for record in valid_records:
            meta = record.get("meta") or {}
            width, height = meta.get("width"), meta.get("height")
            if width and height:
                image_dimensions.append((int(width), int(height)))

        aspect_ratios = [
            width / height for width, height in image_dimensions if height > 0
        ]
        aspect_ratio_spread = (
            (max(aspect_ratios) - min(aspect_ratios))
            if len(aspect_ratios) >= 2 else 0.0
        )
        if aspect_ratio_spread > 0.25:
            warnings.append(
                "The upload contains substantially different image aspect ratios. "
                "Check for crops, rotated images or mixed camera modes."
            )

        if valid_count < 2 or unique_image_count < 2:
            status = "blocked"
            warnings.append(
                "At least two distinct, readable photographs are required for reconstruction."
            )
        elif warnings:
            status = "review"
        else:
            status = "ready"

        quality_mean = round(mean(scores), 1) if scores else None
        return {
            "schema_version": 1,
            "status": status,
            "can_reconstruct": status != "blocked",
            "photo_count": len(records),
            "valid_photo_count": valid_count,
            "unique_photo_count": unique_image_count,
            "invalid_images": invalid_images,
            "duplicate_groups": duplicate_groups,
            "duplicate_image_count": duplicate_images,
            "image_quality": {
                "mean_score": quality_mean,
                "blurry_count": len(blurry_images),
                "rejected_count": len(rejected_images),
                "warning_count": len(warning_images),
                "low_feature_count": len(low_feature_images),
                "rejected_images": rejected_images,
                "warning_images": warning_images,
                "blurry_images": blurry_images,
                "low_feature_images": low_feature_images,
            },
            "camera": {
                "mixed_lenses_detected": bool(mixed_lenses),
                "warnings": camera_warnings,
                "metadata_available_count": exif_count,
                "metadata_coverage_percentage": round(
                    100.0 * exif_count / max(1, valid_count), 1
                ),
                "normalized_focal_lengths_used": True,
            },
            "image_dimensions": {
                "count": len(image_dimensions),
                "aspect_ratio_spread": round(aspect_ratio_spread, 4),
                "dimensions": [
                    {"width": width, "height": height}
                    for width, height in image_dimensions
                ],
            },
            "overlap": {
                "status": "not_measured",
                "available": False,
                "message": (
                    "Pairwise visual overlap is measured during feature matching "
                    "and camera reconstruction; per-image sharpness does not prove overlap."
                ),
            },
            "warnings": warnings,
        }
