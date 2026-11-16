from pathlib import Path

from photo_reconstruction.exif_reader import ExifReader
from photo_reconstruction.photo_set_auditor import PhotoSetQualityAuditor


def _record(path: Path, *, score: float = 90.0, status: str = "EXCELLENT",
            blurry: bool = False, keypoints: int = 1200, exif: dict | None = None):
    return {
        "path": str(path),
        "meta": {
            "file_name": path.name,
            "file_path": str(path),
            "width": 1920,
            "height": 1080,
            "aspect_ratio": 1920 / 1080,
        },
        "quality": {
            "quality_score": score,
            "status": status,
            "is_blurry": blurry,
            "keypoint_count": keypoints,
        },
        "exif": exif or {
            "make": "Example",
            "model": "Phone",
            "estimated_fx": 2304.0,
            "normalized_fx": 1.2,
            "image_width": 1920,
            "image_height": 1080,
            "has_exif": True,
        },
    }


def test_resolution_changes_do_not_trigger_mixed_lens_warning():
    cameras = [
        {
            "model": "Same Phone",
            "estimated_fx": 2304.0,
            "normalized_fx": 1.2,
            "image_width": 1920,
            "image_height": 1080,
        },
        {
            "model": "Same Phone",
            "estimated_fx": 4608.0,
            "normalized_fx": 1.2,
            "image_width": 3840,
            "image_height": 2160,
        },
    ]

    mixed, warnings = ExifReader.check_mixed_lenses(cameras)

    assert mixed is False
    assert warnings == []


def test_photo_set_audit_detects_exact_duplicate_uploads(tmp_path: Path):
    first = tmp_path / "photo_01.jpg"
    duplicate = tmp_path / "photo_02.jpg"
    third = tmp_path / "photo_03.jpg"
    first.write_bytes(b"same image data")
    duplicate.write_bytes(b"same image data")
    third.write_bytes(b"different image data")

    report = PhotoSetQualityAuditor.audit([
        _record(first),
        _record(duplicate),
        _record(third),
    ])

    assert report["status"] == "review"
    assert report["valid_photo_count"] == 3
    assert report["unique_photo_count"] == 2
    assert report["duplicate_image_count"] == 1
    assert any("exact duplicate" in warning for warning in report["warnings"])


def test_photo_set_audit_blocks_fewer_than_two_distinct_photos(tmp_path: Path):
    first = tmp_path / "photo_01.jpg"
    duplicate = tmp_path / "photo_02.jpg"
    first.write_bytes(b"same image data")
    duplicate.write_bytes(b"same image data")

    report = PhotoSetQualityAuditor.audit([
        _record(first),
        _record(duplicate),
    ])

    assert report["status"] == "blocked"
    assert report["can_reconstruct"] is False
    assert report["unique_photo_count"] == 1


def test_photo_set_audit_flags_mixed_lenses_and_low_quality_images(tmp_path: Path):
    first = tmp_path / "wide.jpg"
    second = tmp_path / "tele.jpg"
    first.write_bytes(b"wide image")
    second.write_bytes(b"tele image")

    wide = _record(first)
    tele = _record(
        second,
        score=35.0,
        status="REJECT",
        blurry=True,
        keypoints=100,
        exif={
            "make": "Example",
            "model": "Phone",
            "estimated_fx": 4608.0,
            "normalized_fx": 2.4,
            "image_width": 1920,
            "image_height": 1080,
            "has_exif": True,
        },
    )

    report = PhotoSetQualityAuditor.audit([wide, tele])

    assert report["status"] == "review"
    assert report["camera"]["mixed_lenses_detected"] is True
    assert report["image_quality"]["rejected_count"] == 1
    assert report["image_quality"]["blurry_count"] == 1
    assert report["image_quality"]["low_feature_count"] == 1


def test_photo_set_overlap_is_not_claimed_before_feature_matching(tmp_path: Path):
    first = tmp_path / "photo_01.jpg"
    second = tmp_path / "photo_02.jpg"
    first.write_bytes(b"first")
    second.write_bytes(b"second")

    report = PhotoSetQualityAuditor.audit([
        _record(first),
        _record(second),
    ])

    assert report["status"] == "ready"
    assert report["can_reconstruct"] is True
    assert report["overlap"]["available"] is False
    assert report["overlap"]["status"] == "not_measured"


def test_photo_set_audit_reports_unreadable_uploads(tmp_path: Path):
    first = tmp_path / "valid.jpg"
    missing = tmp_path / "missing.jpg"
    first.write_bytes(b"valid")

    report = PhotoSetQualityAuditor.audit([
        _record(first),
        {
            "path": str(missing),
            "error": "Failed to decode image",
        },
    ])

    assert report["status"] == "blocked"
    assert report["valid_photo_count"] == 1
    assert report["invalid_images"][0]["image"] == "missing.jpg"
