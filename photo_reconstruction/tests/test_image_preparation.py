from pathlib import Path

import numpy as np
from PIL import Image

from photo_reconstruction.image_loader import ImageLoader
from photo_reconstruction.image_preparation import (
    session_relative_path,
    stage_reconstruction_image,
)


def test_native_sfm_image_is_archived_and_copied_without_reencoding(tmp_path: Path):
    source = tmp_path / "source" / "sample.png"
    source.parent.mkdir()
    rgb = np.zeros((48, 64, 3), dtype=np.uint8)
    rgb[:, :, 0] = 200
    Image.fromarray(rgb).save(source, format="PNG")
    original_bytes = source.read_bytes()

    result = stage_reconstruction_image(
        source,
        rgb,
        tmp_path / "session" / "source_uploads",
        tmp_path / "session" / "original_images",
        index=0,
    )

    working = Path(result["sfm_image_path"])
    archived = Path(result["source_archive_path"])
    assert working.suffix == ".png"
    assert result["converted_for_sfm"] is False
    assert working.read_bytes() == original_bytes
    assert archived.read_bytes() == original_bytes
    assert result["original_filename"] == "sample.png"


def test_non_native_sfm_image_is_converted_but_original_is_preserved(tmp_path: Path):
    source = tmp_path / "source" / "sample.bmp"
    source.parent.mkdir()
    rgb = np.zeros((48, 64, 3), dtype=np.uint8)
    rgb[:, :, 1] = 180
    Image.fromarray(rgb).save(source, format="BMP")
    original_bytes = source.read_bytes()

    result = stage_reconstruction_image(
        source,
        rgb,
        tmp_path / "session" / "source_uploads",
        tmp_path / "session" / "original_images",
        index=4,
    )

    working = Path(result["sfm_image_path"])
    archived = Path(result["source_archive_path"])
    assert working.suffix == ".jpg"
    assert result["working_format"] == "jpg"
    assert result["converted_for_sfm"] is True
    assert archived.read_bytes() == original_bytes
    with Image.open(working) as converted:
        assert converted.format == "JPEG"
        assert converted.size == (64, 48)
        pixel = np.asarray(converted.convert("RGB"))[20, 20]
        assert abs(int(pixel[1]) - 180) < 12


def test_tiff_exif_is_copied_to_normalized_jpeg_when_available(tmp_path: Path):
    source = tmp_path / "source" / "camera.tiff"
    source.parent.mkdir()
    rgb = np.full((48, 64, 3), 120, dtype=np.uint8)
    exif = Image.Exif()
    exif[271] = "CLEAR360 Test Maker"
    exif[272] = "Test Camera"
    Image.fromarray(rgb).save(source, format="TIFF", exif=exif)

    result = stage_reconstruction_image(
        source,
        rgb,
        tmp_path / "session" / "source_uploads",
        tmp_path / "session" / "original_images",
        index=1,
    )

    with Image.open(result["sfm_image_path"]) as normalized:
        normalized_exif = normalized.getexif()
        assert normalized_exif.get(271) == "CLEAR360 Test Maker"
        assert normalized_exif.get(272) == "Test Camera"


def test_same_basenames_receive_distinct_sfm_and_archive_names(tmp_path: Path):
    first = tmp_path / "a" / "object.bmp"
    second = tmp_path / "b" / "object.bmp"
    first.parent.mkdir()
    second.parent.mkdir()
    rgb = np.full((48, 64, 3), 90, dtype=np.uint8)
    Image.fromarray(rgb).save(first, format="BMP")
    Image.fromarray(255 - rgb).save(second, format="BMP")

    first_result = stage_reconstruction_image(
        first, rgb, tmp_path / "session" / "source_uploads",
        tmp_path / "session" / "original_images", index=0,
    )
    second_result = stage_reconstruction_image(
        second, 255 - rgb, tmp_path / "session" / "source_uploads",
        tmp_path / "session" / "original_images", index=1,
    )

    assert first_result["sfm_filename"] != second_result["sfm_filename"]
    assert Path(first_result["source_archive_path"]).name != Path(second_result["source_archive_path"]).name

def test_image_loader_accepts_every_advertised_phone_photo_extension():
    for suffix in (".jpg", ".jpeg", ".png", ".heic", ".heif", ".tif", ".tiff", ".bmp"):
        assert ImageLoader.is_supported(Path(f"sample{suffix}"))

def test_session_relative_path_handles_symlinked_session_directory(tmp_path: Path):
    real_session = tmp_path / "real_session"
    archive = real_session / "source_uploads" / "000_photo.jpg"
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b"original")

    # Simulate macOS exposing the same directory via two filesystem aliases.
    session_alias = tmp_path / "session_alias"
    session_alias.symlink_to(real_session, target_is_directory=True)

    relative = session_relative_path(archive, session_alias)

    assert relative == "source_uploads/000_photo.jpg"
