"""Non-destructive staging of uploaded photographs for SfM engines."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from photo_reconstruction.image_loader import ensure_image_format_support


SFM_NATIVE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def _unique_path(directory: Path, filename: str, index: int) -> Path:
    candidate = directory / filename
    if not candidate.exists():
        return candidate
    source = Path(filename)
    candidate = directory / f"{source.stem}_{index:03d}{source.suffix}"
    counter = 1
    while candidate.exists():
        candidate = directory / f"{source.stem}_{index:03d}_{counter}{source.suffix}"
        counter += 1
    return candidate


def stage_reconstruction_image(
    source_path: str | Path,
    rgb_image: np.ndarray,
    source_archive_dir: str | Path,
    sfm_working_dir: str | Path,
    *,
    index: int,
) -> dict[str, Any]:
    """Archive the exact upload and create a SfM-readable working copy.

    JPEG/PNG inputs are copied byte-for-byte into the working folder. Formats
    accepted by CLEAR360's image loader but not by its SfM directory scanner
    (for example HEIC, TIFF or BMP) are converted to JPEG for SfM. The exact
    original bytes are always retained in the source archive.
    """
    source = Path(source_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Source photograph does not exist: {source}")

    rgb = np.asarray(rgb_image)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("SfM staging expects an HxWx3 uint8 RGB image.")

    archive_dir = Path(source_archive_dir)
    working_dir = Path(sfm_working_dir)
    archive_dir.mkdir(parents=True, exist_ok=True)
    working_dir.mkdir(parents=True, exist_ok=True)

    archive_name = f"{index:03d}_{source.name}"
    archive_path = archive_dir / archive_name
    if archive_path.exists():
        archive_path = _unique_path(archive_dir, archive_name, index)
    shutil.copy2(source, archive_path)

    source_suffix = source.suffix.lower()
    output_suffix = source_suffix if source_suffix in SFM_NATIVE_EXTENSIONS else ".jpg"
    working_name = f"{source.stem}{output_suffix}"
    working_path = _unique_path(working_dir, working_name, index)

    converted = source_suffix not in SFM_NATIVE_EXTENSIONS
    if not converted:
        shutil.copy2(source, working_path)
    else:
        exif_bytes = b""
        try:
            ensure_image_format_support(source)
            with Image.open(source) as original:
                exif = original.getexif()
                if exif:
                    exif_bytes = exif.tobytes()
        except Exception:
            exif_bytes = b""

        image = Image.fromarray(rgb, mode="RGB")
        save_options: dict[str, Any] = {
            "format": "JPEG",
            "quality": 95,
            "optimize": True,
        }
        if exif_bytes:
            save_options["exif"] = exif_bytes
        image.save(working_path, **save_options)

    return {
        "source_path": str(source),
        "source_archive_path": str(archive_path.resolve()),
        "original_filename": source.name,
        "sfm_image_path": str(working_path.resolve()),
        "sfm_filename": working_path.name,
        "source_format": source_suffix.lstrip("."),
        "working_format": output_suffix.lstrip("."),
        "converted_for_sfm": converted,
        "source_size_bytes": source.stat().st_size,
        "working_size_bytes": working_path.stat().st_size,
    }
