"""Extract a compact, quality-screened photo set from a walkaround video.

The extractor samples timestamps across the clip rather than decoding every video
frame into the reconstruction set. It rejects unreadable, severely blurred and
near-identical samples, while leaving full cross-image overlap verification to
COLMAP's feature matching and camera-registration checks.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from photo_reconstruction.image_quality import ImageQualityAssessor


SUPPORTED_VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".m4v",
    ".avi",
    ".mkv",
    ".webm",
    ".3gp",
}

DEFAULT_MAX_FRAMES = 40
DEFAULT_SAMPLE_INTERVAL_SECONDS = 0.75
MIN_FRAME_DIMENSION = 320


def candidate_timestamps(
    duration_seconds: float,
    *,
    max_frames: int = DEFAULT_MAX_FRAMES,
    sample_interval_seconds: float = DEFAULT_SAMPLE_INTERVAL_SECONDS,
) -> list[float]:
    """Return evenly spaced sample times, capped to keep the image set compact."""
    duration = float(duration_seconds)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Video duration must be a positive finite number.")
    if not isinstance(max_frames, int) or max_frames < 1:
        raise ValueError("max_frames must be a positive integer.")
    if not math.isfinite(sample_interval_seconds) or sample_interval_seconds <= 0:
        raise ValueError("sample_interval_seconds must be positive and finite.")

    count = min(max_frames, max(1, int(math.ceil(duration / sample_interval_seconds))))
    # Midpoint sampling avoids repeatedly selecting the first or final transition.
    return [((index + 0.5) * duration / count) for index in range(count)]


def _difference_hash(gray: np.ndarray) -> int:
    """Compute a compact 64-bit horizontal difference hash."""
    resized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    comparisons = resized[:, 1:] > resized[:, :-1]
    value = 0
    for bit in comparisons.ravel():
        value = (value << 1) | int(bit)
    return value


def _near_duplicate(
    current_gray: np.ndarray,
    current_hash: int,
    previous_gray: np.ndarray,
    previous_hash: int,
    *,
    hash_distance_threshold: int = 2,
    mean_difference_threshold: float = 3.0,
) -> bool:
    """Detect near-identical adjacent samples without discarding normal camera motion."""
    hash_distance = (current_hash ^ previous_hash).bit_count()
    if hash_distance > hash_distance_threshold:
        return False

    current_small = cv2.resize(current_gray, (64, 64), interpolation=cv2.INTER_AREA)
    previous_small = cv2.resize(previous_gray, (64, 64), interpolation=cv2.INTER_AREA)
    mean_difference = float(
        np.mean(cv2.absdiff(current_small, previous_small))
    )
    return mean_difference <= mean_difference_threshold


class VideoFrameExtractor:
    """Create up to 40 sharp, timestamped JPEG views from a single video."""

    @classmethod
    def extract(
        cls,
        video_path: str | Path,
        output_dir: str | Path,
        *,
        max_frames: int = DEFAULT_MAX_FRAMES,
        sample_interval_seconds: float = DEFAULT_SAMPLE_INTERVAL_SECONDS,
        jpeg_quality: int = 95,
    ) -> dict[str, Any]:
        """Extract sampled frames and write a reproducible JSON manifest.

        The caller owns the output directory and should pass a new, unique
        directory. Existing files are never overwritten.
        """
        source = Path(video_path).expanduser().resolve()
        output = Path(output_dir).expanduser().resolve()

        if not source.is_file():
            raise FileNotFoundError(f"Video file does not exist: {source}")
        if source.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
            supported = ", ".join(sorted(SUPPORTED_VIDEO_EXTENSIONS))
            raise ValueError(
                f"Unsupported video extension '{source.suffix}'. Supported formats: {supported}."
            )
        if not isinstance(jpeg_quality, int) or not 70 <= jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be an integer between 70 and 100.")

        output.mkdir(parents=True, exist_ok=True)
        frames_dir = output / "frames"
        if frames_dir.exists() and any(frames_dir.iterdir()):
            raise FileExistsError(
                f"Video frame output already contains files; refusing to overwrite: {frames_dir}"
            )
        frames_dir.mkdir(parents=True, exist_ok=True)

        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(
                f"OpenCV could not open '{source.name}'. Try an H.264 MP4 or MOV video."
            )

        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            source_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if not math.isfinite(fps) or fps <= 0 or source_frame_count <= 0:
                raise RuntimeError(
                    "The video does not expose a valid frame rate and frame count; "
                    "export it as a constant-frame-rate MP4 and retry."
                )
            if width < MIN_FRAME_DIMENSION or height < MIN_FRAME_DIMENSION:
                raise RuntimeError(
                    f"Video resolution {width}x{height} is too low for reliable photogrammetry."
                )

            duration = source_frame_count / fps
            timestamps = candidate_timestamps(
                duration,
                max_frames=max_frames,
                sample_interval_seconds=sample_interval_seconds,
            )

            saved_frames: list[dict[str, Any]] = []
            skipped_blurry = 0
            skipped_near_duplicate = 0
            skipped_unreadable = 0
            previous_gray: np.ndarray | None = None
            previous_hash: int | None = None

            for sample_index, timestamp in enumerate(timestamps, start=1):
                requested_frame = min(
                    source_frame_count - 1,
                    max(0, int(round(timestamp * fps))),
                )
                if not capture.set(cv2.CAP_PROP_POS_FRAMES, requested_frame):
                    skipped_unreadable += 1
                    continue

                ok, frame_bgr = capture.read()
                if not ok or frame_bgr is None or frame_bgr.size == 0:
                    skipped_unreadable += 1
                    continue

                frame_height, frame_width = frame_bgr.shape[:2]
                if min(frame_width, frame_height) < MIN_FRAME_DIMENSION:
                    skipped_unreadable += 1
                    continue

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                quality = ImageQualityAssessor.evaluate(frame_rgb)
                if float(quality.get("blur_score", 0.0)) < ImageQualityAssessor.BLUR_THRESHOLD_WARN:
                    skipped_blurry += 1
                    continue

                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                frame_hash = _difference_hash(gray)
                if (
                    previous_gray is not None
                    and previous_hash is not None
                    and _near_duplicate(gray, frame_hash, previous_gray, previous_hash)
                ):
                    skipped_near_duplicate += 1
                    continue

                timestamp_ms = int(round(timestamp * 1000.0))
                filename = f"frame_{len(saved_frames) + 1:03d}_t{timestamp_ms:08d}ms.jpg"
                frame_path = frames_dir / filename
                write_ok = cv2.imwrite(
                    str(frame_path),
                    frame_bgr,
                    [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
                )
                if not write_ok or not frame_path.is_file() or frame_path.stat().st_size == 0:
                    raise RuntimeError(f"Could not write extracted frame: {frame_path}")

                saved_frames.append({
                    "path": str(frame_path),
                    "filename": filename,
                    "timestamp_ms": timestamp_ms,
                    "quality": quality,
                })
                previous_gray = gray
                previous_hash = frame_hash

            if not saved_frames:
                raise RuntimeError(
                    "No usable sharp frames were found. Record a slower, steadier walkaround "
                    "with good lighting and try again."
                )

            sampling_interval = (
                round(duration / len(timestamps), 3) if timestamps else None
            )
            warnings: list[str] = []
            if len(saved_frames) < 6:
                warnings.append(
                    "Fewer than six usable frames were extracted. Add more video coverage "
                    "or still photographs before using COLMAP + OpenMVS."
                )
            if skipped_blurry:
                warnings.append(
                    f"{skipped_blurry} sampled frame(s) were skipped for motion/defocus blur."
                )
            if skipped_near_duplicate:
                warnings.append(
                    f"{skipped_near_duplicate} near-identical adjacent frame(s) were skipped."
                )
            warnings.append(
                "Video frames do not guarantee camera overlap: move the phone physically "
                "around the subject; panning from one fixed position does not provide parallax."
            )

            manifest: dict[str, Any] = {
                "schema_version": 1,
                "source_video": str(source),
                "source_filename": source.name,
                "source_size_bytes": source.stat().st_size,
                "duration_seconds": round(duration, 3),
                "fps": round(fps, 3),
                "source_frame_count": source_frame_count,
                "width": width,
                "height": height,
                "candidate_frame_count": len(timestamps),
                "saved_frame_count": len(saved_frames),
                "skipped_blurry_count": skipped_blurry,
                "skipped_near_duplicate_count": skipped_near_duplicate,
                "skipped_unreadable_count": skipped_unreadable,
                "sampling_interval_seconds": sampling_interval,
                "max_frames": max_frames,
                "frames": saved_frames,
                "warnings": warnings,
            }
            (output / "video_extract_manifest.json").write_text(
                json.dumps(manifest, indent=2),
                encoding="utf-8",
            )
            return manifest
        finally:
            capture.release()
