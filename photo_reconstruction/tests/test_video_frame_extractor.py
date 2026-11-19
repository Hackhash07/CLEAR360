from pathlib import Path

import cv2
import numpy as np
import pytest

from photo_reconstruction.video_frame_extractor import (
    VideoFrameExtractor,
    candidate_timestamps,
)


def _write_test_video(path: Path, *, frame_count: int, fps: float, moving: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = 480, 640
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        writer.release()
        pytest.skip("This OpenCV build does not provide the MJPG VideoWriter codec.")

    rng = np.random.default_rng(20261119)
    base = rng.integers(20, 236, size=(height, width, 3), dtype=np.uint8)
    for index in range(frame_count):
        frame = base.copy()
        if moving:
            x = 20 + (index * 17) % (width - 80)
            y = 20 + (index * 11) % (height - 80)
            cv2.rectangle(frame, (x, y), (x + 55, y + 55), (0, 255, 255), 4)
            cv2.putText(
                frame, str(index), (20, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2,
            )
        writer.write(frame)
    writer.release()


def test_candidate_timestamps_are_evenly_spaced_and_respect_cap():
    times = candidate_timestamps(60.0, max_frames=40, sample_interval_seconds=0.75)

    assert len(times) == 40
    assert times == sorted(times)
    assert times[0] > 0
    assert times[-1] < 60.0
    assert np.allclose(np.diff(times), 1.5)


def test_candidate_timestamps_validate_bad_inputs():
    with pytest.raises(ValueError):
        candidate_timestamps(0)
    with pytest.raises(ValueError):
        candidate_timestamps(float("nan"))
    with pytest.raises(ValueError):
        candidate_timestamps(10, max_frames=0)


def test_video_extractor_writes_timestamped_jpegs_and_manifest(tmp_path: Path):
    video = tmp_path / "walkaround.avi"
    _write_test_video(video, frame_count=96, fps=8.0, moving=True)

    result = VideoFrameExtractor.extract(
        video,
        tmp_path / "import_001",
        max_frames=12,
    )

    assert result["source_filename"] == "walkaround.avi"
    assert result["source_frame_count"] == 96
    assert result["candidate_frame_count"] <= 12
    assert 6 <= result["saved_frame_count"] <= 12
    assert len(result["frames"]) == result["saved_frame_count"]

    timestamps = [frame["timestamp_ms"] for frame in result["frames"]]
    assert timestamps == sorted(timestamps)
    assert len(timestamps) == len(set(timestamps))

    for frame in result["frames"]:
        frame_path = Path(frame["path"])
        assert frame_path.is_file()
        assert frame_path.suffix == ".jpg"
        decoded = cv2.imread(str(frame_path))
        assert decoded is not None
        assert decoded.shape[:2] == (480, 640)
        assert frame["quality"]["quality_score"] >= 0

    manifest_path = tmp_path / "import_001" / "video_extract_manifest.json"
    assert manifest_path.is_file()


def test_video_extractor_skips_near_identical_frames(tmp_path: Path):
    video = tmp_path / "stationary.avi"
    _write_test_video(video, frame_count=80, fps=8.0, moving=False)

    result = VideoFrameExtractor.extract(video, tmp_path / "import_002")

    assert result["candidate_frame_count"] > 1
    assert result["saved_frame_count"] == 1
    assert result["skipped_near_duplicate_count"] == result["candidate_frame_count"] - 1


def test_video_extractor_rejects_unsupported_extensions(tmp_path: Path):
    video = tmp_path / "walkaround.txt"
    video.write_text("not a video", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported video extension"):
        VideoFrameExtractor.extract(video, tmp_path / "import_003")


def test_video_extractor_never_overwrites_existing_frame_outputs(tmp_path: Path):
    video = tmp_path / "walkaround.avi"
    _write_test_video(video, frame_count=80, fps=8.0, moving=True)
    output = tmp_path / "import_004"
    frames_dir = output / "frames"
    frames_dir.mkdir(parents=True)
    sentinel = output / "keep.txt"
    sentinel.write_text("preserve", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        VideoFrameExtractor.extract(video, output)

    assert sentinel.read_text(encoding="utf-8") == "preserve"


def test_video_extractor_reports_unreadable_video(tmp_path: Path):
    video = tmp_path / "broken.mp4"
    video.write_bytes(b"not a video container")

    with pytest.raises(RuntimeError, match="could not open"):
        VideoFrameExtractor.extract(video, tmp_path / "import_005")
