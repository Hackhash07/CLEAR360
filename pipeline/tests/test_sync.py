"""
CLEAR360 – Synchronization Tests
=================================
Tests for CameraFrameData, FrameSet, and FrameSetLoader.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from pipeline.sync import CameraFrameData, FrameSet, FrameSetLoader


def _create_dummy_frame_data(
    camera_name: str,
    frame_id: int = 0,
    rgb_shape: tuple[int, int, int] = (720, 1280, 3),
    rgb_dtype: np.dtype = np.uint8,
    depth_shape: tuple[int, int] = (720, 1280),
    depth_dtype: np.dtype = np.float32,
    depth_val: float = 5.0,
    depth_source: str = "ground_truth_blender",
) -> CameraFrameData:
    rgb = np.full(rgb_shape, 128, dtype=rgb_dtype)
    depth = np.full(depth_shape, depth_val, dtype=depth_dtype)
    return CameraFrameData(
        camera_name=camera_name,
        frame_id=frame_id,
        rgb=rgb,
        depth=depth,
        depth_source=depth_source,
    )


def test_frameset_complete() -> None:
    """Build a FrameSet manually with 4 cameras, check is_complete == True."""
    cameras = {
        name: _create_dummy_frame_data(name)
        for name in ("front", "rear", "left", "right")
    }
    fs = FrameSet(frame_id=0, cameras=cameras)
    assert fs.is_complete is True
    errors = fs.validate()
    assert len(errors) == 0, f"Expected no errors, got: {errors}"


def test_frameset_incomplete() -> None:
    """3 cameras -> is_complete == False."""
    cameras = {
        name: _create_dummy_frame_data(name)
        for name in ("front", "rear", "left")  # missing right
    }
    fs = FrameSet(frame_id=0, cameras=cameras)
    assert fs.is_complete is False
    errors = fs.validate()
    assert any("Missing camera: 'right'" in e for e in errors)


def test_frameset_validate_dtype() -> None:
    """Wrong dtype on depth -> validation returns error message."""
    cameras = {
        name: _create_dummy_frame_data(name)
        for name in ("front", "rear", "left", "right")
    }
    # Modify front camera depth to float64 instead of float32
    cameras["front"] = _create_dummy_frame_data(
        "front", depth_dtype=np.float64
    )
    fs = FrameSet(frame_id=0, cameras=cameras)
    errors = fs.validate()
    assert any("depth dtype is float64, expected float32" in e for e in errors)


def test_frameset_validate_shape() -> None:
    """Mismatched shape -> validation returns error."""
    cameras = {
        name: _create_dummy_frame_data(name)
        for name in ("front", "rear", "left", "right")
    }
    # Modify front camera depth shape to (360, 640)
    cameras["front"] = _create_dummy_frame_data(
        "front", depth_shape=(360, 640)
    )
    fs = FrameSet(frame_id=0, cameras=cameras)
    errors = fs.validate()
    assert any("Depth shapes are inconsistent across cameras" in e for e in errors)


def test_loader_available_frames_empty(tmp_path: Path) -> None:
    """FrameSetLoader on a non-existent dir returns empty list (no crash)."""
    loader = FrameSetLoader(data_root=tmp_path / "non_existent_data_root")
    frames = loader.available_frames()
    assert frames == []


def test_loader_load_frame_missing(tmp_path: Path) -> None:
    """Loading a frame that does not exist raises FileNotFoundError."""
    loader = FrameSetLoader(data_root=tmp_path)
    with pytest.raises(FileNotFoundError):
        loader.load_frame(frame_id=999)


def test_camera_frame_data_depth_source_tag() -> None:
    """Check depth_source field is set correctly."""
    cfd_gt = _create_dummy_frame_data("front", depth_source="ground_truth_blender")
    assert cfd_gt.depth_source == "ground_truth_blender"

    cfd_ai = _create_dummy_frame_data("front", depth_source="ai_depth_anything_v2")
    assert cfd_ai.depth_source == "ai_depth_anything_v2"
