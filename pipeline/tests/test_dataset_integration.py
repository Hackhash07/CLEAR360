"""
pipeline/tests/test_dataset_integration.py
===========================================
End-to-end integration test validating the real 4-camera dataset generated for M1.
Tests loading via FrameSetLoader, frame validation, and unprojection consistency.
"""

from pathlib import Path
import numpy as np
import pytest

from pipeline.calibration import CalibrationStore
from pipeline.sync import FrameSetLoader
from validation.gt_loader import verify_depth_vs_position

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
CONFIG_PATH = PROJECT_ROOT / "config" / "cameras.yaml"


@pytest.fixture
def store():
    st = CalibrationStore(CONFIG_PATH)
    st.load()
    return st


@pytest.fixture
def loader():
    return FrameSetLoader(DATA_ROOT)


def test_dataset_available_frames(loader):
    """Verify that all 20 frames (0..19) are detected by FrameSetLoader."""
    available = loader.available_frames()
    assert len(available) == 20, f"Expected 20 frames, found {len(available)}: {available}"
    assert available == list(range(20))


def test_frameset_loader_frame_0_complete(loader):
    """Load FrameSet 0 and verify completeness and validation checks."""
    fs = loader.load_frame(0, load_position=True, load_semantic=True)
    assert fs.is_complete, "FrameSet 0 is not complete across all 4 cameras"

    val_errors = fs.validate()
    assert len(val_errors) == 0, f"FrameSet 0 validation failed with errors: {val_errors}"

    for cam_name in ["front", "rear", "left", "right"]:
        cfd = fs.get(cam_name)
        assert cfd.rgb.shape == (720, 1280, 3)
        assert cfd.rgb.dtype == np.uint8
        assert cfd.depth.shape == (720, 1280)
        assert cfd.depth.dtype == np.float32
        assert cfd.position_map is not None
        assert cfd.position_map.shape == (720, 1280, 3)
        assert cfd.position_map.dtype == np.float32
        assert cfd.semantic_mask is not None
        assert cfd.semantic_mask.shape == (720, 1280)
        assert cfd.semantic_mask.dtype == np.uint8


def test_depth_unprojection_vs_position_map_consistency(store, loader):
    """
    Verify that unprojecting depth pixels with CameraModel.unproject() yields
    world 3D points that exactly match the ground-truth Position map (within 1 cm).
    """
    fs = loader.load_frame(0, load_position=True, load_semantic=False)
    front_cam = store.get("front")
    cfd = fs.get("front")

    # Sample non-sky pixels where depth < 50m
    valid_mask = (cfd.depth > 0.5) & (cfd.depth < 50.0)
    indices = np.argwhere(valid_mask)
    assert len(indices) > 1000, "Insufficient foreground pixels for unprojection check"

    # Test 50 random foreground pixels
    np.random.seed(42)
    sample_idx = indices[np.random.choice(len(indices), size=50, replace=False)]

    max_err = 0.0
    for v, u in sample_idx:
        d = float(cfd.depth[v, u])
        p_unproj = front_cam.unproject(float(u), float(v), d)
        p_gt = cfd.position_map[v, u]

        err = float(np.linalg.norm(p_unproj - p_gt))
        if err > max_err:
            max_err = err

    assert max_err < 0.05, f"Unprojection discrepancy {max_err:.4f} m exceeds 5 cm tolerance"

