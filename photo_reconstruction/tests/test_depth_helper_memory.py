from pathlib import Path
from types import SimpleNamespace

import numpy as np

from photo_reconstruction.depth_helper import IsolatedDepthEstimator


def test_cached_depth_maps_are_returned_as_read_only_memmaps(tmp_path, monkeypatch):
    source = tmp_path / "photo_001.jpg"
    source.write_bytes(b"test image")
    cache_dir = tmp_path / "depth_cache"
    cache_dir.mkdir()
    expected = np.linspace(0.5, 10.0, 120, dtype=np.float32).reshape(10, 12)
    np.save(cache_dir / "photo_001_depth.npy", expected)

    def fake_run(*args, **kwargs):
        return SimpleNamespace(stdout="ISOLATED_DEPTH_SUCCESS", stderr="", returncode=0)

    monkeypatch.setattr("photo_reconstruction.depth_helper.subprocess.run", fake_run)

    result = IsolatedDepthEstimator.estimate_depths_for_images(
        [source],
        cache_dir,
        project_root=tmp_path,
    )

    depth = result["photo_001.jpg"]
    assert isinstance(depth, np.memmap)
    assert depth.shape == expected.shape
    assert depth.dtype == np.float32
    assert not depth.flags.writeable
    np.testing.assert_array_equal(depth, expected)


def test_depth_estimator_skips_nonexistent_image_paths(tmp_path, monkeypatch):
    called = False

    def fake_run(*args, **kwargs):
        nonlocal called
        called = True
        return SimpleNamespace(stdout="ISOLATED_DEPTH_SUCCESS", stderr="", returncode=0)

    monkeypatch.setattr("photo_reconstruction.depth_helper.subprocess.run", fake_run)

    result = IsolatedDepthEstimator.estimate_depths_for_images(
        [tmp_path / "missing.jpg"],
        tmp_path / "depth_cache",
        project_root=tmp_path,
    )

    assert result == {}
    assert called is True
