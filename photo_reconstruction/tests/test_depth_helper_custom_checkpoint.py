"""Tests for custom checkpoint configuration in isolated depth inference."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from photo_reconstruction.depth_helper import IsolatedDepthEstimator


def test_isolated_depth_estimator_passes_custom_checkpoint_and_depth_range(tmp_path, monkeypatch):
    image_path = tmp_path / "room.jpg"
    image_path.write_bytes(b"test image placeholder")
    cache_dir = tmp_path / "depth_cache_hypersim"
    weights_path = tmp_path / "depth_anything_v2_metric_hypersim_vits.pth"
    weights_path.write_bytes(b"checkpoint placeholder")

    def fake_run(cmd, capture_output, text, timeout):
        manifest_path = Path(cmd[-1])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["weights_path"] == str(weights_path)
        assert manifest["max_depth"] == 20.0
        assert timeout == 321

        np.save(cache_dir / "room_depth.npy", np.full((8, 8), 4.0, dtype=np.float32))
        return SimpleNamespace(
            returncode=0,
            stdout="ISOLATED_DEPTH_SUCCESS",
            stderr="",
        )

    monkeypatch.setattr("photo_reconstruction.depth_helper.subprocess.run", fake_run)

    result = IsolatedDepthEstimator.estimate_depths_for_images(
        image_paths=[image_path],
        cache_dir=cache_dir,
        project_root=tmp_path,
        weights_path=weights_path,
        max_depth=20.0,
        timeout_s=321,
        raise_on_error=True,
    )

    assert "room.jpg" in result
    assert result["room.jpg"].shape == (8, 8)
    assert np.allclose(result["room.jpg"], 4.0)


def test_isolated_depth_estimator_surfaces_worker_errors_when_requested(tmp_path, monkeypatch):
    image_path = tmp_path / "room.jpg"
    image_path.write_bytes(b"test image placeholder")

    def fake_run(cmd, capture_output, text, timeout):
        return SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="checkpoint load failed",
        )

    monkeypatch.setattr("photo_reconstruction.depth_helper.subprocess.run", fake_run)

    with pytest.raises(RuntimeError, match="checkpoint load failed"):
        IsolatedDepthEstimator.estimate_depths_for_images(
            image_paths=[image_path],
            cache_dir=tmp_path / "cache",
            project_root=tmp_path,
            weights_path=tmp_path / "missing.pth",
            max_depth=20.0,
            raise_on_error=True,
        )
