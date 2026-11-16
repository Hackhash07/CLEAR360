from types import SimpleNamespace

import pytest

from photo_reconstruction.camera_calibration import (
    pycolmap_camera_mode,
    resolve_colmap_camera_mode,
)


def test_camera_calibration_uses_one_camera_for_matching_dimensions_and_lens():
    report = {
        "camera": {"mixed_lenses_detected": False, "warnings": []},
        "image_dimensions": {
            "dimensions": [
                {"width": 4032, "height": 3024},
                {"width": 4032, "height": 3024},
            ]
        },
    }

    decision = resolve_colmap_camera_mode(report)

    assert decision["strategy"] == "shared_camera"
    assert decision["single_camera"] == "1"
    assert decision["single_camera_per_image"] == "0"


@pytest.mark.parametrize(
    "report",
    [
        {
            "camera": {
                "mixed_lenses_detected": True,
                "warnings": ["Multiple lenses detected."],
            },
            "image_dimensions": {
                "dimensions": [
                    {"width": 4032, "height": 3024},
                    {"width": 4032, "height": 3024},
                ]
            },
        },
        {
            "camera": {"mixed_lenses_detected": False, "warnings": []},
            "image_dimensions": {
                "dimensions": [
                    {"width": 4032, "height": 3024},
                    {"width": 1920, "height": 1080},
                ]
            },
        },
    ],
)
def test_camera_calibration_uses_per_image_mode_for_mixed_inputs(report):
    decision = resolve_colmap_camera_mode(report)

    assert decision["strategy"] == "per_image_camera"
    assert decision["single_camera"] == "0"
    assert decision["single_camera_per_image"] == "1"


def test_pycolmap_camera_mode_maps_shared_and_per_image_strategies():
    camera_enum = SimpleNamespace(SINGLE="single", PER_IMAGE="per_image")
    pycolmap = SimpleNamespace(CameraMode=camera_enum)

    assert pycolmap_camera_mode(pycolmap, "shared_camera") == "single"
    assert pycolmap_camera_mode(pycolmap, "per_image_camera") == "per_image"


def test_pycolmap_camera_mode_fails_clearly_when_enum_is_missing():
    with pytest.raises(RuntimeError, match="does not expose CameraMode.PER_IMAGE"):
        pycolmap_camera_mode(SimpleNamespace(CameraMode=SimpleNamespace()), "per_image_camera")

def test_native_sfm_blocks_mixed_camera_set_without_pycolmap(tmp_path, monkeypatch):
    import photo_reconstruction.sfm_engine as sfm_module
    from photo_reconstruction.sfm_engine import SfMEngine

    monkeypatch.setattr(sfm_module, "HAS_PYCOLMAP", False)
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "wide.jpg").write_bytes(b"not decoded by this preflight test")
    (image_dir / "tele.jpg").write_bytes(b"not decoded by this preflight test")

    engine = SfMEngine(tmp_path / "work")
    result = engine.run_reconstruction(
        image_dir,
        camera_mode="per_image_camera",
    )

    assert result["success"] is False
    assert result["camera_calibration_mode"] == "per_image_camera"
    assert "requires per-image camera calibration" in result["message"]
