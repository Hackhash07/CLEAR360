from types import SimpleNamespace

import pytest

from photo_reconstruction.camera_calibration import (
    can_use_opencv_fallback,
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

def test_opencv_fallback_only_supports_a_shared_camera_matrix():
    assert can_use_opencv_fallback("shared_camera") is True
    assert can_use_opencv_fallback("per_image_camera") is False


def test_opencv_fallback_policy_rejects_unknown_camera_strategy():
    with pytest.raises(ValueError, match="strategy must be"):
        can_use_opencv_fallback("auto")
