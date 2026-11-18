import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from photo_reconstruction.image_quality import ImageQualityAssessor


def test_quality_analysis_normalizes_portrait_and_landscape_longest_side():
    rng = np.random.default_rng(42)
    landscape = rng.integers(0, 256, size=(1600, 900, 3), dtype=np.uint8)
    portrait = np.rot90(landscape).copy()

    landscape_report = ImageQualityAssessor.evaluate(landscape)
    portrait_report = ImageQualityAssessor.evaluate(portrait)

    assert landscape_report["analysis_dimensions"] == {
        "width": 720,
        "height": 1280,
        "max_dimension": 1280,
        "scale_from_input": 0.8,
    }
    assert portrait_report["analysis_dimensions"] == {
        "width": 1280,
        "height": 720,
        "max_dimension": 1280,
        "scale_from_input": 0.8,
    }
    assert landscape_report["normalized_keypoint_count"] >= 0
    assert portrait_report["normalized_keypoint_count"] >= 0


def test_quality_analysis_does_not_upscale_small_images():
    image = np.zeros((360, 640, 3), dtype=np.uint8)

    report = ImageQualityAssessor.evaluate(image)

    assert report["analysis_dimensions"]["width"] == 640
    assert report["analysis_dimensions"]["height"] == 360
    assert report["analysis_dimensions"]["scale_from_input"] == 1.0


def test_quality_analysis_rejects_invalid_rgb_shapes():
    with pytest.raises(ValueError, match="RGB image"):
        ImageQualityAssessor.evaluate(np.zeros((100, 100), dtype=np.uint8))
