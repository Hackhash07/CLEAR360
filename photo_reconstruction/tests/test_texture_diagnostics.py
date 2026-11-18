import numpy as np
import pytest

from photo_reconstruction.texture_diagnostics import texture_color_statistics


def test_texture_statistics_report_mean_rgb_and_near_black_texel_fraction():
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    image[0, :, :] = [180, 70, 30]

    report = texture_color_statistics(image)

    assert report["mean_rgb"] == pytest.approx([18.0, 7.0, 3.0])
    assert report["near_black_texel_percentage"] == pytest.approx(90.0)
    assert report["sampled_pixels"] == 100


def test_texture_statistics_do_not_flag_a_mostly_dark_but_non_black_atlas_as_void():
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    image[:, :2, :] = [5, 5, 5]  # dark, but above the explicit near-black threshold
    image[:, 2:, :] = [40, 30, 20]

    report = texture_color_statistics(image)

    assert report["near_black_texel_percentage"] == pytest.approx(0.0)


def test_texture_statistics_reject_invalid_images():
    with pytest.raises(ValueError, match="at least 3 color channels"):
        texture_color_statistics(np.zeros((8, 8), dtype=np.uint8))
