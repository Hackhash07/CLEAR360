import numpy as np

from photo_reconstruction.run_multiview_consensus_fusion import _cloud_summary


def test_cloud_summary_reports_count_and_bounds_in_scene_units():
    points = np.array([
        [1.0, -2.0, 0.5],
        [3.0, 4.0, 2.5],
        [-1.0, 1.0, -0.5],
    ], dtype=np.float32)

    result = _cloud_summary(points)

    assert result["point_count"] == 3
    np.testing.assert_allclose(result["bounds_min_sfm_units"], [-1.0, -2.0, -0.5])
    np.testing.assert_allclose(result["bounds_max_sfm_units"], [3.0, 4.0, 2.5])
    np.testing.assert_allclose(result["bounds_extent_sfm_units"], [4.0, 6.0, 3.0])


def test_cloud_summary_handles_empty_cloud():
    result = _cloud_summary(np.zeros((0, 3), dtype=np.float32))

    assert result["point_count"] == 0
    assert result["bounds_min_sfm_units"] is None
    assert result["bounds_max_sfm_units"] is None
    assert result["bounds_extent_sfm_units"] is None
