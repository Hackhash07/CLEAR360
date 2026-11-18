import numpy as np
import pytest

from photo_reconstruction.diagnose_fused_cloud import (
    _nearest_neighbor_summary,
    _summarize_labels,
)


def test_cluster_summary_reports_noise_and_largest_component():
    labels = np.array([0, 0, 0, 1, 1, -1, -1], dtype=np.int64)

    result = _summarize_labels(labels, total_points=7)

    assert result["cluster_count"] == 2
    assert result["noise_points"] == 2
    assert result["largest_cluster_points"] == 3
    assert result["largest_cluster_fraction_of_all_points"] == pytest.approx(3 / 7)
    assert result["noise_fraction"] == pytest.approx(2 / 7)
    assert result["top_10_cluster_sizes"] == [3, 2]


def test_cluster_summary_handles_all_noise():
    result = _summarize_labels(np.full(5, -1), total_points=5)

    assert result["cluster_count"] == 0
    assert result["noise_points"] == 5
    assert result["largest_cluster_points"] == 0
    assert result["largest_cluster_fraction_of_all_points"] == 0.0


def test_nearest_neighbor_summary_filters_nonfinite_and_zero_values():
    result = _nearest_neighbor_summary(
        np.array([0.0, 0.1, 0.2, np.nan, np.inf, 0.3])
    )

    assert result["min"] == pytest.approx(0.1)
    assert result["median"] == pytest.approx(0.2)
    assert result["p90"] == pytest.approx(0.28)


def test_nearest_neighbor_summary_rejects_no_positive_distances():
    with pytest.raises(ValueError, match="positive nearest-neighbour"):
        _nearest_neighbor_summary(np.array([0.0, np.nan, np.inf]))
