from types import SimpleNamespace

from photo_reconstruction.compare_depth_calibration_strategies import (
    summarize_view_stats,
)


def test_strategy_summary_counts_accepted_and_preserves_per_view_errors():
    stats = {
        "5.jpg": SimpleNamespace(
            success=True,
            num_anchors=180,
            median_scale=1.58,
            depth_offset=3.15,
            validation_median_relative_error=0.028,
            validation_p90_relative_error=0.081,
            status_message="Accepted",
        ),
        "8.jpg": SimpleNamespace(
            success=False,
            num_anchors=110,
            median_scale=1.82,
            depth_offset=2.64,
            validation_median_relative_error=0.018,
            validation_p90_relative_error=0.14,
            status_message="Rejected: P90 above limit",
        ),
    }

    result = summarize_view_stats(stats)

    assert result["views_checked"] == 2
    assert result["accepted_count"] == 1
    assert result["accepted_views"] == ["5.jpg"]
    assert result["rejected_views"] == ["8.jpg"]
    assert result["views"]["5.jpg"]["validation_p90_relative_error"] == 0.081
    assert result["views"]["8.jpg"]["accepted"] is False


def test_strategy_summary_handles_empty_stats():
    result = summarize_view_stats({})

    assert result["views_checked"] == 0
    assert result["accepted_count"] == 0
    assert result["accepted_views"] == []
    assert result["rejected_views"] == []
