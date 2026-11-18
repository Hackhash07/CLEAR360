import json

from photo_reconstruction.benchmark_report import (
    format_openmvs_summary,
    summarize_openmvs_log,
)


def test_openmvs_log_summary_parses_config_times_and_peak_rss(tmp_path):
    log = tmp_path / "openmvs_pipeline.log"
    config = {
        "quality_preset": "high",
        "performance_profile": "apple_silicon_high_balanced",
        "input_photo_count": 6,
    }
    log.write_text(
        "CLEAR360 OpenMVS pipeline\n"
        "Pipeline configuration: " + json.dumps(config) + "\n"
        "$ /usr/bin/time -l colmap feature_extractor\n"
        "250000000 maximum resident set size\n"
        "[CLEAR360 stage timing] colmap: 12.5 seconds; exit_code=0\n"
        "$ /usr/bin/time -l DensifyPointCloud\n"
        "1200000000 maximum resident set size\n"
        "[CLEAR360 stage timing] DensifyPointCloud: 65.0 seconds; exit_code=0\n",
        encoding="utf-8",
    )

    report = summarize_openmvs_log(log)

    assert report["configuration"] == config
    assert report["external_stage_total_seconds"] == 77.5
    assert report["all_stages_succeeded"] is True
    assert report["memory_profiled_stage_count"] == 2
    assert report["stages"][0]["peak_rss_bytes"] == 250_000_000
    assert report["stages"][1]["peak_rss_bytes"] == 1_200_000_000
    formatted = format_openmvs_summary(report)
    assert "DensifyPointCloud" in formatted
    assert "1144 MiB" in formatted


def test_openmvs_log_summary_handles_logs_without_memory_profile(tmp_path):
    log = tmp_path / "openmvs_pipeline.log"
    log.write_text(
        "[CLEAR360 stage timing] colmap: 2.0 seconds; exit_code=0\n",
        encoding="utf-8",
    )

    report = summarize_openmvs_log(log)

    assert report["stages"][0]["peak_rss_bytes"] is None
    assert report["memory_profiled_stage_count"] == 0
    assert report["all_stages_succeeded"] is True


def test_openmvs_log_summary_marks_failed_stage(tmp_path):
    log = tmp_path / "openmvs_pipeline.log"
    log.write_text(
        "[CLEAR360 stage timing] DensifyPointCloud: 2.0 seconds; exit_code=1\n",
        encoding="utf-8",
    )

    report = summarize_openmvs_log(log)

    assert report["all_stages_succeeded"] is False
