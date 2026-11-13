"""
validation/test_m7_validation.py — Formal Deliverables Validation for Milestone M7
"""

import json
from pathlib import Path
import pytest


ROOT = Path(__file__).resolve().parent.parent


def test_m7_viewer_modules_importable():
    from viewer import (
        Clear360ViewerWindow,
        Clear360Viewport,
        ViewerControlPanel,
        ViewerInfoPanel,
        VehicleModel3D,
        ViewerDataLoader,
        SceneData,
    )
    assert Clear360ViewerWindow is not None
    assert Clear360Viewport is not None
    assert ViewerControlPanel is not None
    assert ViewerInfoPanel is not None
    assert VehicleModel3D is not None
    assert ViewerDataLoader is not None


def test_m7_screenshot_deliverables_exist():
    fig_dir = ROOT / "docs" / "figures"
    expected_screenshots = [
        "m7_viewer_chase_gt_rgb.png",
        "m7_viewer_semantic_mode.png",
        "m7_viewer_occupancy_mode.png",
        "m7_viewer_obstacle_inspection.png",
        "m7_viewer_topdown_perspective.png",
        "m7_viewer_front_view.png",
        "m7_viewer_calibrated_ai.png",
    ]
    for s in expected_screenshots:
        p = fig_dir / s
        assert p.exists(), f"Missing screenshot deliverable: {p}"
        assert p.stat().st_size > 50000, f"Screenshot {p} unexpectedly small: {p.stat().st_size} bytes"


def test_m7_performance_benchmark_deliverable():
    bench_path = ROOT / "data" / "metadata" / "m7_performance_benchmark.json"
    assert bench_path.exists(), "Benchmark results JSON missing"
    with open(bench_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["effective_fps"] >= 60.0, f"Benchmark failed: FPS {data['effective_fps']} < 60.0"
    assert data["target_met_60fps"] is True
    assert data["avg_frame_time_ms"] < 16.67
    assert data["total_frames_tested"] == 100


def test_m7_report_deliverable_exists():
    report_path = ROOT / "docs" / "m7_surround_viewer_report.md"
    assert report_path.exists(), "M7 report missing"
    content = report_path.read_text(encoding="utf-8")
    assert "M7" in content or "Milestone 7" in content
    assert "Hyundai Creta" in content
    assert "FPS" in content
    assert "Clearance" in content

