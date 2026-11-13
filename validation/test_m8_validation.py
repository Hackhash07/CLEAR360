"""
validation/test_m8_validation.py — Automated Verification for CLEAR360 Milestone 8

Validates:
  1. All 10 Blender Cycles test scenario datasets exist and are complete
  2. Benchmark metrics JSON deliverable and generalization error thresholds
  3. Camera calibration generalization stability (zero-shot transfer)
  4. Occupancy grid reliability metrics (false-free vs false-occupied quantification)
  5. Robustness perturbation functions (photometric, motion blur, lens obstruction, sync jitter)
  6. CameraHealthMonitor and SceneQualityDiagnostics alert levels
  7. High-resolution screenshot deliverables
"""

import json
from pathlib import Path
import numpy as np
import pytest

from evaluation.robustness import (
    apply_brightness_change,
    apply_harsh_shadow,
    apply_motion_blur,
    apply_lens_obstruction,
    apply_calibration_perturbation,
    apply_sync_jitter,
    simulate_missing_cameras,
)
from evaluation.health_monitor import (
    CameraHealthMonitor,
    CameraHealthStatus,
    SceneQualityDiagnostics,
)

ROOT = Path(__file__).resolve().parent.parent


def test_m8_scenarios_exist_and_complete():
    """Verify that all 10 novel test scenarios were generated with all camera channels."""
    scenarios_dir = ROOT / "data" / "m8_scenarios"
    assert scenarios_dir.exists(), f"Directory not found: {scenarios_dir}"

    scenario_ids = [
        "scenario_01_narrow_street",
        "scenario_02_double_parked",
        "scenario_03_close_pillars_walls",
        "scenario_04_tight_parking_gap",
        "scenario_05_speed_breakers_kerbs",
        "scenario_06_approaching_pedestrians",
        "scenario_07_harsh_lighting_shadows",
        "scenario_08_partially_occluded",
        "scenario_09_t_junction_turn",
        "scenario_10_construction_zone",
    ]

    expected_cams = ["front", "rear", "left", "right"]

    for sid in scenario_ids:
        s_path = scenarios_dir / sid
        assert s_path.exists(), f"Scenario missing: {sid}"

        # Check metadata
        meta_file = s_path / "metadata.json"
        assert meta_file.exists(), f"Metadata missing in {sid}"
        with open(meta_file, "r") as f:
            meta = json.load(f)
            assert "obstacles" in meta
            assert "lighting" in meta

        # Check camera RGB and GT depth
        for cam in expected_cams:
            assert (s_path / "rgb" / f"{cam}.png").exists(), f"RGB missing: {sid}/{cam}"
            assert (s_path / "gt_depth" / f"{cam}.exr").exists(), f"Depth missing: {sid}/{cam}"
            assert (s_path / "gt_semantic" / f"{cam}.png").exists(), f"Semantic missing: {sid}/{cam}"

        # Check reconstructed point clouds and occupancy
        assert (s_path / "pointcloud_ai.ply").exists(), f"AI PLY missing: {sid}"
        assert (s_path / "pointcloud_gt.ply").exists(), f"GT PLY missing: {sid}"
        assert (s_path / "occupancy_ai.npz").exists(), f"AI occupancy missing: {sid}"
        assert (s_path / "occupancy_gt.npz").exists(), f"GT occupancy missing: {sid}"
        assert (s_path / "obstacles_ai.json").exists(), f"AI obstacles missing: {sid}"


def test_m8_benchmark_metrics_deliverable():
    """Verify benchmark JSON structure and quantitative accuracy thresholds."""
    bench_file = ROOT / "data" / "metadata" / "m8_generalization_benchmark.json"
    assert bench_file.exists(), f"Benchmark deliverable missing: {bench_file}"

    with open(bench_file, "r") as f:
        data = json.load(f)

    assert data["total_scenarios_evaluated"] == 10
    acc = data["overall_reconstruction_accuracy"]

    # Overall calibrated depth MAE must be < 0.80 m across all unseen scenarios
    assert acc["calibrated_depth_mae_m"] < 0.80, f"MAE too high: {acc['calibrated_depth_mae_m']}"
    # Delta 1.25 metric accuracy must be > 75%
    assert acc["calibrated_delta_1_25_pct"] > 0.75, f"Delta 1.25 too low: {acc['calibrated_delta_1_25_pct']}"
    # Vehicle clearance error must be < 0.60 m
    assert acc["vehicle_clearance_error_m"] < 0.60, f"Clearance error too high: {acc['vehicle_clearance_error_m']}"
    # Ground plane tilt error must be < 3.0 degrees
    assert acc["ground_plane_tilt_error_deg"] < 3.0, f"Tilt error too high: {acc['ground_plane_tilt_error_deg']}"

    # Per-camera calibrated MAE must be < 0.65 m for every camera
    per_cam = data["per_camera_accuracy"]
    for cam in ["front", "rear", "left", "right"]:
        assert per_cam[cam]["calibrated_mae_m"] < 0.65, f"{cam} MAE too high: {per_cam[cam]['calibrated_mae_m']}"
        # Calibrated MAE must be drastically lower than raw uncalibrated MAE (>80% reduction)
        raw_mae = per_cam[cam]["uncalibrated_raw_mae_m"]
        cal_mae = per_cam[cam]["calibrated_mae_m"]
        assert cal_mae < 0.20 * raw_mae, f"{cam} calibration did not improve MAE sufficiently"


def test_m8_occupancy_grid_reliability():
    """Verify occupancy reliability quantification."""
    bench_file = ROOT / "data" / "metadata" / "m8_generalization_benchmark.json"
    with open(bench_file, "r") as f:
        data = json.load(f)

    occ = data["occupancy_grid_reliability"]
    assert occ["voxel_resolution_m"] == 0.05
    assert occ["mean_free_recall"] > 0.60, f"Free recall too low: {occ['mean_free_recall']}"
    assert occ["total_false_free_voxels"] > 0, "False-free voxels must be quantified"
    assert occ["mean_unknown_space_pct"] > 65.0, "Unknown space must account for occlusion"


def test_m8_robustness_perturbations():
    """Verify mathematical behavior of sensor degradation perturbation functions."""
    img = np.full((100, 100, 3), 128, dtype=np.uint8)

    # 1. Brightness
    dark = apply_brightness_change(img, factor=0.5)
    assert np.mean(dark) < np.mean(img)
    bright = apply_brightness_change(img, factor=1.5)
    assert np.mean(bright) > np.mean(img)

    # 2. Shadow
    shadowed = apply_harsh_shadow(img, shadow_factor=0.3)
    assert np.mean(shadowed) < np.mean(img)

    # 3. Motion blur
    noisy = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    blurred = apply_motion_blur(noisy, kernel_size=15)
    # Variance of Laplacian decreases under blur
    var_orig = float(np.var(noisy))
    var_blur = float(np.var(blurred))
    assert var_blur <= var_orig

    # 4. Lens obstruction
    obstructed = apply_lens_obstruction(img, severity=0.5, num_splatters=5)
    assert not np.array_equal(obstructed, img)

    # 5. Calibration perturbation
    T_ext = np.eye(4, dtype=np.float32)
    T_perturbed = apply_calibration_perturbation(
        T_ext,
        roll_deg=1.5,
        pitch_deg=1.0,
        yaw_deg=-1.5,
        trans_perturb_m=(0.02, -0.01, 0.03),
    )
    assert not np.allclose(T_ext, T_perturbed)
    assert T_perturbed.shape == (4, 4)

    # 6. Sync jitter
    frames = {"front": img, "rear": img, "left": img, "right": img}
    jitter_map = {"front": 0.0, "rear": 15.0, "left": -25.0, "right": 30.0}
    jittered = apply_sync_jitter(frames, jitter_ms=jitter_map)
    assert jittered["rear"]["jitter_ms"] == 15.0
    assert jittered["right"]["jitter_ms"] == 30.0

    # 7. Missing camera
    dropped = simulate_missing_cameras(frames, dropped_cameras=["rear"])
    assert "rear" not in dropped
    assert "front" in dropped


def test_m8_camera_health_monitor():
    """Verify real-time sensor health monitoring categorization."""
    monitor = CameraHealthMonitor()

    # Nominal healthy feed
    clean_img = np.random.randint(40, 200, (360, 640, 3), dtype=np.uint8)
    rep_clean = monitor.inspect_feed(clean_img, "front", latency_ms=10.0)
    assert rep_clean.status == CameraHealthStatus.HEALTHY
    assert rep_clean.health_score > 0.70

    # Offline / missing feed
    rep_off = monitor.inspect_feed(None, "rear", latency_ms=0.0)
    assert rep_off.status == CameraHealthStatus.OFFLINE
    assert rep_off.health_score == 0.0

    # Degraded feed (pitch dark)
    dark_img = np.full((360, 640, 3), 10, dtype=np.uint8)
    rep_dark = monitor.inspect_feed(dark_img, "left", latency_ms=10.0)
    assert rep_dark.status in [CameraHealthStatus.DEGRADED, CameraHealthStatus.OFFLINE]
    assert len(rep_dark.warnings) > 0


def test_m8_scene_quality_diagnostics():
    """Verify surround system diagnostic alert levels."""
    diag = SceneQualityDiagnostics()
    clean_img = np.random.randint(40, 200, (180, 320, 3), dtype=np.uint8)

    # 1. All healthy -> NORMAL
    res_normal = diag.evaluate_scene({"front": clean_img, "rear": clean_img, "left": clean_img, "right": clean_img})
    assert res_normal.alert_level == "NORMAL"
    assert res_normal.healthy_cameras_count == 4

    # 2. 1 camera offline -> WARNING
    res_warn = diag.evaluate_scene({"front": clean_img, "rear": None, "left": clean_img, "right": clean_img})
    assert res_warn.alert_level == "WARNING"
    assert res_warn.offline_cameras_count == 1

    # 3. 2 cameras offline -> CRITICAL
    res_crit = diag.evaluate_scene({"front": clean_img, "rear": None, "left": None, "right": clean_img})
    assert res_crit.alert_level == "CRITICAL"
    assert res_crit.offline_cameras_count == 2


def test_m8_screenshot_deliverables_exist():
    """Verify that all required M8 high-resolution screenshots are generated."""
    fig_dir = ROOT / "docs" / "figures"
    expected_files = [
        "m8_viewer_unknown_space.png",
        "m8_viewer_confidence_mode.png",
        "m8_viewer_camera_health_hud.png",
        "m8_scenario_narrow_street.png",
        "m8_scenario_construction_zone.png",
    ]
    for fn in expected_files:
        fp = fig_dir / fn
        assert fp.exists(), f"Screenshot deliverable missing: {fp}"
        assert fp.stat().st_size > 50000, f"Screenshot file suspiciously small: {fp}"
