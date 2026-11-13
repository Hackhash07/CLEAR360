"""
evaluation/ — CLEAR360 Generalization, Reliability, Robustness & Evaluation Suite (M8)
"""

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
    CameraHealthStatus,
    CameraHealthReport,
    CameraHealthMonitor,
    SceneQualityDiagnostics,
    SceneQualityReport,
)

__all__ = [
    "apply_brightness_change",
    "apply_harsh_shadow",
    "apply_motion_blur",
    "apply_lens_obstruction",
    "apply_calibration_perturbation",
    "apply_sync_jitter",
    "simulate_missing_cameras",
    "CameraHealthStatus",
    "CameraHealthReport",
    "CameraHealthMonitor",
    "SceneQualityDiagnostics",
    "SceneQualityReport",
]

