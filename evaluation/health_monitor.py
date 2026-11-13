"""
evaluation/health_monitor.py — Camera Health Monitoring & Scene Quality Diagnostics (M8)
========================================================================================

Detects and quantifies sensor degradation in real time:
  - Underexposure / Overexposure / Glare
  - Motion blur (Laplacian variance)
  - Physical lens obstruction (mud/droplets dead-zones)
  - Inter-camera frame latency and synchronization jitter
  - Dropped or missing camera feeds
  - System-level reconstruction confidence & degradation alerts
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np


class CameraHealthStatus(str, Enum):
    HEALTHY = "HEALTHY"    # Normal nominal feed
    DEGRADED = "DEGRADED"  # Degraded (blur, dirt, glare, sync jitter)
    OFFLINE = "OFFLINE"    # Feed missing or unreadable


@dataclass
class CameraHealthReport:
    """Individual camera health diagnostic report."""
    camera_name: str
    status: CameraHealthStatus
    health_score: float  # [0.0, 1.0] where 1.0 is pristine
    mean_brightness: float  # [0, 255]
    contrast: float
    sharpness_score: float  # Laplacian variance
    obstruction_score: float  # [0.0, 1.0] fraction of obstructed field
    latency_ms: float
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "camera_name": self.camera_name,
            "status": self.status.value,
            "health_score": round(float(self.health_score), 3),
            "mean_brightness": round(float(self.mean_brightness), 1),
            "contrast": round(float(self.contrast), 1),
            "sharpness_score": round(float(self.sharpness_score), 1),
            "obstruction_score": round(float(self.obstruction_score), 3),
            "latency_ms": round(float(self.latency_ms), 1),
            "warnings": self.warnings,
        }


@dataclass
class SceneQualityReport:
    """Surround multi-camera system quality & reliability assessment."""
    overall_quality_score: float  # [0, 100]%
    healthy_cameras_count: int
    degraded_cameras_count: int
    offline_cameras_count: int
    alert_level: str  # "NORMAL", "ADVISORY", "WARNING", "CRITICAL"
    recommendation: str
    camera_reports: Dict[str, CameraHealthReport]

    def to_dict(self) -> dict:
        return {
            "overall_quality_score": round(float(self.overall_quality_score), 1),
            "healthy_cameras_count": self.healthy_cameras_count,
            "degraded_cameras_count": self.degraded_cameras_count,
            "offline_cameras_count": self.offline_cameras_count,
            "alert_level": self.alert_level,
            "recommendation": self.recommendation,
            "cameras": {k: v.to_dict() for k, v in self.camera_reports.items()},
        }


class CameraHealthMonitor:
    """
    Automotive real-time vision sensor health analyzer.
    """

    def __init__(
        self,
        min_brightness: float = 30.0,
        max_brightness: float = 230.0,
        min_sharpness: float = 80.0,
        max_obstruction: float = 0.18,
        max_sync_jitter_ms: float = 40.0,
    ):
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.min_sharpness = min_sharpness
        self.max_obstruction = max_obstruction
        self.max_sync_jitter_ms = max_sync_jitter_ms

    def inspect_feed(
        self,
        image: Optional[np.ndarray],
        camera_name: str,
        latency_ms: float = 0.0,
    ) -> CameraHealthReport:
        """Analyze a single camera image feed and determine its health state."""
        warnings: List[str] = []

        if image is None or image.size == 0:
            return CameraHealthReport(
                camera_name=camera_name,
                status=CameraHealthStatus.OFFLINE,
                health_score=0.0,
                mean_brightness=0.0,
                contrast=0.0,
                sharpness_score=0.0,
                obstruction_score=1.0,
                latency_ms=latency_ms,
                warnings=["Camera feed missing or disconnected"],
            )

        # Convert to grayscale if RGB
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY if image.shape[2] == 3 else cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        # 1. Photometric analysis
        mean_b = float(np.mean(gray))
        contrast = float(np.std(gray))

        if mean_b < self.min_brightness:
            warnings.append(f"Severe underexposure / darkness (mean={mean_b:.1f})")
        elif mean_b > self.max_brightness:
            warnings.append(f"Severe overexposure / direct sun glare (mean={mean_b:.1f})")

        # 2. Sharpness analysis (Laplacian variance)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = float(np.var(lap))

        if sharpness < self.min_sharpness:
            warnings.append(f"Motion blur or defocus detected (sharpness={sharpness:.1f})")

        # 3. Lens obstruction / contamination analysis
        # Downsample and look for localized low-variance or dark blobs
        small = cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA)
        local_std = cv2.blur((small.astype(np.float32) - cv2.blur(small.astype(np.float32), (15, 15))) ** 2, (15, 15))
        local_std = np.sqrt(local_std)
        # Obstructed regions have low local detail and anomalous brightness
        obstruction_mask = (local_std < 5.0) & ((small < 40) | (small > 240))
        obstruction_fraction = float(np.mean(obstruction_mask))

        if obstruction_fraction > self.max_obstruction:
            warnings.append(f"Partial lens obstruction detected ({obstruction_fraction*100:.1f}% obscured)")

        # 4. Synchronization jitter check
        if latency_ms > self.max_sync_jitter_ms:
            warnings.append(f"Hardware clock synchronization jitter ({latency_ms:.1f}ms > {self.max_sync_jitter_ms:.1f}ms)")

        # Compute unified health score [0.0, 1.0]
        score = 1.0
        if mean_b < self.min_brightness:
            score -= 0.3 * (self.min_brightness - mean_b) / self.min_brightness
        elif mean_b > self.max_brightness:
            score -= 0.3 * (mean_b - self.max_brightness) / (255 - self.max_brightness)

        if sharpness < self.min_sharpness:
            score -= 0.35 * (1.0 - min(1.0, sharpness / self.min_sharpness))

        if obstruction_fraction > self.max_obstruction:
            score -= 0.40 * min(1.0, obstruction_fraction / 0.5)

        if latency_ms > self.max_sync_jitter_ms:
            score -= 0.20 * min(1.0, latency_ms / 100.0)

        score = max(0.0, min(1.0, score))

        # Status categorization
        if score >= 0.75 and len(warnings) == 0:
            status = CameraHealthStatus.HEALTHY
        elif score >= 0.25:
            status = CameraHealthStatus.DEGRADED
        else:
            status = CameraHealthStatus.OFFLINE

        return CameraHealthReport(
            camera_name=camera_name,
            status=status,
            health_score=score,
            mean_brightness=mean_b,
            contrast=contrast,
            sharpness_score=sharpness,
            obstruction_score=obstruction_fraction,
            latency_ms=latency_ms,
            warnings=warnings,
        )


class SceneQualityDiagnostics:
    """
    Evaluates multi-camera surround reconstruction integrity and health.
    """

    def __init__(self, health_monitor: Optional[CameraHealthMonitor] = None):
        self.monitor = health_monitor or CameraHealthMonitor()

    def evaluate_scene(
        self,
        camera_images: Dict[str, Optional[np.ndarray]],
        camera_latencies: Optional[Dict[str, float]] = None,
    ) -> SceneQualityReport:
        """Run diagnostics on all 4 cameras and generate system-level assessment."""
        camera_latencies = camera_latencies or {}
        expected_cams = ["front", "rear", "left", "right"]
        reports: Dict[str, CameraHealthReport] = {}

        healthy_count = 0
        degraded_count = 0
        offline_count = 0
        scores: List[float] = []

        for cam in expected_cams:
            img = camera_images.get(cam)
            lat = camera_latencies.get(cam, 0.0)
            rep = self.monitor.inspect_feed(img, cam, lat)
            reports[cam] = rep

            if rep.status == CameraHealthStatus.HEALTHY:
                healthy_count += 1
            elif rep.status == CameraHealthStatus.DEGRADED:
                degraded_count += 1
            else:
                offline_count += 1

            scores.append(rep.health_score)

        # Overall quality score: mean of camera health weighted by importance
        # Front and rear have slightly higher safety weight in automotive surround
        weights = {"front": 0.30, "rear": 0.30, "left": 0.20, "right": 0.20}
        total_quality = sum(reports[c].health_score * weights[c] for c in expected_cams) * 100.0

        if offline_count >= 2 or total_quality < 40.0:
            alert = "CRITICAL"
            rec = "Multiple camera failure! 3D surround vision unavailable. Transition to manual control."
        elif offline_count == 1:
            offline_name = next(c for c in expected_cams if reports[c].status == CameraHealthStatus.OFFLINE)
            alert = "WARNING"
            rec = f"{offline_name.capitalize()} camera offline. Operating with degraded surround coverage."
        elif degraded_count >= 2 or total_quality < 70.0:
            alert = "ADVISORY"
            rec = "Reduced optical clarity detected (shadows/motion blur/dirt). Exercise caution near obstacles."
        else:
            alert = "NORMAL"
            rec = "All 4 surround cameras healthy and nominal. 3D reconstruction operating at full fidelity."

        return SceneQualityReport(
            overall_quality_score=total_quality,
            healthy_cameras_count=healthy_count,
            degraded_cameras_count=degraded_count,
            offline_cameras_count=offline_count,
            alert_level=alert,
            recommendation=rec,
            camera_reports=reports,
        )

