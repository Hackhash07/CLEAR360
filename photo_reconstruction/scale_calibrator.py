"""
photo_reconstruction/scale_calibrator.py — Metric Scale Calibration Tool
========================================================================

Recovers true metric scale for unscaled Structure-from-Motion photogrammetry models.
Allows users to define a known physical reference distance (e.g. wheelbase, parking bay width,
door height, or kerb distance) between two 3D points, scaling points, meshes, and camera poses.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np


class ScaleCalibrator:
    """Recovers absolute metric scale from user reference measurements."""

    def __init__(self):
        self.scale_factor: float = 1.0
        self.reference_name: str = "Uncalibrated"
        self.target_distance_m: Optional[float] = None
        self.measured_distance_units: Optional[float] = None
        self.is_calibrated: bool = False

    def compute_scale(
        self,
        point_a: np.ndarray,
        point_b: np.ndarray,
        target_distance_metres: float,
        reference_label: str = "User Reference Distance",
    ) -> float:
        """
        Compute metric scale factor from two 3D points and target distance.

        Parameters
        ----------
        point_a, point_b : np.ndarray (3,)
            Two distinct points in arbitrary reconstruction coordinate space.
        target_distance_metres : float
            Known physical ground distance between points in metres.
        reference_label : str
            Human-readable description (e.g. "Car Wheelbase", "Parking Bay Width").

        Returns
        -------
        scale_factor : float
        """
        dist_units = float(np.linalg.norm(point_a - point_b))
        if dist_units < 1e-6:
            raise ValueError("Selected reference points are coincident (distance ~ 0).")
        if target_distance_metres <= 0:
            raise ValueError(f"Target distance must be positive, got {target_distance_metres} m.")

        self.scale_factor = float(target_distance_metres) / dist_units
        self.reference_name = str(reference_label)
        self.target_distance_m = float(target_distance_metres)
        self.measured_distance_units = dist_units
        self.is_calibrated = True

        return self.scale_factor

    def apply_scale(
        self,
        vertices: np.ndarray,
        camera_poses: Optional[Dict[str, np.ndarray]] = None,
    ) -> Tuple[np.ndarray, Optional[Dict[str, np.ndarray]]]:
        """
        Apply global metric scaling to 3D vertices and camera translation components.

        Parameters
        ----------
        vertices : np.ndarray (N, 3)
        camera_poses : Optional[Dict[str, np.ndarray]] (4x4 T_world_cam matrices)

        Returns
        -------
        scaled_vertices : np.ndarray (N, 3)
        scaled_camera_poses : Dict[str, np.ndarray]
        """
        if not self.is_calibrated or abs(self.scale_factor - 1.0) < 1e-9:
            return vertices, camera_poses

        scaled_v = vertices * self.scale_factor

        scaled_poses = None
        if camera_poses:
            scaled_poses = {}
            for name, T in camera_poses.items():
                T_scaled = T.copy()
                # Scale translation vector (t = T[:3, 3])
                T_scaled[:3, 3] *= self.scale_factor
                scaled_poses[name] = T_scaled

        return scaled_v, scaled_poses

    def get_calibration_info(self) -> Dict[str, Any]:
        """Return structured calibration provenance metadata."""
        return {
            "is_calibrated": self.is_calibrated,
            "scale_factor": round(self.scale_factor, 6),
            "reference_name": self.reference_name,
            "target_distance_m": self.target_distance_m,
            "measured_distance_units": round(self.measured_distance_units, 4) if self.measured_distance_units else None,
            "metric_accuracy_certified": bool(self.is_calibrated and self.target_distance_m is not None),
        }

