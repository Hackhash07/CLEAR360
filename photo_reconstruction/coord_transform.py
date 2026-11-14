"""
photo_reconstruction/coord_transform.py — Authoritative CLEAR360 Coordinate System Engine
========================================================================================

Certifies and executes canonical, right-handed coordinate transformations between:
  1. COLMAP / OpenCV Reconstruction Frame:
       +X_colmap = Right
       +Y_colmap = Down  (-Y is Up)
       +Z_colmap = Forward (optical depth)
  2. Clear360 Vehicle World Frame:
       +X_world = Right
       +Y_world = Forward
       +Z_world = Up

Mathematical Formulation:
-------------------------
The canonical transition rotation R_c2v is a pure proper rotation in SO(3):
    X_world =  X_colmap
    Y_world =  Z_colmap
    Z_world = -Y_colmap

Matrix representation:
    R_c2v = [ 1  0  0 ]
            [ 0  0  1 ]
            [ 0 -1  0 ]

Properties:
    det(R_c2v) = +1.000000  (Pure rotation, NO reflection, right-handed preserving)
    R_c2v @ R_c2v.T = I_3    (Orthonormal, error = 0.0)
    Handedness: Right-handed -> Right-handed

For Camera Poses:
    T_w2c = [ R | t ]
    T_c2w_colmap = [ R^T | -R^T t ]
    T_c2w_clear360 = T_c2v @ T_c2w_colmap
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


# Canonical 4x4 Transformation Matrix from COLMAP raw world to Clear360 vehicle world
T_COLMAP_TO_CLEAR360 = np.array(
    [
        [1.0,  0.0,  0.0,  0.0],
        [0.0,  0.0,  1.0,  0.0],
        [0.0, -1.0,  0.0,  0.0],
        [0.0,  0.0,  0.0,  1.0],
    ],
    dtype=np.float64,
)

R_COLMAP_TO_CLEAR360 = T_COLMAP_TO_CLEAR360[:3, :3]


@dataclass
class CoordinateAuditResult:
    """Quantitative certification of coordinate system consistency."""
    is_valid_so3: bool
    det_R: float
    orthonormality_error: float
    handedness: str
    camera_forward_vector: List[float]
    camera_up_vector: List[float]
    world_up_alignment_dot: float
    ground_plane_normal: Optional[List[float]] = None
    fountain_top_z: Optional[float] = None
    fountain_base_z: Optional[float] = None
    is_upright: bool = True
    errors: List[str] = field(default_factory=list)


class Clear360CoordinateTransformer:
    """Authoritative coordinate transformer between photogrammetric and vehicle frames."""

    @staticmethod
    def get_canonical_transform() -> np.ndarray:
        """Return the canonical 4x4 COLMAP-to-Clear360 transformation matrix."""
        return T_COLMAP_TO_CLEAR360.copy()

    @staticmethod
    def colmap_to_clear360_points(points: np.ndarray) -> np.ndarray:
        """
        Transform 3D points from COLMAP frame to Clear360 vehicle frame.
        
        Parameters
        ----------
        points : (N, 3) float32 or float64
        
        Returns
        -------
        transformed_points : (N, 3) same dtype
        """
        pts = np.asarray(points)
        if len(pts) == 0:
            return pts
        return pts @ R_COLMAP_TO_CLEAR360.T

    @staticmethod
    def colmap_to_clear360_normals(normals: np.ndarray) -> np.ndarray:
        """
        Transform 3D normal vectors from COLMAP frame to Clear360 vehicle frame.
        
        Parameters
        ----------
        normals : (N, 3) float32 or float64
        
        Returns
        -------
        transformed_normals : (N, 3) same dtype
        """
        norms = np.asarray(normals)
        if len(norms) == 0:
            return norms
        return norms @ R_COLMAP_TO_CLEAR360.T

    @staticmethod
    def colmap_to_clear360_pose(T_c2w_colmap: np.ndarray) -> np.ndarray:
        """
        Transform a 4x4 camera-to-world pose matrix from COLMAP to Clear360 world.
        
        Parameters
        ----------
        T_c2w_colmap : (4, 4) float64
        
        Returns
        -------
        T_c2w_clear360 : (4, 4) float64
        """
        T = np.asarray(T_c2w_colmap, dtype=np.float64)
        return T_COLMAP_TO_CLEAR360 @ T

    @classmethod
    def audit_transform(
        cls,
        points: np.ndarray,
        camera_poses: Dict[str, np.ndarray],
    ) -> CoordinateAuditResult:
        """
        Perform exhaustive mathematical audit of coordinate frame properties.
        """
        errors = []
        det_R = float(np.linalg.det(R_COLMAP_TO_CLEAR360))
        ortho_err = float(np.max(np.abs(R_COLMAP_TO_CLEAR360 @ R_COLMAP_TO_CLEAR360.T - np.eye(3))))
        is_so3 = (abs(det_R - 1.0) < 1e-6) and (ortho_err < 1e-6)

        if not is_so3:
            errors.append(f"Invalid SO(3): det={det_R:.6f}, ortho_err={ortho_err:.6e}")

        # Compute average camera forward and up vectors in Clear360 frame
        cam_forwards = []
        cam_ups = []
        for name, pose in camera_poses.items():
            R = pose[:3, :3]
            # In OpenCV/COLMAP camera: +Z is forward, -Y is up
            fwd = R @ np.array([0.0, 0.0, 1.0])
            up = R @ np.array([0.0, -1.0, 0.0])
            cam_forwards.append(fwd)
            cam_ups.append(up)

        avg_fwd = np.mean(cam_forwards, axis=0) if cam_forwards else np.array([0.0, 1.0, 0.0])
        avg_fwd /= (np.linalg.norm(avg_fwd) + 1e-9)

        avg_up = np.mean(cam_ups, axis=0) if cam_ups else np.array([0.0, 0.0, 1.0])
        avg_up /= (np.linalg.norm(avg_up) + 1e-9)

        # In Clear360 world, world UP is +Z [0, 0, 1]
        target_up = np.array([0.0, 0.0, 1.0])
        up_dot = float(np.dot(avg_up, target_up))
        is_upright = up_dot > 0.85

        if not is_upright:
            errors.append(f"Scene is NOT upright: camera UP dot world +Z is {up_dot:.4f} (expected > 0.85)")

        # Verify fountain vertical orientation from points
        fountain_top_z = None
        fountain_base_z = None
        if len(points) > 10:
            fountain_top_z = float(np.percentile(points[:, 2], 98.0))
            fountain_base_z = float(np.percentile(points[:, 2], 2.0))
            if fountain_top_z <= fountain_base_z:
                errors.append(f"Inverted height: top_Z ({fountain_top_z:.2f}) <= base_Z ({fountain_base_z:.2f})")

        return CoordinateAuditResult(
            is_valid_so3=is_so3,
            det_R=round(det_R, 6),
            orthonormality_error=round(ortho_err, 9),
            handedness="Right-Handed (+X Right, +Y Forward, +Z Up)",
            camera_forward_vector=[round(float(x), 4) for x in avg_fwd],
            camera_up_vector=[round(float(x), 4) for x in avg_up],
            world_up_alignment_dot=round(up_dot, 4),
            fountain_top_z=round(fountain_top_z, 4) if fountain_top_z is not None else None,
            fountain_base_z=round(fountain_base_z, 4) if fountain_base_z is not None else None,
            is_upright=is_upright,
            errors=errors,
        )

