"""
obstacles/clustering.py — 3D Obstacle Clustering & Accurate Vehicle Clearance
=============================================================================

Implements spatial clustering (DBSCAN) and 3D obstacle property extraction:
  - 3D centroid and spatial bounding box (AABB & dimensions: width, length, height)
  - Exact Euclidean clearance to the Hyundai Creta vehicle geometry (body + side mirrors)
  - CRITICAL: Clearance is NOT distance from vehicle origin (0, 0, 0)!
  - Semantic classification, point count, and observation quality score
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN

from scene.semantics import SemanticClass, SEMANTIC_CLASS_NAMES


@dataclass
class BoundingBox3D:
    min_point: np.ndarray  # [min_x, min_y, min_z]
    max_point: np.ndarray  # [max_x, max_y, max_z]
    dimensions: np.ndarray # [width_x, length_y, height_z]
    volume_m3: float

    def to_dict(self) -> dict:
        return {
            "min_point": self.min_point.tolist(),
            "max_point": self.max_point.tolist(),
            "dimensions": self.dimensions.tolist(),
            "volume_m3": float(self.volume_m3),
        }


@dataclass
class Obstacle3D:
    """
    Structured representation of a detected 3D obstacle.

    Attributes
    ----------
    obstacle_id           : unique integer identifier
    centroid              : (3,) float64 [X, Y, Z]
    bbox                  : BoundingBox3D
    clearance_to_vehicle_m: minimum Euclidean distance from obstacle to vehicle outer envelope
    semantic_class        : SemanticClass integer
    semantic_name         : human-readable class name
    point_count           : number of points in cluster
    confidence            : detection confidence in [0, 1]
    is_dynamic            : True if object is dynamic/moving
    """
    obstacle_id: int
    centroid: np.ndarray
    bbox: BoundingBox3D
    clearance_to_vehicle_m: float
    semantic_class: int
    semantic_name: str
    point_count: int
    confidence: float
    is_dynamic: bool = False

    def to_dict(self) -> dict:
        return {
            "obstacle_id": int(self.obstacle_id),
            "centroid": self.centroid.tolist(),
            "bbox": self.bbox.to_dict(),
            "clearance_to_vehicle_m": float(self.clearance_to_vehicle_m),
            "semantic_class": int(self.semantic_class),
            "semantic_name": self.semantic_name,
            "point_count": int(self.point_count),
            "confidence": float(self.confidence),
            "is_dynamic": bool(self.is_dynamic),
        }


class VehicleFootprint:
    """
    Composite geometric envelope of the virtual Hyundai Creta vehicle.

    Accounting for:
      - Main body: Length = 4.30 m, Width = 1.79 m, Height = 1.635 m
        (Bounds: X in [-0.895, +0.895], Y in [-2.15, +2.15], Z in [0.0, 1.635])
      - Left wing mirror: Center (-1.00, 0.40, 1.00), Extent X in [-1.05, -0.895]
      - Right wing mirror: Center (+1.00, 0.40, 1.00), Extent X in [+0.895, +1.05]
      - Overall width with mirrors = 2.10 m (X in [-1.05, +1.05])
    """

    def __init__(
        self,
        half_width: float = 0.895,
        half_length: float = 2.15,
        height: float = 1.635,
        mirror_tip_x: float = 1.05,
        mirror_y_bounds: Tuple[float, float] = (0.30, 0.50),
        mirror_z_bounds: Tuple[float, float] = (0.85, 1.15),
    ):
        self.body_min = np.array([-half_width, -half_length, 0.0], dtype=np.float64)
        self.body_max = np.array([+half_width, +half_length, height], dtype=np.float64)

        # Left mirror box (X in [-1.05, -0.895])
        self.left_mirror_min = np.array([-mirror_tip_x, mirror_y_bounds[0], mirror_z_bounds[0]], dtype=np.float64)
        self.left_mirror_max = np.array([-half_width, mirror_y_bounds[1], mirror_z_bounds[1]], dtype=np.float64)

        # Right mirror box (X in [+0.895, +1.05])
        self.right_mirror_min = np.array([+half_width, mirror_y_bounds[0], mirror_z_bounds[0]], dtype=np.float64)
        self.right_mirror_max = np.array([+mirror_tip_x, mirror_y_bounds[1], mirror_z_bounds[1]], dtype=np.float64)

        self.boxes = [
            (self.body_min, self.body_max),
            (self.left_mirror_min, self.left_mirror_max),
            (self.right_mirror_min, self.right_mirror_max),
        ]

    @staticmethod
    def _point_to_box_distance(pts: np.ndarray, b_min: np.ndarray, b_max: np.ndarray) -> np.ndarray:
        """Vectorized minimum Euclidean distance from (N, 3) points to an axis-aligned 3D box."""
        dx = np.maximum(0.0, np.maximum(b_min[0] - pts[:, 0], pts[:, 0] - b_max[0]))
        dy = np.maximum(0.0, np.maximum(b_min[1] - pts[:, 1], pts[:, 1] - b_max[1]))
        dz = np.maximum(0.0, np.maximum(b_min[2] - pts[:, 2], pts[:, 2] - b_max[2]))
        return np.sqrt(dx**2 + dy**2 + dz**2)

    def compute_clearance(self, points: np.ndarray) -> float:
        """
        Compute true minimum Euclidean distance from points to the composite vehicle envelope.
        Returns clearance in metres (0.0 if points penetrate vehicle envelope).
        """
        if len(points) == 0:
            return float("inf")

        d_body = self._point_to_box_distance(points, self.body_min, self.body_max)
        d_l_mirr = self._point_to_box_distance(points, self.left_mirror_min, self.left_mirror_max)
        d_r_mirr = self._point_to_box_distance(points, self.right_mirror_min, self.right_mirror_max)

        d_composite = np.minimum(d_body, np.minimum(d_l_mirr, d_r_mirr))
        min_clearance = float(np.min(d_composite))
        return max(0.0, min_clearance)


class ObstacleClusterer:
    """
    Extracts structured 3D obstacles from non-ground point clouds via DBSCAN.

    Parameters
    ----------
    eps : float
        DBSCAN clustering radius in metres (default 0.30 m).
    min_samples : int
        DBSCAN minimum points per cluster (default 25).
    min_cluster_points : int
        Minimum cluster size to keep (default 30).
    max_cluster_points : int
        Maximum points before splitting or classifying as boundary wall (default 100,000).
    """

    def __init__(
        self,
        eps: float = 0.30,
        min_samples: int = 25,
        min_cluster_points: int = 30,
        max_cluster_points: int = 100000,
    ):
        self.eps = eps
        self.min_samples = min_samples
        self.min_cluster_points = min_cluster_points
        self.max_cluster_points = max_cluster_points
        self.vehicle_footprint = VehicleFootprint()

    def cluster_obstacles(
        self,
        points: np.ndarray,
        labels: Optional[np.ndarray] = None,
        colors: Optional[np.ndarray] = None,
    ) -> List[Obstacle3D]:
        """
        Run DBSCAN spatial clustering and return list of Obstacle3D objects.
        """
        if len(points) < self.min_cluster_points:
            return []

        # Run DBSCAN
        db = DBSCAN(eps=self.eps, min_samples=self.min_samples, n_jobs=-1)
        cluster_labels = db.fit_predict(points)

        unique_clusters = set(cluster_labels)
        unique_clusters.discard(-1)  # Discard noise points

        obstacles: List[Obstacle3D] = []
        obs_id = 1

        for c_id in sorted(unique_clusters):
            c_mask = cluster_labels == c_id
            c_pts = points[c_mask]

            if len(c_pts) < self.min_cluster_points:
                continue

            # Bounding box & centroid
            b_min = c_pts.min(axis=0)
            b_max = c_pts.max(axis=0)
            dims = b_max - b_min
            vol = float(dims[0] * dims[1] * dims[2])
            centroid = c_pts.mean(axis=0)

            bbox = BoundingBox3D(
                min_point=b_min,
                max_point=b_max,
                dimensions=dims,
                volume_m3=vol,
            )

            # Compute true vehicle clearance
            clearance = self.vehicle_footprint.compute_clearance(c_pts)

            # Determine semantic classification
            if labels is not None:
                c_labs = labels[c_mask]
                # Majority vote
                vals, counts = np.unique(c_labs, return_counts=True)
                sem_class = int(vals[np.argmax(counts)])
            else:
                sem_class = SemanticClass.UNKNOWN

            sem_name = SEMANTIC_CLASS_NAMES.get(sem_class, "Unknown")

            # Observation confidence based on density and point count
            confidence = float(np.clip(len(c_pts) / 200.0, 0.4, 0.99))

            # Dynamic mover classification: compact obstacle in forward sector
            is_dynamic = (
                (sem_class == SemanticClass.PEDESTRIAN_CYCLIST)
                or (dims[0] < 1.2 and dims[1] < 1.2 and dims[2] < 1.8 and clearance > 1.0)
            )

            obs = Obstacle3D(
                obstacle_id=obs_id,
                centroid=centroid,
                bbox=bbox,
                clearance_to_vehicle_m=clearance,
                semantic_class=sem_class,
                semantic_name=sem_name,
                point_count=len(c_pts),
                confidence=confidence,
                is_dynamic=is_dynamic,
            )
            obstacles.append(obs)
            obs_id += 1

        # Sort obstacles by clearance (closest to vehicle first)
        obstacles.sort(key=lambda o: o.clearance_to_vehicle_m)
        return obstacles

