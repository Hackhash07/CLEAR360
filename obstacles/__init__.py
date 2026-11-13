"""
CLEAR360 3D Obstacle Detection, Clustering, Clearance, and Tracking Module (Milestone M6).
========================================================================================

Provides:
  - Obstacle3D: 3D bounding box, dimensions, semantics, clearance, and quality
  - BoundingBox3D: Spatial extent and volume container
  - VehicleFootprint: Hyundai Creta body + side mirrors geometry with true clearance calculation
  - ObstacleClusterer: DBSCAN-based spatial obstacle detector
  - DynamicObstacleTracker: Dynamic object association and velocity tracking
  - DynamicTrack: Temporal track history container
"""

from obstacles.clustering import (
    BoundingBox3D,
    Obstacle3D,
    ObstacleClusterer,
    VehicleFootprint,
)
from obstacles.tracker import (
    DynamicObstacleTracker,
    DynamicTrack,
)

__all__ = [
    "BoundingBox3D",
    "Obstacle3D",
    "ObstacleClusterer",
    "VehicleFootprint",
    "DynamicObstacleTracker",
    "DynamicTrack",
]
