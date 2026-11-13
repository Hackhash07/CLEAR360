"""
CLEAR360 Multi-Camera Point-Cloud Fusion Module (Milestone M5).
==============================================================

Provides:
  - PointCloudFusionEngine: Rigid multi-camera point cloud merger
  - FusionResult: Container for fused 3D point clouds
  - OverlapAnalyzer: Pairwise camera overlap consistency analyzer
  - OverlapMetrics: Overlap statistics container
  - GeometricDistortionAnalyzer: Geometric integrity and bounding box evaluation
"""

from fusion.engine import (
    CAMERA_OVERLAP_COLORS,
    FusionResult,
    PointCloudFusionEngine,
)
from fusion.overlap import (
    OVERLAP_REGIONS,
    OverlapAnalyzer,
    OverlapMetrics,
)
from fusion.distortion import (
    CloudComparisonStats,
    GeometricDistortionAnalyzer,
    PillarStats,
    RoadProfileStats,
    SkyLeakageStats,
    WallStats,
)

__all__ = [
    "CAMERA_OVERLAP_COLORS",
    "FusionResult",
    "PointCloudFusionEngine",
    "OVERLAP_REGIONS",
    "OverlapAnalyzer",
    "OverlapMetrics",
    "CloudComparisonStats",
    "GeometricDistortionAnalyzer",
    "PillarStats",
    "RoadProfileStats",
    "SkyLeakageStats",
    "WallStats",
]
