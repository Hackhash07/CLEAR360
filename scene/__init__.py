"""
CLEAR360 3D Scene Graph, Filtering, and Spatial Understanding Module (Milestone M6).
===================================================================================

Provides:
  - SceneFilter: Point cloud filtering, deduplication, outlier removal, ground segmentation
  - GroundPlaneResult, FilterStats: Data structures for scene filtering
  - SemanticClass, SEMANTIC_COLOR_MAP: Standardized semantic schema and palette
  - create_semantic_point_cloud, classify_ai_scene_semantics: Semantic scene generators
"""

from scene.filter import (
    FilterStats,
    GroundPlaneResult,
    SceneFilter,
)
from scene.semantics import (
    SEMANTIC_CLASS_NAMES,
    SEMANTIC_COLOR_MAP,
    SemanticClass,
    classify_ai_scene_semantics,
    create_semantic_point_cloud,
    labels_to_colors,
)

__all__ = [
    "FilterStats",
    "GroundPlaneResult",
    "SceneFilter",
    "SemanticClass",
    "SEMANTIC_COLOR_MAP",
    "SEMANTIC_CLASS_NAMES",
    "classify_ai_scene_semantics",
    "create_semantic_point_cloud",
    "labels_to_colors",
]
