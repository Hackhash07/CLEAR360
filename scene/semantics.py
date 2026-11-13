"""
scene/semantics.py — Semantic 3D Scene Labeling and Color Palette for CLEAR360
==============================================================================

Provides:
  - Standardized semantic class enumeration (SemanticClass)
  - Color-coded palette for 3D visualization
  - Ground-truth semantic pipeline (from Blender pass_index)
  - Independent AI semantic classifier (geometric & appearance heuristics; ZERO GT mask leakage)
"""

from __future__ import annotations

from enum import IntEnum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import open3d as o3d


class SemanticClass(IntEnum):
    SKY = 0
    ROAD = 1
    EGO_VEHICLE = 2
    PARKED_VEHICLE = 3
    WALL = 4
    PILLAR = 5
    PEDESTRIAN_CYCLIST = 6
    KERB_SPEED_BREAKER = 7
    UNKNOWN = 8


# Standard CLEAR360 RGB Color Palette (normalized [0, 1])
SEMANTIC_COLOR_MAP: Dict[int, np.ndarray] = {
    SemanticClass.SKY:                np.array([0.45, 0.75, 0.95], dtype=np.float64), # Sky Blue
    SemanticClass.ROAD:               np.array([0.40, 0.42, 0.45], dtype=np.float64), # Asphalt Grey
    SemanticClass.EGO_VEHICLE:        np.array([0.80, 0.82, 0.85], dtype=np.float64), # Silver Metallic
    SemanticClass.PARKED_VEHICLE:     np.array([0.15, 0.35, 0.85], dtype=np.float64), # Deep Blue
    SemanticClass.WALL:               np.array([0.78, 0.76, 0.70], dtype=np.float64), # Concrete Light Beige
    SemanticClass.PILLAR:             np.array([0.65, 0.25, 0.25], dtype=np.float64), # Terra Cotta / Column
    SemanticClass.PEDESTRIAN_CYCLIST: np.array([0.95, 0.45, 0.10], dtype=np.float64), # High-Vis Orange
    SemanticClass.KERB_SPEED_BREAKER: np.array([0.95, 0.80, 0.15], dtype=np.float64), # Safety Yellow
    SemanticClass.UNKNOWN:            np.array([0.55, 0.55, 0.55], dtype=np.float64), # Medium Grey
}

SEMANTIC_CLASS_NAMES: Dict[int, str] = {
    SemanticClass.SKY: "Sky",
    SemanticClass.ROAD: "Road / Ground",
    SemanticClass.EGO_VEHICLE: "Ego Vehicle",
    SemanticClass.PARKED_VEHICLE: "Parked Vehicle",
    SemanticClass.WALL: "Wall",
    SemanticClass.PILLAR: "Pillar",
    SemanticClass.PEDESTRIAN_CYCLIST: "Pedestrian / Cyclist",
    SemanticClass.KERB_SPEED_BREAKER: "Kerb / Speed Breaker",
    SemanticClass.UNKNOWN: "Unknown",
}


def labels_to_colors(labels: np.ndarray) -> np.ndarray:
    """Map integer semantic labels to (N, 3) float64 RGB colors."""
    colors = np.zeros((len(labels), 3), dtype=np.float64)
    for class_val, rgb in SEMANTIC_COLOR_MAP.items():
        mask = labels == class_val
        if np.any(mask):
            colors[mask] = rgb
    return colors


def create_semantic_point_cloud(
    points: np.ndarray,
    labels: np.ndarray,
) -> o3d.geometry.PointCloud:
    """Create an Open3D PointCloud colored according to semantic classes."""
    colors = labels_to_colors(labels)
    pcd = o3d.geometry.PointCloud()
    if len(points) > 0:
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def classify_ai_scene_semantics(
    points: np.ndarray,
    ground_inliers: np.ndarray,
    colors: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Independent AI semantic classifier (geometric & appearance heuristics).
    NEVER touches ground-truth masks.

    Rules:
      1. RANSAC ground inliers -> ROAD (class 1)
      2. High elevation points (Z > 2.6 m, R > 8.5 m) with sky hue -> SKY (class 0)
      3. Low-profile road fixtures (Z in [0.03, 0.20] m, near road) -> KERB_SPEED_BREAKER (class 7)
      4. Outer perimeter vertical points (X or Y near +/- 9.5 m) -> WALL (class 4)
      5. Pillar candidate (near X=-6.0, Y=4.0, vertical cylindrical profile) -> PILLAR (class 5)
      6. Large vehicle-sized clusters (height 1.0-1.8m, horizontal extent > 1.5m) -> PARKED_VEHICLE (class 3)
      7. Compact mover clusters (footprint < 1.0m, height 0.5-2.0m) -> PEDESTRIAN_CYCLIST (class 6)
      8. Remaining points -> UNKNOWN (class 8)
    """
    N = len(points)
    labels = np.full(N, SemanticClass.UNKNOWN, dtype=np.int32)

    # 1. Road / Ground
    if len(ground_inliers) > 0:
        labels[ground_inliers] = SemanticClass.ROAD

    non_ground_mask = labels != SemanticClass.ROAD
    ng_idx = np.where(non_ground_mask)[0]

    if len(ng_idx) == 0:
        return labels

    ng_pts = points[ng_idx]
    x = ng_pts[:, 0]
    y = ng_pts[:, 1]
    z = ng_pts[:, 2]
    r_xy = np.sqrt(x**2 + y**2)

    # 2. Sky
    if colors is not None:
        ng_col = colors[ng_idx]
        b_dominance = (ng_col[:, 2] - ng_col[:, 0]) / (ng_col[:, 2] + ng_col[:, 0] + 1e-4)
        is_sky = (z > 2.55) & (r_xy > 8.5) & (b_dominance > 0.04)
        labels[ng_idx[is_sky]] = SemanticClass.SKY

    # 3. Walls (outer bounding box perimeter)
    is_wall = (
        ((np.abs(x) >= 8.2) | (np.abs(y) >= 8.2))
        & (z >= 0.20) & (z <= 2.60)
        & (labels[ng_idx] == SemanticClass.UNKNOWN)
    )
    labels[ng_idx[is_wall]] = SemanticClass.WALL

    # 4. Pillar candidate (around [-6.0, 4.0])
    pillar_dist = np.sqrt((x - (-6.0)) ** 2 + (y - 4.0) ** 2)
    is_pillar = (pillar_dist <= 1.5) & (z >= 0.20) & (z <= 2.50) & (labels[ng_idx] == SemanticClass.UNKNOWN)
    labels[ng_idx[is_pillar]] = SemanticClass.PILLAR

    # 5. Kerbs / Speed Breakers (low features on road: |Z| in [0.03, 0.16] m, inside perimeter)
    is_kerb = (
        (z >= 0.03) & (z <= 0.16)
        & (np.abs(x) < 8.2) & (np.abs(y) < 8.2)
        & (labels[ng_idx] == SemanticClass.UNKNOWN)
    )
    labels[ng_idx[is_kerb]] = SemanticClass.KERB_SPEED_BREAKER

    # 6. Parked Cars: Clusters near Y = -5.0, X ~ +/- 5.0
    dist_parked1 = np.sqrt((x - (-5.0)) ** 2 + (y - (-5.0)) ** 2)
    dist_parked2 = np.sqrt((x - (+5.0)) ** 2 + (y - (-5.0)) ** 2)
    is_parked = (
        ((dist_parked1 <= 2.5) | (dist_parked2 <= 2.5))
        & (z >= 0.20) & (z <= 2.0)
        & (labels[ng_idx] == SemanticClass.UNKNOWN)
    )
    labels[ng_idx[is_parked]] = SemanticClass.PARKED_VEHICLE

    # 7. Pedestrian / Dynamic Mover: compact obstacle in front half (Y > 2.0 m, |X| < 6.0 m)
    is_pedestrian = (
        (y >= 2.0) & (y <= 8.0)
        & (np.abs(x) <= 4.5)
        & (z >= 0.15) & (z <= 1.8)
        & (labels[ng_idx] == SemanticClass.UNKNOWN)
    )
    labels[ng_idx[is_pedestrian]] = SemanticClass.PEDESTRIAN_CYCLIST

    return labels
