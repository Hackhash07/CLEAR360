"""
scene/tests/test_semantics.py — Unit Tests for 3D Semantic Mapping
==================================================================

Tests:
  - Standardized semantic class schema
  - Color palette mapping
  - Independent AI heuristic semantic classification (no ground-truth mask inputs)
  - Strict pipeline separation
"""

import numpy as np
import pytest

from scene.semantics import (
    SEMANTIC_CLASS_NAMES,
    SEMANTIC_COLOR_MAP,
    SemanticClass,
    classify_ai_scene_semantics,
    create_semantic_point_cloud,
    labels_to_colors,
)


def test_semantic_classes_defined():
    """Verify all 8 mandatory semantic classes are defined."""
    required = [
        SemanticClass.ROAD,
        SemanticClass.PARKED_VEHICLE,
        SemanticClass.WALL,
        SemanticClass.PILLAR,
        SemanticClass.PEDESTRIAN_CYCLIST,
        SemanticClass.KERB_SPEED_BREAKER,
        SemanticClass.SKY,
        SemanticClass.UNKNOWN,
    ]
    for c in required:
        assert c in SEMANTIC_COLOR_MAP
        assert c in SEMANTIC_CLASS_NAMES
        col = SEMANTIC_COLOR_MAP[c]
        assert col.shape == (3,)
        assert np.all(col >= 0.0) and np.all(col <= 1.0)


def test_labels_to_colors_mapping():
    """Verify mapping of label arrays to valid RGB colors."""
    labels = np.array([SemanticClass.ROAD, SemanticClass.WALL, SemanticClass.PEDESTRIAN_CYCLIST])
    colors = labels_to_colors(labels)
    assert colors.shape == (3, 3)
    assert np.allclose(colors[0], SEMANTIC_COLOR_MAP[SemanticClass.ROAD])
    assert np.allclose(colors[1], SEMANTIC_COLOR_MAP[SemanticClass.WALL])
    assert np.allclose(colors[2], SEMANTIC_COLOR_MAP[SemanticClass.PEDESTRIAN_CYCLIST])


def test_ai_semantic_classifier_independence():
    """Verify that AI semantic classifier assigns labels without GT masks."""
    # Synthetic scene points
    road_pts = np.array([[x, y, 0.0] for x in np.linspace(-2, 2, 10) for y in np.linspace(-2, 2, 10)])
    wall_pts = np.array([[9.5, y, z] for y in np.linspace(-3, 3, 10) for z in np.linspace(0.5, 2.0, 10)])
    mover_pts = np.array([[0.0, 5.0, z] for z in np.linspace(0.25, 1.5, 10)])

    all_pts = np.vstack([road_pts, wall_pts, mover_pts])
    ground_inliers = np.arange(len(road_pts))

    labels = classify_ai_scene_semantics(all_pts, ground_inliers=ground_inliers)

    assert len(labels) == len(all_pts)
    # Road points must be labeled ROAD
    assert np.all(labels[:len(road_pts)] == SemanticClass.ROAD)
    # Wall points near X=9.5 must be labeled WALL
    wall_start = len(road_pts)
    wall_end = wall_start + len(wall_pts)
    assert np.all(labels[wall_start:wall_end] == SemanticClass.WALL)
    # Mover in front sector must be labeled PEDESTRIAN_CYCLIST
    assert np.all(labels[wall_end:] == SemanticClass.PEDESTRIAN_CYCLIST)
