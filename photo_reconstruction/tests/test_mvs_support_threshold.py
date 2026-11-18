import numpy as np
import pytest

from photo_reconstruction.m7_3_6_fusion import M736GeometricConsensusEngine


def test_default_support_threshold_keeps_samples_with_one_or_more_other_views():
    engine = M736GeometricConsensusEngine()
    mask = engine._verified_support_mask(np.array([0, 1, 2, 3]))
    np.testing.assert_array_equal(mask, [False, True, True, True])


def test_strict_support_threshold_keeps_only_samples_seen_by_two_other_views():
    engine = M736GeometricConsensusEngine(minimum_support_count=2)
    mask = engine._verified_support_mask(np.array([0, 1, 2, 3]))
    np.testing.assert_array_equal(mask, [False, False, True, True])


def test_support_threshold_rejects_values_outside_supported_modes():
    with pytest.raises(ValueError, match="must be 1 or 2"):
        M736GeometricConsensusEngine(minimum_support_count=3)
