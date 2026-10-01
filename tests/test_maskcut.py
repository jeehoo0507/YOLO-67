import numpy as np
import pytest

from solo.config import MaskCutConfig
from solo.maskcut import discover_masks, refine_instances


def test_maskcut_discovers_seed_connected_object():
    features = np.zeros((8, 8, 2), dtype=np.float32)
    features[..., 0] = 1
    features[2:6, 2:6] = [0, 1]
    masks = discover_masks(features.reshape(64, 2), (8, 8), MaskCutConfig(max_objects=1))
    expected = np.zeros((8, 8), dtype=bool)
    expected[2:6, 2:6] = True
    assert len(masks) == 1
    np.testing.assert_array_equal(masks[0], expected)


def test_multiple_masks_do_not_overlap():
    features = np.random.default_rng(8).normal(size=(144, 32)).astype(np.float32)
    masks = discover_masks(features, (12, 12), MaskCutConfig(max_objects=2))
    assert 0 < len(masks) <= 2
    if len(masks) == 2:
        assert not (masks[0] & masks[1]).any()


def test_nonfinite_and_shape_are_rejected():
    with pytest.raises(ValueError, match="grid"):
        discover_masks(np.zeros((5, 2)), (4, 4), MaskCutConfig())
    with pytest.raises(ValueError, match="NaN/Inf"):
        discover_masks(np.full((16, 2), np.nan), (4, 4), MaskCutConfig())


def test_disconnected_identical_objects_are_retained_in_one_cut():
    features = np.zeros((12, 12, 2), dtype=np.float32)
    features[..., 0] = 1
    expected = []
    for col in (1, 5, 9):
        mask = np.zeros((12, 12), dtype=bool)
        mask[3:8, col:col+2] = True
        features[mask] = [0, 1]
        expected.append(mask)
    masks = discover_masks(features.reshape(144, 2), (12, 12), MaskCutConfig(
        max_objects=3, all_components=True,
    ))
    assert len(masks) == 3
    assert all(any(np.array_equal(a, b) for b in masks) for a in expected)


def test_touching_distinct_features_split_without_losing_pixels():
    mask = np.ones((8, 8), dtype=bool)
    features = np.empty((8, 8, 2))
    features[:, :4] = [1, 0]
    features[:, 4:] = [0.7, 0.7]  # Similar enough to merge at foreground tau=0.15.
    config = MaskCutConfig(split_depth=2)
    masks = refine_instances([mask], features.reshape(64, 2), config)
    assert len(masks) == 2
    assert all(m.sum() == 32 for m in masks)
    assert not (masks[0] & masks[1]).any()
    np.testing.assert_array_equal(np.logical_or.reduce(masks), mask)


def test_homogeneous_object_is_not_split_and_instance_budget_is_respected():
    from dataclasses import replace

    mask = np.ones((8, 8), dtype=bool)
    features = np.tile([1., 0.], (64, 1))
    config = MaskCutConfig(split_depth=2)
    assert len(refine_instances([mask], features, config)) == 1
    features.reshape(8, 8, 2)[:, 4:] = [0, 1]
    assert len(refine_instances([mask], features, replace(config, max_instances=1))) == 1


def test_fragmented_feature_partition_is_not_accepted():
    mask = np.ones((8, 8), dtype=bool)
    features = np.zeros((8, 8, 2))
    checker = np.indices((8, 8)).sum(axis=0) % 2 == 0
    features[checker] = [1, 0]
    features[~checker] = [0, 1]
    assert len(refine_instances([mask], features.reshape(64, 2), MaskCutConfig(split_depth=2))) == 1
