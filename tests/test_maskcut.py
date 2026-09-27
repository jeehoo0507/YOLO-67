import numpy as np
import pytest

from solo.config import MaskCutConfig
from solo.maskcut import discover_masks


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
