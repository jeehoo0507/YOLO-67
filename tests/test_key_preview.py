import numpy as np
import pytest

from solo.key_preview import keys_to_rgb


def test_key_pca_is_finite_and_scale_invariant():
    keys = np.random.default_rng(12).normal(size=(64, 16))
    rgb, variance = keys_to_rgb(keys, (8, 8))
    scaled, _ = keys_to_rgb(keys * 7, (8, 8))
    assert rgb.shape == (8, 8, 3) and rgb.dtype == np.uint8
    np.testing.assert_allclose(rgb, scaled, atol=1)
    assert 0 < sum(variance) <= 1
    constant, _ = keys_to_rgb(np.ones((64, 16)), (8, 8))
    assert not constant.any()


def test_key_pca_rejects_nonfinite_or_incorrect_grid():
    with pytest.raises(ValueError, match="Invalid keys"):
        keys_to_rgb(np.ones((10, 16)), (8, 8))
    with pytest.raises(ValueError, match="Invalid keys"):
        keys_to_rgb(np.full((64, 16), np.nan), (8, 8))
