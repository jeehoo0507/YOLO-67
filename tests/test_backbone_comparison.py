import numpy as np
import pytest

from solo.backbone_comparison import ComparisonBackbone, compare_backbones
from solo.config import ROOT, Config, DinoConfig


@pytest.mark.model
def test_comparison_v1_keys_match_production_extractor():
    if not (ROOT / "weights/dino_vits16.pth").is_file():
        pytest.skip("Local DINO weights unavailable")
    from solo.dino import DinoBackbone

    comparison = ComparisonBackbone("v1", 128)
    production = DinoBackbone(DinoConfig(resolution=128, device="cpu"))
    sample = np.random.default_rng(7).normal(size=(3, 128, 128)).astype(np.float32)
    try:
        features = comparison.extract(sample)
        np.testing.assert_allclose(features["keys"], production.extract(sample[None])[0], atol=1e-6)
        assert features["tokens"].shape == (64, 384)
        assert not comparison.model.training
        assert all(not p.requires_grad for p in comparison.model.parameters())
    finally:
        comparison.close()


def test_comparison_rejects_invalid_limits_before_loading_models(tmp_path):
    with pytest.raises(ValueError, match="limit"):
        compare_backbones(tmp_path, Config(), limit=0)
