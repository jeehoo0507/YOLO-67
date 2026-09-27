from dataclasses import replace

import numpy as np
import pytest

from solo.config import ROOT, DinoConfig


@pytest.mark.model
def test_pretrained_keys_match_upstream_hook_and_are_batched():
    if not (ROOT / "weights/dino_vits16.pth").is_file():
        pytest.skip("Official checkpoint not yet downloaded; ./solo benchmark prepares it")
    import torch

    from solo.dino import DinoBackbone

    model = DinoBackbone(replace(DinoConfig(), resolution=128, device="cpu"))
    assert not model.model.training
    assert not any(p.requires_grad for p in model.model.parameters())
    images = np.random.default_rng(4).normal(size=(2, 3, 128, 128)).astype(np.float32)
    actual = model.extract(images)
    captured = []
    hook = model.model.blocks[-1].attn.qkv.register_forward_hook(
        lambda _module, _inputs, output: captured.append(output)
    )
    try:
        with torch.inference_mode():
            model.model.get_last_selfattention(torch.from_numpy(images))
    finally:
        hook.remove()
    reference = captured[0].reshape(2, 65, 3, 6, 64)[:, 1:, 1].reshape(2, 64, 384).numpy()
    np.testing.assert_allclose(actual, reference, rtol=1e-6, atol=1e-6)
    assert actual.shape == (2, 64, 384)
    assert actual.dtype == np.float32
    assert model.metadata["parameters"] == 21665664
