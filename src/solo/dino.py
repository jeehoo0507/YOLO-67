from __future__ import annotations

import os
import tempfile
import urllib.request
from pathlib import Path
from typing import Protocol

import numpy as np
import torch

from .config import ROOT, DinoConfig
from .storage import output_lock, sha256_file
from .vendor.vision_transformer import vit_small

CHECKPOINT_URL = (
    "https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
)
CHECKPOINT_SHA256 = "1566d50496f27f52f07fea6094fa29b2fdd6fae89da65bdd3ebc3b24ef6b7eb7"


class DenseBackbone(Protocol):
    """Replacement backbone contract; GPU ownership stays in the producer process."""

    grid: tuple[int, int]
    device: torch.device
    metadata: dict

    def extract(self, images: np.ndarray) -> np.ndarray:
        """Normalized RGB NCHW -> float32 CPU [batch, patches, channels]."""
        ...


def prepare_checkpoint(path: Path | None = None) -> Path:
    path = path or ROOT / "weights/dino_vits16.pth"
    with output_lock(path.parent / ".download.lock"):
        if path.is_file() and sha256_file(path) == CHECKPOINT_SHA256:
            return path
        print(f"Downloading official DINO ViT-S/16 to {path}", flush=True)
        fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".download")
        try:
            with os.fdopen(fd, "wb") as target:
                with urllib.request.urlopen(CHECKPOINT_URL, timeout=60) as response:
                    for chunk in iter(lambda: response.read(1024 * 1024), b""):
                        target.write(chunk)
                target.flush()
                os.fsync(target.fileno())
            if sha256_file(Path(temporary)) != CHECKPOINT_SHA256:
                raise RuntimeError("Official DINO checkpoint checksum mismatch")
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)
    return path


class DinoBackbone:
    def __init__(self, config: DinoConfig):
        self.config = config
        device = "cuda" if config.device == "auto" and torch.cuda.is_available() else config.device
        self.device = torch.device("cpu" if device == "auto" else device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; check NVIDIA driver / PyTorch")
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        checkpoint = prepare_checkpoint()
        self.model = vit_small(patch_size=16)
        self.model.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True
        )
        self.model.eval().requires_grad_(False).to(self.device)
        self.grid = (config.resolution // 16, config.resolution // 16)
        self.metadata = {
            "model": config.model,
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "patch_size": 16,
            "feature_type": "last_attention_keys",
            "frozen": True,
            "checkpoint_url": CHECKPOINT_URL,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "device": str(self.device),
            "mixed_precision": self.amp,
            "resolution": config.resolution,
            "amp_float32_retry_batches": 0,
        }

    @property
    def amp(self) -> bool:
        return self.config.mixed_precision and self.device.type == "cuda"

    def _keys(self, images: torch.Tensor) -> torch.Tensor:
        # Same qkv projection as upstream MaskCut; CLS is excluded. No per-image forward.
        tokens = self.model.prepare_tokens(images)
        for block in self.model.blocks[:-1]:
            tokens = block(tokens)
        last = self.model.blocks[-1]
        qkv = last.attn.qkv(last.norm1(tokens))
        batch, count, _ = qkv.shape
        return qkv.reshape(batch, count, 3, 384)[:, 1:, 1, :]

    @torch.inference_mode()
    def extract(self, images: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(images).to(self.device)
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.amp):
            keys = self._keys(tensor)
        if not torch.isfinite(keys).all():
            # Unsafe fp16 values get a float32 retry, never enter the CPU eigensolver.
            keys = self._keys(tensor.float())
            self.metadata["amp_float32_retry_batches"] += 1
        if not torch.isfinite(keys).all():
            raise ValueError("DINO produced non-finite features even in float32")
        return keys.float().cpu().numpy().copy()

    def warmup(self, batch_size: int) -> None:
        self.extract(
            np.zeros(
                (batch_size, 3, self.config.resolution, self.config.resolution), dtype=np.float32
            )
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def reset_peak_memory(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def peak_memory(self) -> int:
        return torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else 0

    def clear_memory(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
