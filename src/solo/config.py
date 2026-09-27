from __future__ import annotations

import hashlib
import json
import os
import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

ROOT = Path(os.environ.get("SOLO_ROOT", Path(__file__).resolve().parents[2])).resolve()


@dataclass(frozen=True)
class DinoConfig:
    model: str = "dino_vits16"
    resolution: int = 384
    device: str = "auto"
    mixed_precision: bool = True


@dataclass(frozen=True)
class MaskCutConfig:
    tau: float = 0.15
    epsilon: float = 1e-5
    max_objects: int = 2
    min_mask_area: float = 0.01
    max_mask_iou: float = 0.5
    crf: bool = False


@dataclass(frozen=True)
class FilterConfig:
    min_box_area: float = 0.005
    max_box_area: float = 0.95
    max_aspect_ratio: float = 10.0
    duplicate_iou: float = 0.9


@dataclass(frozen=True)
class PipelineConfig:
    output_dir: str = "pseudo/imagenet"
    decode_workers: int = 4
    queue_batches: int = 2
    progress_interval: float = 1.0
    rolling_seconds: float = 30.0
    fsync: bool = True
    seed: int = 0


@dataclass(frozen=True)
class BenchmarkConfig:
    smoke_images: int = 100
    scaling_images: int = 1000
    confirmation_images: int = 10000
    repeats: int = 2
    min_stability_ratio: float = 0.85
    batch_sizes: list[int] = field(default_factory=lambda: [8, 16, 32, 64, 128])
    worker_counts: list[int] = field(default_factory=list)
    max_workers: int = 0
    preview_images: int = 64
    imagenet_images: int = 1281167
    target_images_per_sec: float = 14.83


@dataclass(frozen=True)
class Config:
    dino: DinoConfig = field(default_factory=DinoConfig)
    maskcut: MaskCutConfig = field(default_factory=MaskCutConfig)
    filtering: FilterConfig = field(default_factory=FilterConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    def fingerprint(self, checkpoint_sha256: str) -> str:
        # Scheduling does not change annotations. Algorithm/schema changes must bump this version.
        payload = {
            "schema": "solo-pseudo-v1",
            "affinity_dtype": "float64",
            "dino": {k: v for k, v in asdict(self.dino).items() if k != "device"},
            "maskcut": asdict(self.maskcut),
            "filtering": asdict(self.filtering),
            "checkpoint_sha256": checkpoint_sha256,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def load_config(path: Path | None = None) -> Config:
    path = path or ROOT / "configs/baseline.toml"
    with path.open("rb") as stream:
        raw = tomllib.load(stream)
    sections = {
        "dino": DinoConfig,
        "maskcut": MaskCutConfig,
        "filtering": FilterConfig,
        "pipeline": PipelineConfig,
        "benchmark": BenchmarkConfig,
    }
    unknown = raw.keys() - sections.keys()
    if unknown:
        raise ValueError(f"Unknown config sections: {sorted(unknown)}")
    values = {}
    for name, cls in sections.items():
        section = raw.get(name, {})
        unknown = section.keys() - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown {name} settings: {sorted(unknown)}")
        values[name] = cls(**section)
    config = Config(**values)
    validate(config)
    return config


def validate(c: Config) -> None:
    if c.dino.model != "dino_vits16" or c.dino.device not in {"auto", "cpu", "cuda"}:
        raise ValueError("Milestone 1 supports dino_vits16 with auto/cpu/cuda only")
    if c.dino.resolution < 32 or c.dino.resolution % 16:
        raise ValueError("DINO resolution must be >=32 and divisible by patch size 16")
    if c.maskcut.crf:
        raise ValueError("CRF is intentionally disabled in this bounding-box baseline")
    if not -1 <= c.maskcut.tau < 1 or not 0 < c.maskcut.epsilon < 1:
        raise ValueError("Invalid MaskCut graph parameters")
    if c.maskcut.max_objects < 1:
        raise ValueError("max_objects must be positive")
    for value in (c.maskcut.min_mask_area, c.maskcut.max_mask_iou, c.filtering.duplicate_iou):
        if not 0 <= value <= 1:
            raise ValueError("Area/IoU ratios must lie in [0,1]")
    if not 0 <= c.filtering.min_box_area < c.filtering.max_box_area <= 1:
        raise ValueError("Require 0 <= min_box_area < max_box_area <= 1")
    if c.filtering.max_aspect_ratio < 1:
        raise ValueError("max_aspect_ratio must be >= 1")
    if c.pipeline.decode_workers < 1 or c.pipeline.queue_batches < 1:
        raise ValueError("Decode workers and queue_batches must be positive")
    if c.pipeline.progress_interval <= 0 or c.pipeline.rolling_seconds <= 0:
        raise ValueError("Progress timing must be positive")
    b = c.benchmark
    if min(b.smoke_images, b.scaling_images, b.confirmation_images, b.repeats) < 1:
        raise ValueError("Benchmark image counts and repeats must be positive")
    if not b.batch_sizes or any(x < 1 for x in b.batch_sizes + b.worker_counts):
        raise ValueError("Batch sizes / worker counts must be positive")
    if not 0 < b.min_stability_ratio <= 1 or b.max_workers < 0:
        raise ValueError("Invalid stability ratio / max_workers")
    if not 1 <= b.preview_images <= 100:
        raise ValueError("preview_images must be in [1,100]")
    if b.imagenet_images < 1 or b.target_images_per_sec <= 0:
        raise ValueError("Invalid full-dataset estimate settings")
    local_path(c.pipeline.output_dir)


def local_path(value: str | Path) -> Path:
    path = (ROOT / value).resolve()
    if path == ROOT or not path.is_relative_to(ROOT):
        raise ValueError(f"SOLO outputs must be below repository root: {path}")
    return path


def protect_dataset(dataset: Path, output: Path) -> None:
    if output.is_relative_to(dataset) or dataset.is_relative_to(output):
        raise ValueError(f"Input dataset and output must not overlap: {dataset}, {output}")
