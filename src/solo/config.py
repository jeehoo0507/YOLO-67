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
    all_components: bool = False
    split_depth: int = 0
    split_min_patches: int = 8
    split_temperature: float = 0.05
    split_max_ncut: float = 0.15
    split_min_cosine_distance: float = 0.1
    max_instances: int = 24


@dataclass(frozen=True)
class FilterConfig:
    min_box_area: float = 0.005
    max_box_area: float = 0.95
    max_aspect_ratio: float = 10.0
    duplicate_iou: float = 0.9


@dataclass(frozen=True)
class PipelineConfig:
    output_dir: str = "pseudo/coco"
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
class TrainConfig:
    run_name: str = "coco-yolo11n"
    epochs: int = 100
    image_size: int = 640
    batch_size: int = 16
    workers: int = 8
    device: str = "auto"
    seed: int = 0
    patience: int = 30
    progress_interval: float = 10.0


@dataclass(frozen=True)
class Config:
    dino: DinoConfig = field(default_factory=DinoConfig)
    maskcut: MaskCutConfig = field(default_factory=MaskCutConfig)
    filtering: FilterConfig = field(default_factory=FilterConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    def fingerprint(self, checkpoint_sha256: str) -> str:
        # Scheduling does not change annotations. Algorithm/schema changes must bump this version.
        maskcut = asdict(self.maskcut)
        enhanced = self.maskcut.all_components or self.maskcut.split_depth > 0
        if not enhanced:
            # Preserve receipts and benchmark compatibility for the unchanged baseline.
            maskcut = {k: v for k, v in maskcut.items() if k in {
                "tau", "epsilon", "max_objects", "min_mask_area", "max_mask_iou", "crf",
            }}
        payload = {
            "schema": "solo-pseudo-instances-v1" if enhanced else "solo-pseudo-v1",
            "affinity_dtype": "float64",
            "dino": {k: v for k, v in asdict(self.dino).items() if k != "device"},
            "maskcut": maskcut,
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
        "train": TrainConfig,
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
    m = c.maskcut
    if (not 0 <= m.split_depth <= 4 or m.split_min_patches < 2
            or (m.split_depth and m.max_instances < m.max_objects)):
        raise ValueError(
            "Require split_depth in 0..4, split_min_patches >=2, max_instances >= max_objects"
        )
    if not 0 < m.split_temperature <= 1 or not 0 < m.split_max_ncut < 2:
        raise ValueError("Invalid split temperature / normalized-cut threshold")
    if not 0 < m.split_min_cosine_distance <= 2:
        raise ValueError("split_min_cosine_distance must lie in (0,2]")
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
    t = c.train
    if not t.run_name or any(
        ch not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for ch in t.run_name
    ):
        raise ValueError("train.run_name must contain only lowercase letters, digits, - and _")
    if t.epochs < 1 or t.image_size < 32 or t.image_size % 32:
        raise ValueError("Training epochs must be positive; image_size must be a multiple of 32")
    if t.batch_size < 1 or t.workers < 0 or t.patience < 0 or t.progress_interval <= 0:
        raise ValueError("Invalid training batch/workers/patience/progress interval")
    if t.device not in {"auto", "cpu", "cuda"}:
        raise ValueError("Training device must be auto, cpu, or cuda")


def local_path(value: str | Path) -> Path:
    path = (ROOT / value).resolve()
    if path == ROOT or not path.is_relative_to(ROOT):
        raise ValueError(f"SOLO outputs must be below repository root: {path}")
    return path


def protect_dataset(dataset: Path, output: Path) -> None:
    if output.is_relative_to(dataset) or dataset.is_relative_to(output):
        raise ValueError(f"Input dataset and output must not overlap: {dataset}, {output}")
