from __future__ import annotations

import csv
import hashlib
import json
import os
import shlex
import sys
import time
import uuid
from pathlib import Path

from PIL import Image

from .config import Config, local_path, protect_dataset
from .dataset import ImageRecord, scan_images
from .dino import CHECKPOINT_SHA256
from .progress import duration
from .pseudo_labels import LabelStore
from .reporting import git_context
from .storage import atomic_write, output_lock, sha256_file, write_json
from .system_info import effective_cpus, system_info


def _link(source: Path, destination: Path) -> None:
    if destination.is_symlink() and os.readlink(destination) == str(source):
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.symlink_to(source)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_dataset(dataset: Path, config: Config, records: list[ImageRecord]) -> dict:
    output = local_path(config.pipeline.output_dir)
    adapter = local_path("work/coco_yolo_dataset")
    protect_dataset(dataset, output)
    protect_dataset(dataset, adapter)
    if not output.is_dir():
        raise RuntimeError("No pseudo labels found. Run ./solo generate first.")
    store = LabelStore(output, config.fingerprint(CHECKPOINT_SHA256), str(dataset))
    if len(records) < 2:
        raise RuntimeError("YOLO training needs COCO train2017 and val2017 images")
    counts = {"train": 0, "val": 0, "zero_box": 0, "train_boxes": 0, "val_boxes": 0}
    manifests: dict[str, list[str]] = {"train": [], "val": []}
    started = time.monotonic()
    previous = started
    for index, record in enumerate(records, 1):
        boxes = store.valid_boxes(record)
        if boxes is None:
            raise RuntimeError(
                f"Missing, corrupt, or incompatible pseudo label: {record.relative}. "
                "Run ./solo generate with the same dataset and config first."
            )
        source_split = Path(record.relative).parts[0]
        if source_split not in {"train2017", "val2017"}:
            raise ValueError(f"Expected COCO train2017/ or val2017/ image: {record.relative}")
        split = "train" if source_split == "train2017" else "val"
        # Ultralytics can repair truncated JPEGs in place. Reject them before linking
        # so the original dataset stays read-only. Reject EXIF rotation to keep the
        # raw DINO and OpenCV training coordinate systems identical.
        with Image.open(record.path) as source:
            if source.getexif().get(274, 1) not in (None, 1):
                raise ValueError(f"Unsupported EXIF rotation: {record.relative}")
            if source.format == "JPEG":
                with open(record.path, "rb") as stream:
                    stream.seek(-2, 2)
                    if stream.read() != b"\xff\xd9":
                        raise ValueError(f"Truncated JPEG (left unchanged): {record.relative}")
        counts[split] += 1
        counts[split + "_boxes"] += len(boxes)
        counts["zero_box"] += not bool(boxes)
        _link(Path(record.path), adapter / "images" / split / record.relative)
        _link(store.label_path(record), adapter / "labels" / split / record.key)
        manifests[split].append(str(adapter / "images" / split / record.relative))
        now = time.monotonic()
        if now - previous >= config.train.progress_interval or index == len(records):
            rate = index / max(now - started, 1e-9)
            print(
                f"Preparing YOLO dataset: {index:,}/{len(records):,} "
                f"({index / len(records):.1%}) {rate:.1f} images/s "
                f"ETA {duration((len(records) - index) / rate)}",
                flush=True,
            )
            previous = now
    if not counts["train"] or not counts["val"]:
        raise RuntimeError("COCO train2017 and val2017 images are both required")
    if not counts["train_boxes"] or not counts["val_boxes"]:
        raise RuntimeError(
            "Both COCO splits need non-empty DINO pseudo boxes; inspect ./solo preview"
        )
    for split, paths in manifests.items():
        atomic_write(adapter / f"{split}.txt", ("\n".join(paths) + "\n").encode())
    # Upstream cache hashes sizes rather than label contents. Rebuild it after
    # checking receipts, including when changed boxes have the same byte length.
    for cache in (adapter / "labels").rglob("*.cache"):
        cache.unlink()
    yaml_path = adapter / "data.yaml"
    # JSON is valid YAML; avoids an additional YAML writer dependency.
    atomic_write(
        yaml_path,
        (
            json.dumps(
                {
                    "path": str(adapter),
                    "train": "train.txt",
                    "val": "val.txt",
                    "names": ["object"],
                },
                indent=2,
            )
            + "\n"
        ).encode(),
    )
    return {
        "path": str(yaml_path),
        **counts,
        "images": len(records),
        "preparation_seconds": time.monotonic() - started,
    }


class TrainETA:
    def __init__(self, interval: float) -> None:
        self.interval = interval
        self.started = 0.0
        self.last_print = 0.0
        self.start_epoch = 0
        self.batches = 0
        self.last_epoch = -1

    def on_start(self, trainer) -> None:
        self.started = self.last_print = time.monotonic()
        self.start_epoch = trainer.start_epoch
        print(f"YOLO11n training: epoch {self.start_epoch + 1}/{trainer.epochs}", flush=True)

    def on_batch(self, trainer) -> None:
        if trainer.epoch != self.last_epoch:
            self.last_epoch = trainer.epoch
            self.batches = 0
        self.batches += 1
        now = time.monotonic()
        if now - self.last_print >= self.interval:
            self._print(trainer, now, completed_epoch=False)

    def on_epoch(self, trainer) -> None:
        # This callback also runs once after final best-model validation.
        self._print(trainer, time.monotonic(), completed_epoch=True)

    def _print(self, trainer, now: float, completed_epoch: bool) -> None:
        total_epochs = trainer.epochs - self.start_epoch
        if total_epochs <= 0:
            return
        batches_per_epoch = max(1, len(trainer.train_loader))
        progress = trainer.epoch - self.start_epoch
        progress += 1 if completed_epoch else min(1, self.batches / batches_per_epoch)
        progress = min(total_epochs, max(0, progress))
        elapsed = max(0, now - self.started)
        eta = elapsed * (total_epochs - progress) / progress if progress > 0 else 0
        print(
            f"SOLO train: epoch {trainer.epoch + 1}/{trainer.epochs}, "
            f"batch {self.batches}/{batches_per_epoch}, "
            f"{(self.start_epoch + progress) / trainer.epochs:.1%}, "
            f"elapsed {duration(elapsed)}, ETA ~{duration(eta) if progress else '?'}",
            flush=True,
        )
        self.last_print = now


def _last_metrics(path: Path) -> dict:
    try:
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        return {key.strip(): value for key, value in rows[-1].items()} if rows else {}
    except OSError:
        return {}


def train_detector(dataset: Path, config: Config) -> Path:
    dataset = dataset.resolve(strict=True)
    output = local_path("outputs/coco-yolo11n")
    report = local_path("reports") / "coco-yolo11n-training"
    for destination in (
        output,
        report,
        local_path("work/coco_yolo_dataset"),
        local_path("weights"),
    ):
        protect_dataset(dataset, destination)
    with (
        output_lock(local_path("work/.coco-train.lock")),
        output_lock(local_path(config.pipeline.output_dir) / ".generation.lock"),
    ):
        records = scan_images(dataset)
        fingerprint = config.fingerprint(CHECKPOINT_SHA256)
        identity = hashlib.sha256(
            json.dumps(
                {
                    "schema": "solo-coco-random-v1",
                    "dataset": str(dataset),
                    "images": [(r.relative, r.size, r.mtime_ns) for r in records],
                    "fingerprint": fingerprint,
                    "train": {
                        k: v
                        for k, v in config.to_dict()["train"].items()
                        if k not in {"device", "batch_size", "workers", "progress_interval"}
                    },
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        state_path = local_path("work/coco-train-state.json")
        last = output / "weights/last.pt"
        if last.is_file() and not state_path.is_file():
            raise RuntimeError(
                "Found a YOLO checkpoint without SOLO training state. "
                "Move outputs/coco-yolo11n aside before starting a new run."
            )
        if state_path.exists():
            state = json.loads(state_path.read_text())
            if state.get("identity") != identity:
                raise RuntimeError(
                    "Existing training run uses different images or configuration. "
                    "Preserve outputs/coco-yolo11n and work/coco-train-state.json, "
                    "then remove them before starting a new run."
                )
            if state.get("status") == "complete" and (output / "weights/best.pt").is_file():
                print(f"Training already complete: {output / 'weights/best.pt'}", flush=True)
                return report
        adapter = prepare_dataset(dataset, config, records)
        import torch
        from ultralytics import YOLO

        if config.train.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA training requested but CUDA is unavailable")
        device = (
            0
            if (
                config.train.device == "cuda"
                or config.train.device == "auto"
                and torch.cuda.is_available()
            )
            else "cpu"
        )
        if isinstance(device, int):
            torch.cuda.set_device(device)
        resume = last.is_file()
        weight = str(last) if resume else "yolo11n.yaml"
        weight_sha = sha256_file(last) if resume else None
        report.mkdir(parents=True, exist_ok=True)
        payload = {
            "status": "running",
            "git": git_context(),
            "command": shlex.join(["./solo", *sys.argv[1:]]),
            "dataset": str(dataset),
            "dataset_identity": identity,
            "fingerprint": fingerprint,
            "adapter": adapter,
            "configuration": config.to_dict()["train"],
            "pseudo_configuration": {
                key: value for key, value in config.to_dict().items() if key != "train"
            },
            "device": str(device),
            "system": system_info(),
            "initial_model": str(weight),
            "initial_checkpoint_sha256": weight_sha,
            "initialization": "resume checkpoint" if resume else "random initialization",
            "resumed": resume,
            "warnings": [
                "COCO images only: COCO original boxes and class annotations are not used. "
                "YOLO starts from random weights; validation measures "
                "agreement with DINO/MaskCut pseudo boxes, not ground-truth accuracy."
            ],
        }
        write_json(state_path, {"identity": identity, "status": "running"})
        write_json(report / "training.json", payload)
        eta = TrainETA(config.train.progress_interval)
        started = time.monotonic()
        try:
            model = YOLO(str(weight))
            model.add_callback("on_train_start", eta.on_start)
            model.add_callback("on_train_batch_end", eta.on_batch)
            model.add_callback("on_fit_epoch_end", eta.on_epoch)
            model.train(
                data=adapter["path"],
                epochs=config.train.epochs,
                imgsz=config.train.image_size,
                batch=config.train.batch_size,
                workers=min(config.train.workers, effective_cpus()),
                device=device,
                amp=False,
                seed=config.train.seed,
                patience=config.train.patience,
                project=str(local_path("outputs")),
                name="coco-yolo11n",
                exist_ok=True,
                resume=resume,
                pretrained=False,
                val=True,
                plots=False,
                cache=False,
                save=True,
            )
            if not (output / "weights/best.pt").is_file() or not last.is_file():
                raise RuntimeError(
                    f"YOLO completed without expected checkpoints in {output / 'weights'}"
                )
            payload["status"] = "complete"
        except BaseException as exc:
            payload["status"] = "failed"
            payload["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            payload["elapsed_seconds"] = time.monotonic() - started
            payload["metrics"] = _last_metrics(output / "results.csv")
            if (output / "results.csv").is_file():
                atomic_write(report / "metrics.csv", (output / "results.csv").read_bytes())
            payload["best_checkpoint"] = str(output / "weights/best.pt")
            payload["last_checkpoint"] = str(last)
            write_json(report / "training.json", payload)
            hardware = ", ".join(gpu["name"] for gpu in payload["system"].get("gpus", []))
            atomic_write(
                report / "summary.md",
                (
                    f"# SOLO COCO YOLO11n object training\n\nStatus: **{payload['status']}**\n\n"
                    f"Git commit: `{payload['git']['commit_sha']}`\n\n"
                    f"Command: `{payload['command']}`\n\n"
                    f"Initialization: {payload['initialization']}\n\n"
                    f"Hardware: {hardware or 'CPU'}\n\n"
                    f"Images: {adapter['images']} ({adapter['train']} train, {adapter['val']} val; "
                    f"{adapter['zero_box']} zero-box)\n\n"
                    f"Device: {device}; epochs: {config.train.epochs}; batch: "
                    f"{config.train.batch_size}; image size: {config.train.image_size}\n\n"
                    f"Elapsed: {duration(payload['elapsed_seconds'])}\n\n"
                    f"Best checkpoint: `{payload['best_checkpoint']}`\n\n"
                    "All training labels are `0 = object` DINO/MaskCut boxes. COCO annotations "
                    "are unused. Validation metrics compare with pseudo boxes.\n\n"
                    f"Metrics: `{json.dumps(payload['metrics'])}`\n"
                    f"\nError: {payload.get('error', 'none')}\n"
                ).encode(),
            )
            print(f"Training report: {report / 'summary.md'}", flush=True)
        if payload["status"] == "complete":
            write_json(state_path, {"identity": identity, "status": "complete"})
    return report
