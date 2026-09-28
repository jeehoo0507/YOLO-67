from __future__ import annotations

import json
import time
from pathlib import Path

from .config import Config, local_path, protect_dataset
from .dataset import deterministic_subset, scan_images
from .pipeline import PipelineFailure, run_pipeline
from .pseudo_labels import LabelStore
from .reporting import bottleneck, estimate, new_report, preview, save_report
from .storage import output_lock
from .system_info import system_info, worker_limit


def hardware_signature(system: dict) -> tuple:
    return tuple((gpu["name"], gpu["vram_bytes"]) for gpu in system["gpus"]), system["cuda_runtime"]


def load_best(config: Config, fingerprint: str, system: dict) -> dict:
    paths = [local_path("work/best_config.json")]
    paths.extend(sorted(local_path("reports").glob("*/best_config.json"), reverse=True))
    for path in paths:
        try:
            candidate = json.loads(path.read_text())
            if candidate["fingerprint"] != fingerprint or not candidate["qualified_confirmation"]:
                continue
            if hardware_signature(candidate["hardware"]) != hardware_signature(system):
                continue
            if candidate["cpu_workers"] > worker_limit(config.benchmark.max_workers):
                continue
            # Performance-sensitive scheduling / write policy must match the measured run.
            measured = candidate["configuration"]["pipeline"]
            current = config.to_dict()["pipeline"]
            if any(measured[k] != current[k] for k in ("decode_workers", "queue_batches", "fsync")):
                continue
            return candidate
        except (OSError, ValueError, KeyError, TypeError):
            continue
    raise RuntimeError(
        "No compatible successful 10k benchmark. Run ./solo benchmark DATASET first."
    )


def generate(dataset: Path, config: Config) -> Path:
    from .dino import CHECKPOINT_SHA256, DinoBackbone

    dataset = dataset.resolve(strict=True)
    output = local_path(config.pipeline.output_dir)
    for destination in (output, local_path("reports"), local_path("weights"), local_path("work")):
        protect_dataset(dataset, destination)
    system = system_info()
    fingerprint = config.fingerprint(CHECKPOINT_SHA256)
    best = load_best(config, fingerprint, system)
    directory, payload = new_report("generate", config, system, dataset)
    payload["best_config"] = best
    store = None
    result = None
    samples = []
    try:
        with output_lock(output / ".generation.lock"):
            started = time.perf_counter()
            records = scan_images(dataset)
            payload["discovery_seconds"] = time.perf_counter() - started
            payload["dataset_images"] = len(records)
            samples = deterministic_subset(
                records, config.benchmark.preview_images * 8, config.pipeline.seed
            )
            store = LabelStore(output, fingerprint, str(dataset), config.pipeline.fsync)
            backbone = DinoBackbone(config.dino)
            payload["backbone"] = backbone.metadata
            try:
                result = run_pipeline(
                    records,
                    backbone,
                    config,
                    store,
                    best["cpu_workers"],
                    best["batch_size"],
                    resume=True,
                )
            except PipelineFailure as exc:
                result = exc.result
                raise
            if result["images_failed"]:
                raise RuntimeError(f"{result['images_failed']} images failed; rerun to retry them")
            payload["status"] = "complete"
    except BaseException as exc:
        payload["status"] = "failed"
        payload["errors"].append(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if result:
            result["stage"] = "generation"
            result["repeat"] = 0
            payload["runs"].append(result)
            payload["quality"] = result["quality"]
            if result["images_processed"]:
                payload["estimate"] = estimate(result, config, payload.get("dataset_images"))
                payload["bottleneck"] = bottleneck(result)
            payload["errors"].extend(e["error"] for e in result["errors"])
        if store:
            try:
                payload["preview_samples"] = preview(
                    samples,
                    store,
                    directory / "pseudo_preview.jpg",
                    config.benchmark.preview_images,
                )
            except (OSError, ValueError) as exc:
                payload["warnings"].append(f"Preview: {exc}")
        save_report(directory, payload)
        print(f"Report: {directory}", flush=True)
        preview_path = directory / "pseudo_preview.jpg"
        if preview_path.is_file():
            print(
                f"Bounding-box preview ({payload['preview_samples']} images): {preview_path}",
                flush=True,
            )
    return directory
