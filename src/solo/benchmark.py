from __future__ import annotations

import json
import shutil
import statistics
import time
from pathlib import Path

from .config import Config, local_path, protect_dataset
from .dataset import deterministic_subset, scan_images, subset_digest
from .pipeline import PipelineFailure, run_pipeline
from .pseudo_labels import LabelStore
from .reporting import bottleneck, estimate, new_report, preview, save_report
from .storage import output_lock, write_json
from .system_info import system_info, worker_candidates, worker_limit


def choose_best(candidates: list[dict], min_stability_ratio: float) -> dict:
    eligible = [c for c in candidates if c["ok"] and c["stability_ratio"] >= min_stability_ratio]
    if not eligible:
        raise RuntimeError(
            "No stable successful candidate. Inspect report; increase scaling subset."
        )
    return max(eligible, key=lambda c: c["stable_throughput"])


def benchmark(dataset: Path, config: Config) -> Path:
    from .dino import CHECKPOINT_SHA256, DinoBackbone

    dataset = dataset.resolve(strict=True)
    for destination in ("reports", "work", "weights"):
        protect_dataset(dataset, local_path(destination))
    directory, payload = new_report("benchmark", config, system_info(), dataset)
    config_id = config.fingerprint(CHECKPOINT_SHA256)
    scratch = local_path(Path("work/benchmarks") / directory.name)
    scratch.mkdir(parents=True)
    last_store = None
    records = []
    try:
        scan_started = time.perf_counter()
        all_records = scan_images(dataset)
        payload["discovery_seconds"] = time.perf_counter() - scan_started
        payload["dataset_images"] = len(all_records)
        if dataset == local_path("data/imagenet/train"):
            download_receipt = local_path("data/imagenet/.download-complete.json")
            if download_receipt.is_file():
                try:
                    payload["dataset_source"] = json.loads(download_receipt.read_text())
                except (OSError, ValueError):
                    payload["warnings"].append("ImageNet subset receipt is unreadable")
        b = config.benchmark
        count = max(b.smoke_images, b.scaling_images, b.confirmation_images)
        records = deterministic_subset(all_records, count, config.pipeline.seed)
        if len(all_records) < count:
            payload["warnings"].append(
                f"Dataset has only {len(all_records)} images; requested {count}."
            )
        payload["subsets"] = {}
        for name, wanted in (
            ("smoke", b.smoke_images),
            ("scaling", b.scaling_images),
            ("confirmation", b.confirmation_images),
        ):
            subset = records[:wanted]
            payload["subsets"][name] = {
                "requested": wanted,
                "actual": len(subset),
                "sha256": subset_digest(subset),
            }
        # At most 10k paths in a small report; the full ImageNet manifest remains untracked.
        write_json(
            directory / "subset.json",
            {"seed": config.pipeline.seed, "relative_paths": [r.relative for r in records]},
        )
        backbone = DinoBackbone(config.dino)
        payload["backbone"] = backbone.metadata
        if backbone.device.type != "cuda":
            payload["warnings"].append(
                "CPU correctness run: this is not an A5000 throughput result."
            )
        limit = worker_limit(b.max_workers)
        counts = worker_candidates(b.worker_counts, limit)
        if not counts:
            raise ValueError(f"No requested worker count fits safe limit {limit}")
        payload["worker_limit"] = limit
        payload["candidate_workers"] = counts
        payload["candidate_batch_sizes"] = sorted(set(b.batch_sizes))

        def trial(stage: str, subset: list, workers: int, batch: int, repeat: int = 0):
            nonlocal last_store
            trial_dir = scratch / f"{stage}-w{workers}-b{batch}-r{repeat}"
            store = LabelStore(trial_dir, config_id, str(dataset), config.pipeline.fsync)
            result = None
            try:
                result = run_pipeline(
                    subset,
                    backbone,
                    config,
                    store,
                    workers,
                    batch,
                    resume=False,
                    title=f"SOLO {stage} w={workers} b={batch}",
                )
            except PipelineFailure as exc:
                result = exc.result
                payload["errors"].append(str(exc))
            result.update(
                {"stage": stage, "repeat": repeat, "subset_sha256": subset_digest(subset)}
            )
            payload["runs"].append(result)
            last_store = store
            save_report(directory, payload)
            return result, store

        # Smoke checks real decode/features/MaskCut/serialized YOLO labels and JPEG visualization.
        smoke, store = trial("smoke", records[: b.smoke_images], 1, 1)
        if smoke["status"] != "ok" or smoke["images_processed"] != len(records[: b.smoke_images]):
            raise RuntimeError("Smoke test failed; autotuning was not started")
        if not smoke["quality"]["total_boxes"]:
            raise RuntimeError(
                "Smoke produced zero boxes on every image; inspect config / features"
            )
        payload["preview_samples"] = preview(
            records, store, directory / "pseudo_preview.jpg", b.preview_images
        )
        if any(store.valid_boxes(r) is None for r in records[: b.smoke_images]):
            raise RuntimeError("Smoke label/receipt verification failed")
        shutil.rmtree(store.root)
        candidates = []
        scaling_records = records[: b.scaling_images]
        stop_batches = False
        for batch in sorted(set(b.batch_sizes)):
            for workers in counts:
                repeats = []
                for repeat in range(b.repeats):
                    result, store = trial("scaling", scaling_records, workers, batch, repeat)
                    repeats.append(result)
                    shutil.rmtree(store.root)
                    if result.get("oom"):
                        payload["warnings"].append(f"OOM at batch {batch}; larger batches skipped.")
                        stop_batches = True
                        backbone.clear_memory()
                        break
                    if result["status"] != "ok":
                        break
                rates = [r["throughput_images_per_sec"] for r in repeats]
                candidate = {
                    "cpu_workers": workers,
                    "batch_size": batch,
                    "ok": len(repeats) == b.repeats and all(r["status"] == "ok" for r in repeats),
                    "repeat_throughputs": rates,
                    "median_throughput": statistics.median(rates),
                    "stable_throughput": min(rates),
                    "stability_ratio": min(rates) / max(rates) if max(rates) else 0,
                }
                candidates.append(candidate)
                payload["candidates"] = candidates
                save_report(directory, payload)
                if stop_batches:
                    break
            if stop_batches:
                break
        best = choose_best(candidates, b.min_stability_ratio)
        payload["best_stable_throughput"] = best["stable_throughput"]
        payload["best_config"] = {
            "schema_version": 1,
            "source_run_id": directory.name,
            "cpu_workers": best["cpu_workers"],
            "batch_size": best["batch_size"],
            "fingerprint": config_id,
            "configuration": config.to_dict(),
            "hardware": payload["system"],
            "qualified_confirmation": False,
        }
        result, store = trial(
            "confirmation",
            records[: b.confirmation_images],
            best["cpu_workers"],
            best["batch_size"],
        )
        payload["estimate"] = estimate(result, config, len(all_records))
        payload["bottleneck"] = bottleneck(result)
        payload["quality"] = result["quality"]
        payload["preview_samples"] = preview(
            records, store, directory / "pseudo_preview.jpg", b.preview_images
        )
        if result["status"] != "ok":
            raise RuntimeError("Confirmation had failures; best configuration was not activated")
        qualified = len(records[: b.confirmation_images]) >= 10000 and b.repeats >= 2
        payload["best_config"]["qualified_confirmation"] = qualified
        if not qualified:
            payload["warnings"].append(
                "Reduced confirmation: <10k images or <2 repeats; "
                "full generation requires the default server benchmark."
            )
        payload["status"] = "complete"
        save_report(directory, payload)
        # Only activate configurations with a real 10k confirmation and repeat stability.
        if qualified:
            with output_lock(local_path("work/.best-config.lock")):
                write_json(local_path("work/best_config.json"), payload["best_config"])
        print_confirmation(result, payload["estimate"], b.confirmation_images)
    except BaseException as exc:
        payload["status"] = "failed"
        payload["errors"].append(f"{type(exc).__name__}: {exc}")
        if last_store is not None and last_store.root.exists():
            try:
                payload["preview_samples"] = preview(
                    records,
                    last_store,
                    directory / "pseudo_preview.jpg",
                    config.benchmark.preview_images,
                )
            except (ValueError, OSError):
                pass
        save_report(directory, payload)
        print(f"Failure report: {directory}", flush=True)
        raise
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    print(f"Report: {directory}", flush=True)
    return directory


def print_confirmation(result: dict, outcome: dict, requested: int) -> None:
    from .progress import duration

    print(f"Average throughput: {outcome['average_throughput_images_per_sec']:.2f} images/sec")
    print(f"Images processed: {result['images_processed']} / {requested}")
    print(f"Elapsed: {duration(result['elapsed_seconds'])}")
    hours = outcome["estimated_imagenet_hours"]
    dataset_hours = outcome["estimated_dataset_hours"]
    print(
        f"Estimated current dataset ({outcome['dataset_images']} images): "
        f"{f'{dataset_hours:.2f} hours' if dataset_hours is not None else 'unknown'}"
    )
    print(f"Current dataset 24h target: {outcome['dataset_24h_target']}")
    print(f"Estimated ImageNet total: {f'{hours:.2f} hours' if hours is not None else 'unknown'}")
    print(f"Full ImageNet 24h target: {outcome['24h_target']}")
