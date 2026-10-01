from __future__ import annotations

import csv
import io
import json
import shlex
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from .config import ROOT, Config, local_path, protect_dataset
from .dataset import ImageRecord, deterministic_subset, scan_images
from .progress import duration
from .pseudo_labels import LabelStore
from .storage import atomic_write, write_json


def git_context() -> dict:
    def git(*args):
        result = subprocess.run(
            ["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else None

    return {
        "commit_sha": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "dirty": bool(
            git("status", "--porcelain", "--untracked-files=all", "--", ".", ":(exclude)reports")
        ),
    }


def new_report(kind: str, config: Config, system: dict, dataset: Path) -> tuple[Path, dict]:
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + f"-{kind}-{uuid.uuid4().hex[:6]}"
    directory = local_path(Path("reports") / run_id)
    directory.mkdir(parents=True)
    payload = {
        "schema_version": 1,
        "run_id": run_id,
        "kind": kind,
        "status": "running",
        "created_utc": datetime.now(UTC).isoformat(),
        "git": git_context(),
        "command": shlex.join(["./solo", *sys.argv[1:]]),
        "configuration": config.to_dict(),
        "system": system,
        "dataset_root": str(dataset),
        "runs": [],
        "warnings": [],
        "errors": [],
        "timing_notes": (
            "End-to-end timed wall includes decode, DINO + host/device transfers, queue/IPC, "
            "CPU MaskCut, bbox conversion, atomic label/receipt writes and final worker drain. "
            "Pool startup, kernel warmup, discovery and resume scanning are separately recorded. "
            "Stage ms/image are summed service times across workers and overlap; do not add "
            "them as wall time. Filesystem cache is not flushed between candidates."
        ),
    }
    if payload["git"]["commit_sha"] is None or payload["git"]["dirty"]:
        payload["warnings"].append(
            "Source is uncommitted or dirty; commit code before server runs."
        )
    atomic_write(directory / "system_info.txt", (json.dumps(system, indent=2) + "\n").encode())
    return directory, payload


def estimate(result: dict, config: Config, dataset_images: int | None = None) -> dict:
    rate = result.get("throughput_images_per_sec", 0)
    return {
        "average_throughput_images_per_sec": rate,
        "dataset_images": dataset_images,
        "estimated_dataset_hours": dataset_images / rate / 3600
        if rate and dataset_images is not None
        else None,
        "dataset_24h_target": (
            "PASS" if rate >= dataset_images / 86400 else "FAIL"
        ) if dataset_images is not None else "NOT MEASURED",
        "dataset_target_images_per_sec": dataset_images / 86400
        if dataset_images is not None
        else None,
        "imagenet_images": config.benchmark.imagenet_images,
        "estimated_imagenet_hours": config.benchmark.imagenet_images / rate / 3600
        if rate
        else None,
        "24h_target": "PASS" if rate >= config.benchmark.target_images_per_sec else "FAIL",
        "target_images_per_sec": config.benchmark.target_images_per_sec,
    }


def bottleneck(result: dict) -> dict:
    seconds = result.get("timing_seconds", {})
    workers = max(1, result.get("cpu_workers", 1))
    capacities = {
        "DINO / transfers": seconds.get("dino_seconds", 0),
        "CPU MaskCut": seconds.get("maskcut_seconds", 0) / workers,
        "CPU bbox conversion": seconds.get("bbox_seconds", 0) / workers,
        "IO writes / fsync": seconds.get("write_seconds", 0) / workers,
        "decode starvation": seconds.get("decode_wait_seconds", 0),
    }
    major = max(capacities, key=capacities.get)
    elapsed = max(result.get("elapsed_seconds", 0), 1e-9)
    return {
        "major_bottleneck": major,
        "method": "Heuristic: producer wall vs aggregate worker service / workers; overlap applies",
        "equivalent_wall_seconds": capacities,
        "producer_backpressure_ratio": seconds.get("backpressure_seconds", 0) / elapsed,
        "mean_queue_latency_ms": result.get("timing_ms_per_image", {}).get("queue", 0),
    }


def preview(records: list[ImageRecord], store: LabelStore, path: Path, count: int) -> int:
    boxed = []
    unboxed = []
    # The caller passes a deterministic subset; no large per-image previews are stored.
    for record in records:
        boxes = store.valid_boxes(record)
        if boxes is None:
            continue
        if not boxes and len(unboxed) >= count:
            continue
        try:
            with Image.open(record.path) as source:
                image = source.convert("RGB")
            width, height = image.size
            draw = ImageDraw.Draw(image)
            for x, y, w, h in boxes:
                draw.rectangle(
                    (
                        (x - w / 2) * width,
                        (y - h / 2) * height,
                        (x + w / 2) * width,
                        (y + h / 2) * height,
                    ),
                    outline=(255, 72, 32),
                    width=max(2, round(width / 200)),
                )
            tile = Image.new("RGB", (192, 216), (24, 24, 24))
            tile.paste(ImageOps.pad(image, (192, 192), color=(24, 24, 24)), (0, 0))
            ImageDraw.Draw(tile).text(
                (4, 198), f"object boxes: {len(boxes)}", fill="white"
            )
            if boxes:
                boxed.append(tile)
            elif len(unboxed) < count:
                unboxed.append(tile)
            if len(boxed) >= count:
                break
        except OSError:
            continue
    samples = (boxed + unboxed)[:count]
    if not samples:
        raise ValueError("No valid pseudo-labels available for visualization")
    cols = min(8, len(samples))
    rows = (len(samples) + cols - 1) // cols
    grid = Image.new("RGB", (cols * 192, rows * 216), (24, 24, 24))
    for index, tile in enumerate(samples):
        grid.paste(tile, (index % cols * 192, index // cols * 216))
    buffer = io.BytesIO()
    grid.save(buffer, format="JPEG", quality=75, optimize=True)
    atomic_write(path, buffer.getvalue())
    return len(samples)


def create_preview(dataset: Path, config: Config) -> Path:
    from .dino import CHECKPOINT_SHA256

    dataset = dataset.resolve(strict=True)
    labels = local_path(config.pipeline.output_dir)
    name = "pseudo_preview" if config.train.run_name == "coco-yolo11n" else (
        f"pseudo_preview_{config.train.run_name}"
    )
    path = local_path(f"reports/{name}.jpg")
    protect_dataset(dataset, labels)
    protect_dataset(dataset, path.parent)
    if not labels.is_dir():
        raise RuntimeError("No pseudo labels yet. Run ./solo generate first.")
    records = scan_images(dataset)
    records = deterministic_subset(records, len(records), config.pipeline.seed)
    store = LabelStore(labels, config.fingerprint(CHECKPOINT_SHA256), str(dataset))
    try:
        count = preview(records, store, path, config.benchmark.preview_images)
    except ValueError as exc:
        raise RuntimeError(
            "No matching pseudo labels found. Run ./solo generate with this dataset and config."
        ) from exc
    print(f"Bounding-box preview ({count} images): {path}", flush=True)
    return path


def save_report(directory: Path, payload: dict) -> None:
    write_json(directory / "benchmark.json", payload)
    buffer = io.StringIO(newline="")
    names = [
        "stage",
        "repeat",
        "status",
        "cpu_workers",
        "batch_size",
        "images_processed",
        "images_failed",
        "images_skipped",
        "elapsed_seconds",
        "throughput_images_per_sec",
        "dino_ms_image",
        "maskcut_ms_image",
        "io_ms_image",
        "bbox_ms_image",
        "backpressure_seconds",
        "gpu_peak_vram_bytes",
    ]
    writer = csv.DictWriter(buffer, fieldnames=names, lineterminator="\n")
    writer.writeheader()
    for run in payload["runs"]:
        row = {name: run.get(name) for name in names}
        timings = run.get("timing_ms_per_image", {})
        row.update(
            {
                "dino_ms_image": timings.get("dino"),
                "maskcut_ms_image": timings.get("maskcut"),
                "io_ms_image": timings.get("decode", 0) + timings.get("write", 0),
                "bbox_ms_image": timings.get("bbox"),
                "backpressure_seconds": run.get("timing_seconds", {}).get("backpressure_seconds"),
            }
        )
        writer.writerow(row)
    atomic_write(directory / "benchmark.csv", buffer.getvalue().encode())
    best = payload.get("best_config")
    if best:
        write_json(directory / "best_config.json", best)
    system = payload["system"]
    outcome = payload.get("estimate", {})
    hours = outcome.get("estimated_imagenet_hours")
    dataset_hours = outcome.get("estimated_dataset_hours")
    hardware = ", ".join(gpu["name"] for gpu in system.get("gpus", [])) or "CPU only"
    sustained = outcome.get("average_throughput_images_per_sec", 0)
    major = payload.get("bottleneck", {}).get("major_bottleneck", "not measured")
    lines = [
        f"# SOLO {payload['kind']} — {payload['run_id']}",
        "",
        f"Status: **{payload['status']}**",
        "",
        f"Git commit: `{payload['git']['commit_sha']}` (dirty: {payload['git']['dirty']})",
        f"Command: `{payload['command']}`",
        "",
        f"GPU: {hardware}; CUDA runtime: {system.get('cuda_runtime')}",
        f"CPU: {system.get('cpu')} ({system.get('physical_cores')} physical / "
        f"{system.get('logical_cores')} logical / {system.get('effective_cpus')} effective cores)",
        f"RAM: {system.get('ram_total_bytes', 0) / 1024**3:.1f} GiB",
        f"Python: {system.get('python', '').splitlines()[0]}",
        f"PyTorch: {system.get('pytorch')}",
        "",
        f"DINO: {payload['configuration']['dino']}",
        f"MaskCut: {payload['configuration']['maskcut']}",
        f"Box filters: {payload['configuration']['filtering']}",
        f"Dataset source: {payload.get('dataset_source', 'external or not recorded')}",
        "",
        f"Tested worker counts: {sorted({r['cpu_workers'] for r in payload['runs']})}",
        f"Tested batch sizes: {sorted({r['batch_size'] for r in payload['runs']})}",
        f"Best parallel config: workers={best['cpu_workers']}, batch={best['batch_size']}"
        if best
        else "Best parallel config: n/a",
        "",
        f"Best stable scaling throughput: {payload.get('best_stable_throughput', 'n/a')} img/s",
        f"Average sustained throughput: {sustained:.2f} img/s",
        f"Estimated current dataset ({outcome.get('dataset_images', 'n/a')} images): "
        f"{f'{dataset_hours:.2f} hours' if dataset_hours is not None else 'n/a'}",
        f"Current dataset 24h target: **{outcome.get('dataset_24h_target', 'NOT MEASURED')}**",
        f"Estimated ImageNet total: {f'{hours:.2f} hours' if hours is not None else 'n/a'}",
        f"Full ImageNet 24h target: **{outcome.get('24h_target', 'NOT MEASURED')}**",
        "",
        f"Major bottleneck: {major}",
        "",
        "| Stage | Workers | Batch | Images | Time | img/s | Status |",
        "|---|---:|---:|---:|---|---:|---|",
    ]
    for run in payload["runs"]:
        lines.append(
            f"| {run['stage']} | {run['cpu_workers']} | {run['batch_size']} | "
            f"{run['images_processed']} | {duration(run['elapsed_seconds'])} | "
            f"{run['throughput_images_per_sec']:.2f} | {run['status']} |"
        )
    lines.extend(
        [
            "",
            "Quality statistics (last confirmation / generation run):",
            "",
            "```json",
            json.dumps(payload.get("quality", {}), indent=2),
            "```",
            "",
            f"Preview samples: {payload.get('preview_samples', 0)}",
            "",
            "![Pseudo boxes](pseudo_preview.jpg)",
            "",
            "Measurement details:",
            "",
            payload["timing_notes"],
            "",
            "Warnings/errors:",
            "",
        ]
    )
    lines.extend(f"- {message}" for message in payload["warnings"] + payload["errors"])
    if not payload["warnings"] and not payload["errors"]:
        lines.append("None.")
    atomic_write(directory / "summary.md", ("\n".join(lines) + "\n").encode())
