from __future__ import annotations

import json
import multiprocessing as mp
import os
import threading
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image
from threadpoolctl import threadpool_limits
from tqdm import tqdm

from .config import Config
from .dataset import ImageRecord
from .maskcut import discover_masks
from .progress import Progress
from .pseudo_labels import LabelStore, masks_to_boxes
from .quality import QualityStats

if TYPE_CHECKING:
    from .dino import DinoBackbone

_THREAD_LIMITER = None


def initialize_worker(barrier):
    global _THREAD_LIMITER
    # Spawned CPU workers never initialize CUDA. Also limit already loaded BLAS libraries.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    _THREAD_LIMITER = threadpool_limits(limits=1)
    barrier.wait(timeout=60)


def ready_worker(_):
    return os.getpid()


def decode_image(record: ImageRecord, resolution: int) -> dict:
    started = time.perf_counter()
    try:
        with Image.open(record.path) as source:
            image = source.convert("RGB")
            size = image.size
            image = image.resize((resolution, resolution), Image.Resampling.LANCZOS)
            array = np.asarray(image, dtype=np.float32) / 255.0
        array = (array - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
            [0.229, 0.224, 0.225], dtype=np.float32
        )
        return {
            "record": record,
            "array": array.transpose(2, 0, 1).copy(),
            "size": size,
            "seconds": time.perf_counter() - started,
        }
    except Exception as exc:
        return {
            "record": record,
            "error": f"decode: {type(exc).__name__}: {exc}",
            "seconds": time.perf_counter() - started,
        }


def process_feature(
    record: ImageRecord,
    features: np.ndarray,
    size: tuple[int, int],
    grid: tuple[int, int],
    config: Config,
    store: LabelStore,
    submitted: float,
) -> dict:
    started = time.perf_counter()
    result = {"image": record.relative, "queue_seconds": max(0, started - submitted)}
    try:
        masks = discover_masks(features, grid, config.maskcut)
        after_mask = time.perf_counter()
        boxes = masks_to_boxes(masks, size, config.filtering)
        after_boxes = time.perf_counter()
        store.save(record, boxes, size)
        result.update(
            {
                "boxes": boxes,
                "size": size,
                "maskcut_seconds": after_mask - started,
                "bbox_seconds": after_boxes - after_mask,
                "write_seconds": time.perf_counter() - after_boxes,
            }
        )
    except Exception as exc:
        result["error"] = f"CPU: {type(exc).__name__}: {exc}"
    return result


class Metrics:
    def __init__(self, total: int):
        self.total = total
        self.success = self.skipped = self.failed = self.pending = 0
        self.started = time.perf_counter()
        self.timings = dict.fromkeys(
            (
                "decode_seconds",
                "dino_seconds",
                "maskcut_seconds",
                "bbox_seconds",
                "write_seconds",
                "queue_seconds",
                "backpressure_seconds",
                "decode_wait_seconds",
            ),
            0.0,
        )
        self.errors = []
        self.quality = QualityStats()
        self.lock = threading.Lock()
        self.fatal = None

    def fail(self, image: str, message: str):
        self.failed += 1
        if len(self.errors) < 100:
            self.errors.append({"image": image, "error": message[:1000]})

    def consume(self, future):
        with self.lock:
            self.pending -= 1
            try:
                result = future.result()
            except Exception as exc:
                self.fatal = f"CPU worker pool failed: {type(exc).__name__}: {exc}"
                self.fail("worker", self.fatal)
                return
            if "error" in result:
                self.fail(result["image"], result["error"])
                return
            self.success += 1
            self.quality.add(result["boxes"], result["size"])
            for key in ("maskcut_seconds", "bbox_seconds", "write_seconds", "queue_seconds"):
                self.timings[key] += result[key]

    def report(self, elapsed: float) -> dict:
        with self.lock:
            count = self.success
            return {
                "images_total": self.total,
                "images_processed": count,
                "images_skipped": self.skipped,
                "images_failed": self.failed,
                "images_completed": count + self.skipped,
                "elapsed_seconds": elapsed,
                # Recovered successes must never inflate benchmark / resume throughput.
                "throughput_images_per_sec": count / elapsed if elapsed else 0,
                "timing_seconds": self.timings.copy(),
                "timing_ms_per_image": {
                    k.removesuffix("_seconds"): 1000 * v / max(count, 1)
                    for k, v in self.timings.items()
                },
                "quality": self.quality.report(),
                "errors": self.errors.copy(),
                "errors_truncated": max(0, self.failed - len(self.errors)),
            }


class PipelineFailure(RuntimeError):
    def __init__(self, message: str, result: dict, oom: bool = False):
        super().__init__(message)
        self.result, self.oom = result, oom


def run_pipeline(
    records: list[ImageRecord],
    backbone: DinoBackbone,
    config: Config,
    store: LabelStore,
    workers: int,
    batch_size: int,
    *,
    resume: bool,
    title: str = "SOLO pseudo-label generation",
) -> dict:
    import torch

    startup = time.perf_counter()
    metrics = Metrics(len(records))
    todo = []
    recovery_started = time.perf_counter()
    for record in tqdm(records, desc="Checking resume receipts", unit="img", disable=not resume):
        boxes = store.valid_boxes(record) if resume else None
        if boxes is None:
            todo.append(record)
        else:
            metrics.skipped += 1
            # Already verified receipt includes dimensions; no source image decode needed.
            receipt = json.loads(store.receipt_path(record).read_text())
            metrics.quality.add(boxes, tuple(receipt["image_size"]))
    recovery_seconds = time.perf_counter() - recovery_started
    if not todo:
        report = metrics.report(0)
        report.update(
            {
                "status": "ok",
                "cpu_workers": workers,
                "batch_size": batch_size,
                "gpu_peak_vram_bytes": 0,
                "startup_seconds": 0,
                "recovery_seconds": recovery_seconds,
            }
        )
        return report
    context = mp.get_context("spawn")
    barrier = context.Barrier(workers + 1)
    pool = ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=initialize_worker,
        initargs=(barrier,),
    )
    decoder = ThreadPoolExecutor(
        max_workers=config.pipeline.decode_workers, thread_name_prefix="solo-decode"
    )
    pending = set()
    failure = None
    oom = False
    progress = None
    peak = 0
    try:
        # Starting workers/warming kernels is measured separately from timed image processing.
        initialized = [pool.submit(ready_worker, i) for i in range(workers)]
        barrier.wait(timeout=60)
        for future in initialized:
            future.result()
        backbone.warmup(batch_size)
        backbone.reset_peak_memory()
        startup_seconds = time.perf_counter() - startup
        metrics.started = time.perf_counter()
        progress = Progress(
            metrics,
            workers,
            batch_size,
            config.pipeline.progress_interval,
            config.pipeline.rolling_seconds,
            title,
        )
        batches = (todo[i : i + batch_size] for i in range(0, len(todo), batch_size))
        decoded = deque()

        def prefetch():
            try:
                batch = next(batches)
            except StopIteration:
                return
            decoded.append([decoder.submit(decode_image, r, config.dino.resolution) for r in batch])

        prefetch()
        prefetch()
        limit = workers + config.pipeline.queue_batches * batch_size
        while decoded:
            if metrics.fatal:
                raise RuntimeError(metrics.fatal)
            waiting = time.perf_counter()
            batch = [future.result() for future in decoded.popleft()]
            metrics.timings["decode_wait_seconds"] += time.perf_counter() - waiting
            prefetch()
            valid = []
            with metrics.lock:
                for item in batch:
                    metrics.timings["decode_seconds"] += item["seconds"]
                    if "error" in item:
                        metrics.fail(item["record"].relative, item["error"])
                    else:
                        valid.append(item)
            if not valid:
                continue
            # Backpressure before extracting the next GPU batch; bounded host feature memory.
            pending = {future for future in pending if not future.done()}
            waiting = time.perf_counter()
            while len(pending) + len(valid) > limit:
                _, pending = wait(pending, return_when="FIRST_COMPLETED")
            metrics.timings["backpressure_seconds"] += time.perf_counter() - waiting
            dino_started = time.perf_counter()
            features = backbone.extract(np.stack([item["array"] for item in valid]))
            metrics.timings["dino_seconds"] += time.perf_counter() - dino_started
            if features.ndim != 3 or features.shape[:2] != (
                len(valid),
                backbone.grid[0] * backbone.grid[1],
            ):
                raise ValueError(f"Unexpected dense DINO feature shape: {features.shape}")
            for item, feature in zip(valid, features, strict=True):
                with metrics.lock:
                    metrics.pending += 1
                future = pool.submit(
                    process_feature,
                    item["record"],
                    feature,
                    item["size"],
                    backbone.grid,
                    config,
                    store,
                    time.perf_counter(),
                )
                future.add_done_callback(metrics.consume)
                pending.add(future)
        wait(pending)
        if metrics.fatal:
            raise RuntimeError(metrics.fatal)
    except BaseException as exc:
        failure = exc
        oom = isinstance(exc, torch.cuda.OutOfMemoryError)
    finally:
        decoder.shutdown(wait=True, cancel_futures=True)
        # Drain submitted writes before reporting; receipts keep successful work recoverable.
        pool.shutdown(wait=True, cancel_futures=False)
        peak = backbone.peak_memory()
        if progress:
            progress.close()
    elapsed = time.perf_counter() - metrics.started
    report = metrics.report(elapsed)
    report.update(
        {
            "status": "failed" if failure or metrics.failed else "ok",
            "cpu_workers": workers,
            "batch_size": batch_size,
            "gpu_peak_vram_bytes": peak,
            "startup_seconds": locals().get("startup_seconds", elapsed),
            "recovery_seconds": recovery_seconds,
        }
    )
    if failure:
        report["failure"] = f"{type(failure).__name__}: {failure}"
        report["oom"] = oom
        raise PipelineFailure(report["failure"], report, oom) from failure
    return report
