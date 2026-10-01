"""Qualitative DINO/MaskCut vs official SAM 2.1 automatic-mask comparison."""
from __future__ import annotations

import gc
import tempfile
import time
import urllib.request
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from .config import Config, FilterConfig, local_path, protect_dataset
from .dataset import deterministic_subset, scan_images
from .key_preview import COLORS
from .maskcut import discover_masks
from .pseudo_labels import box_iou, masks_to_boxes
from .reporting import new_report
from .storage import atomic_write, output_lock, sha256_file, write_json
from .system_info import system_info

SAM_REVISION = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"
SAM_SHA256 = "7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69"
SAM_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"
AMG_SETTINGS = dict(
    points_per_side=16, points_per_batch=16, pred_iou_thresh=0.8,
    stability_score_thresh=0.95, box_nms_thresh=0.7, crop_n_layers=0,
    min_mask_region_area=0, output_mode="binary_mask", use_m2m=False,
)


def prepare_sam_checkpoint() -> Path:
    path = local_path("weights/sam2.1_hiera_tiny.pt")
    with output_lock(path.parent / ".sam-download.lock"):
        if path.exists() and sha256_file(path) == SAM_SHA256:
            return path
        print(f"Downloading official SAM 2.1 Hiera-Tiny to {path}", flush=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".download", delete=False) as f:
            temporary = Path(f.name)
            try:
                with urllib.request.urlopen(SAM_URL, timeout=60) as response:
                    while chunk := response.read(1024 * 1024):
                        f.write(chunk)
                f.close()
                if sha256_file(temporary) != SAM_SHA256:
                    raise RuntimeError("SAM 2.1 checkpoint checksum mismatch")
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
    return path


def filter_sam_masks(annotations: list[dict], size: tuple[int, int], config: FilterConfig):
    """Same geometric filter as DINO, highest predicted mask-IoU wins duplicates.

    SAM's predicted IoU is mask quality, not a foreground/object confidence.
    There is no box count cap or contained-part removal.
    """
    boxes, selected = [], []
    for annotation in sorted(annotations, key=lambda x: -x["predicted_iou"]):
        candidate = masks_to_boxes([annotation["segmentation"]], size, config)
        if candidate and all(box_iou(candidate[0], box) <= config.duplicate_iou for box in boxes):
            boxes.append(candidate[0])
            selected.append(annotation)
    return boxes, selected


def render_sample(original: Image.Image, name: str, dino_boxes, sam_boxes, selected):
    font = ImageFont.load_default(size=16)
    panel = Image.new("RGB", (1152, 340), "#10151e")
    for col, (title, boxes) in enumerate((
        (f"DINO v1 + MaskCut: {len(dino_boxes)}", dino_boxes),
        (f"SAM 2.1 automatic: {len(sam_boxes)}", sam_boxes),
        ("SAM retained masks", []),
    )):
        picture = ImageOps.contain(original, (368, 282))
        if col == 2:
            array = np.array(picture).astype(np.float32)
            # Large regions first, then small masks so nested masks remain visible.
            for i in sorted(range(len(selected)), key=lambda n: -selected[n]["area"]):
                mask = Image.fromarray(selected[i]["segmentation"]).resize(
                    picture.size, Image.Resampling.NEAREST,
                )
                mask = np.asarray(mask, dtype=bool)
                array[mask] = array[mask] * .45 + np.array(COLORS[i % len(COLORS)]) * .55
            picture = Image.fromarray(array.astype(np.uint8))
        draw = ImageDraw.Draw(picture)
        for i, (x, y, w, h) in enumerate(boxes):
            xy = ((x-w/2)*picture.width, (y-h/2)*picture.height,
                  (x+w/2)*picture.width, (y+h/2)*picture.height)
            color = COLORS[i % len(COLORS)]
            draw.rectangle(xy, outline=color, width=2)
            draw.text((xy[0]+2, xy[1]+2), str(i+1), fill=color, font=font)
        offset = 384*col
        ImageDraw.Draw(panel).text((offset+8, 5), title, fill="white", font=font)
        ImageDraw.Draw(panel).text((offset+8, 26), name, fill="#b9c6d8", font=font)
        panel.paste(picture, (offset+(384-picture.width)//2, 54+(282-picture.height)//2))
    return panel


def compare_sam(
    dataset: Path, config: Config, limit: int = 20, points_per_side: int = 16,
    cpu_threads: int = 4,
) -> Path:
    import torch
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2

    from .dino import DinoBackbone

    if not 1 <= limit <= 20:
        raise ValueError("compare-sam --limit must be between 1 and 20")
    if not 1 <= points_per_side <= 64 or not 1 <= cpu_threads <= 32:
        raise ValueError("Require points-per-side in 1..64 and cpu-threads in 1..32")
    # Same float32 arithmetic for both methods, including CUDA comparison runs.
    config = replace(config, dino=replace(config.dino, mixed_precision=False))
    amg_settings = dict(AMG_SETTINGS, points_per_side=points_per_side)
    dataset = dataset.resolve(strict=True)
    for destination in ("reports", "weights"):
        protect_dataset(dataset, local_path(destination))
    records = deterministic_subset(scan_images(dataset), limit, config.pipeline.seed)
    if not records:
        raise ValueError("No images found")
    directory, payload = new_report("sam-comparison", config, system_info(), dataset)
    payload["samples"] = []
    payload["timing_notes"] = (
        "One unmeasured complete warmup per method. One timed run/image, "
        f"torch threads={cpu_threads}, "
        "FP32. Common PIL decode timed separately. DINO includes its resize/normalization, "
        "feature extraction, MaskCut and boxes. SAM includes preprocessing, image encoder, "
        f"{points_per_side**2} automatic point prompts in batches of 16, decoder, AMG and "
        "shared geometric "
        "filters. Loading/download/render/report writes excluded. Not a sustained benchmark."
    )
    payload["warnings"] = [
        "Qualitative comparison; no scored ground truth, mAP, or precision/recall.",
        "Different architectures, pretraining supervision and resolutions: not a pure ViT test.",
        "SAM is mask-supervised. No target annotations or manual points/boxes are used here.",
        "AMG may retain background, nested parts and merged groups; mask IoU is not objectness.",
        "No crops, CUDA extension, connected-component cleanup or dynamic mask fallback.",
        "16x16 grid is a CPU preview setting (official default 32x32); may miss small objects.",
    ]
    try:
        dino = DinoBackbone(config.dino)
        torch.set_num_threads(cpu_threads)
        payload["dino"] = dino.metadata
        device = dino.device
        sam = build_sam2(SAM_CONFIG, str(prepare_sam_checkpoint()), device=str(device),
                         apply_postprocessing=False).eval().requires_grad_(False)
        generator = SAM2AutomaticMaskGenerator(sam, **amg_settings)
        payload["sam"] = {
            "model": "SAM 2.1 Hiera-Tiny", "parameters": sum(p.numel() for p in sam.parameters()),
            "source_revision": SAM_REVISION, "checkpoint_url": SAM_URL,
            "checkpoint_sha256": SAM_SHA256, "model_config": SAM_CONFIG,
            "resolution": sam.image_size, "device": str(device), "precision": "float32",
            "amg": amg_settings, "apply_postprocessing": False, "torch_threads": cpu_threads,
        }

        def synchronize():
            if device.type == "cuda":
                torch.cuda.synchronize(device)

        def run_dino(original):
            resized = original.resize((config.dino.resolution,)*2, Image.Resampling.LANCZOS)
            array = np.asarray(resized, dtype=np.float32)/255.0
            array = (array-np.array([.485, .456, .406], dtype=np.float32)) / np.array(
                [.229, .224, .225], dtype=np.float32,
            )
            keys = dino.extract(array.transpose(2, 0, 1).copy()[None])[0]
            return masks_to_boxes(discover_masks(keys, dino.grid, config.maskcut),
                                  original.size, config.filtering)

        print(f"Warming up DINO and SAM automatic masks ({points_per_side**2} point prompts)",
              flush=True)
        with Image.open(records[0].path) as source:
            original = source.convert("RGB")
        run_dino(original)
        with torch.inference_mode():
            generator.generate(np.array(original))
        synchronize()
        for index, record in enumerate(records):
            started = time.perf_counter()
            with Image.open(record.path) as source:
                original = source.convert("RGB")
            rgb = np.array(original)
            decode_seconds = time.perf_counter()-started
            synchronize()
            started = time.perf_counter()
            dino_boxes = run_dino(original)
            synchronize()
            dino_seconds = time.perf_counter()-started
            started = time.perf_counter()
            with torch.inference_mode():
                annotations = generator.generate(rgb)
            sam_boxes, selected = filter_sam_masks(annotations, original.size, config.filtering)
            synchronize()
            sam_seconds = time.perf_counter()-started
            name = f"sample-{index+1:02}.jpg"
            render_sample(original, record.relative, dino_boxes, sam_boxes, selected).save(
                directory / name, quality=88, optimize=True,
            )
            payload["samples"].append({
                "image": record.relative, "sha256": sha256_file(Path(record.path)),
                "visualization": name, "decode_seconds": decode_seconds,
                "dino_seconds": dino_seconds, "sam_seconds": sam_seconds,
                "dino_boxes": dino_boxes, "sam_boxes": sam_boxes,
                "sam_amg_count_before_shared_filter": len(annotations),
                "sam_mask_quality": [{"predicted_iou": a["predicted_iou"],
                                      "stability_score": a["stability_score"], "area": a["area"]}
                                     for a in selected],
            })
            write_json(directory / "comparison.json", payload)
            print(f"{index+1}/{len(records)} {record.relative}: DINO {len(dino_boxes)} boxes "
                  f"{dino_seconds:.2f}s | SAM {len(sam_boxes)} boxes {sam_seconds:.2f}s",
                  flush=True)
            del annotations, selected
            gc.collect()
        for offset in range(0, len(records), 5):
            samples = payload["samples"][offset:offset+5]
            page = Image.new("RGB", (1152, 340*len(samples)), "#10151e")
            for row, sample in enumerate(samples):
                with Image.open(directory / sample["visualization"]) as panel:
                    page.paste(panel, (0, 340*row))
            page.save(directory / f"comparison-{offset//5+1}.jpg", quality=88, optimize=True)
        payload["status"] = "complete"
    except BaseException as exc:
        payload["status"] = "failed"
        payload["errors"].append(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_json(directory / "comparison.json", payload)
        samples = payload["samples"]
        lines = ["# DINO v1/MaskCut vs SAM 2.1 Hiera-Tiny", "",
                 f"Status: {payload['status']}; images completed: {len(samples)}/{len(records)}",
                 f"Command: `{payload['command']}`", f"Source: {payload['git']}", ""]
        lines.extend(f"- {note}" for note in payload["warnings"])
        lines.extend(["", payload["timing_notes"], "",
                      "| Method | Total boxes | Mean seconds/image |",
                      "|---|---:|---:|"])
        for method in ("dino", "sam"):
            if samples:
                total = sum(len(s[f"{method}_boxes"]) for s in samples)
                mean = sum(s[f"{method}_seconds"] for s in samples)/len(samples)
                lines.append(f"| {method} | {total} | {mean:.3f} |")
        for page in sorted(directory.glob("comparison-*.jpg")):
            lines.extend(["", f"![DINO boxes, SAM boxes, SAM masks]({page.name})"])
        lines.extend(f"Error: {e}" for e in payload["errors"])
        atomic_write(directory / "summary.md", "\n".join(lines).encode())
        print(f"SAM comparison report: {directory}", flush=True)
    return directory
