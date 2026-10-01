"""Visualize the exact DINO keys used by MaskCut, without changing training labels."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from .config import Config, local_path, protect_dataset
from .dataset import deterministic_subset, scan_images
from .maskcut import discover_masks
from .pipeline import decode_image
from .pseudo_labels import masks_to_boxes
from .reporting import new_report
from .storage import atomic_write, sha256_file, write_json
from .system_info import system_info

COLORS = [
    (255, 90, 70), (50, 225, 140), (85, 150, 255), (255, 205, 60),
    (225, 95, 225), (40, 210, 220), (255, 150, 50), (165, 130, 255),
    (140, 215, 75), (245, 120, 170), (125, 180, 200), (225, 215, 160),
]


def keys_to_rgb(keys: np.ndarray, grid: tuple[int, int]) -> tuple[np.ndarray, list[float]]:
    """Per-image PCA of L2-normalized keys; visualization only, never mask input."""
    values = np.asarray(keys, dtype=np.float64)
    if values.ndim != 2 or len(values) != grid[0] * grid[1] or not np.isfinite(values).all():
        raise ValueError("Invalid keys/grid for PCA visualization")
    values = values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
    values -= values.mean(axis=0)
    u, singular, components = np.linalg.svd(values, full_matrices=False)
    count = min(3, len(singular))
    scores = u[:, :count] * singular[:count]
    for axis in range(count):
        if components[axis, np.argmax(np.abs(components[axis]))] < 0:
            scores[:, axis] *= -1
    low, high = np.percentile(scores, [2, 98], axis=0)
    rgb = np.zeros((len(keys), 3))
    rgb[:, :count] = np.clip((scores-low) / np.maximum(high-low, 1e-12), 0, 1)
    variance = singular[:count] ** 2 / max(float(np.sum(singular**2)), 1e-12)
    return (rgb.reshape(*grid, 3) * 255).astype(np.uint8), variance.tolist()


def key_preview(dataset: Path, config: Config, limit: int = 5) -> Path:
    from .dino import DinoBackbone

    if not 1 <= limit <= 20:
        raise ValueError("key-preview --limit must be between 1 and 20")
    dataset = dataset.resolve(strict=True)
    for destination in ("reports", "weights"):
        protect_dataset(dataset, local_path(destination))
    records = deterministic_subset(scan_images(dataset), limit, config.pipeline.seed)
    directory, payload = new_report("key-preview", config, system_info(), dataset)
    payload["samples"] = []
    payload["notes"] = [
        "DINO v1 ViT-S/16, frozen. Last-block key projection, excluding CLS.",
        "PCA RGB shows only three components of normalized 384D keys. Colors are per-image, "
        "not classes, object IDs, attention weights, or probabilities.",
        "MaskCut consumes all 384 key channels, NOT the PCA RGB image.",
        "Mask colors match retained box IDs; gray masks were filtered out. "
        "Different images may reuse colors. Boxes are class 0 = object, not YOLO predictions.",
        "Square-resized model input maps back to original coordinates; no crop DINO calls.",
        "Qualitative examples only; no instance ground truth or recognition accuracy measured.",
    ]
    font = ImageFont.load_default(size=18)
    small = ImageFont.load_default(size=14)
    canvas = Image.new("RGB", (1536, 54 + len(records)*344 + 54), "#10151e")
    draw = ImageDraw.Draw(canvas)
    draw.text((16, 12), f"DINO v1 ViT-S/16 | {config.dino.resolution}px | frozen | "
              "original > key features > MaskCut > object boxes", font=font, fill="white")
    try:
        backbone = DinoBackbone(config.dino)
        payload["backbone"] = backbone.metadata
        for index, record in enumerate(records):
            started = time.perf_counter()
            decoded = decode_image(record, config.dino.resolution)
            if "error" in decoded:
                raise RuntimeError(decoded["error"])
            keys = backbone.extract(decoded["array"][None])[0]
            masks = discover_masks(keys, backbone.grid, config.maskcut)
            boxes = masks_to_boxes(masks, decoded["size"], config.filtering)
            rgb, variance = keys_to_rgb(keys, backbone.grid)
            with Image.open(record.path) as source:
                original = source.convert("RGB")
            key_image = Image.fromarray(rgb).resize(original.size, Image.Resampling.NEAREST)
            mask_rgb = np.zeros((*backbone.grid, 3), dtype=np.uint8)
            for mask in masks:
                box = masks_to_boxes([mask], decoded["size"], config.filtering)
                color = COLORS[boxes.index(box[0]) % len(COLORS)] if box and box[0] in boxes else (
                    95, 95, 95
                )
                mask_rgb[mask] = color
            mask_image = Image.fromarray(mask_rgb).resize(original.size, Image.Resampling.NEAREST)
            panel = Image.new("RGB", (1536, 344), "#10151e")
            for col, (label, picture) in enumerate(zip(
                (f"{index+1}. {Path(record.relative).name}", "Key map: PCA RGB",
                 f"MaskCut: {len(masks)} masks", f"Filtered: {len(boxes)} object boxes"),
                (original, key_image, mask_image, original), strict=True,
            )):
                tile = Image.new("RGB", (384, 344), "#10151e")
                ImageDraw.Draw(tile).text((8, 8), label, font=font, fill="white")
                thumb = ImageOps.contain(picture, (368, 288))
                if col == 3:
                    brush = ImageDraw.Draw(thumb)
                    for i, (x, y, w, h) in enumerate(boxes):
                        corners = ((x-w/2)*thumb.width, (y-h/2)*thumb.height,
                                   (x+w/2)*thumb.width, (y+h/2)*thumb.height)
                        color = COLORS[i % len(COLORS)]
                        brush.rectangle(corners, outline=color, width=2)
                        brush.text((corners[0]+2, corners[1]+2), str(i+1), font=small, fill=color)
                tile.paste(thumb, ((384-thumb.width)//2, 38+(288-thumb.height)//2))
                panel.paste(tile, (384*col, 0))
            panel.save(directory / f"sample-{index+1:02}.jpg", quality=90, optimize=True)
            canvas.paste(panel, (0, 54+index*344))
            payload["samples"].append({
                "image": record.relative, "sha256": sha256_file(Path(record.path)),
                "key_shape": list(keys.shape), "grid": list(backbone.grid),
                "pca_explained_variance_ratio": variance, "masks": len(masks),
                "boxes_xywh_normalized": boxes, "seconds": time.perf_counter()-started,
                "visualization": f"sample-{index+1:02}.jpg",
            })
            print(f"Key preview {index+1}/{len(records)}: {record.relative}, {len(boxes)} boxes",
                  flush=True)
        draw = ImageDraw.Draw(canvas)
        draw.text((16, canvas.height-44), "PCA colors are NOT class labels or object identities. "
                  "MaskCut uses the full keys; gray masks were filtered out.",
                  font=small, fill="white")
        draw.text((16, canvas.height-23), "No human boxes, no fine-tuning, no YOLO inference. "
                  "Some instances may merge or fragment.", font=small, fill="#b9c6d8")
        canvas.save(directory / "key_pipeline.jpg", quality=90, optimize=True)
        payload["status"] = "complete"
    except BaseException as exc:
        payload["status"] = "failed"
        payload["errors"].append(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_json(directory / "features.json", payload)
        lines = ["# DINO key pipeline visualization", "", f"Status: {payload['status']}",
                 f"Source: {payload['git']}", f"Command: `{payload['command']}`", ""]
        lines.extend(f"- {note}" for note in payload["notes"])
        lines.extend(["", "![Original, keys, masks, boxes](key_pipeline.jpg)", ""])
        lines.extend(f"Error: {error}" for error in payload["errors"])
        atomic_write(directory / "summary.md", "\n".join(lines).encode())
        print(f"Key map report: {directory}", flush=True)
    return directory
