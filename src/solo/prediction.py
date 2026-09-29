from __future__ import annotations

import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from .config import Config, local_path, protect_dataset
from .dataset import EXTENSIONS
from .progress import duration
from .storage import sha256_file, write_json


def predict_objects(
    source: Path, weights: Path, config: Config, limit: int = 16, confidence: float = 0.25,
) -> Path:
    if limit < 0 or not 0 < confidence <= 1:
        raise ValueError("Require --limit >= 0 and 0 < --conf <= 1")
    if not weights.is_file():
        raise FileNotFoundError(
            f"Trained checkpoint not found: {weights}. Run ./solo train first."
        )
    weights = weights.resolve(strict=True)
    if not source.exists():
        raise FileNotFoundError(
            f"Image input not found: {source}. Pass a photo/folder or run ./solo download-coco."
        )
    source = source.resolve(strict=True)
    paths = [source] if source.is_file() else sorted(
        path for path in source.rglob("*") if path.is_file() and path.suffix.lower() in EXTENSIONS
    )
    if not paths or any(path.suffix.lower() not in EXTENSIONS for path in paths):
        raise ValueError(f"No supported images: {source}")
    paths = paths[:limit] if limit else paths
    root = source.parent if source.is_file() else source
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    output = local_path(Path("outputs/predictions") / run_id)
    protect_dataset(root, output)

    import torch
    from ultralytics import YOLO

    if config.train.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA prediction requested but CUDA is unavailable")
    device = 0 if config.train.device != "cpu" and torch.cuda.is_available() else "cpu"
    model = YOLO(str(weights))
    if model.task != "detect" or model.names != {0: "object"}:
        raise ValueError("Expected a SOLO single-class object detector checkpoint")
    output.mkdir(parents=True)
    started = time.monotonic()
    total_boxes = 0
    print(f"YOLO predictions: {output}", flush=True)
    for index, path in enumerate(paths, 1):
        with Image.open(path) as original:
            picture = ImageOps.exif_transpose(original).convert("RGB")
        result = model.predict(
            source=picture, imgsz=config.train.image_size, conf=confidence,
            device=device, half=False, save=False, verbose=False,
        )[0]
        boxes = result.boxes.xyxy.cpu().tolist()
        scores = result.boxes.conf.cpu().tolist()
        draw = ImageDraw.Draw(picture)
        for box, score in zip(boxes, scores, strict=True):
            draw.rectangle(box, outline=(255, 60, 30), width=3)
            draw.text((box[0], max(0, box[1] - 12)), f"object {score:.2f}", fill=(255, 60, 30))
        relative = path.relative_to(root)
        # Keep the original extension to avoid same-stem JPG/PNG collisions.
        rendered = output / "images" / relative.parent / (relative.name + ".jpg")
        rendered.parent.mkdir(parents=True, exist_ok=True)
        picture.save(rendered, quality=90)
        write_json(output / "boxes" / relative.parent / (relative.name + ".json"), {
            "source": str(path), "width": picture.width, "height": picture.height,
            "coordinate_system": "xyxy pixels in EXIF-oriented image",
            "detections": [
                {"class_id": 0, "label": "object", "confidence": score, "xyxy": box}
                for box, score in zip(boxes, scores, strict=True)
            ],
        })
        total_boxes += len(boxes)
        elapsed = time.monotonic() - started
        print(
            f"Predict {index}/{len(paths)} ({index / len(paths):.1%}) | "
            f"{len(boxes)} objects | elapsed {duration(elapsed)} | "
            f"ETA {duration(elapsed / index * (len(paths) - index))}", flush=True,
        )
    write_json(output / "summary.json", {
        "weights": str(weights), "checkpoint_sha256": sha256_file(weights),
        "source": str(source), "images": len(paths), "boxes": total_boxes,
        "confidence_threshold": confidence, "image_size": config.train.image_size,
        "device": str(device), "elapsed_seconds": time.monotonic() - started,
        "note": "YOLO object predictions only; no ground-truth evaluation or few-shot matching.",
    })
    print(f"Saved box images: {output / 'images'}", flush=True)
    return output
