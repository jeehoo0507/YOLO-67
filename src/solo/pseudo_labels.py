from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import FilterConfig
from .dataset import ImageRecord
from .storage import atomic_write, write_json

# YOLO xywh in normalized original-image coordinates.
Box = tuple[float, float, float, float]


def box_iou(a: Box, b: Box) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    intersection = max(0.0, min(ax + aw / 2, bx + bw / 2) - max(ax - aw / 2, bx - bw / 2)) * max(
        0.0, min(ay + ah / 2, by + bh / 2) - max(ay - ah / 2, by - bh / 2)
    )
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def masks_to_boxes(
    masks: list[np.ndarray], image_size: tuple[int, int], config: FilterConfig
) -> list[Box]:
    width, height = image_size
    boxes = []
    for mask in masks:
        y, x = np.nonzero(mask)
        if not len(x):
            continue
        rows, cols = mask.shape
        x0, x1 = float(x.min()) / cols, float(x.max() + 1) / cols
        y0, y1 = float(y.min()) / rows, float(y.max() + 1) / rows
        w, h = x1 - x0, y1 - y0
        aspect = (w * width) / (h * height)
        if not config.min_box_area <= w * h <= config.max_box_area:
            continue
        if max(aspect, 1 / aspect) > config.max_aspect_ratio:
            continue
        box = ((x0 + x1) / 2, (y0 + y1) / 2, w, h)
        if all(box_iou(box, previous) <= config.duplicate_iou for previous in boxes):
            boxes.append(box)
    return boxes


def encode_labels(boxes: list[Box]) -> bytes:
    return "".join(
        "0 " + " ".join(f"{value:.8f}" for value in box) + "\n" for box in boxes
    ).encode()


def decode_labels(data: bytes) -> list[Box]:
    boxes = []
    for line in data.decode("utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5 or parts[0] != "0":
            raise ValueError("Invalid YOLO row; expected single class 0 with xywh")
        box = tuple(float(value) for value in parts[1:])
        if not all(math.isfinite(value) for value in box):
            raise ValueError("Non-finite box")
        x, y, w, h = box
        tolerance = 2e-8
        if not (0 < w <= 1 and 0 < h <= 1 and 0 <= x <= 1 and 0 <= y <= 1):
            raise ValueError("Box outside normalized range")
        if min(x - w / 2, y - h / 2) < -tolerance or max(x + w / 2, y + h / 2) > 1 + tolerance:
            raise ValueError("Box extends outside image")
        boxes.append(box)
    return boxes


@dataclass
class LabelStore:
    root: Path
    fingerprint: str
    dataset_root: str
    durable: bool = True

    def label_path(self, record: ImageRecord) -> Path:
        return self.root / "labels" / record.key

    def receipt_path(self, record: ImageRecord) -> Path:
        return self.root / "receipts" / (record.relative + ".json")

    def save(self, record: ImageRecord, boxes: list[Box], image_size: tuple[int, int]) -> None:
        data = encode_labels(boxes)
        decode_labels(data)  # validates output, including rounding at image edges
        atomic_write(self.label_path(record), data, self.durable)
        # The receipt is the commit point. Any interruption before it forces regeneration.
        write_json(
            self.receipt_path(record),
            {
                "schema": 1,
                "fingerprint": self.fingerprint,
                "dataset_root": self.dataset_root,
                "image": record.relative,
                "source_size": record.size,
                "source_mtime_ns": record.mtime_ns,
                "image_size": list(image_size),
                "label_sha256": hashlib.sha256(data).hexdigest(),
                "boxes": len(boxes),
            },
            self.durable,
        )

    def valid_boxes(self, record: ImageRecord) -> list[Box] | None:
        try:
            receipt = json.loads(self.receipt_path(record).read_text())
            data = self.label_path(record).read_bytes()
            if (
                receipt["schema"] != 1
                or receipt["fingerprint"] != self.fingerprint
                or receipt["dataset_root"] != self.dataset_root
                or receipt["image"] != record.relative
                or receipt["source_size"] != record.size
                or receipt["source_mtime_ns"] != record.mtime_ns
                or receipt["label_sha256"] != hashlib.sha256(data).hexdigest()
            ):
                return None
            boxes = decode_labels(data)
            width, height = receipt["image_size"]
            if not (
                isinstance(width, int) and isinstance(height, int) and width > 0 and height > 0
            ):
                return None
            return boxes if len(boxes) == receipt["boxes"] else None
        except (OSError, ValueError, KeyError, TypeError, UnicodeError):
            return None
