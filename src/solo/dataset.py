from __future__ import annotations

import hashlib
import heapq
import os
from dataclasses import dataclass
from pathlib import Path

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass(frozen=True)
class ImageRecord:
    path: str
    relative: str
    size: int
    mtime_ns: int

    @property
    def key(self) -> str:
        return Path(self.relative).with_suffix(".txt").as_posix()


def scan_images(root: Path) -> list[ImageRecord]:
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"Not an image directory: {root}")
    records = []
    label_keys = set()
    for directory, dirs, files in os.walk(root, followlinks=False):
        dirs.sort()
        for name in sorted(files):
            path = Path(directory) / name
            if path.suffix.lower() not in EXTENSIONS:
                continue
            stat = path.stat()
            record = ImageRecord(
                str(path), path.relative_to(root).as_posix(), stat.st_size, stat.st_mtime_ns
            )
            if record.key in label_keys:
                raise ValueError(
                    f"YOLO filename collision (same stem, different extensions): {path}"
                )
            label_keys.add(record.key)
            records.append(record)
    if not records:
        raise ValueError(f"No supported images found below {root}")
    return records


def deterministic_subset(records: list[ImageRecord], count: int, seed: int) -> list[ImageRecord]:
    # Hash relative paths only: directory names are identifiers, never class labels.
    def rank(record: ImageRecord):
        return hashlib.sha256(f"{seed}:{record.relative}".encode()).digest(), record.relative

    return heapq.nsmallest(min(count, len(records)), records, key=rank)


def subset_digest(records: list[ImageRecord]) -> str:
    return hashlib.sha256("\n".join(r.relative for r in records).encode()).hexdigest()
