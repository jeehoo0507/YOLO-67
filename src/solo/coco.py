from __future__ import annotations

import os
import re
import shutil
import time
import urllib.request
import uuid
import zipfile
import zlib
from pathlib import Path

from .config import local_path
from .progress import duration
from .storage import output_lock, write_json

SPLITS = {
    "train2017": (
        "https://s3.amazonaws.com/images.cocodataset.org/zips/train2017.zip",
        118_287,
        19_336_861_798,
        '"62ff7d7fbcc7e0c0604cbb0f9047ce77-2306"',
    ),
    "val2017": (
        "https://s3.amazonaws.com/images.cocodataset.org/zips/val2017.zip",
        5_000,
        815_585_330,
        '"d366be60d3dc737327160d62453e3973-98"',
    ),
}


def _download(url: str, path: Path, total: int, etag: str) -> None:
    if path.is_file():
        try:
            with zipfile.ZipFile(path):
                if path.stat().st_size == total:
                    return
            path.unlink()
        except zipfile.BadZipFile:
            path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset == total:
        with zipfile.ZipFile(partial):
            pass
        os.replace(partial, path)
        return
    if offset > total:
        raise RuntimeError(f"Oversized partial archive; remove only this file and retry: {partial}")
    headers = {"User-Agent": "SOLO/0.1"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
        headers["If-Range"] = etag
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=90) as response:
        if offset:
            match = re.fullmatch(
                r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", "")
            )
            if (
                response.status != 206
                or not match
                or tuple(map(int, match.groups())) != (offset, total - 1, total)
            ):
                raise RuntimeError(
                    f"COCO server rejected resume for {partial}; keeping partial file"
                )
        else:
            if response.status != 200:
                raise RuntimeError(f"COCO download returned HTTP {response.status}: {url}")
        if (
            response.headers.get("ETag") != etag
            or int(response.headers.get("Content-Length", -1)) != total - offset
        ):
            raise RuntimeError(f"Official COCO archive changed: {url}")
        started = last = time.monotonic()
        received = offset
        with partial.open("ab") as stream:
            while block := response.read(4 * 1024 * 1024):
                stream.write(block)
                received += len(block)
                now = time.monotonic()
                if now - last >= 10 or total and received == total:
                    rate = (received - offset) / max(now - started, 1e-9)
                    fraction = f"{received / total:.1%}" if total else "size unknown"
                    eta = duration((total - received) / rate) if total and rate else "?"
                    print(
                        f"COCO {path.name}: {received / 1e9:.2f} GB "
                        f"({fraction}), {rate / 1e6:.1f} MB/s, ETA {eta}",
                        flush=True,
                    )
                    last = now
            stream.flush()
            os.fsync(stream.fileno())
    if total and received != total:
        raise RuntimeError(f"Incomplete COCO download: {path.name} ({received}/{total} bytes)")
    try:
        with zipfile.ZipFile(partial):
            pass  # Extraction verifies each member's CRC without a second 19 GB read.
    except zipfile.BadZipFile as exc:
        raise RuntimeError(f"Incomplete or invalid COCO zip: {partial}") from exc
    os.replace(partial, path)


def _crc32(path: Path) -> int:
    checksum = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum = zlib.crc32(block, checksum)
    return checksum


def _extract(archive_path: Path, split: str, image_root: Path, expected: int) -> dict:
    started = last = time.monotonic()
    count = 0
    bytes_written = 0
    with zipfile.ZipFile(archive_path) as archive:
        members = [
            item
            for item in archive.infolist()
            if item.filename.startswith(split + "/") and item.filename.lower().endswith(".jpg")
        ]
        if len(members) != expected:
            raise RuntimeError(
                f"Unexpected {split} archive image count: {len(members)} (expected {expected})"
            )
        if len({item.filename for item in members}) != expected:
            raise RuntimeError(f"Duplicate image names in {archive_path}")
        for index, item in enumerate(members, 1):
            relative = Path(item.filename)
            if len(relative.parts) != 2 or relative.parts[0] != split:
                raise RuntimeError(f"Unsafe COCO zip path: {item.filename}")
            target = image_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            valid_existing = (
                target.is_file()
                and not target.is_symlink()
                and target.stat().st_size == item.file_size
                and _crc32(target) == item.CRC
            )
            if not valid_existing:
                temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
                try:
                    with archive.open(item) as source, temporary.open("wb") as destination:
                        shutil.copyfileobj(source, destination, 1024 * 1024)
                        destination.flush()
                        os.fsync(destination.fileno())
                    if temporary.stat().st_size != item.file_size:
                        raise RuntimeError(f"COCO image size mismatch: {item.filename}")
                    os.replace(temporary, target)
                finally:
                    temporary.unlink(missing_ok=True)
                bytes_written += item.file_size
            count += 1
            now = time.monotonic()
            if now - last >= 10 or index == expected:
                rate = index / max(now - started, 1e-9)
                print(
                    f"COCO {split} extract: {index:,}/{expected:,} ({index / expected:.1%}), "
                    f"{rate:.1f} images/s, ETA {duration((expected - index) / rate)}",
                    flush=True,
                )
                last = now
    return {"images": count, "extracted_bytes": bytes_written}


def download_coco() -> Path:
    root = local_path("data/coco")
    images = local_path(root / "images")
    with output_lock(root / ".download.lock"):
        for split, (url, expected, archive_bytes, etag) in SPLITS.items():
            marker = root / f".{split}-complete.json"
            folder = images / split
            if (
                marker.is_file()
                and folder.is_dir()
                and sum(1 for _ in folder.glob("*.jpg")) == expected
            ):
                print(f"COCO {split}: already prepared ({expected:,} images)", flush=True)
                continue
            archive = local_path(root / "downloads" / f"{split}.zip")
            print(f"Downloading official COCO 2017 {split} images", flush=True)
            _download(url, archive, archive_bytes, etag)
            stats = _extract(archive, split, images, expected)
            write_json(marker, {"source": url, "split": split, "etag": etag, **stats})
            archive.unlink()
    total = sum(spec[1] for spec in SPLITS.values())
    print(f"COCO images ready: {images} ({total:,} images; no COCO annotations used)", flush=True)
    return images
