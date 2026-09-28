from __future__ import annotations

import hashlib
import http.client
import json
import os
import shutil
import tarfile
import time
import urllib.error
import urllib.request
from contextlib import nullcontext
from pathlib import Path, PurePosixPath

from .config import ROOT
from .dataset import EXTENSIONS
from .storage import atomic_write, output_lock, write_json

TRAIN_URL = "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar"
ARCHIVE_NAME = "ILSVRC2012_img_train.tar"
ARCHIVE_BYTES = 147_897_477_120
ARCHIVE_ETAG = '"226f606800-4c27352eaf4c0"'
FALLBACK_TRANSFER_BYTES = 99_000_000_000
FULL_DAY_SECONDS = 24 * 3600
FULL_IMAGE_COUNT = 1_281_167
DEFAULT_IMAGES_PER_GROUP = 100
ESTIMATE_SAFETY_FACTOR = 1.2
DISK_SAFETY_BYTES = 5_000_000_000
TRAIN_DIR = ROOT / "data/imagenet/train"
DOWNLOAD_DIR = ROOT / "data/imagenet/downloads"
BUNDLED_INDEX = ROOT / "assets/imagenet_train_index.json"
RECEIPT_DIR = ROOT / "data/imagenet/download_receipts"
PLAN_FILE = ROOT / "data/imagenet/download_plan.json"
COMPLETE_FILE = ROOT / "data/imagenet/.download-complete.json"
LOCK_FILE = ROOT / "data/imagenet/.download.lock"
SELECTION_FILE = TRAIN_DIR / ".solo-selection.txt"


def _padded(size: int) -> int:
    return ((size + 511) // 512) * 512


def _index_remote() -> list[dict]:
    payload = json.loads(BUNDLED_INDEX.read_text())
    entries = payload["entries"]
    if (
        payload.get("archive_bytes") != ARCHIVE_BYTES
        or payload.get("archive_etag") != ARCHIVE_ETAG
        or len(entries) != 1000
        or len({entry["name"] for entry in entries}) != 1000
    ):
        raise RuntimeError("Bundled ImageNet-1K TAR index does not match the official archive")
    return entries


def _index_local(archive: Path) -> list[dict]:
    entries = []
    with tarfile.open(archive, "r:") as outer:
        for member in outer:
            if member.isfile() and member.name.endswith(".tar"):
                entries.append(
                    {"name": PurePosixPath(member.name).name,
                     "offset": member.offset_data, "size": member.size}
                )
    if len(entries) != 1000:
        raise RuntimeError(f"Expected 1,000 ImageNet-1K groups; found {len(entries)}")
    return entries


def _remote_range(start: int, size: int):
    end = start + size - 1
    request = urllib.request.Request(
        TRAIN_URL,
        headers={"Range": f"bytes={start}-{end}", "If-Range": ARCHIVE_ETAG},
    )
    response = urllib.request.urlopen(request, timeout=90)
    expected = f"bytes {start}-{end}/{ARCHIVE_BYTES}"
    if (
        response.status != 206
        or response.headers.get("Content-Range") != expected
        or response.headers.get("ETag") != ARCHIVE_ETAG
        or int(response.headers.get("Content-Length", -1)) != size
    ):
        response.close()
        raise RuntimeError("Official ImageNet server changed or rejected HTTP byte ranges")
    return response


def _copy_exact(source, destination, size: int) -> None:
    remaining = size
    while remaining:
        block = source.read(min(1024 * 1024, remaining))
        if not block:
            raise EOFError("ImageNet TAR transfer ended before the requested byte range")
        if destination is not None:
            destination.write(block)
        remaining -= len(block)


def _write_image(source, destination: Path, size: int) -> None:
    if destination.is_file() and destination.stat().st_size == size:
        _copy_exact(source, None, size)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".solo-part")
    try:
        with temporary.open("wb") as handle:
            _copy_exact(source, handle, size)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _extract_group(source, item: dict) -> dict:
    synset = item["name"].removesuffix(".tar")
    if not synset.startswith("n") or not synset[1:].isdigit():
        raise RuntimeError(f"Invalid ImageNet group name: {item['name']}")
    limit = item["take_bytes"]
    position = 0
    count = 0
    image_bytes = 0
    files = []
    max_images = item.get("max_images")
    while position + 512 <= limit:
        header = source.read(512)
        if len(header) != 512:
            raise EOFError("ImageNet TAR header was truncated")
        position += 512
        if header == bytes(512):
            break
        member = tarfile.TarInfo.frombuf(header, "utf-8", "surrogateescape")
        padded = _padded(member.size)
        if position + padded > limit:
            break  # Stop before a partially transferred image.
        name = PurePosixPath(member.name)
        if member.isfile() and len(name.parts) == 1 and name.suffix.lower() in EXTENSIONS:
            _write_image(source, TRAIN_DIR / synset / name.name, member.size)
            count += 1
            image_bytes += member.size
            if max_images is not None:
                files.append({"name": name.name, "size": member.size})
        else:
            _copy_exact(source, None, member.size)
        _copy_exact(source, None, padded - member.size)
        position += padded
        if max_images is not None and count == max_images:
            break
    if not count:
        raise RuntimeError(f"No complete images in ImageNet group: {item['name']}")
    if max_images is not None and count != max_images:
        raise RuntimeError(f"Only {count}/{max_images} images in ImageNet group: {item['name']}")
    receipt = {**item, "images": count, "image_bytes": image_bytes, "bytes_read": position}
    if max_images is not None:
        receipt["files"] = files
    return receipt


def _group_done(item: dict) -> dict | None:
    path = RECEIPT_DIR / (item["name"] + ".json")
    if not path.is_file():
        return None
    try:
        receipt = json.loads(path.read_text())
        keys = ("name", "offset", "size", "take_bytes")
        if "max_images" in item:
            keys += ("max_images",)
        if any(receipt.get(key) != item[key] for key in keys):
            return None
        folder = TRAIN_DIR / item["name"].removesuffix(".tar")
        if "max_images" in item:
            files = receipt["files"]
            if len(files) != item["max_images"]:
                return None
            if any((folder / f["name"]).stat().st_size != f["size"] for f in files):
                return None
            return receipt
        images = [p for p in folder.iterdir() if p.suffix.lower() in EXTENSIONS]
        if len(images) != receipt["images"]:
            return None
        if sum(p.stat().st_size for p in images) != receipt["image_bytes"]:
            return None
        return receipt
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _fetch_group(item: dict, outer=None, force: bool = False) -> tuple[dict, float]:
    if not force and (receipt := _group_done(item)) is not None:
        return receipt, 0.0
    started = time.perf_counter()
    if outer is None:
        for attempt in range(3):
            try:
                with _remote_range(item["offset"], item["take_bytes"]) as source:
                    receipt = _extract_group(source, item)
                    if "max_images" not in item:
                        source.read()  # Detect a short response after TAR end markers.
                break
            except (
                urllib.error.URLError,
                http.client.IncompleteRead,
                TimeoutError,
                ConnectionError,
                EOFError,
            ) as exc:
                if attempt == 2:
                    raise
                print(f"Retrying {item['name']} after transfer error: {exc}", flush=True)
                time.sleep(2 ** attempt)
    else:
        outer.fileobj.seek(item["offset"])
        receipt = _extract_group(outer.fileobj, item)
    elapsed = time.perf_counter() - started
    write_json(RECEIPT_DIR / (item["name"] + ".json"), receipt)
    return receipt, elapsed


def _plan_entries(entries: list[dict], mode: str, probes: list[dict]) -> list[dict]:
    if mode == "full":
        return [{**item, "take_bytes": item["size"]} for item in entries]
    probe_names = {item["name"] for item in probes}
    fixed = sum(item["size"] for item in probes)
    remaining = sum(item["size"] for item in entries if item["name"] not in probe_names)
    if fixed >= FALLBACK_TRANSFER_BYTES or not remaining:
        raise RuntimeError("100 GB ImageNet subset budget is too small for the probe groups")
    available = FALLBACK_TRANSFER_BYTES - fixed
    planned = []
    for item in entries:
        take = (
            item["size"] if item["name"] in probe_names
            else min(item["size"], _padded(available * item["size"] // remaining) - 512)
        )
        if take < 1024:
            raise RuntimeError("ImageNet subset allocation is too small for a class archive")
        planned.append({**item, "take_bytes": take})
    return planned


def _index_digest(entries: list[dict]) -> str:
    return hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()


def _load_plan(entries: list[dict], requested_mode: str) -> dict | None:
    if not PLAN_FILE.is_file():
        return None
    try:
        plan = json.loads(PLAN_FILE.read_text())
        if (
            plan["schema_version"] == 1
            and plan["archive_etag"] == ARCHIVE_ETAG
            and plan["index_sha256"] == _index_digest(entries)
            and len(plan["entries"]) == len(entries)
            and (requested_mode == "auto" or requested_mode == plan["mode"])
            and all(
                planned["name"] == indexed["name"]
                and planned["offset"] == indexed["offset"]
                and planned["size"] == indexed["size"]
                and 0 < planned["take_bytes"] <= indexed["size"]
                for planned, indexed in zip(plan["entries"], entries, strict=True)
            )
        ):
            return plan
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def _image_bytes() -> int:
    total = 0
    for directory, _, files in os.walk(TRAIN_DIR):
        for name in files:
            if Path(name).suffix.lower() in EXTENSIONS:
                total += (Path(directory) / name).stat().st_size
    return total


def _disk_available(target_bytes: int) -> bool:
    existing = _image_bytes()
    required = max(0, int(target_bytes * 1.05) - existing) + DISK_SAFETY_BYTES
    return shutil.disk_usage(ROOT).free >= required


def _choose_plan(entries: list[dict], requested_mode: str, outer=None) -> dict:
    probes = [entries[index] for index in sorted({0, len(entries) // 2, len(entries) - 1})]
    sample_bytes = 0
    sample_seconds = 0.0
    print("Measuring download + extraction on three ImageNet groups ...", flush=True)
    for group in probes:
        item = {**group, "take_bytes": group["size"]}
        _, elapsed = _fetch_group(item, outer, force=True)
        sample_bytes += item["take_bytes"]
        sample_seconds += elapsed
    rate = sample_bytes / max(sample_seconds, 1e-9)
    full_bytes = sum(item["size"] for item in entries)
    predicted_hours = full_bytes / rate * ESTIMATE_SAFETY_FACTOR / 3600
    full_disk_ok = _disk_available(full_bytes)
    if requested_mode == "auto":
        mode = "full" if predicted_hours * 3600 <= FULL_DAY_SECONDS and full_disk_ok else "100gb"
        reason = (
            "full acquisition estimated within 24 hours and disk is sufficient"
            if mode == "full" else
            "full acquisition exceeds 24 hours or disk is insufficient"
        )
    else:
        mode = requested_mode
        reason = f"user selected {requested_mode}"
    planned = _plan_entries(entries, mode, probes)
    planned_bytes = sum(item["take_bytes"] for item in planned)
    if not _disk_available(planned_bytes):
        raise RuntimeError(
            f"Not enough free space for the {mode} ImageNet plan. "
            "Use a larger repository filesystem or an existing external dataset path."
        )
    plan = {
        "schema_version": 1,
        "archive_etag": ARCHIVE_ETAG,
        "index_sha256": _index_digest(entries),
        "mode": mode,
        "selection_reason": reason,
        "entries": planned,
        "transfer_bytes": planned_bytes,
        "probe_bytes": sample_bytes,
        "probe_seconds": sample_seconds,
        "probe_bytes_per_sec": rate,
        "estimated_full_hours_with_margin": predicted_hours,
        "full_disk_available": full_disk_ok,
    }
    write_json(PLAN_FILE, plan)
    print(
        f"Full ImageNet estimate: {predicted_hours:.1f} h (20% margin); "
        f"disk {'OK' if full_disk_ok else 'insufficient'}",
        flush=True,
    )
    print(f"Selected: {mode}, transfer {planned_bytes / 1e9:.2f} GB ({reason})", flush=True)
    return plan


def _prepare_imagenet(archive: Path | None, mode: str) -> None:
    if archive is None:
        prior_archive = DOWNLOAD_DIR / ARCHIVE_NAME
        if prior_archive.is_file() and prior_archive.stat().st_size == ARCHIVE_BYTES:
            archive = prior_archive
            print(f"Using existing project archive: {archive}", flush=True)
    if archive is not None:
        archive = archive.expanduser().resolve(strict=True)
        if archive.suffix.lower() != ".tar":
            raise ValueError("Expected ILSVRC2012_img_train.tar")
        entries = _index_local(archive)
    else:
        entries = _index_remote()
    RECEIPT_DIR.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:") if archive else nullcontext() as outer:
        plan = _load_plan(entries, mode) or _choose_plan(entries, mode, outer)
        total_images = 0
        completed_bytes = 0
        transferred_bytes = 0
        started = time.perf_counter()
        for number, item in enumerate(plan["entries"], 1):
            receipt, elapsed = _fetch_group(item, outer)
            total_images += receipt["images"]
            completed_bytes += item["take_bytes"]
            if elapsed:
                transferred_bytes += item["take_bytes"]
            if number % 10 == 0 or number == len(entries):
                rate = transferred_bytes / max(time.perf_counter() - started, 1e-9)
                remaining = (plan["transfer_bytes"] - completed_bytes) / rate if rate else 0
                print(
                    f"ImageNet {plan['mode']}: {number}/{len(entries)} groups, "
                    f"{total_images:,} images, {completed_bytes / 1e9:.2f}/"
                    f"{plan['transfer_bytes'] / 1e9:.2f} GB, "
                    f"{rate / 1e6:.1f} MB/s, ETA {remaining / 3600:.1f} h",
                    flush=True,
                )
    actual_bytes = _image_bytes()
    if plan["mode"] == "full" and total_images != FULL_IMAGE_COUNT:
        raise RuntimeError(
            f"Full ImageNet-1K has {total_images:,} images; expected {FULL_IMAGE_COUNT:,}"
        )
    if plan["mode"] == "100gb" and actual_bytes > 100_000_000_000:
        raise RuntimeError("Existing extra images pushed the ImageNet subset over 100 GB")
    write_json(
        COMPLETE_FILE,
        {
            "source": str(archive) if archive else TRAIN_URL,
            "mode": plan["mode"],
            "groups": len(plan["entries"]),
            "images": total_images,
            "directory_image_bytes": actual_bytes,
            "transfer_bytes": plan["transfer_bytes"],
            "estimated_full_hours_with_margin": plan["estimated_full_hours_with_margin"],
            "plan_sha256": hashlib.sha256(json.dumps(plan["entries"]).encode()).hexdigest(),
        },
    )
    print(
        f"ImageNet-1K {plan['mode']} ready: {total_images:,} images, "
        f"{actual_bytes / 1e9:.2f} GB in {TRAIN_DIR}",
        flush=True,
    )


def _existing_sample(item: dict) -> dict | None:
    folder = TRAIN_DIR / item["name"].removesuffix(".tar")
    if not folder.is_dir():
        return None
    files = sorted(
        (path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in EXTENSIONS),
        key=lambda path: path.name,
    )[:item["max_images"]]
    if len(files) != item["max_images"]:
        return None
    selected = [{"name": path.name, "size": path.stat().st_size} for path in files]
    receipt = {
        **item,
        "images": len(selected),
        "image_bytes": sum(entry["size"] for entry in selected),
        "bytes_read": 0,
        "files": selected,
    }
    write_json(RECEIPT_DIR / (item["name"] + ".json"), receipt)
    return receipt


def _prepare_sample(archive: Path | None, images_per_group: int) -> None:
    if archive is None:
        prior_archive = DOWNLOAD_DIR / ARCHIVE_NAME
        if prior_archive.is_file() and prior_archive.stat().st_size == ARCHIVE_BYTES:
            archive = prior_archive
            print(f"Using existing project archive: {archive}", flush=True)
    if archive is not None:
        archive = archive.expanduser().resolve(strict=True)
        entries = _index_local(archive)
    else:
        entries = _index_remote()
    target_images = len(entries) * images_per_group
    if target_images > FULL_IMAGE_COUNT:
        raise ValueError("Requested sample exceeds full ImageNet-1K train image count")
    RECEIPT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(
        PLAN_FILE,
        {
            "schema_version": 2,
            "mode": "sample",
            "images_per_group": images_per_group,
            "target_images": target_images,
            "source": str(archive) if archive else TRAIN_URL,
            "index_sha256": _index_digest(entries),
            "selection_file": str(SELECTION_FILE),
        },
    )
    selected_paths = []
    selected_bytes = 0
    bytes_read = 0
    new_groups = 0
    started = time.perf_counter()
    with tarfile.open(archive, "r:") if archive else nullcontext() as outer:
        for number, group in enumerate(entries, 1):
            item = {**group, "take_bytes": group["size"], "max_images": images_per_group}
            receipt = _group_done(item) or _existing_sample(item)
            if receipt is None:
                receipt, _ = _fetch_group(item, outer)
                new_groups += 1
            synset = item["name"].removesuffix(".tar")
            selected_paths.extend(f"{synset}/{file['name']}" for file in receipt["files"])
            selected_bytes += receipt["image_bytes"]
            bytes_read += receipt["bytes_read"]
            if number % 10 == 0 or number == len(entries):
                group_rate = new_groups / max(time.perf_counter() - started, 1e-9)
                eta = (len(entries) - number) / group_rate / 3600 if group_rate else 0
                print(
                    f"ImageNet sample: {number}/{len(entries)} groups, "
                    f"{len(selected_paths):,}/{target_images:,} images, "
                    f"{bytes_read / 1e9:.2f} GB read, ETA {eta:.1f} h",
                    flush=True,
                )
    if len(selected_paths) != target_images:
        raise RuntimeError(f"ImageNet sample has {len(selected_paths):,}/{target_images:,} images")
    atomic_write(SELECTION_FILE, ("\n".join(selected_paths) + "\n").encode())
    write_json(
        COMPLETE_FILE,
        {
            "source": str(archive) if archive else TRAIN_URL,
            "mode": "sample",
            "groups": len(entries),
            "images_per_group": images_per_group,
            "images": target_images,
            "selected_image_bytes": selected_bytes,
            "archive_bytes_read": bytes_read,
            "selection_sha256": hashlib.sha256(SELECTION_FILE.read_bytes()).hexdigest(),
        },
    )
    print(
        f"ImageNet-1K sample ready: {target_images:,} images, "
        f"{selected_bytes / 1e9:.2f} GB selected in {TRAIN_DIR}",
        flush=True,
    )


def download_imagenet(
    archive: Path | None = None,
    mode: str = "sample",
    images_per_group: int = DEFAULT_IMAGES_PER_GROUP,
) -> None:
    if mode not in {"sample", "auto", "full", "100gb"}:
        raise ValueError(f"Unknown ImageNet download mode: {mode}")
    if images_per_group < 1:
        raise ValueError("images_per_group must be positive")
    with output_lock(LOCK_FILE):
        if mode == "sample":
            _prepare_sample(archive, images_per_group)
        else:
            _prepare_imagenet(archive, mode)
            SELECTION_FILE.unlink(missing_ok=True)
