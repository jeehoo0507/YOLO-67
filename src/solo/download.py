from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path, PurePosixPath

from .config import ROOT
from .dataset import EXTENSIONS

TRAIN_URL = "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar"
ARCHIVE_NAME = "ILSVRC2012_img_train.tar"
ARCHIVE_BYTES = 147_897_477_120
ARCHIVE_MD5 = "1d675b47d978889d74fa0da5fadfb00e"
EXPECTED_IMAGES = 1_281_167
TRAIN_DIR = ROOT / "data/imagenet/train"
DOWNLOAD_DIR = ROOT / "data/imagenet/downloads"
COMPLETE_FILE = ROOT / "data/imagenet/.download-complete.json"


def _md5(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - published archive integrity checksum
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download_archive() -> Path:
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    archive = DOWNLOAD_DIR / ARCHIVE_NAME
    partial = DOWNLOAD_DIR / (ARCHIVE_NAME + ".part")
    if archive.is_file():
        if archive.stat().st_size == ARCHIVE_BYTES and _md5(archive) == ARCHIVE_MD5:
            print(f"Using verified archive: {archive}", flush=True)
            return archive
        print("Existing archive is incomplete or corrupt; downloading again.", flush=True)
        archive.unlink()
    start = partial.stat().st_size if partial.exists() else 0
    if start >= ARCHIVE_BYTES:
        partial.unlink()
        start = 0
    headers = {"Range": f"bytes={start}-"} if start else {}
    request = urllib.request.Request(TRAIN_URL, headers=headers)
    print(f"Downloading official ImageNet train archive ({ARCHIVE_BYTES / 1e9:.1f} GB)", flush=True)
    if start:
        print(f"Resuming at {start / 1e9:.1f} GB", flush=True)
    try:
        response = urllib.request.urlopen(request, timeout=90)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"ImageNet server returned HTTP {exc.code}. See {TRAIN_URL} "
            "and use --archive if you already downloaded the train TAR."
        ) from exc
    with response:
        if start and response.status != 206:
            raise RuntimeError("ImageNet server did not honor the resume Range request")
        if not start and response.status != 200:
            raise RuntimeError(f"Unexpected ImageNet response: HTTP {response.status}")
        if start:
            content_range = response.headers.get("Content-Range", "")
            if not content_range.startswith(f"bytes {start}-"):
                raise RuntimeError(f"Unexpected ImageNet Content-Range: {content_range}")
        if "text/html" in response.headers.get("Content-Type", ""):
            raise RuntimeError("ImageNet returned a web page instead of the train archive")
        downloaded = start
        last_report = time.monotonic()
        with partial.open("ab" if start else "wb") as output:
            while block := response.read(8 * 1024 * 1024):
                output.write(block)
                downloaded += len(block)
                now = time.monotonic()
                if now - last_report >= 5:
                    print(
                        f"Download: {downloaded / 1e9:.1f} / {ARCHIVE_BYTES / 1e9:.1f} GB",
                        flush=True,
                    )
                    last_report = now
    if partial.stat().st_size != ARCHIVE_BYTES:
        raise RuntimeError("ImageNet download stopped early; rerun ./solo download to resume")
    print("Verifying ImageNet archive checksum ...", flush=True)
    if _md5(partial) != ARCHIVE_MD5:
        partial.unlink()
        raise RuntimeError("ImageNet archive checksum mismatch; rerun ./solo download")
    partial.replace(archive)
    return archive


def _write_image(source, destination: Path, expected_size: int) -> None:
    if destination.is_file() and destination.stat().st_size == expected_size:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".solo-part")
    try:
        with temporary.open("wb") as handle:
            shutil.copyfileobj(source, handle, length=1024 * 1024)
        if temporary.stat().st_size != expected_size:
            raise RuntimeError(f"Incomplete image in archive: {destination}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _extract_tar(archive: Path) -> int:
    count = 0
    with tarfile.open(archive, "r|") as outer:
        for group in outer:
            group_name = PurePosixPath(group.name).name
            if not group.isfile() or not group_name.endswith(".tar"):
                continue
            synset = group_name.removesuffix(".tar")
            if not synset.startswith("n") or not synset[1:].isdigit():
                raise RuntimeError(f"Unexpected ImageNet training archive member: {group.name}")
            group_stream = outer.extractfile(group)
            if group_stream is None:
                raise RuntimeError(f"Unreadable member: {group.name}")
            with tarfile.open(fileobj=group_stream, mode="r|") as inner:
                for image in inner:
                    name = PurePosixPath(image.name)
                    if not image.isfile() or len(name.parts) != 1:
                        continue
                    if name.suffix.lower() not in EXTENSIONS:
                        continue
                    source = inner.extractfile(image)
                    if source is None:
                        raise RuntimeError(f"Unreadable image: {image.name}")
                    with source:
                        _write_image(source, TRAIN_DIR / synset / name.name, image.size)
                    count += 1
                    if count % 10_000 == 0:
                        print(f"Prepared {count:,} train images", flush=True)
    return count


def _count_images() -> int:
    count = 0
    for _directory, _, files in os.walk(TRAIN_DIR):
        count += sum(Path(name).suffix.lower() in EXTENSIONS for name in files)
    return count


def download_imagenet(archive: Path | None = None) -> None:
    if archive is None and _count_images() == EXPECTED_IMAGES:
        print(f"ImageNet train already prepared at {TRAIN_DIR}")
        return
    downloaded = archive is None
    archive = _download_archive() if downloaded else archive.expanduser().resolve(strict=True)
    if archive.suffix.lower() != ".tar":
        raise ValueError("Expected the official ILSVRC2012_img_train.tar archive")
    if not downloaded and archive.name == ARCHIVE_NAME and _md5(archive) != ARCHIVE_MD5:
        raise RuntimeError("ImageNet train archive checksum mismatch")
    count = _extract_tar(archive)
    if count != EXPECTED_IMAGES:
        raise RuntimeError(
            f"Archive contains {count:,} training images; expected {EXPECTED_IMAGES:,}. "
            "Check that this is the full ImageNet-1K train archive."
        )
    COMPLETE_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary = COMPLETE_FILE.with_suffix(".tmp")
    temporary.write_text(json.dumps({"images": count, "source": archive.name}) + "\n")
    temporary.replace(COMPLETE_FILE)
    if downloaded:
        archive.unlink()
    print(f"ImageNet train ready: {count:,} images in {TRAIN_DIR}")
