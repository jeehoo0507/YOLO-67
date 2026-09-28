from __future__ import annotations

import io
import tarfile
import urllib.error
from pathlib import Path

from solo import download
from solo.cli import main
from solo.config import ROOT
from solo.dataset import scan_images


def _inner_archive() -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as inner:
        for number in range(3):
            data = f"image-{number}".encode()
            image = tarfile.TarInfo(f"n00000001_{number}.JPEG")
            image.size = len(data)
            inner.addfile(image, io.BytesIO(data))
    return buffer.getvalue()


def _one_group(contents: bytes) -> dict:
    return {"name": "n00000001.tar", "offset": 512, "size": len(contents)}


def test_bundled_index_covers_imagenet_1k():
    entries = download._index_remote()
    assert len(entries) == 1000
    assert len({entry["name"] for entry in entries}) == 1000
    assert entries[-1]["offset"] + entries[-1]["size"] < download.ARCHIVE_BYTES
    probes = [entries[index] for index in (0, 500, 999)]
    subset = download._plan_entries(entries, "100gb", probes)
    assert len(subset) == 1000
    assert sum(item["take_bytes"] for item in subset) <= download.FALLBACK_TRANSFER_BYTES
    assert all(1024 <= item["take_bytes"] <= item["size"] for item in subset)


def test_partial_group_keeps_only_complete_images(tmp_path, monkeypatch):
    contents = _inner_archive()
    monkeypatch.setattr(download, "TRAIN_DIR", tmp_path / "train")
    partial = {**_one_group(contents), "take_bytes": 2560}
    receipt = download._extract_group(io.BytesIO(contents[:2560]), partial)
    assert receipt["images"] == 2
    assert not (download.TRAIN_DIR / "n00000001/n00000001_2.JPEG").exists()
    full = {**_one_group(contents), "take_bytes": len(contents)}
    receipt = download._extract_group(io.BytesIO(contents), full)
    assert receipt["images"] == 3
    assert (download.TRAIN_DIR / "n00000001/n00000001_2.JPEG").read_bytes() == b"image-2"


def test_auto_plan_chooses_full_only_when_fast_and_disk_fits(tmp_path, monkeypatch):
    entries = [
        {"name": f"n{i:08d}.tar", "offset": i * 10240, "size": 10240}
        for i in range(5)
    ]
    monkeypatch.setattr(download, "PLAN_FILE", tmp_path / "plan.json")
    monkeypatch.setattr(download, "FALLBACK_TRANSFER_BYTES", 4 * 10240)
    monkeypatch.setattr(download, "_fetch_group", lambda item, outer, force: ({}, 1.0))
    monkeypatch.setattr(download, "_disk_available", lambda target: True)
    monkeypatch.setattr(download, "FULL_DAY_SECONDS", 10)
    full = download._choose_plan(entries, "auto")
    assert full["mode"] == "full"
    assert full["transfer_bytes"] == 5 * 10240
    monkeypatch.setattr(download, "FULL_DAY_SECONDS", 1)
    subset = download._choose_plan(entries, "auto")
    assert subset["mode"] == "100gb"
    assert len(subset["entries"]) == 5
    assert subset["transfer_bytes"] <= 4 * 10240
    assert all(item["take_bytes"] > 0 for item in subset["entries"])


def test_group_download_receipt_skips_completed_transfer(tmp_path, monkeypatch):
    contents = _inner_archive()
    item = _one_group(contents)
    monkeypatch.setattr(download, "TRAIN_DIR", tmp_path / "train")
    monkeypatch.setattr(download, "DOWNLOAD_DIR", tmp_path / "downloads")
    monkeypatch.setattr(download, "RECEIPT_DIR", tmp_path / "receipts")
    monkeypatch.setattr(download, "PLAN_FILE", tmp_path / "plan.json")
    monkeypatch.setattr(download, "COMPLETE_FILE", tmp_path / "complete.json")
    monkeypatch.setattr(download, "LOCK_FILE", tmp_path / ".download.lock")
    monkeypatch.setattr(download, "FULL_IMAGE_COUNT", 3)
    monkeypatch.setattr(download, "_disk_available", lambda target: True)
    monkeypatch.setattr(download, "_index_remote", lambda: [item])
    calls = []

    def remote_range(start, size):
        calls.append((start, size))
        return io.BytesIO(contents)

    monkeypatch.setattr(download, "_remote_range", remote_range)
    download.download_imagenet(mode="full")
    assert calls == [(512, len(contents))]
    download.download_imagenet(mode="full")
    assert len(calls) == 1
    assert download.COMPLETE_FILE.is_file()


def test_sample_stops_after_requested_images_and_resumes(tmp_path, monkeypatch):
    contents = _inner_archive()
    group = _one_group(contents)
    monkeypatch.setattr(download, "TRAIN_DIR", tmp_path / "train")
    monkeypatch.setattr(download, "SELECTION_FILE", tmp_path / "train/.solo-selection.txt")
    monkeypatch.setattr(download, "RECEIPT_DIR", tmp_path / "receipts")
    monkeypatch.setattr(download, "PLAN_FILE", tmp_path / "plan.json")
    monkeypatch.setattr(download, "COMPLETE_FILE", tmp_path / "complete.json")
    monkeypatch.setattr(download, "LOCK_FILE", tmp_path / ".download.lock")
    monkeypatch.setattr(download, "DOWNLOAD_DIR", tmp_path / "downloads")
    monkeypatch.setattr(download, "_index_remote", lambda: [group])
    calls = []

    def remote_range(start, size):
        calls.append((start, size))

        class Stream(io.BytesIO):
            def close(self):
                calls.append(self.tell())
                super().close()

        return Stream(contents)

    monkeypatch.setattr(download, "_remote_range", remote_range)
    download.download_imagenet(images_per_group=2)
    assert len(calls) == 2
    assert calls[1] < len(contents)
    assert len(scan_images(download.TRAIN_DIR)) == 2
    assert download.COMPLETE_FILE.is_file()
    download.download_imagenet(images_per_group=2)
    assert len(calls) == 2


def test_sample_selects_existing_images_without_redownloading(tmp_path, monkeypatch):
    contents = _inner_archive()
    group = _one_group(contents)
    folder = tmp_path / "train/n00000001"
    folder.mkdir(parents=True)
    for number in range(3):
        (folder / f"n00000001_{number}.JPEG").write_bytes(f"image-{number}".encode())
    monkeypatch.setattr(download, "TRAIN_DIR", tmp_path / "train")
    monkeypatch.setattr(download, "SELECTION_FILE", tmp_path / "train/.solo-selection.txt")
    monkeypatch.setattr(download, "RECEIPT_DIR", tmp_path / "receipts")
    monkeypatch.setattr(download, "PLAN_FILE", tmp_path / "plan.json")
    monkeypatch.setattr(download, "COMPLETE_FILE", tmp_path / "complete.json")
    monkeypatch.setattr(download, "LOCK_FILE", tmp_path / ".download.lock")
    monkeypatch.setattr(download, "DOWNLOAD_DIR", tmp_path / "downloads")
    monkeypatch.setattr(download, "_index_remote", lambda: [group])
    monkeypatch.setattr(download, "_remote_range", lambda start, size: None)
    download.download_imagenet(images_per_group=2)
    assert len(scan_images(download.TRAIN_DIR)) == 2
    assert len(list(folder.glob("*.JPEG"))) == 3  # Preserve already downloaded images.


def test_transient_range_error_retries_group(tmp_path, monkeypatch):
    contents = _inner_archive()
    item = {**_one_group(contents), "take_bytes": len(contents)}
    monkeypatch.setattr(download, "TRAIN_DIR", tmp_path / "train")
    monkeypatch.setattr(download, "RECEIPT_DIR", tmp_path / "receipts")
    monkeypatch.setattr(download.time, "sleep", lambda seconds: None)
    calls = []

    def remote_range(start, size):
        calls.append((start, size))
        if len(calls) == 1:
            raise urllib.error.URLError("temporary failure")
        return io.BytesIO(contents)

    monkeypatch.setattr(download, "_remote_range", remote_range)
    receipt, _ = download._fetch_group(item)
    assert receipt["images"] == 3
    assert len(calls) == 2


def test_existing_archive_is_read_only(tmp_path, monkeypatch):
    contents = _inner_archive()
    archive = tmp_path / download.ARCHIVE_NAME
    with tarfile.open(archive, "w") as outer:
        group = tarfile.TarInfo("n00000001.tar")
        group.size = len(contents)
        outer.addfile(group, io.BytesIO(contents))
    monkeypatch.setattr(download, "TRAIN_DIR", tmp_path / "train")
    monkeypatch.setattr(download, "RECEIPT_DIR", tmp_path / "receipts")
    monkeypatch.setattr(download, "PLAN_FILE", tmp_path / "plan.json")
    monkeypatch.setattr(download, "COMPLETE_FILE", tmp_path / "complete.json")
    monkeypatch.setattr(download, "LOCK_FILE", tmp_path / ".download.lock")
    monkeypatch.setattr(download, "FULL_IMAGE_COUNT", 3)
    monkeypatch.setattr(download, "_disk_available", lambda target: True)
    item = _one_group(contents)
    monkeypatch.setattr(download, "_index_local", lambda path: [item])
    original = archive.read_bytes()
    download.download_imagenet(archive, "full")
    assert archive.read_bytes() == original


def test_download_cli_defaults_to_100k_sample(monkeypatch):
    calls: list[tuple[Path | None, str, int]] = []
    monkeypatch.setenv("SOLO_ROOT", str(ROOT))
    monkeypatch.setattr(
        download, "download_imagenet",
        lambda archive, mode, images_per_group: calls.append((archive, mode, images_per_group))
    )
    assert main(["download"]) == 0
    assert calls == [(None, "sample", 100)]
