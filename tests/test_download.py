from __future__ import annotations

import hashlib
import io
import tarfile
from pathlib import Path

from solo import download
from solo.cli import main
from solo.config import ROOT


def test_official_download_resumes_and_verifies_checksum(tmp_path, monkeypatch):
    payload = b"test-official-archive"
    monkeypatch.setattr(download, "DOWNLOAD_DIR", tmp_path)
    monkeypatch.setattr(download, "ARCHIVE_BYTES", len(payload))
    monkeypatch.setattr(download, "ARCHIVE_MD5", hashlib.md5(payload).hexdigest())
    partial = tmp_path / (download.ARCHIVE_NAME + ".part")
    partial.write_bytes(payload[:5])

    class Response(io.BytesIO):
        status = 206
        headers = {"Content-Range": f"bytes 5-{len(payload) - 1}/{len(payload)}"}

    def urlopen(request, timeout):
        assert request.get_header("Range") == "bytes=5-"
        assert timeout == 90
        return Response(payload[5:])

    monkeypatch.setattr(download.urllib.request, "urlopen", urlopen)
    archive = download._download_archive()
    assert archive.read_bytes() == payload
    assert download._download_archive() == archive


def test_official_nested_tar_extracts_images(tmp_path, monkeypatch):
    monkeypatch.setattr(download, "TRAIN_DIR", tmp_path / "train")
    inner_buffer = io.BytesIO()
    with tarfile.open(fileobj=inner_buffer, mode="w") as inner:
        data = b"training-image"
        info = tarfile.TarInfo("n00000001_1.JPEG")
        info.size = len(data)
        inner.addfile(info, io.BytesIO(data))
    archive = tmp_path / "ILSVRC2012_img_train.tar"
    with tarfile.open(archive, "w") as outer:
        contents = inner_buffer.getvalue()
        info = tarfile.TarInfo("n00000001.tar")
        info.size = len(contents)
        outer.addfile(info, io.BytesIO(contents))
    assert download._extract_tar(archive) == 1
    assert (download.TRAIN_DIR / "n00000001/n00000001_1.JPEG").read_bytes() == data
    assert archive.is_file()


def test_download_command_does_not_need_dataset_path(monkeypatch):
    calls: list[Path | None] = []
    monkeypatch.setenv("SOLO_ROOT", str(ROOT))
    monkeypatch.setattr(download, "download_imagenet", calls.append)
    assert main(["download"]) == 0
    assert calls == [None]
