import io
import zipfile

import pytest

from solo import coco


def zip_bytes(split="train2017"):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr(f"{split}/000000000001.jpg", b"image one")
        archive.writestr(f"{split}/000000000002.jpg", b"image two")
    return stream.getvalue()


def test_resume_download_validates_range_and_remote_identity(tmp_path, monkeypatch):
    data = zip_bytes()
    archive = tmp_path / "train.zip"
    partial = archive.with_suffix(".zip.part")
    partial.write_bytes(data[:20])
    captured = []

    def response(request, timeout):
        captured.append(request)
        result = io.BytesIO(data[20:])
        result.status = 206
        result.headers = {
            "Content-Range": f"bytes 20-{len(data) - 1}/{len(data)}",
            "Content-Length": str(len(data) - 20),
            "ETag": '"fixed"',
        }
        return result

    monkeypatch.setattr(coco.urllib.request, "urlopen", response)
    coco._download("https://example.test/train.zip", archive, len(data), '"fixed"')
    assert captured[0].get_header("Range") == "bytes=20-"
    assert captured[0].get_header("If-range") == '"fixed"'
    assert archive.read_bytes() == data
    assert not partial.exists()


def test_completed_partial_is_recovered_without_network(tmp_path, monkeypatch):
    data = zip_bytes()
    archive = tmp_path / "train.zip"
    archive.with_suffix(".zip.part").write_bytes(data)
    monkeypatch.setattr(coco.urllib.request, "urlopen", lambda *_a, **_k: pytest.fail("network"))
    coco._download("https://example.test/train.zip", archive, len(data), '"fixed"')
    assert archive.read_bytes() == data


def test_changed_remote_archive_keeps_partial_intact(tmp_path, monkeypatch):
    data = zip_bytes()
    archive = tmp_path / "train.zip"
    partial = archive.with_suffix(".zip.part")
    partial.write_bytes(data[:20])
    response = io.BytesIO(data[20:])
    response.status = 206
    response.headers = {
        "Content-Range": f"bytes 20-{len(data) - 1}/{len(data)}",
        "Content-Length": str(len(data) - 20),
        "ETag": '"changed"',
    }
    monkeypatch.setattr(coco.urllib.request, "urlopen", lambda *_a, **_k: response)
    with pytest.raises(RuntimeError, match="archive changed"):
        coco._download("https://example.test/train.zip", archive, len(data), '"fixed"')
    assert partial.read_bytes() == data[:20]


def test_zip_path_cannot_escape_dataset(tmp_path):
    archive = tmp_path / "train.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("train2017/../../escaped.jpg", b"bad")
    with pytest.raises(RuntimeError, match="Unsafe COCO zip path"):
        coco._extract(archive, "train2017", tmp_path / "images", 1)
    assert not (tmp_path / "escaped.jpg").exists()


def test_extract_repairs_same_size_corruption_and_is_resumable(tmp_path):
    archive = tmp_path / "train.zip"
    archive.write_bytes(zip_bytes())
    images = tmp_path / "images"
    coco._extract(archive, "train2017", images, 2)
    target = images / "train2017/000000000001.jpg"
    target.write_bytes(b"bad bytes")
    coco._extract(archive, "train2017", images, 2)
    assert target.read_bytes() == b"image one"
    assert not list(images.rglob("*.tmp"))


def test_coco_download_only_images_and_reuses_completed_splits(tmp_path, monkeypatch):
    monkeypatch.setattr(coco, "local_path", lambda value: tmp_path / value)
    monkeypatch.setattr(
        coco,
        "SPLITS",
        {
            split: (f"https://example.test/{split}.zip", 2, 100, '"fixed"')
            for split in ("train2017", "val2017")
        },
    )
    calls = []

    def download(url, path, total, etag):
        calls.append(url)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(zip_bytes(path.stem))

    monkeypatch.setattr(coco, "_download", download)
    images = coco.download_coco()
    coco.download_coco()
    assert len(calls) == 2
    assert len(list(images.rglob("*.jpg"))) == 4
    assert not list((tmp_path / "data/coco/downloads").glob("*.zip"))
    assert not (tmp_path / "data/coco/annotations").exists()
