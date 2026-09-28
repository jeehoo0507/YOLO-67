import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from solo.config import Config
from solo.dataset import ImageRecord, scan_images
from solo.detector import prepare_dataset, train_detector
from solo.dino import CHECKPOINT_SHA256
from solo.pseudo_labels import LabelStore


def test_training_adapter_links_valid_pseudo_labels_without_copy(tmp_path, monkeypatch):
    dataset = tmp_path / "source"
    for index in range(40):
        split = "val2017" if index < 5 else "train2017"
        path = dataset / split / f"{index:03}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (100, 100), "white").save(path)
    records = scan_images(dataset)
    config = Config()
    monkeypatch.setattr("solo.detector.local_path", lambda value: tmp_path / value)
    store = LabelStore(
        tmp_path / "pseudo/coco",
        config.fingerprint(CHECKPOINT_SHA256),
        str(dataset.resolve()),
        durable=False,
    )
    for record in records:
        store.save(record, [(0.5, 0.5, 0.25, 0.25)], (100, 100))

    adapter = prepare_dataset(dataset.resolve(), config, records)
    assert adapter["train"] + adapter["val"] == 40
    assert adapter["train"] == 35 and adapter["val"] == 5
    assert json.loads(Path(adapter["path"]).read_text())["names"] == ["object"]
    for split in ("train", "val"):
        for image in (tmp_path / "work/coco_yolo_dataset/images" / split).rglob("*.jpg"):
            label = (
                tmp_path
                / "work/coco_yolo_dataset/labels"
                / split
                / image.relative_to(tmp_path / "work/coco_yolo_dataset/images" / split).with_suffix(
                    ".txt"
                )
            )
            assert image.is_symlink() and label.is_symlink()
            assert label.read_text().startswith("0 ")

    store.receipt_path(records[0]).unlink()
    with pytest.raises(RuntimeError, match="Missing, corrupt, or incompatible pseudo label"):
        prepare_dataset(dataset.resolve(), config, records)


def test_train_starts_from_random_yolo11n_and_reports_eta(tmp_path, monkeypatch, capsys):
    dataset = tmp_path / "source"
    dataset.mkdir()
    output = tmp_path / "outputs/coco-yolo11n"
    seen = {}

    class FakeYOLO:
        def __init__(self, model):
            seen["model"] = model
            self.callbacks = {}

        def add_callback(self, event, callback):
            self.callbacks[event] = callback

        def train(self, **kwargs):
            seen["kwargs"] = kwargs
            trainer = SimpleNamespace(
                start_epoch=0, epochs=kwargs["epochs"], epoch=0, train_loader=range(2)
            )
            self.callbacks["on_train_start"](trainer)
            self.callbacks["on_train_batch_end"](trainer)
            self.callbacks["on_fit_epoch_end"](trainer)
            weights = output / "weights"
            weights.mkdir(parents=True)
            (weights / "last.pt").write_bytes(b"checkpoint")
            (weights / "best.pt").write_bytes(b"checkpoint")

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeYOLO))
    monkeypatch.setattr("solo.detector.local_path", lambda value: tmp_path / value)
    monkeypatch.setattr(
        "solo.detector.scan_images",
        lambda _: [ImageRecord(str(dataset / "train2017/x.jpg"), "train2017/x.jpg", 1, 1)],
    )
    monkeypatch.setattr(
        "solo.detector.prepare_dataset",
        lambda *_: {
            "path": str(tmp_path / "work/coco_yolo_dataset/data.yaml"),
            "train": 1,
            "val": 1,
            "zero_box": 0,
            "images": 2,
        },
    )
    monkeypatch.setattr("solo.detector.system_info", lambda: {"gpus": []})
    monkeypatch.setattr("solo.detector.git_context", lambda: {"commit_sha": "test"})

    report = train_detector(dataset, Config())
    assert seen["model"] == "yolo11n.yaml"
    assert seen["kwargs"]["amp"] is False
    assert seen["kwargs"]["pretrained"] is False
    assert seen["kwargs"]["resume"] is False
    assert "ETA ~" in capsys.readouterr().out
    assert (
        json.loads((report / "training.json").read_text())["initialization"]
        == "random initialization"
    )
    assert json.loads((tmp_path / "work/coco-train-state.json").read_text())["status"] == "complete"
