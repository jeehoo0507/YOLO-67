import json
import sys
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from solo.config import Config
from solo.prediction import predict_objects


def test_predict_saves_boxes_and_empty_results_without_changing_inputs(tmp_path, monkeypatch):
    source = tmp_path / "input"
    source.mkdir()
    for name in ("a.jpg", "a.png"):
        Image.new("RGB", (80, 60), "white").save(source / name)
    original = {p: p.read_bytes() for p in source.iterdir()}
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"test checkpoint")
    monkeypatch.setattr("solo.config.ROOT", tmp_path)
    calls = []

    class FakeYOLO:
        names = {0: "object"}
        task = "detect"

        def __init__(self, path):
            assert path == str(weights)

        def predict(self, **kwargs):
            calls.append(kwargs)
            return [SimpleNamespace(boxes=SimpleNamespace(
                xyxy=torch.tensor([[10., 15., 50., 45.]]) if len(calls) == 1 else torch.empty(0, 4),
                conf=torch.tensor([0.8]) if len(calls) == 1 else torch.empty(0),
            ))]

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeYOLO))
    output = predict_objects(source, weights, Config(), limit=0)
    assert len(calls) == 2
    assert all(not call["save"] for call in calls)
    assert all(p.read_bytes() == content for p, content in original.items())
    box = json.loads((output / "boxes/a.jpg.json").read_text())["detections"][0]
    assert box["class_id"] == 0
    assert box["xyxy"] == [10, 15, 50, 45]
    assert json.loads((output / "boxes/a.png.json").read_text())["detections"] == []
    assert Image.open(output / "images/a.jpg.jpg").size == (80, 60)
    assert json.loads((output / "summary.json").read_text())["images"] == 2
    second = predict_objects(source, weights, Config(), limit=1)
    assert second != output
    assert json.loads((second / "summary.json").read_text())["images"] == 1


def test_missing_weights_does_not_download(tmp_path):
    with pytest.raises(FileNotFoundError, match="Run ./solo train first"):
        predict_objects(tmp_path, tmp_path / "missing.pt", Config())
