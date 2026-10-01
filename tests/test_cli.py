from pathlib import Path

from solo.cli import main
from solo.config import ROOT


def test_dataset_path_defaults_to_repository_folder(monkeypatch):
    captured = []
    monkeypatch.setenv("SOLO_ROOT", str(ROOT))
    monkeypatch.setattr("solo.benchmark.benchmark", lambda path, config: captured.append(path))
    monkeypatch.setattr("solo.generation.generate", lambda path, config: captured.append(path))
    monkeypatch.setattr("solo.reporting.create_preview", lambda path, config: captured.append(path))
    monkeypatch.setattr("solo.detector.train_detector", lambda path, config: captured.append(path))
    assert main(["benchmark"]) == 0
    assert main(["generate"]) == 0
    assert main(["preview"]) == 0
    assert main(["train"]) == 0
    assert captured == [ROOT / "data/coco/images"] * 4


def test_explicit_external_path_still_works(monkeypatch):
    captured = []
    monkeypatch.setenv("SOLO_ROOT", str(ROOT))
    monkeypatch.setattr("solo.benchmark.benchmark", lambda path, config: captured.append(path))
    path = Path("/example/existing/imagenet")
    assert main(["benchmark", str(path)]) == 0
    assert captured == [path]


def test_coco_download_command(monkeypatch):
    calls = []
    monkeypatch.setenv("SOLO_ROOT", str(ROOT))
    monkeypatch.setattr("solo.coco.download_coco", lambda: calls.append("images only"))
    assert main(["download-coco"]) == 0
    assert calls == ["images only"]


def test_predict_defaults_and_custom_image(monkeypatch):
    calls = []
    monkeypatch.setenv("SOLO_ROOT", str(ROOT))
    monkeypatch.setattr("solo.prediction.predict_objects", lambda *args: calls.append(args))
    assert main(["predict"]) == 0
    assert calls[0][0] == ROOT / "data/coco/images/val2017"
    assert calls[0][1] == ROOT / "outputs/coco-yolo11n/weights/best.pt"
    assert calls[0][3:] == (16, 0.25)
    assert main(["predict", "/example/photo.jpg", "--conf", "0.1"]) == 0
    assert calls[1][0] == Path("/example/photo.jpg")
    assert calls[1][4] == 0.1


def test_crowded_profile_selects_separate_checkpoint_and_comparison(monkeypatch):
    calls = []
    monkeypatch.setenv("SOLO_ROOT", str(ROOT))
    monkeypatch.setattr("solo.prediction.predict_objects", lambda *args: calls.append(args))
    assert main(["predict", "--crowded"]) == 0
    assert calls[0][1] == ROOT / "outputs/coco-yolo11n-crowded/weights/best.pt"
    monkeypatch.setattr("solo.comparison.compare", lambda *args: calls.append(args))
    assert main(["compare"]) == 0
    assert calls[1][2].maskcut.all_components
    assert calls[1][2].maskcut.split_depth == 0
    assert calls[1][3] == 16
