from pathlib import Path

from solo.cli import main
from solo.config import ROOT


def test_dataset_path_defaults_to_repository_folder(monkeypatch):
    captured = []
    monkeypatch.setenv("SOLO_ROOT", str(ROOT))
    monkeypatch.setattr("solo.benchmark.benchmark", lambda path, config: captured.append(path))
    monkeypatch.setattr("solo.generation.generate", lambda path, config: captured.append(path))
    assert main(["benchmark"]) == 0
    assert main(["generate"]) == 0
    assert captured == [ROOT / "data/imagenet/train"] * 2


def test_explicit_external_path_still_works(monkeypatch):
    captured = []
    monkeypatch.setenv("SOLO_ROOT", str(ROOT))
    monkeypatch.setattr("solo.benchmark.benchmark", lambda path, config: captured.append(path))
    path = Path("/example/existing/imagenet")
    assert main(["benchmark", str(path)]) == 0
    assert captured == [path]
