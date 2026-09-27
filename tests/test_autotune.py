import importlib
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from PIL import Image

from solo.config import Config
from solo.generation import load_best


def setup_benchmark(tmp_path, monkeypatch):
    module = importlib.import_module("solo.benchmark")
    dataset = tmp_path / "input"
    dataset.mkdir()
    for index in range(4):
        Image.new("RGB", (50, 50)).save(dataset / f"{index}.jpg")
    monkeypatch.setattr(module, "local_path", lambda path: tmp_path / path)
    monkeypatch.setattr(module, "worker_limit", lambda _: 2)
    system = {"gpus": [], "cuda_runtime": None, "cpu": "test", "python": "test"}
    monkeypatch.setattr(module, "system_info", lambda: system)
    report = tmp_path / "reports/test"
    report.mkdir(parents=True)
    payload = {"runs": [], "warnings": [], "errors": [], "system": system, "status": "running"}
    monkeypatch.setattr(module, "new_report", lambda *_: (report, payload))
    monkeypatch.setattr(module, "save_report", lambda *_: None)
    monkeypatch.setattr(module, "preview", lambda *_: 4)
    monkeypatch.setattr(
        "solo.dino.DinoBackbone",
        lambda _: SimpleNamespace(
            metadata={}, device=SimpleNamespace(type="cpu"), clear_memory=lambda: None
        ),
    )
    c = Config()
    c = replace(
        c,
        benchmark=replace(
            c.benchmark,
            smoke_images=4,
            scaling_images=4,
            confirmation_images=4,
            worker_counts=[1, 2],
            batch_sizes=[1, 2, 4, 8],
            preview_images=4,
        ),
    )
    return module, dataset, c, payload


def successful_result(count, workers, batch):
    return {
        "status": "ok",
        "images_processed": count,
        "images_failed": 0,
        "throughput_images_per_sec": 10,
        "elapsed_seconds": count / 10,
        "cpu_workers": workers,
        "batch_size": batch,
        "timing_seconds": {},
        "timing_ms_per_image": {},
        "quality": {"total_boxes": count},
        "errors": [],
    }


def test_smoke_failure_stops_autotune_and_preserves_failure_status(tmp_path, monkeypatch):
    module, dataset, c, payload = setup_benchmark(tmp_path, monkeypatch)
    calls = []

    def failed_smoke(records, _backbone, _config, _store, workers, batch, **_kwargs):
        calls.append(batch)
        result = successful_result(len(records), workers, batch)
        result["status"] = "failed"
        return result

    monkeypatch.setattr(module, "run_pipeline", failed_smoke)
    with pytest.raises(RuntimeError, match="Smoke test failed"):
        module.benchmark(dataset, c)
    assert calls == [1]
    assert payload["status"] == "failed"
    assert [r["stage"] for r in payload["runs"]] == ["smoke"]


def test_oom_stops_larger_batches_and_speed_fail_still_completes(tmp_path, monkeypatch):
    module, dataset, c, payload = setup_benchmark(tmp_path, monkeypatch)
    calls = []

    def trial(records, _backbone, _config, store, workers, batch, **_kwargs):
        calls.append(batch)
        store.root.mkdir(parents=True)
        for record in records:
            store.save(record, [(0.5, 0.5, 0.2, 0.2)], (50, 50))
        result = successful_result(len(records), workers, batch)
        if batch == 4:
            result.update({"status": "failed", "oom": True})
            raise module.PipelineFailure("test OOM", result, oom=True)
        return result

    monkeypatch.setattr(module, "run_pipeline", trial)
    module.benchmark(dataset, c)
    assert 4 in calls and 8 not in calls
    assert payload["status"] == "complete"
    assert payload["estimate"]["24h_target"] == "FAIL"
    assert not payload["best_config"]["qualified_confirmation"]
    assert not (tmp_path / "work/best_config.json").exists()
    assert any("OOM" in warning for warning in payload["warnings"])


def test_full_generation_rejects_unqualified_or_wrong_hardware(tmp_path, monkeypatch):
    monkeypatch.setattr("solo.generation.local_path", lambda path: tmp_path / path)
    monkeypatch.setattr("solo.generation.worker_limit", lambda _: 4)
    c = Config()
    system = {"gpus": [{"name": "A5000", "vram_bytes": 24 * 1024**3}], "cuda_runtime": "12.6"}
    path = tmp_path / "work/best_config.json"
    path.parent.mkdir(parents=True)
    candidate = {
        "qualified_confirmation": False,
        "fingerprint": "test",
        "cpu_workers": 2,
        "hardware": system,
        "configuration": c.to_dict(),
    }
    path.write_text(json.dumps(candidate))
    with pytest.raises(RuntimeError, match="10k benchmark"):
        load_best(c, "test", system)
    candidate["qualified_confirmation"] = True
    path.write_text(json.dumps(candidate))
    assert load_best(c, "test", system)["cpu_workers"] == 2
    with pytest.raises(RuntimeError, match="10k benchmark"):
        load_best(c, "test", {"gpus": [], "cuda_runtime": None})
