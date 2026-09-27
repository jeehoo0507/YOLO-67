from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from solo.config import Config
from solo.dataset import scan_images
from solo.pipeline import PipelineFailure, run_pipeline
from solo.pseudo_labels import LabelStore


class SyntheticBackbone:
    grid = (8, 8)

    def __init__(self, failure=None):
        self.calls = []
        self.failure = failure

    def extract(self, images):
        self.calls.append(len(images))
        if self.failure == "oom":
            import torch

            raise torch.cuda.OutOfMemoryError("simulated CUDA OOM")
        feature = np.zeros((8, 8, 2), dtype=np.float32)
        feature[..., 0] = 1
        feature[2:6, 2:6] = [0, 1]
        if self.failure == "nan":
            feature[:] = np.nan
        return np.repeat(feature.reshape(1, 64, 2), len(images), axis=0)

    def warmup(self, _):
        pass

    def reset_peak_memory(self):
        pass

    def peak_memory(self):
        return 0


def dataset(tmp_path, count=7):
    source = tmp_path / "source"
    source.mkdir()
    for index in range(count):
        Image.new("RGB", (100, 80), color=(index * 20, 90, 60)).save(source / f"{index}.jpg")
    return scan_images(source)


def config():
    c = Config()
    return replace(
        c, dino=replace(c.dino, resolution=128), maskcut=replace(c.maskcut, max_objects=1)
    )


def test_batched_producer_parallel_workers_resume_and_corruption(tmp_path):
    records = dataset(tmp_path)
    before = [(r.path, open(r.path, "rb").read()) for r in records]
    store = LabelStore(tmp_path / "labels", "test", str(tmp_path / "source"))
    backbone = SyntheticBackbone()
    result = run_pipeline(records, backbone, config(), store, workers=2, batch_size=3, resume=True)
    assert result["status"] == "ok"
    assert result["images_processed"] == 7
    assert backbone.calls == [3, 3, 1]
    assert result["quality"]["mean_boxes_per_image"] == 1
    for record in records:
        assert store.valid_boxes(record) == [(0.5, 0.5, 0.5, 0.5)]
    assert all(open(path, "rb").read() == data for path, data in before)
    mtimes = [store.label_path(record).stat().st_mtime_ns for record in records]
    backbone.calls.clear()
    resumed = run_pipeline(records, backbone, config(), store, 2, 3, resume=True)
    assert resumed["images_skipped"] == 7
    assert resumed["images_processed"] == 0
    assert resumed["throughput_images_per_sec"] == 0
    assert not backbone.calls
    store.label_path(records[1]).write_text("0 .5")
    retry = run_pipeline(records, backbone, config(), store, 2, 3, resume=True)
    assert retry["images_processed"] == 1
    assert retry["images_skipped"] == 6
    assert all(
        store.label_path(record).stat().st_mtime_ns == stamp
        for index, (record, stamp) in enumerate(zip(records, mtimes, strict=True))
        if index != 1
    )


def test_decode_and_cpu_failures_never_create_success_receipts(tmp_path):
    records = dataset(tmp_path, 2)
    store = LabelStore(tmp_path / "labels", "test", str(tmp_path / "source"))
    result = run_pipeline(records, SyntheticBackbone("nan"), config(), store, 1, 2, resume=False)
    assert result["status"] == "failed"
    assert result["images_failed"] == 2
    assert all(store.valid_boxes(r) is None for r in records)
    (tmp_path / "source/1.jpg").write_bytes(b"bad jpeg")
    records = scan_images(tmp_path / "source")
    result = run_pipeline(records, SyntheticBackbone(), config(), store, 1, 2, resume=False)
    assert result["images_failed"] == 1 and result["images_processed"] == 1
    assert "decode" in result["errors"][0]["error"]


def test_cuda_oom_is_marked_in_partial_result(tmp_path):
    records = dataset(tmp_path, 1)
    store = LabelStore(tmp_path / "labels", "test", str(tmp_path / "source"))
    with pytest.raises(PipelineFailure) as failure:
        run_pipeline(records, SyntheticBackbone("oom"), config(), store, 1, 1, resume=False)
    assert failure.value.oom
    assert failure.value.result["images_processed"] == 0
    assert store.valid_boxes(records[0]) is None
