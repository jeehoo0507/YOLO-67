import json
from dataclasses import replace

import numpy as np
import pytest

from solo.config import Config, FilterConfig
from solo.dataset import ImageRecord
from solo.pseudo_labels import LabelStore, decode_labels, encode_labels, masks_to_boxes
from solo.storage import atomic_write


@pytest.fixture
def record():
    return ImageRecord("/readonly/object.jpg", "sub/object.jpg", 1234, 5678)


def test_patch_bbox_coordinates_and_original_aspect():
    mask = np.zeros((24, 24), dtype=bool)
    mask[6:18, 3:9] = True
    boxes = masks_to_boxes([mask], (400, 200), FilterConfig())
    assert boxes == [(0.25, 0.5, 0.25, 0.5)]
    assert decode_labels(encode_labels(boxes)) == boxes
    assert masks_to_boxes([mask], (4000, 100), FilterConfig()) == []


def test_box_filters_and_duplicates():
    mask = np.zeros((24, 24), dtype=bool)
    mask[4:14, 4:14] = True
    full = np.ones_like(mask)
    tiny = np.zeros_like(mask)
    tiny[0, 0] = True
    assert len(masks_to_boxes([mask, mask, full, tiny], (200, 200), FilterConfig())) == 1


@pytest.mark.parametrize(
    "data",
    [
        b"1 .5 .5 .1 .1\n",
        b"0 nan .5 .1 .1\n",
        b"0 .5 .5 0 .1\n",
        b"0 .01 .5 .2 .2\n",
        b"0 .5 .5 1.1 .1\n",
        b"0 .5 .5 .1\n",
        b"\n",
    ],
)
def test_reject_invalid_labels(data):
    with pytest.raises(ValueError):
        decode_labels(data)


def test_empty_success_is_distinct_from_partial_output(tmp_path, record):
    store = LabelStore(tmp_path, "fingerprint", "/readonly")
    store.label_path(record).parent.mkdir(parents=True)
    store.label_path(record).write_bytes(b"")
    assert store.valid_boxes(record) is None
    store.save(record, [], (200, 100))
    assert store.valid_boxes(record) == []
    store.receipt_path(record).write_text("{incomplete")
    assert store.valid_boxes(record) is None


def test_receipt_hash_and_configuration_guard(tmp_path, record):
    store = LabelStore(tmp_path, "fingerprint", "/readonly")
    boxes = [(0.5, 0.5, 0.25, 0.25)]
    store.save(record, boxes, (200, 100))
    assert store.valid_boxes(record) == boxes
    assert replace(store, fingerprint="new").valid_boxes(record) is None
    assert replace(store, dataset_root="/other").valid_boxes(record) is None
    assert store.valid_boxes(replace(record, size=2222)) is None
    store.label_path(record).write_bytes(b"")
    assert store.valid_boxes(record) is None
    store.save(record, boxes, (200, 100))
    receipt = json.loads(store.receipt_path(record).read_text())
    receipt["image_size"] = [-1, 0]
    store.receipt_path(record).write_text(json.dumps(receipt))
    assert store.valid_boxes(record) is None


def test_atomic_write_preserves_previous_file_on_interruption(tmp_path, monkeypatch):
    target = tmp_path / "label.txt"
    atomic_write(target, b"previous")

    def interrupted(*_):
        raise OSError("simulated interruption")

    monkeypatch.setattr("solo.storage.os.replace", interrupted)
    with pytest.raises(OSError, match="interruption"):
        atomic_write(target, b"new")
    assert target.read_bytes() == b"previous"
    assert list(tmp_path.iterdir()) == [target]


def test_interruption_between_label_and_receipt_requires_retry(tmp_path, record, monkeypatch):
    store = LabelStore(tmp_path, "fingerprint", "/readonly")
    store.save(record, [(0.5, 0.5, 0.25, 0.25)], (200, 100))

    def interrupted(*_):
        raise OSError("receipt interrupted")

    monkeypatch.setattr("solo.pseudo_labels.write_json", interrupted)
    with pytest.raises(OSError):
        store.save(record, [], (200, 100))
    assert store.valid_boxes(record) is None


def test_fingerprint_ignores_scheduling_but_tracks_algorithm():
    c = Config()
    assert c.fingerprint("sha") == replace(
        c, pipeline=replace(c.pipeline, queue_batches=9)
    ).fingerprint("sha")
    assert c.fingerprint("sha") != replace(c, maskcut=replace(c.maskcut, tau=0.2)).fingerprint(
        "sha"
    )
