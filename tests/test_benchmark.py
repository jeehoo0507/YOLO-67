from dataclasses import replace

import pytest

from solo.benchmark import choose_best
from solo.config import ROOT, Config, local_path, protect_dataset
from solo.dataset import ImageRecord, deterministic_subset, scan_images, subset_digest
from solo.quality import QualityStats
from solo.reporting import estimate
from solo.system_info import worker_candidates


def test_stable_throughput_selection_rejects_fast_unstable_candidate():
    candidates = [
        {"ok": True, "stable_throughput": 30, "stability_ratio": 0.5},
        {"ok": True, "stable_throughput": 20, "stability_ratio": 0.95},
        {"ok": False, "stable_throughput": 40, "stability_ratio": 1},
    ]
    assert choose_best(candidates, 0.85) is candidates[1]
    with pytest.raises(RuntimeError, match="stable"):
        choose_best(candidates[:1], 0.85)


def test_subset_is_order_independent_and_has_no_class_targets():
    records = [ImageRecord(f"/unlabeled/{i}.jpg", f"folder/{i}.jpg", 1, 1) for i in range(20)]
    a = deterministic_subset(records, 5, 10)
    b = deterministic_subset(list(reversed(records)), 5, 10)
    assert a == b
    assert subset_digest(a) == subset_digest(b)
    assert a != deterministic_subset(records, 5, 11)


def test_workers_cannot_exceed_limit():
    assert worker_candidates([], 24) == [1, 2, 4, 8, 16, 24]
    assert worker_candidates([1, 4, 32], 8) == [1, 4]


def test_standard_yolo_stem_and_filename_collision(tmp_path):
    from PIL import Image

    Image.new("RGB", (20, 20)).save(tmp_path / "object.jpg")
    assert scan_images(tmp_path)[0].key == "object.txt"
    Image.new("RGB", (20, 20)).save(tmp_path / "object.png")
    with pytest.raises(ValueError, match="collision"):
        scan_images(tmp_path)


def test_target_fail_is_an_estimate_not_an_exception():
    c = Config()
    assert estimate({"throughput_images_per_sec": 10}, c)["24h_target"] == "FAIL"
    passed = estimate({"throughput_images_per_sec": 20}, c)
    assert passed["24h_target"] == "PASS"
    assert passed["estimated_imagenet_hours"] < 24


def test_quality_median_zero_boxes_and_pixel_aspect():
    stats = QualityStats()
    stats.add([], (100, 100))
    stats.add([(0.5, 0.5, 0.2, 0.2)], (200, 100))
    report = stats.report()
    assert report["median_boxes_per_image"] == 0.5
    assert report["zero_box_image_ratio"] == 0.5
    assert sum(report["bbox_aspect_ratio_distribution"]["counts"]) == 1


def test_output_is_local_and_cannot_overlap_input():
    with pytest.raises(ValueError, match="repository"):
        local_path("../outside")
    with pytest.raises(ValueError, match="overlap"):
        protect_dataset(ROOT / "data", ROOT / "data/labels")
    c = Config()
    assert c.fingerprint("sha") != replace(c, dino=replace(c.dino, resolution=512)).fingerprint(
        "sha"
    )
