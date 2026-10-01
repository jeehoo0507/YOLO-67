import hashlib
import json
from dataclasses import asdict, replace

import pytest

from solo.config import ROOT, Config, load_config, validate
from solo.detector import adapter_path


def test_baseline_fingerprint_stays_compatible_with_existing_receipts():
    config = Config()
    old_maskcut = {k: v for k, v in asdict(config.maskcut).items() if k in {
        "tau", "epsilon", "max_objects", "min_mask_area", "max_mask_iou", "crf",
    }}
    payload = {
        "schema": "solo-pseudo-v1", "affinity_dtype": "float64",
        "dino": {k: v for k, v in asdict(config.dino).items() if k != "device"},
        "maskcut": old_maskcut, "filtering": asdict(config.filtering),
        "checkpoint_sha256": "abc",
    }
    expected = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    assert config.fingerprint("abc") == expected
    enhanced = replace(config, maskcut=replace(config.maskcut, split_depth=2))
    assert enhanced.fingerprint("abc") != expected


def test_crowded_outputs_are_separate_and_config_is_valid():
    baseline = load_config()
    crowded = load_config(ROOT / "configs/crowded.toml")
    assert crowded.pipeline.output_dir != baseline.pipeline.output_dir
    assert crowded.train.run_name != baseline.train.run_name
    assert adapter_path(crowded) != adapter_path(baseline)
    assert crowded.fingerprint("abc") != baseline.fingerprint("abc")
    with pytest.raises(ValueError, match="run_name"):
        validate(replace(crowded, train=replace(crowded.train, run_name="../outside")))
