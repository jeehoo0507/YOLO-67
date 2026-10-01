"""Small, fixed PlantDoc few-shot experiment; annotations are evaluation-only."""

from __future__ import annotations

import gc
import json
import time
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from scipy.optimize import linear_sum_assignment

from .config import ROOT, load_config, local_path
from .maskcut import discover_masks
from .matcher import match_prototypes
from .prototypes import make_prototypes
from .pseudo_labels import box_iou, masks_to_boxes
from .reporting import new_report
from .roi_pooling import roi_pool
from .storage import atomic_write, sha256_file, write_json
from .system_info import system_info

LABEL_MAP = {
    "Tomato leaf": "healthy",
    "Tomato Early blight leaf": "early_blight",
    "Tomato leaf yellow virus": "yellow_virus",
}
COLORS = {"healthy": "#36db90", "early_blight": "#ff745a", "yellow_virus": "#ffd448"}
YOLO_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"
YOLO_SHA = "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1"


def fetch_verified(url: str, path: Path, checksum: str):
    if path.exists() and sha256_file(path) == checksum:
        return
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read()
    import hashlib

    if hashlib.sha256(data).hexdigest() != checksum:
        raise ValueError(f"Checksum mismatch: {url}")
    atomic_write(path, data)


def read_reference(item, actual_size):
    root = ET.parse(ROOT / item["evaluation_xml"]).getroot()
    width, height = float(root.findtext("size/width")), float(root.findtext("size/height"))
    if abs(width / height - actual_size[0] / actual_size[1]) > 0.01:
        raise ValueError("Annotation and image aspect ratios disagree")
    boxes, labels = [], []
    for obj in root.findall("object"):
        label = LABEL_MAP.get(obj.findtext("name"))
        if label is None:
            continue
        x0, y0, x1, y1 = [
            float(obj.findtext("bndbox/" + k)) for k in ("xmin", "ymin", "xmax", "ymax")
        ]
        # VOC inclusive one-based pixels -> half-open original-image coordinates.
        x0, y0 = max(0.0, x0 - 1), max(0.0, y0 - 1)
        x1, y1 = min(width, x1), min(height, y1)
        if x1 <= x0 or y1 <= y0:
            raise ValueError("Invalid reference box")
        boxes.append(
            ((x0 + x1) / 2 / width, (y0 + y1) / 2 / height, (x1 - x0) / width, (y1 - y0) / height)
        )
        labels.append(label)
    return boxes, labels


def pair_boxes(proposals, reference, threshold=0.5):
    """Maximum-cardinality IoU>=threshold matching, then maximize overlap; no labels."""
    if not proposals or not reference:
        return []
    ious = np.array([[box_iou(p, r) for r in reference] for p in proposals])
    rows, cols = linear_sum_assignment(-((ious >= threshold) * 1000 + ious))
    return [(int(p), int(r)) for p, r in zip(rows, cols, strict=True) if ious[p, r] >= threshold]


def render_tile(picture, title, boxes, labels, scores=None):
    tile = Image.new("RGB", (384, 340), "#10151e")
    font = ImageFont.load_default(size=14)
    thumb = ImageOps.contain(picture, (368, 284))
    d = ImageDraw.Draw(thumb)
    for i, (box, label) in enumerate(zip(boxes, labels, strict=True)):
        x, y, w, h = box
        xy = (
            (x - w / 2) * thumb.width,
            (y - h / 2) * thumb.height,
            (x + w / 2) * thumb.width,
            (y + h / 2) * thumb.height,
        )
        color = COLORS[label]
        d.rectangle(xy, outline=color, width=2)
        text = str(i + 1) if scores is None else f"{i + 1}:{scores[i]:.2f}"
        d.text(
            (xy[0] + 1, xy[1] + 1), text, fill=color, stroke_width=1, stroke_fill="black", font=font
        )
    tile.paste(thumb, ((384 - thumb.width) // 2, 50 + (284 - thumb.height) // 2))
    ImageDraw.Draw(tile).text((8, 7), title, fill="white", font=font)
    ImageDraw.Draw(tile).text(
        (8, 27), "green:healthy red:blight yellow:virus", fill="#bcc8dc", font=font
    )
    return tile


def crop_demo():
    import torch
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2
    from ultralytics import YOLO

    from .backbone_comparison import ComparisonBackbone
    from .sam_comparison import (
        AMG_SETTINGS,
        SAM_CONFIG,
        SAM_REVISION,
        SAM_SHA256,
        filter_sam_masks,
        prepare_sam_checkpoint,
    )

    manifest = json.loads((ROOT / "configs/crop-demo.json").read_text())
    config = load_config(ROOT / "configs/crowded.toml")
    config = replace(
        config, dino=replace(config.dino, resolution=448, mixed_precision=False, device="cpu")
    )
    directory, payload = new_report("crop-demo", config, system_info(), ROOT / "data/crop-study")
    payload.update(
        manifest=manifest, samples=[], backbones={}, metrics={}, feature_forwards={},
        matching_feature="final LayerNorm patch tokens; fractional mean ROI pooling",
        support_pooling="whole support image; mean normalized support embeddings per class",
        proposal_matching="maximum count of IoU>=0.5 one-to-one pairs, then maximum total IoU",
        max_input_long_side=1280,
    )
    payload["warnings"] = [
        "3 support images, one per class; 9 visually selected natural-background query images.",
        "Dataset labels are not independently verified agronomic diagnoses.",
        "Queries have garden/pot/field-like backgrounds, not a verified commercial-field dataset.",
        "COCO-pretrained YOLO11n is a control, NOT the untrained/unavailable SOLO object detector.",
        "YOLO class labels are ignored, which does not make its proposals class agnostic.",
        "Reference boxes are evaluation-only, including the marked oracle-ROI diagnostic.",
        "Known-three-class top1 matching; no calibrated unknown/background rejection.",
        "Annotation incompleteness means unmatched proposals are not all proven false positives.",
        "No weights or feature-fusion layers are trained. Query DINO forward only once/model.",
    ]
    payload["timing_notes"] = (
        "CPU FP32, feature/SAM threads=4; actual YOLO threads recorded per sample. "
        "Unwarmed exploratory service timings, not a throughput benchmark."
    )
    features, images, sizes, proposals = {}, {}, {}, {}
    supports = [x for x in manifest["images"] if x["role"] == "support"]
    queries = [x for x in manifest["images"] if x["role"] == "query"]
    try:
        for item in manifest["images"]:
            path = local_path(item["file"])
            fetch_verified(item["source_url"], path, item["sha256"])
            with Image.open(path) as source:
                picture = source.convert("RGB")
            sizes[item["id"]] = picture.size
            picture.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
            images[item["id"]] = picture
        support_grid = Image.new("RGB", (1152, 340), "#10151e")
        for i, item in enumerate(supports):
            support_grid.paste(
                render_tile(images[item["id"]], "Support: " + item["class"], [], []), (384 * i, 0)
            )
        support_grid.save(directory / "support.jpg", quality=90, optimize=True)
        prototypes = {}
        for version in ("v1", "v2", "v3"):
            backbone = ComparisonBackbone(version, 448)
            torch.set_num_threads(4)
            payload["backbones"][version] = backbone.metadata
            features[version] = {}
            payload["feature_forwards"][version] = 0
            embeddings = []
            for item in manifest["images"]:
                picture = images[item["id"]]
                array = (
                    np.asarray(
                        picture.resize((448, 448), Image.Resampling.LANCZOS), dtype=np.float32
                    )
                    / 255.0
                )
                array = (array - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
                    [0.229, 0.224, 0.225],
                    dtype=np.float32,
                )
                started = time.perf_counter()
                dense = backbone.extract(array.transpose(2, 0, 1).copy())
                payload["feature_forwards"][version] += 1
                features[version][item["id"]] = dense["tokens"]
                if item["role"] == "support":
                    embeddings.append(
                        roi_pool(dense["tokens"], backbone.grid, [(0.5, 0.5, 1.0, 1.0)])[0]
                    )
                elif version == "v1":
                    masks = discover_masks(dense["keys"], backbone.grid, config.maskcut)
                    proposals[item["id"]] = {
                        "maskcut": masks_to_boxes(
                            masks,
                            picture.size,
                            config.filtering,
                        )
                    }
                print(
                    f"DINO {version}: {item['id']} ({time.perf_counter() - started:.2f}s)",
                    flush=True,
                )
            prototypes[version] = make_prototypes(
                np.stack(embeddings), [s["class"] for s in supports]
            )
            backbone.close()
            del backbone
            gc.collect()
        weights = local_path("weights/yolo11n.pt")
        fetch_verified(YOLO_URL, weights, YOLO_SHA)
        yolo = YOLO(str(weights))
        if len(yolo.names) != 80:
            raise ValueError("Expected the pinned COCO YOLO11n comparison checkpoint")
        sam = (
            build_sam2(
                SAM_CONFIG, str(prepare_sam_checkpoint()), device="cpu", apply_postprocessing=False
            )
            .eval()
            .requires_grad_(False)
        )
        amg = dict(AMG_SETTINGS, points_per_batch=8)
        generator = SAM2AutomaticMaskGenerator(sam, **amg)
        payload["proposals"] = {
            "yolo": {
                "weights": str(weights),
                "sha256": YOLO_SHA,
                "pretraining": "COCO80",
                "conf": 0.05,
                "iou": 0.7,
                "max_det": 100,
                "imgsz": 640,
                "agnostic_nms": True,
            },
            "sam": {
                "model": "SAM 2.1 Hiera-Tiny",
                "sha256": SAM_SHA256,
                "revision": SAM_REVISION,
                "amg": amg,
                "image_size": 1024,
            },
            "maskcut": config.to_dict(),
        }
        # All proposals are generated before reading any query reference annotation.
        for i, item in enumerate(queries):
            picture = images[item["id"]]
            started = time.perf_counter()
            result = yolo.predict(
                picture,
                imgsz=640,
                conf=0.05,
                iou=0.7,
                max_det=100,
                agnostic_nms=True,
                device="cpu",
                save=False,
                verbose=False,
            )[0]
            proposals[item["id"]]["yolo"] = result.boxes.xywhn.cpu().tolist()
            yolo_seconds = time.perf_counter() - started
            yolo_threads = torch.get_num_threads()
            torch.set_num_threads(4)
            started = time.perf_counter()
            with torch.inference_mode():
                annotations = generator.generate(np.array(picture))
            proposals[item["id"]]["sam"], _ = filter_sam_masks(
                annotations,
                picture.size,
                config.filtering,
            )
            sam_seconds = time.perf_counter() - started
            counts = {name: len(boxes) for name, boxes in proposals[item["id"]].items()}
            payload["samples"].append(
                {
                    "id": item["id"],
                    "proposals": proposals[item["id"]],
                    "yolo_seconds": yolo_seconds,
                    "yolo_threads": yolo_threads,
                    "sam_seconds": sam_seconds,
                }
            )
            write_json(directory / "results.json", payload)
            print(f"Proposals {i + 1}/{len(queries)}: {counts}", flush=True)
            del annotations, result
            gc.collect()
        for item, sample in zip(queries, payload["samples"], strict=True):
            fetch_verified(
                item["annotation_url"],
                local_path(item["evaluation_xml"]),
                item["annotation_sha256"],
            )
            reference, truth = read_reference(item, sizes[item["id"]])
            sample.update(reference_boxes=reference, reference_labels=truth, predictions={})
            for version in features:
                dense = features[version][item["id"]]
                grid = tuple(payload["backbones"][version]["grid"])
                classes, prototype = prototypes[version]
                predictions = {}
                for method, boxes in {**proposals[item["id"]], "reference_roi": reference}.items():
                    predictions[method] = match_prototypes(
                        roi_pool(dense, grid, boxes), classes, prototype
                    )
                    metric = payload["metrics"].setdefault(
                        version + "/" + method,
                        {
                            "reference_leaves": 0,
                            "proposed_boxes": 0,
                            "localized": 0,
                            "localized_and_correct": 0,
                            "by_class": {},
                        },
                    )
                    pairs = (
                        list(enumerate(range(len(reference))))
                        if method == "reference_roi"
                        else (pair_boxes(boxes, reference))
                    )
                    metric["reference_leaves"] += len(reference)
                    metric["proposed_boxes"] += len(boxes)
                    metric["localized"] += len(pairs)
                    for label in truth:
                        metric["by_class"].setdefault(label, {"total": 0, "correct": 0})[
                            "total"
                        ] += 1
                    for p, r in pairs:
                        if predictions[method][p]["label"] == truth[r]:
                            metric["localized_and_correct"] += 1
                            metric["by_class"][truth[r]]["correct"] += 1
                sample["predictions"][version] = predictions
            for version in features:
                panel = Image.new("RGB", (1536, 340), "#10151e")
                panel.paste(
                    render_tile(images[item["id"]], "Reference: EVALUATION ONLY", reference, truth),
                    (0, 0),
                )
                for col, method in enumerate(("yolo", "maskcut", "sam"), 1):
                    prediction = sample["predictions"][version][method]
                    panel.paste(
                        render_tile(
                            images[item["id"]], method + " + DINO " + version,
                            proposals[item["id"]][method],
                            [p["label"] for p in prediction], [p["cosine"] for p in prediction],
                        ), (384 * col, 0),
                    )
                prefix = "" if version == "v1" else version + "-"
                panel.save(directory / (prefix + item["id"] + ".jpg"), quality=88, optimize=True)
        for metric in payload["metrics"].values():
            total = metric["reference_leaves"]
            metric["proposal_recall_iou50"] = metric["localized"] / total
            metric["correct_state_recall_iou50"] = metric["localized_and_correct"] / total
            metric["macro_correct_state_recall"] = float(
                np.mean([c["correct"] / c["total"] for c in metric["by_class"].values()])
            )
        payload["status"] = "complete"
    except BaseException as exc:
        payload["status"] = "failed"
        payload["errors"].append(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_json(directory / "results.json", payload)
        lines = [
            "# Crop few-shot diagnostic",
            "",
            f"Status: {payload['status']}",
            f"Source: {payload['git']}",
            f"Command: `{payload['command']}`",
            "",
        ]
        lines.extend(f"- {w}" for w in payload["warnings"])
        lines.extend(
            [
                "",
                "![Support images](support.jpg)",
                "",
                "| Features / proposals | Localized / leaves | Correct state + location |",
                "|---|---:|---:|",
            ]
        )
        for key, m in payload["metrics"].items():
            lines.append(
                f"| {key} | {m['localized']}/{m['reference_leaves']} | "
                f"{m['localized_and_correct']}/{m['reference_leaves']} |"
            )
        lines.extend(["", "| Reference ROI diagnostic | Healthy | Early blight | Yellow virus |",
                      "|---|---:|---:|---:|"])
        for version in ("v1", "v2", "v3"):
            metric = payload["metrics"].get(version+"/reference_roi")
            if metric:
                cells = []
                for name in ("healthy", "early_blight", "yellow_virus"):
                    counts = metric["by_class"].get(name, {"correct": 0, "total": 0})
                    cells.append(f"{counts['correct']}/{counts['total']}")
                lines.append(f"| {version} | " + " | ".join(cells) + " |")
        lines.extend(
            [
                "",
                "Reference ROI rows assume known leaf boxes: matching diagnostic only, "
                "NOT end-to-end detection performance. Known-class argmax; no unknown gate.",
            ]
        )
        for item in queries:
            if (directory / (item["id"] + ".jpg")).exists():
                lines.extend(["", f"![{item['id']}]({item['id']}.jpg)"])
        lines.extend(f"Error: {e}" for e in payload["errors"])
        atomic_write(directory / "summary.md", "\n".join(lines).encode())
        print(f"Crop experiment report: {directory}", flush=True)
    return directory
