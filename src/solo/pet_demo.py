"""Fixed everyday-object demo: YOLO dogs, followed by one-shot DINO breed matching."""

from __future__ import annotations

import gc
import json
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from .config import ROOT, Config, local_path
from .crop_demo import YOLO_SHA, YOLO_URL, fetch_verified
from .matcher import match_prototypes
from .prototypes import make_prototypes
from .reporting import new_report
from .roi_pooling import roi_pool
from .storage import atomic_write, sha256_file, write_json
from .system_info import system_info

COLORS = {"beagle": "#ffb454", "pug": "#73d4ff", "samoyed": "#87edb0"}


def prepare_image(item: dict, manifest: dict) -> Path:
    path = local_path(item["file"])
    if path.exists() and sha256_file(path) == item["sha256"]:
        return path
    params = urllib.parse.urlencode(
        dict(
            dataset=manifest["dataset"],
            config="default",
            split=item["split"],
            offset=item["row_index"],
            length=1,
        )
    )
    with urllib.request.urlopen(
        "https://datasets-server.huggingface.co/rows?" + params, timeout=60
    ) as response:
        row = json.load(response)["rows"][0]["row"]
    if row["image_id"] != item["id"] or row["label"] != item["label"]:
        raise ValueError("Dataset row changed; refusing a different sample")
    url = row["image"]["src"]
    if f"/--/{manifest['revision']}/--/" not in url:
        raise ValueError("Dataset revision changed; refusing an unpinned asset")
    fetch_verified(url, path, item["sha256"])
    return path


def tile(picture, title, boxes=(), labels=(), subtitle=""):
    canvas = Image.new("RGB", (384, 340), "#10151e")
    thumb = ImageOps.contain(picture, (368, 276))
    draw = ImageDraw.Draw(thumb)
    font = ImageFont.load_default(size=14)
    for box, label in zip(boxes, labels, strict=True):
        x, y, w, h = box
        xy = (
            (x - w / 2) * thumb.width,
            (y - h / 2) * thumb.height,
            (x + w / 2) * thumb.width,
            (y + h / 2) * thumb.height,
        )
        color = COLORS.get(label.split()[0], "#eeeeee")
        draw.rectangle(xy, outline=color, width=3)
        draw.text(
            (max(0, xy[0]) + 2, max(0, xy[1]) + 2),
            label,
            font=font,
            fill=color,
            stroke_width=1,
            stroke_fill="black",
        )
    canvas.paste(thumb, ((384 - thumb.width) // 2, 58 + (276 - thumb.height) // 2))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 7), title, fill="white", font=font)
    draw.text((8, 29), subtitle, fill="#bcc8dc", font=font)
    return canvas


def pet_demo():
    import torch
    from ultralytics import YOLO

    from .backbone_comparison import ComparisonBackbone

    manifest = json.loads((ROOT / "configs/pet-demo.json").read_text())
    directory, payload = new_report("pet-demo", Config(), system_info(), ROOT / "data/pet-study")
    payload.update(manifest=manifest, backbones={}, samples=[], metrics={}, feature_forwards={})
    payload["configuration"] = {
        "resolution": 448,
        "device": "cpu",
        "dtype": "float32",
        "dino_threads": 4,
        "yolo": {"weights_sha256": YOLO_SHA, "imgsz": 640, "conf": 0.25, "iou": 0.7},
        "support": "one whole-image mean patch embedding per breed",
        "query": "fractional mean ROI pooling of final LayerNorm patch tokens",
        "matching": "cosine argmax among three breeds; no unknown rejection",
        "evaluation": "highest-confidence YOLO dog proposal vs image breed; no IoU metric",
    }
    payload["warnings"] = [
        "COCO-pretrained YOLO11n control, not a trained SOLO class-agnostic detector.",
        "YOLO runs all classes; breed matching explicitly routes only its dog predictions.",
        "Three visually distinct breeds, one support and three test photos per breed.",
        "Breed recognition, NOT identity recognition of the same individual dog.",
        "No reference boxes used; candidate presence is NOT ground-truth localization recall.",
        "No verification of train/test animal identity or pretraining data overlap.",
        "Known-three-class matching: unknown dogs/background rejection untested.",
        "Synthetic collage is an extra diagnostic, excluded from the nine-photo metrics.",
        "CPU demonstration only; not a throughput or field benchmark.",
    ]
    payload["timing_notes"] = "No throughput measurements; model forwards counted explicitly."
    supports = [x for x in manifest["images"] if x["role"] == "support"]
    queries = [x for x in manifest["images"] if x["role"] == "query"]
    images, detections, features = {}, {}, {}
    try:
        for item in manifest["images"]:
            with Image.open(prepare_image(item, manifest)) as source:
                images[item["id"]] = source.convert("RGB")
        support_grid = Image.new("RGB", (1152, 340))
        for i, item in enumerate(supports):
            support_grid.paste(
                tile(images[item["id"]], "Support: " + item["class"], subtitle=item["id"]),
                (384 * i, 0),
            )
        support_grid.save(directory / "support.jpg", quality=87, optimize=True)
        # Fixed layout before inference; no compositing based on favorable predictions.
        collage = Image.new("RGB", (1152, 384), "#262626")
        sources = [next(q for q in queries if q["class"] == s["class"]) for s in supports]
        for i, item in enumerate(sources):
            thumb = ImageOps.contain(images[item["id"]], (384, 384))
            collage.paste(thumb, (384 * i + (384 - thumb.width) // 2, (384 - thumb.height) // 2))
        images["collage"] = collage
        payload["collage_sources"] = [x["id"] for x in sources]
        weights = local_path("weights/yolo11n.pt")
        fetch_verified(YOLO_URL, weights, YOLO_SHA)
        yolo = YOLO(str(weights))
        dog_class = next(k for k, name in yolo.names.items() if name == "dog")
        for item in [*queries, {"id": "collage"}]:
            ident = item["id"]
            result = yolo.predict(
                images[ident],
                imgsz=640,
                conf=0.25,
                iou=0.7,
                max_det=100,
                device="cpu",
                save=False,
                verbose=False,
            )[0]
            boxes = result.boxes
            detections[ident] = [
                {
                    "box": box,
                    "class_id": int(cls),
                    "label": yolo.names[int(cls)],
                    "confidence": float(conf),
                }
                for box, cls, conf in zip(
                    boxes.xywhn.cpu().tolist(),
                    boxes.cls.cpu().tolist(),
                    boxes.conf.cpu().tolist(),
                    strict=True,
                )
            ]
            print(f"YOLO {ident}: {[d['label'] for d in detections[ident]]}", flush=True)
        for version in ("v1", "v2", "v3"):
            torch.set_num_threads(4)
            backbone = ComparisonBackbone(version, 448)
            payload["backbones"][version] = backbone.metadata
            payload["feature_forwards"][version] = 0
            features[version] = {}
            for ident, picture in images.items():
                array = (
                    np.asarray(
                        picture.resize((448, 448), Image.Resampling.LANCZOS), dtype=np.float32
                    )
                    / 255
                )
                array = (array - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
                    [0.229, 0.224, 0.225], dtype=np.float32
                )
                features[version][ident] = backbone.extract(array.transpose(2, 0, 1).copy())[
                    "tokens"
                ]
                payload["feature_forwards"][version] += 1
            classes, prototypes = make_prototypes(
                np.stack(
                    [
                        roi_pool(features[version][s["id"]], backbone.grid, [(0.5, 0.5, 1.0, 1.0)])[
                            0
                        ]
                        for s in supports
                    ]
                ),
                [s["class"] for s in supports],
            )
            metric = {
                "query_photos": len(queries),
                "photos_with_dog_candidate": 0,
                "top_dog_correct_breed": 0,
                "by_class": {},
            }
            for item in [*queries, {"id": "collage", "class": "synthetic"}]:
                ident = item["id"]
                dogs = sorted(
                    [d for d in detections[ident] if d["class_id"] == dog_class],
                    key=lambda d: d["confidence"],
                    reverse=True,
                )
                prediction = match_prototypes(
                    roi_pool(features[version][ident], backbone.grid, [d["box"] for d in dogs]),
                    classes,
                    prototypes,
                )
                payload["samples"].append(
                    {
                        "id": ident,
                        "version": version,
                        "truth": item["class"],
                        "yolo_all": detections[ident],
                        "dogs": dogs,
                        "predictions": prediction,
                    }
                )
                if ident != "collage":
                    metric["photos_with_dog_candidate"] += bool(dogs)
                    correct = bool(prediction and prediction[0]["label"] == item["class"])
                    metric["top_dog_correct_breed"] += correct
                    tally = metric["by_class"].setdefault(item["class"], {"total": 0, "correct": 0})
                    tally["total"] += 1
                    tally["correct"] += correct
            payload["metrics"][version] = metric
            backbone.close()
            del backbone
            gc.collect()
            print(f"DINO {version}: {metric}", flush=True)
        indexed = {(s["id"], s["version"]): s for s in payload["samples"]}
        for item in [*queries, {"id": "collage", "class": "SYNTHETIC COLLAGE"}]:
            ident = item["id"]
            panel = Image.new("RGB", (1536, 340))
            det = detections[ident]
            panel.paste(
                tile(
                    images[ident],
                    "YOLO only",
                    [d["box"] for d in det],
                    [d["label"] for d in det],
                    "Source: " + item["class"],
                ),
                (0, 0),
            )
            for i, version in enumerate(("v1", "v2", "v3"), 1):
                sample = indexed[ident, version]
                panel.paste(
                    tile(
                        images[ident],
                        "YOLO + DINO " + version,
                        [d["box"] for d in sample["dogs"]],
                        [p["label"] + f" {p['cosine']:.2f}" for p in sample["predictions"]],
                        "cosine score; not probability",
                    ),
                    (384 * i, 0),
                )
            panel.save(directory / (ident + ".jpg"), quality=87, optimize=True)
        focus = Image.new("RGB", (1536, 1020))
        for i, item in enumerate(sources):
            with Image.open(directory / (item["id"] + ".jpg")) as panel:
                focus.paste(panel, (0, i * 340))
        focus.save(directory / "focus.jpg", quality=87, optimize=True)
        payload["status"] = "complete"
    except BaseException as exc:
        payload["status"] = "failed"
        payload["errors"].append(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_json(directory / "results.json", payload)
        lines = [
            "# Everyday pet few-shot demo",
            "",
            f"Status: {payload['status']}",
            f"Source: {payload['git']}",
            f"Command: `{payload['command']}`",
            "",
        ]
        lines.extend("- " + w for w in payload["warnings"])
        lines.extend(
            [
                "",
                "![Support](support.jpg)",
                "",
                "| DINO | Photos with dog candidate | Top dog correct breed |",
                "|---|---:|---:|",
            ]
        )
        for version, m in payload["metrics"].items():
            lines.append(
                f"| {version} | {m['photos_with_dog_candidate']}/{m['query_photos']} | "
                f"{m['top_dog_correct_breed']}/{m['query_photos']} |"
            )
        for item in [*queries, {"id": "collage"}]:
            lines.extend(["", f"![{item['id']}]({item['id']}.jpg)"])
        atomic_write(directory / "summary.md", ("\n".join(lines) + "\n").encode())
        print(f"Pet demo report: {directory}", flush=True)
    return directory
