"""Small paired visual check, deliberately separate from throughput qualification."""
from __future__ import annotations

import gc
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from .config import Config, local_path, protect_dataset
from .dataset import deterministic_subset, scan_images
from .pipeline import PipelineFailure, run_pipeline
from .pseudo_labels import LabelStore
from .reporting import new_report, save_report
from .storage import atomic_write, write_json
from .system_info import system_info, worker_limit


def compare(dataset: Path, baseline: Config, candidate: Config, limit: int = 16) -> Path:
    from .dino import CHECKPOINT_SHA256, DinoBackbone

    if not 1 <= limit <= 64:
        raise ValueError("Comparison --limit must be between 1 and 64")
    dataset = dataset.resolve(strict=True)
    for destination in ("reports", "work", "weights"):
        protect_dataset(dataset, local_path(destination))
    records = deterministic_subset(scan_images(dataset), limit, candidate.pipeline.seed)
    directory, payload = new_report("compare", candidate, system_info(), dataset)
    payload["warnings"].append(
        "Experimental splitting may divide one object into parts. More boxes is not proof of "
        "better instance recall. Visually check missed objects, merged people, body fragments, "
        "and background boxes. This small batch-1 run does NOT qualify full generation."
    )
    write_json(directory / "subset.json", {"relative_paths": [r.relative for r in records]})
    stores = []
    try:
        for name, config in (("baseline", baseline), ("crowded", candidate)):
            store = LabelStore(
                local_path(Path("work/comparisons") / directory.name / name),
                config.fingerprint(CHECKPOINT_SHA256), str(dataset), config.pipeline.fsync,
            )
            backbone = DinoBackbone(config.dino)
            try:
                result = run_pipeline(
                    records, backbone, config, store, min(4, worker_limit(0)), 1,
                    resume=False, title=f"SOLO compare {name}",
                )
                result["backbone"] = backbone.metadata
            except PipelineFailure as exc:
                exc.result.update(stage=name, repeat=0, configuration=config.to_dict())
                payload["runs"].append(exc.result)
                raise
            finally:
                backbone.clear_memory()
                del backbone
                gc.collect()
            result.update(stage=name, repeat=0, configuration=config.to_dict())
            payload["runs"].append(result)
            if result["status"] != "ok" or result["images_failed"]:
                raise RuntimeError(f"{name} comparison failed: {result['errors']}")
            stores.append(store)
        grid = Image.new("RGB", (768, 280 * len(records)), "#181818")
        colors = ("#ff5030", "#20df80", "#4080ff", "#ffcf20", "#fa60d0", "#40e0e0")
        for row, record in enumerate(records):
            with Image.open(record.path) as source:
                original = source.convert("RGB")
            for col, (name, store) in enumerate(zip(("baseline", "crowded"), stores, strict=True)):
                boxes = store.valid_boxes(record)
                if boxes is None:
                    raise RuntimeError(f"Missing comparison label: {record.relative}")
                tile = Image.new("RGB", (384, 280), "#181818")
                picture = ImageOps.contain(original, (384, 240))
                draw = ImageDraw.Draw(picture)
                for index, (x, y, w, h) in enumerate(boxes):
                    xy = ((x-w/2)*picture.width, (y-h/2)*picture.height,
                          (x+w/2)*picture.width, (y+h/2)*picture.height)
                    color = colors[index % len(colors)]
                    draw.rectangle(xy, outline=color, width=2)
                    draw.text((xy[0], xy[1]), str(index + 1), fill=color)
                tile.paste(picture, ((384-picture.width)//2, 0))
                draw = ImageDraw.Draw(tile)
                draw.text((4, 242), f"{name}: {len(boxes)} boxes", fill="white")
                draw.text((4, 258), record.relative[-55:], fill="white")
                grid.paste(tile, (384*col, 280*row))
        # At most 64 pairs. JPEG is small enough for reports committed from the server.
        grid.save(directory / "comparison.jpg", quality=75, optimize=True)
        payload["status"] = "complete"
        payload["quality"] = {r["stage"]: r["quality"] for r in payload["runs"]}
    except BaseException as exc:
        payload["status"] = "failed"
        payload["errors"].append(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save_report(directory, payload)
        summary = directory / "summary.md"
        text = summary.read_text().replace("![Pseudo boxes](pseudo_preview.jpg)",
                                           "![Baseline vs crowded](comparison.jpg)")
        atomic_write(summary, text.encode())
        print(f"Paired quality report: {directory}", flush=True)
    return directory
