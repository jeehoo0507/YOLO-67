"""Small, controlled backbone experiment; never changes production pseudo-labels."""

from __future__ import annotations

import csv
import gc
import io
import statistics
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from threadpoolctl import threadpool_limits

from .config import Config, local_path, protect_dataset
from .dataset import deterministic_subset, scan_images
from .key_preview import COLORS, keys_to_rgb
from .maskcut import discover_masks
from .pipeline import decode_image
from .pseudo_labels import encode_labels, masks_to_boxes
from .reporting import new_report
from .storage import atomic_write, sha256_file, write_json
from .system_info import system_info

MODELS = {
    "v2": ("vit_small_patch14_dinov2.lvd142m", "4610ca143709d58a633b6397a74412c2c3842454"),
    "v3": ("vit_small_patch16_dinov3.lvd1689m", "3bf4720a82ec2066db88137180ff1f83a675cef0"),
}


class ComparisonBackbone:
    """Extract raw last-block keys and final normalized patch tokens in one forward."""

    def __init__(self, version: str, resolution: int):
        import torch

        self.version = version
        self.captured = None
        if version == "v1":
            from .dino import CHECKPOINT_SHA256, DinoBackbone

            base = DinoBackbone(replace(Config().dino, device="cpu", resolution=resolution))
            self.model = base.model
            self.prefix, self.patch = 1, 16
            self.metadata = dict(base.metadata)
            self.metadata["checkpoint_sha256"] = CHECKPOINT_SHA256
        else:
            import timm
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file
            from timm.layers import set_fused_attn

            name, revision = MODELS[version]
            set_fused_attn(False)  # Match the existing v1 explicit attention implementation.
            checkpoint = hf_hub_download(
                f"timm/{name}",
                "model.safetensors",
                revision=revision,
                cache_dir=str(local_path(".cache/huggingface/hub")),
                token=False,
            )
            self.model = timm.create_model(name, pretrained=False, dynamic_img_size=True)
            self.model.load_state_dict(load_file(checkpoint), strict=True)
            self.prefix = self.model.num_prefix_tokens
            self.patch = self.model.patch_embed.patch_size[0]
            self.metadata = {
                "model": name,
                "repository": f"https://huggingface.co/timm/{name}",
                "revision": revision,
                "checkpoint_sha256": sha256_file(Path(checkpoint)),
                "implementation": f"timm {timm.__version__}",
                "mean": self.model.pretrained_cfg["mean"],
                "std": self.model.pretrained_cfg["std"],
            }
            if tuple(self.metadata["mean"]) != (0.485, 0.456, 0.406) or tuple(
                self.metadata["std"]
            ) != (0.229, 0.224, 0.225):
                raise ValueError("Model normalization differs from the shared preprocessing")
        self.model.eval().requires_grad_(False)
        self.grid = (resolution // self.patch, resolution // self.patch)
        self.metadata.update(
            version=version,
            resolution=resolution,
            patch_size=self.patch,
            grid=self.grid,
            prefix_tokens=self.prefix,
            parameters=sum(p.numel() for p in self.model.parameters()),
            device="cpu",
            dtype="float32",
            fused_attention=False,
        )
        self.hook = self.model.blocks[-1].attn.qkv.register_forward_hook(self._capture)
        self.torch = torch

    def _capture(self, _module, _inputs, output):
        self.captured = output

    def extract(self, array: np.ndarray) -> dict[str, np.ndarray]:
        self.captured = None
        with self.torch.inference_mode():
            tensor = self.torch.from_numpy(array[None])
            tokens = (
                self.model.get_intermediate_layers(tensor, n=1)[0]
                if self.version == "v1"
                else (self.model.forward_features(tensor))
            )
            if self.captured is None:
                raise RuntimeError("Last-block QKV hook did not execute")
            batch, length, width = self.captured.shape
            keys = self.captured.reshape(batch, length, 3, width // 3)[:, self.prefix :, 1]
            values = {
                "keys": keys[0].numpy().copy(),
                "tokens": tokens[0, self.prefix :].numpy().copy(),
            }
        self.captured = None
        for feature in values.values():
            if (
                feature.shape != (self.grid[0] * self.grid[1], 384)
                or not np.isfinite(feature).all()
            ):
                raise ValueError(f"Invalid {self.version} patch features: {feature.shape}")
        return values

    def close(self):
        self.hook.remove()


def _draw_boxes(image: Image.Image, boxes: list) -> Image.Image:
    image = image.copy()
    draw = ImageDraw.Draw(image)
    for i, (x, y, w, h) in enumerate(boxes):
        xy = (
            (x - w / 2) * image.width,
            (y - h / 2) * image.height,
            (x + w / 2) * image.width,
            (y + h / 2) * image.height,
        )
        draw.rectangle(xy, outline=COLORS[i % len(COLORS)], width=2)
        draw.text((xy[0] + 2, xy[1] + 2), str(i + 1), fill=COLORS[i % len(COLORS)])
    return image


def _panel(pictures: list[Image.Image], labels: list[str]) -> Image.Image:
    canvas = Image.new("RGB", (384 * len(pictures), 344), "#10151e")
    font = ImageFont.load_default(size=17)
    draw = ImageDraw.Draw(canvas)
    for col, (picture, label) in enumerate(zip(pictures, labels, strict=True)):
        thumb = ImageOps.contain(picture, (368, 292))
        draw.text((384 * col + 8, 8), label, fill="white", font=font)
        canvas.paste(thumb, (384 * col + (384 - thumb.width) // 2, 40 + (292 - thumb.height) // 2))
    return canvas


def compare_backbones(dataset: Path, config: Config, limit: int = 5, repeats: int = 3) -> Path:
    import torch

    if not 1 <= limit <= 20 or not 1 <= repeats <= 10:
        raise ValueError("Require limit 1..20 and repeats 1..10")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    dataset = dataset.resolve(strict=True)
    for destination in ("reports", "weights", ".cache", "work"):
        protect_dataset(dataset, local_path(destination))
    config = replace(config, dino=replace(config.dino, resolution=448, device="cpu"))
    records = deterministic_subset(scan_images(dataset), limit, config.pipeline.seed)
    directory, payload = new_report("backbones", config, system_info(), dataset)
    payload["models"] = {}
    payload["token_threshold_sweep"] = {}
    payload["samples"] = [
        {"image": r.relative, "sha256": sha256_file(Path(r.path))} for r in records
    ]
    payload["notes"] = [
        "Same images, 448x448 LANCZOS, ImageNet normalization, CPU FP32, 1 thread, batch 1. "
        "ViT-S models (~22M parameters), explicit non-fused attention, one warmup per model.",
        "v1/v3: 28x28 grid (patch16). v2: 32x32 (patch14). "
        "Resolution is matched, token count is not.",
        "Two feature probes: raw last-block QKV keys (v3 pre-RoPE) and final LayerNorm patch "
        "tokens. CLS/register tokens excluded. Both extracted in ONE full forward per image.",
        "MaskCut/filter configuration fixed for all models/features. Tau was not tuned per model. "
        "This is a transfer sanity check, NOT a ranking of optimally tuned backbone performance.",
        "Median per-image timings across repeated identical inputs, then mean over images. "
        "Startup/download/PCA/rendering excluded. Common decode/forward plus each feature's "
        "MaskCut/box/write time describes serial latency, NOT GPU or batched throughput.",
        "No ground truth, no mAP or objective accuracy. Box count is not instance correctness. "
        "PCA colors have no class/identity meaning and are fitted separately for each image/model.",
        "v1 uses the project's original implementation; v2/v3 use pinned timm conversions. "
        "The v3 timm RoPE implementation has documented numerical differences from Meta's.",
        "Secondary diagnostic: final patch tokens at fixed tau values 0.5, 0.7, 0.9, "
        "identical for all models. Reuses extracted features; no retraining or extra forward. "
        "These five examples are not an independent validation set for threshold selection.",
    ]
    results = {}
    originals = []
    for record in records:
        with Image.open(record.path) as image:
            originals.append(image.convert("RGB"))
    try:
        with threadpool_limits(limits=1):
            for version in ("v1", "v2", "v3"):
                print(f"Preparing DINO {version} ...", flush=True)
                backbone = ComparisonBackbone(version, 448)
                entries = []
                results[version] = {}
                try:
                    backbone.extract(np.zeros((3, 448, 448), dtype=np.float32))
                    for index, record in enumerate(records):
                        timings = {k: [] for k in ("decode", "forward", "keys", "tokens")}
                        first_boxes = None
                        for _repeat in range(repeats):
                            start = time.perf_counter()
                            decoded = decode_image(record, 448)
                            if "error" in decoded:
                                raise RuntimeError(decoded["error"])
                            after_decode = time.perf_counter()
                            features = backbone.extract(decoded["array"])
                            after_forward = time.perf_counter()
                            timings["decode"].append(after_decode - start)
                            timings["forward"].append(after_forward - after_decode)
                            probes = {}
                            for kind, feature in features.items():
                                start = time.perf_counter()
                                masks = discover_masks(feature, backbone.grid, config.maskcut)
                                boxes = masks_to_boxes(masks, decoded["size"], config.filtering)
                                target = local_path(
                                    Path("work/backbone-comparisons")
                                    / directory.name
                                    / version
                                    / kind
                                    / f"{index}.txt"
                                )
                                atomic_write(target, encode_labels(boxes))
                                timings[kind].append(time.perf_counter() - start)
                                probes[kind] = {"boxes": boxes, "masks": masks, "feature": feature}
                            if first_boxes is None:
                                first_boxes = {k: v["boxes"] for k, v in probes.items()}
                            elif first_boxes != {k: v["boxes"] for k, v in probes.items()}:
                                raise RuntimeError("Repeated CPU runs produced different boxes")
                        results[version][index] = probes
                        entry = {
                            "image": record.relative,
                            "timings_seconds": timings,
                            "median_ms": {
                                k: 1000 * statistics.median(v) for k, v in timings.items()
                            },
                            "probes": {
                                k: {"boxes_xywh": v["boxes"], "mask_count": len(v["masks"])}
                                for k, v in probes.items()
                            },
                        }
                        entries.append(entry)
                        print(
                            f"{version} {index + 1}/{len(records)} {record.relative}: "
                            f"keys={len(probes['keys']['boxes'])}, "
                            f"tokens={len(probes['tokens']['boxes'])}",
                            flush=True,
                        )
                    payload["models"][version] = {
                        "metadata": backbone.metadata,
                        "samples": entries,
                        "mean_median_ms": {
                            k: statistics.mean(e["median_ms"][k] for e in entries) for k in timings
                        },
                    }
                finally:
                    backbone.close()
                    del backbone
                    gc.collect()
                write_json(directory / "comparison.json", payload)
            # Diagnostic grid, not a per-image optimized or automatically selected result.
            for tau in (0.5, 0.7, 0.9):
                sheet = Image.new("RGB", (1536, 344 * len(records)), "#10151e")
                sweep = {}
                for index, original in enumerate(originals):
                    pictures, labels = [original], [records[index].relative]
                    for version in results:
                        feature = results[version][index]["tokens"]["feature"]
                        grid = tuple(payload["models"][version]["metadata"]["grid"])
                        masks = discover_masks(feature, grid, replace(config.maskcut, tau=tau))
                        boxes = masks_to_boxes(masks, original.size, config.filtering)
                        values = feature.astype(np.float64)
                        normalized = values / np.maximum(
                            np.linalg.norm(values, axis=1, keepdims=True),
                            1e-12,
                        )
                        # Verify actual values rather than stale macOS BLAS flags (as in MaskCut).
                        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                            similarity = normalized @ normalized.T
                        if not np.isfinite(similarity).all():
                            raise ValueError("Non-finite cosine diagnostic")
                        off_diagonal = similarity[~np.eye(len(feature), dtype=bool)]
                        sweep.setdefault(version, []).append(
                            {
                                "image": records[index].relative,
                                "boxes_xywh": boxes,
                                "affinity_edge_fraction": float(np.mean(off_diagonal > tau)),
                                "cosine_quantiles_05_50_95": np.quantile(
                                    off_diagonal,
                                    [0.05, 0.5, 0.95],
                                ).tolist(),
                            }
                        )
                        pictures.append(_draw_boxes(ImageOps.contain(original, (368, 292)), boxes))
                        labels.append(f"{version} tokens t={tau}: {len(boxes)} boxes")
                    sheet.paste(_panel(pictures, labels), (0, 344 * index))
                sheet.save(directory / f"tokens_tau_{tau}.jpg", quality=85, optimize=True)
                payload["token_threshold_sweep"][str(tau)] = sweep
                print(f"Token threshold diagnostic tau={tau} complete", flush=True)
        # Each row uses the same source image; no favorable sample selection.
        for kind in ("keys", "tokens"):
            canvas = Image.new("RGB", (1536, 344 * len(records)), "#10151e")
            maps = Image.new("RGB", canvas.size, "#10151e")
            pipelines = {v: Image.new("RGB", canvas.size, "#10151e") for v in results}
            for index, original in enumerate(originals):
                pictures, key_maps = [original], [original]
                labels = [records[index].relative]
                for version in results:
                    probe = results[version][index][kind]
                    grid = tuple(payload["models"][version]["metadata"]["grid"])
                    rgb, _ = keys_to_rgb(probe["feature"], grid)
                    pca = Image.fromarray(rgb).resize(original.size, Image.Resampling.NEAREST)
                    mask_rgb = np.zeros((*grid, 3), dtype=np.uint8)
                    for i, mask in enumerate(probe["masks"]):
                        mask_rgb[mask] = COLORS[i % len(COLORS)]
                    mask_image = Image.fromarray(mask_rgb).resize(
                        original.size, Image.Resampling.NEAREST
                    )
                    # Draw on thumbnail for legible box boundaries at display resolution.
                    boxed = _draw_boxes(ImageOps.contain(original, (368, 292)), probe["boxes"])
                    pictures.append(boxed)
                    key_maps.append(pca)
                    labels.append(f"{version} {kind}: {len(probe['boxes'])} boxes")
                    panel = _panel(
                        [original, pca, mask_image, boxed],
                        [
                            records[index].relative,
                            f"{version} {kind}: PCA RGB",
                            "Raw MaskCut masks",
                            f"Filtered: {len(probe['boxes'])} boxes",
                        ],
                    )
                    pipelines[version].paste(panel, (0, 344 * index))
                canvas.paste(_panel(pictures, labels), (0, 344 * index))
                maps.paste(
                    _panel(key_maps, [labels[0], "v1 PCA", "v2 PCA", "v3 PCA"]), (0, 344 * index)
                )
            for name, image in {
                f"{kind}_boxes": canvas,
                f"{kind}_maps": maps,
                **{f"{v}_{kind}_pipeline": p for v, p in pipelines.items()},
            }.items():
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=85, optimize=True)
                atomic_write(directory / f"{name}.jpg", buffer.getvalue())
        payload["status"] = "complete"
    except BaseException as exc:
        payload["status"] = "failed"
        payload["errors"].append(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_json(directory / "comparison.json", payload)
        buffer = io.StringIO(newline="")
        writer = csv.writer(buffer)
        writer.writerow(["model", "image", "decode_ms", "forward_ms", "keys_cut_box_write_ms",
                         "tokens_cut_box_write_ms", "key_boxes", "token_boxes"])
        for name, model in payload["models"].items():
            for sample in model["samples"]:
                times = sample["median_ms"]
                writer.writerow([name, sample["image"], times["decode"], times["forward"],
                                 times["keys"], times["tokens"],
                                 len(sample["probes"]["keys"]["boxes_xywh"]),
                                 len(sample["probes"]["tokens"]["boxes_xywh"])])
        atomic_write(directory / "timings.csv", buffer.getvalue().encode())
        lines = [
            "# DINO backbone comparison",
            "",
            f"Status: {payload['status']}",
            f"Git: {payload['git']}",
            f"Command: `{payload['command']}`",
            "",
        ]
        lines.extend(f"- {note}" for note in payload["notes"])
        lines.extend(
            [
                "",
                "| Model | Forward ms | Keys cut/box/write ms | Tokens cut/box/write ms |",
                "|---|---:|---:|---:|",
            ]
        )
        for name, model in payload["models"].items():
            times = model["mean_median_ms"]
            lines.append(
                f"| {name} | {times['forward']:.1f} | {times['keys']:.1f} | {times['tokens']:.1f} |"
            )
        lines.extend(
            [
                "",
                "## Raw key probe",
                "",
                "![Key boxes](keys_boxes.jpg)",
                "",
                "## Final patch-token probe",
                "",
                "![Token boxes](tokens_boxes.jpg)",
            ]
        )
        for tau in payload["token_threshold_sweep"]:
            lines.extend(
                [
                    "",
                    f"## Fixed token threshold {tau}",
                    "",
                    f"![Token boxes tau {tau}](tokens_tau_{tau}.jpg)",
                ]
            )
        lines.extend(f"\nError: {e}" for e in payload["errors"])
        atomic_write(directory / "summary.md", "\n".join(lines).encode())
        print(f"Backbone comparison: {directory}", flush=True)
    return directory
