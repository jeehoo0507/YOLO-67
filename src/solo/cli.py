from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .config import ROOT, load_config


def config_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--config", type=Path, default=ROOT / "configs/baseline.toml")
    group.add_argument(
        "--crowded", dest="config", action="store_const", const=ROOT / "configs/crowded.toml",
        help="Use the crowded-scene profile with separate labels and trained weights",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="./solo", description="SOLO: COCO images → DINO/MaskCut boxes → YOLO object detector"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("crop-demo", help="Run a fixed three-state PlantDoc few-shot comparison")
    sam = commands.add_parser("compare-sam", help="Compare DINO/MaskCut and automatic SAM 2.1")
    sam.add_argument("dataset", type=Path, nargs="?", default=ROOT / "data/coco/images/val2017")
    sam.add_argument("--limit", type=int, default=20)
    sam.add_argument("--points-per-side", type=int, default=16)
    sam.add_argument("--cpu-threads", type=int, default=4)
    sam.add_argument("--config", type=Path, default=ROOT / "configs/crowded.toml")
    backbones = commands.add_parser("compare-backbones", help="Compare DINO v1/v2/v3 on CPU")
    backbones.add_argument(
        "dataset", type=Path, nargs="?", default=ROOT / "data/coco/images/val2017",
    )
    backbones.add_argument("--limit", type=int, default=5)
    backbones.add_argument("--repeats", type=int, default=3)
    backbones.add_argument("--config", type=Path, default=ROOT / "configs/crowded.toml")
    keys = commands.add_parser("key-preview", help="Show original -> DINO keys -> masks -> boxes")
    keys.add_argument(
        "dataset", type=Path, nargs="?", default=ROOT / "data/coco/images/val2017",
    )
    keys.add_argument("--limit", type=int, default=5)
    config_arguments(keys)
    comparison = commands.add_parser("compare", help="Compare baseline/crowded pseudo boxes")
    comparison.add_argument(
        "dataset", type=Path, nargs="?", default=ROOT / "data/coco/images/val2017",
    )
    comparison.add_argument("--limit", type=int, default=16)
    comparison.add_argument("--config", type=Path, default=ROOT / "configs/crowded.toml")
    comparison.add_argument("--baseline", type=Path, default=ROOT / "configs/baseline.toml")
    for name in ("benchmark", "generate", "preview"):
        command = commands.add_parser(name)
        command.add_argument(
            "dataset",
            type=Path,
            nargs="?",
            default=ROOT / "data/coco/images",
            help="Image directory (default: data/coco/images inside this repository)",
        )
        config_arguments(command)
    download = commands.add_parser("download", help="Prepare a 100k-image ImageNet-1K sample")
    download.add_argument(
        "--archive", type=Path, help="Extract an existing official ILSVRC2012_img_train.tar"
    )
    commands.add_parser("download-coco", help="Download official COCO 2017 train/val images only")
    download.add_argument(
        "--mode",
        choices=("sample", "full", "100gb", "auto"),
        default="sample",
        help="sample selects 100 images per group by default; full/100gb/auto are optional",
    )
    download.add_argument(
        "--images-per-group",
        type=int,
        default=100,
        help="Images per archive in sample mode (default: 100; 1,000 groups)",
    )
    commands.add_parser("doctor", help="Print environment and hardware without model downloads")
    train = commands.add_parser("train", help="Train object-only YOLO11n from DINO pseudo boxes")
    train.add_argument(
        "dataset",
        type=Path,
        nargs="?",
        default=ROOT / "data/coco/images",
        help="Image directory used for ./solo generate",
    )
    config_arguments(train)
    predict = commands.add_parser("predict", help="Test trained YOLO and save object box images")
    predict.add_argument(
        "source", type=Path, nargs="?", default=ROOT / "data/coco/images/val2017",
        help="Image or directory (default: COCO val2017)",
    )
    predict.add_argument(
        "--weights", type=Path, help="Checkpoint (default: configured training run best.pt)",
    )
    predict.add_argument(
        "--limit", type=int, default=16, help="Image limit; 0 means all (default: 16)",
    )
    predict.add_argument(
        "--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)",
    )
    config_arguments(predict)
    infer = commands.add_parser("infer", help="Reserved for Milestone 3: few-shot ROI matching")
    infer.add_argument("--support", type=Path)
    infer.add_argument("--query", type=Path)
    args = parser.parse_args(argv)
    if args.command == "infer":
        parser.error(
            "Few-shot inference is the next milestone; "
            "validate pseudo boxes and YOLO training first."
        )
    if os.environ.get("SOLO_ROOT") != str(ROOT):
        parser.error("Use the ./solo wrapper so all environments / caches stay project-local")
    try:
        if args.command == "crop-demo":
            from .crop_demo import crop_demo

            crop_demo()
        elif args.command == "doctor":
            import json

            from .system_info import system_info

            print(json.dumps(system_info(), indent=2))
        elif args.command == "download":
            from .download import download_imagenet

            download_imagenet(args.archive, args.mode, args.images_per_group)
        elif args.command == "download-coco":
            from .coco import download_coco

            download_coco()
        else:
            config = load_config(args.config)
            if args.command == "compare-sam":
                from .sam_comparison import compare_sam

                compare_sam(
                    args.dataset, config, args.limit, args.points_per_side, args.cpu_threads,
                )
            elif args.command == "compare-backbones":
                from .backbone_comparison import compare_backbones

                compare_backbones(args.dataset, config, args.limit, args.repeats)
            elif args.command == "key-preview":
                from .key_preview import key_preview

                key_preview(args.dataset, config, args.limit)
            elif args.command == "compare":
                from .comparison import compare

                compare(args.dataset, load_config(args.baseline), config, args.limit)
            elif args.command == "benchmark":
                from .benchmark import benchmark

                benchmark(args.dataset, config)
            elif args.command == "generate":
                from .generation import generate

                generate(args.dataset, config)
            elif args.command == "train":
                from .detector import train_detector

                train_detector(args.dataset, config)
            elif args.command == "predict":
                from .prediction import predict_objects

                weights = args.weights or (
                    ROOT / "outputs" / config.train.run_name / "weights/best.pt"
                )
                predict_objects(args.source, weights, config, args.limit, args.conf)
            else:
                from .reporting import create_preview

                create_preview(args.dataset, config)
    except KeyboardInterrupt:
        print(
            "Interrupted; rerun to resume completed files or the last saved training epoch.",
            file=sys.stderr,
        )
        return 130
    except Exception as exc:
        print(f"SOLO: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0
