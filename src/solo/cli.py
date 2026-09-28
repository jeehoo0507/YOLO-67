from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .config import ROOT, load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="./solo", description="SOLO: COCO images → DINO/MaskCut boxes → YOLO object detector"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("benchmark", "generate", "preview"):
        command = commands.add_parser(name)
        command.add_argument(
            "dataset",
            type=Path,
            nargs="?",
            default=ROOT / "data/coco/images",
            help="Image directory (default: data/coco/images inside this repository)",
        )
        command.add_argument("--config", type=Path, default=ROOT / "configs/baseline.toml")
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
    train.add_argument("--config", type=Path, default=ROOT / "configs/baseline.toml")
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
        if args.command == "doctor":
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
            if args.command == "benchmark":
                from .benchmark import benchmark

                benchmark(args.dataset, config)
            elif args.command == "generate":
                from .generation import generate

                generate(args.dataset, config)
            elif args.command == "train":
                from .detector import train_detector

                train_detector(args.dataset, config)
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
