from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .config import ROOT, load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="./solo", description="SOLO: frozen DINO + CPU MaskCut baseline"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("benchmark", "generate", "preview"):
        command = commands.add_parser(name)
        command.add_argument(
            "dataset",
            type=Path,
            nargs="?",
            default=ROOT / "data/imagenet/train",
            help="Image directory (default: data/imagenet/train inside this repository)",
        )
        command.add_argument("--config", type=Path, default=ROOT / "configs/baseline.toml")
    download = commands.add_parser("download", help="Prepare a 100k-image ImageNet-1K sample")
    download.add_argument(
        "--archive", type=Path, help="Extract an existing official ILSVRC2012_img_train.tar"
    )
    download.add_argument(
        "--mode", choices=("sample", "full", "100gb", "auto"), default="sample",
        help="sample selects 100 images per group by default; full/100gb/auto are optional",
    )
    download.add_argument(
        "--images-per-group", type=int, default=100,
        help="Images per archive in sample mode (default: 100; 1,000 groups)",
    )
    commands.add_parser("doctor", help="Print environment and hardware without model downloads")
    commands.add_parser("train", help="Reserved for Milestone 2: single-class YOLO11n")
    infer = commands.add_parser("infer", help="Reserved for Milestone 3: few-shot ROI matching")
    infer.add_argument("--support", type=Path)
    infer.add_argument("--query", type=Path)
    args = parser.parse_args(argv)
    if args.command in {"train", "infer"}:
        parser.error(
            f"{args.command} is reserved for the next milestone. "
            "Validate Milestone 1 on the server first (README.md)."
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
        else:
            config = load_config(args.config)
            if args.command == "benchmark":
                from .benchmark import benchmark

                benchmark(args.dataset, config)
            elif args.command == "generate":
                from .generation import generate

                generate(args.dataset, config)
            else:
                from .reporting import create_preview

                create_preview(args.dataset, config)
    except KeyboardInterrupt:
        print("Interrupted; completed image receipts remain resumable.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"SOLO: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0
