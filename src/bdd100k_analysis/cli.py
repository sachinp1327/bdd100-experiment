"""Command-line entrypoints for dataset analysis."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from .analysis import analyze_split
from .reports import compare_summaries
from .visuals import render_extreme_samples


def _default_labels_path(split: str) -> Path:
    return Path(
        f"data/bdd100k_labels_release/bdd100k/labels/"
        f"bdd100k_labels_images_{split}.json"
    )


def _default_images_root(split: str) -> Path:
    return Path(f"data/bdd100k_images_100k/bdd100k/images/100k/{split}")


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def cmd_analyze(args: argparse.Namespace) -> None:
    labels_path = Path(args.labels) if args.labels else _default_labels_path(args.split)
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels not found: {labels_path}")

    images_root = (
        Path(args.images_root) if args.images_root else _default_images_root(args.split)
    )
    summary = analyze_split(labels_path, split=args.split, images_root=images_root)
    output_dir = Path(args.output)
    summary_path = output_dir / "summary.json"
    _write_json(summary_path, summary)

    if args.render_visuals:
        if not images_root.exists():
            raise FileNotFoundError(f"Images root not found: {images_root}")
        render_extreme_samples(
            summary,
            images_root=images_root,
            output_dir=output_dir / "visuals",
            limit_per_group=args.visuals_limit,
        )


def cmd_compare(args: argparse.Namespace) -> None:
    train_summary = Path(args.train_summary)
    val_summary = Path(args.val_summary)
    with train_summary.open("r", encoding="utf-8") as handle:
        train = json.load(handle)
    with val_summary.open("r", encoding="utf-8") as handle:
        val = json.load(handle)

    comparison = compare_summaries(train, val)
    _write_json(Path(args.output), comparison)


def cmd_analyze_all(args: argparse.Namespace) -> None:
    output_root = Path(args.output)
    for split in ("train", "val"):
        split_args = argparse.Namespace(
            split=split,
            labels=None,
            output=str(output_root / split),
            images_root=None,
            render_visuals=args.render_visuals,
            visuals_limit=args.visuals_limit,
        )
        cmd_analyze(split_args)

    comparison_path = output_root / "compare.json"
    compare_args = argparse.Namespace(
        train_summary=str(output_root / "train" / "summary.json"),
        val_summary=str(output_root / "val" / "summary.json"),
        output=str(comparison_path),
    )
    cmd_compare(compare_args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BDD100K detection analysis")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="Analyze a single split")
    analyze.add_argument("--split", required=True, choices=["train", "val"])
    analyze.add_argument("--labels", help="Path to labels JSON")
    analyze.add_argument("--output", required=True, help="Output directory")
    analyze.add_argument("--images-root", help="Root directory of images")
    analyze.add_argument("--render-visuals", action="store_true")
    analyze.add_argument("--visuals-limit", type=int, default=3)
    analyze.set_defaults(func=cmd_analyze)

    compare = subparsers.add_parser("compare", help="Compare train/val summaries")
    compare.add_argument("--train-summary", required=True)
    compare.add_argument("--val-summary", required=True)
    compare.add_argument("--output", required=True)
    compare.set_defaults(func=cmd_compare)

    analyze_all = subparsers.add_parser("analyze-all", help="Analyze train and val splits")
    analyze_all.add_argument("--output", required=True, help="Output root directory")
    analyze_all.add_argument("--render-visuals", action="store_true")
    analyze_all.add_argument("--visuals-limit", type=int, default=3)
    analyze_all.set_defaults(func=cmd_analyze_all)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
