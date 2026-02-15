"""CLI for BDD100K evaluation diagnostics."""
from __future__ import annotations

import argparse
from pathlib import Path

from .diagnostics import DiagnosticsConfig, EvaluationDiagnostics


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for diagnostics."""
    parser = argparse.ArgumentParser(
        description="BDD100K evaluation diagnostics"
    )
    parser.add_argument(
        "--predictions",
        required=True,
        help="Path to COCO-style predictions.json.",
    )
    parser.add_argument(
        "--coco",
        default="data/bdd100k_coco/instances_val.json",
        help="Path to COCO ground truth annotations.",
    )
    parser.add_argument(
        "--labels",
        default=(
            "data/bdd100k_labels_release/bdd100k/labels/"
            "bdd100k_labels_images_val.json"
        ),
        help="Path to BDD100K label metadata JSON.",
    )
    parser.add_argument(
        "--images",
        default="data/bdd100k_images_100k/bdd100k/images/100k/val",
        help="Path to images root.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/eval_diagnostics",
        help="Directory to write diagnostics artifacts.",
    )
    parser.add_argument(
        "--analysis-summary",
        default="outputs/val/summary.json",
        help="Path to analysis summary for linking metrics.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.05,
        help="Score threshold for predictions.",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for matching.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-K samples per class for best/worst.",
    )
    parser.add_argument(
        "--cluster-k",
        type=int,
        default=5,
        help="Number of clusters for image-level analysis.",
    )
    return parser


def main() -> None:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args()

    config = DiagnosticsConfig(
        predictions_path=Path(args.predictions),
        coco_path=Path(args.coco),
        labels_path=Path(args.labels),
        images_root=Path(args.images),
        output_dir=Path(args.output_dir),
        score_threshold=args.score_threshold,
        iou_threshold=args.iou_threshold,
        top_k=args.top_k,
        cluster_k=args.cluster_k,
        analysis_summary=Path(args.analysis_summary)
        if args.analysis_summary
        else None,
    )

    diagnostics = EvaluationDiagnostics(config)
    diagnostics.run()


if __name__ == "__main__":
    main()
