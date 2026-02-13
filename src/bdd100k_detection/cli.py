"""Command-line interface for training and evaluation."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from bdd100k_analysis.constants import BDD100K_DET_CLASSES
from .config import (
    DatasetConfig,
    EvaluationConfig,
    ModelConfig,
    PipelineConfig,
    TrainingConfig,
)
from .coco_utils import convert_bdd_to_coco
from .datasets import CocoDetectionDataset, collate_fn
from .evaluator import Evaluator
from .model import build_model
from .pipeline import DetectionPipeline
from .transforms import build_transforms


def _default_labels(split: str) -> Path:
    """Return default labels path for a split."""
    return Path(
        "data/bdd100k_labels_release/bdd100k/labels/"
        f"bdd100k_labels_images_{split}.json"
    )


def _default_images(split: str) -> Path:
    """Return default images directory for a split."""
    return Path(f"data/bdd100k_images_100k/bdd100k/images/100k/{split}")


def _default_coco(split: str) -> Path:
    """Return default COCO output path for a split."""
    return Path(f"data/bdd100k_coco/instances_{split}.json")


def _extract_state_dict(checkpoint: object) -> dict:
    """Return a model state dict from a checkpoint payload."""
    state_dict = checkpoint
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get(
            "model_state",
            checkpoint.get("model", checkpoint.get("state_dict", checkpoint)),
        )
    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint does not contain a valid state_dict.")
    if any(key.startswith("module.") for key in state_dict.keys()):
        state_dict = {
            key.replace("module.", "", 1): value
            for key, value in state_dict.items()
        }
    return state_dict


def _infer_backbone(state_dict: dict) -> str | None:
    """Infer ResNet backbone depth from torchvision-style keys."""
    prefix = "backbone.body.layer3."
    blocks = {
        int(key.split(prefix)[1].split(".")[0])
        for key in state_dict.keys()
        if key.startswith(prefix)
        and key.split(prefix)[1].split(".")[0].isdigit()
    }
    if not blocks:
        return None
    max_block = max(blocks)
    return "resnet101" if max_block >= 10 else "resnet50"


def _build_pipeline_config(args: argparse.Namespace) -> PipelineConfig:
    """Build PipelineConfig from CLI args."""
    classes = BDD100K_DET_CLASSES.copy()
    train = DatasetConfig(
        split="train",
        labels_path=Path(args.train_labels),
        images_root=Path(args.train_images),
        coco_output=Path(args.train_coco),
        classes=classes,
    )
    val = DatasetConfig(
        split="val",
        labels_path=Path(args.val_labels),
        images_root=Path(args.val_images),
        coco_output=Path(args.val_coco),
        classes=classes,
    )
    model = ModelConfig(
        num_classes=len(classes) + 1,
        backbone=args.backbone,
        weights=args.weights,
        trainable_backbone_layers=args.trainable_backbone_layers,
    )
    training = TrainingConfig(
        output_dir=Path(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        step_size=args.step_size,
        gamma=args.gamma,
        device=args.device,
        amp=args.amp,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        log_every=args.log_every,
        save_every=args.save_every,
        resume_checkpoint=Path(args.resume) if args.resume else None,
    )
    evaluation = EvaluationConfig(
        score_threshold=args.score_threshold,
        iou_type=args.iou_type,
        save_predictions=not args.no_save_predictions,
    )
    return PipelineConfig(train, val, model, training, evaluation)


def cmd_train(args: argparse.Namespace) -> None:
    """Run the full training pipeline."""
    config = _build_pipeline_config(args)
    pipeline = DetectionPipeline(config)
    pipeline.run()


def cmd_eval(args: argparse.Namespace) -> None:
    """Evaluate a checkpoint on the validation split."""
    classes = BDD100K_DET_CLASSES.copy()
    val_labels = Path(args.val_labels)
    val_images = Path(args.val_images)
    val_coco = Path(args.val_coco)

    val_coco = convert_bdd_to_coco(
        val_labels,
        val_images,
        val_coco,
        classes,
    )

    val_dataset = CocoDetectionDataset(
        val_images,
        val_coco,
        transforms=build_transforms(train=False),
    )
    num_workers = args.num_workers
    pin_memory = args.pin_memory and str(args.device).startswith("cuda")
    persistent_workers = args.persistent_workers and num_workers > 0
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = _extract_state_dict(checkpoint)
    inferred_backbone = _infer_backbone(state_dict)
    backbone = args.backbone
    if backbone == "auto" and inferred_backbone:
        backbone = inferred_backbone
        print(f"Inferred backbone: {backbone}")
    elif inferred_backbone and inferred_backbone != backbone:
        print(
            "Warning: checkpoint looks like "
            f"{inferred_backbone} but CLI requested {backbone}."
        )

    model = build_model(
        ModelConfig(
            num_classes=len(classes) + 1,
            backbone=backbone,
            weights=None,
        )
    )

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(
            "Checkpoint load summary:",
            {
                "missing_keys": len(missing),
                "unexpected_keys": len(unexpected),
            },
        )
    model.to(torch.device(args.device))

    evaluator = Evaluator(
        val_dataset.coco,
        Path(args.output_dir),
        split="val",
        score_threshold=args.score_threshold,
        iou_type=args.iou_type,
        save_predictions=not args.no_save_predictions,
    )
    evaluator.evaluate(model, val_loader, epoch=0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="BDD100K detection training pipeline"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train and evaluate")
    train.add_argument("--train-labels", default=_default_labels("train"))
    train.add_argument("--val-labels", default=_default_labels("val"))
    train.add_argument("--train-images", default=_default_images("train"))
    train.add_argument("--val-images", default=_default_images("val"))
    train.add_argument("--train-coco", default=_default_coco("train"))
    train.add_argument("--val-coco", default=_default_coco("val"))
    train.add_argument("--output-dir", default="outputs/training")
    train.add_argument("--epochs", type=int, default=12)
    train.add_argument("--batch-size", type=int, default=4)
    train.add_argument("--num-workers", type=int, default=4)
    train.add_argument("--lr", type=float, default=0.005)
    train.add_argument("--momentum", type=float, default=0.9)
    train.add_argument("--weight-decay", type=float, default=0.0005)
    train.add_argument("--step-size", type=int, default=8)
    train.add_argument("--gamma", type=float, default=0.1)
    train.add_argument("--device", default="cuda")
    train.add_argument("--amp", action="store_true")
    train.add_argument("--log-every", type=int, default=50)
    train.add_argument("--save-every", type=int, default=1)
    train.add_argument("--pin-memory", action="store_true", default=True)
    train.add_argument(
        "--no-pin-memory",
        action="store_false",
        dest="pin_memory",
        help="Disable DataLoader pin_memory.",
    )
    train.add_argument("--persistent-workers", action="store_true", default=True)
    train.add_argument(
        "--no-persistent-workers",
        action="store_false",
        dest="persistent_workers",
        help="Disable DataLoader persistent workers.",
    )
    train.add_argument("--weights", default="DEFAULT")
    train.add_argument(
        "--resume",
        default=None,
        help="Path to a training checkpoint to resume from.",
    )
    train.add_argument(
        "--backbone",
        choices=["resnet50", "resnet101"],
        default="resnet50",
    )
    train.add_argument("--trainable-backbone-layers", type=int, default=3)
    train.add_argument("--score-threshold", type=float, default=0.05)
    train.add_argument("--iou-type", default="bbox")
    train.add_argument("--no-save-predictions", action="store_true")
    train.set_defaults(func=cmd_train)

    eval_cmd = subparsers.add_parser("eval", help="Evaluate a checkpoint")
    eval_cmd.add_argument("--val-labels", default=_default_labels("val"))
    eval_cmd.add_argument("--val-images", default=_default_images("val"))
    eval_cmd.add_argument("--val-coco", default=_default_coco("val"))
    eval_cmd.add_argument(
        "--checkpoint",
        default="pretrained-models/model.pth",
        help="Path to a torchvision-style checkpoint (.pth).",
    )
    eval_cmd.add_argument("--output-dir", default="outputs/eval")
    eval_cmd.add_argument("--num-workers", type=int, default=4)
    eval_cmd.add_argument("--pin-memory", action="store_true", default=True)
    eval_cmd.add_argument(
        "--no-pin-memory",
        action="store_false",
        dest="pin_memory",
        help="Disable DataLoader pin_memory.",
    )
    eval_cmd.add_argument(
        "--persistent-workers", action="store_true", default=True
    )
    eval_cmd.add_argument(
        "--no-persistent-workers",
        action="store_false",
        dest="persistent_workers",
        help="Disable DataLoader persistent workers.",
    )
    eval_cmd.add_argument("--device", default="cuda")
    eval_cmd.add_argument(
        "--backbone",
        choices=["resnet50", "resnet101", "auto"],
        default="auto",
    )
    eval_cmd.add_argument("--score-threshold", type=float, default=0.05)
    eval_cmd.add_argument("--iou-type", default="bbox")
    eval_cmd.add_argument("--no-save-predictions", action="store_true")
    eval_cmd.set_defaults(func=cmd_eval)

    return parser


def main() -> None:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
