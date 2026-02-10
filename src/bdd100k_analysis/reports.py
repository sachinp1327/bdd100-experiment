"""Reporting utilities for comparing splits."""

from __future__ import annotations

from typing import Any, Dict


def compare_summaries(
    train: Dict[str, Any], val: Dict[str, Any]
) -> Dict[str, Any]:
    """Create a comparison report for train/val split summaries."""
    train_classes = train.get("classes", {})
    val_classes = val.get("classes", {})
    all_classes = sorted(set(train_classes) | set(val_classes))

    class_comparison = {}
    for class_name in all_classes:
        train_boxes = train_classes.get(class_name, {}).get("box_count", 0)
        val_boxes = val_classes.get(class_name, {}).get("box_count", 0)
        train_images = train_classes.get(class_name, {}).get("image_count", 0)
        val_images = val_classes.get(class_name, {}).get("image_count", 0)
        class_comparison[class_name] = {
            "train_boxes": train_boxes,
            "val_boxes": val_boxes,
            "box_ratio": (train_boxes / val_boxes) if val_boxes else None,
            "train_images": train_images,
            "val_images": val_images,
            "image_ratio": (train_images / val_images) if val_images else None,
        }

    return {
        "train": {
            "image_count": train.get("image_count", 0),
            "total_boxes": train.get("total_boxes", 0),
        },
        "val": {
            "image_count": val.get("image_count", 0),
            "total_boxes": val.get("total_boxes", 0),
        },
        "classes": class_comparison,
    }
