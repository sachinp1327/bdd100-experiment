"""Rendering utilities for evaluation diagnostics."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

from PIL import Image, ImageDraw, ImageFont


def _bbox_to_xyxy(bbox: Iterable[float]) -> List[float]:
    """Convert COCO bbox [x, y, w, h] to [x1, y1, x2, y2]."""
    x, y, w, h = bbox
    return [float(x), float(y), float(x + w), float(y + h)]


def render_image_with_boxes(
    image_path: Path,
    gt_boxes: List[Dict],
    pred_boxes: List[Dict],
    output_path: Path,
    class_map: Dict[int, str],
) -> None:
    """Draw ground-truth and predicted boxes on an image."""
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for gt in gt_boxes:
        coords = _bbox_to_xyxy(gt["bbox"])
        label = class_map.get(int(gt["category_id"]), "gt")
        draw.rectangle(coords, outline="lime", width=3)
        draw.text((coords[0] + 4, coords[1] + 4), f"GT {label}", fill="lime", font=font)

    for pred in pred_boxes:
        coords = _bbox_to_xyxy(pred["bbox"])
        label = class_map.get(int(pred["category_id"]), "pred")
        score = pred.get("score")
        score_text = f"{score:.2f}" if score is not None else ""
        draw.rectangle(coords, outline="red", width=2)
        draw.text(
            (coords[0] + 4, coords[1] + 4),
            f"Pred {label} {score_text}",
            fill="red",
            font=font,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
