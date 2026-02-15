"""Evaluation utilities for detection models."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class EvaluationResult:
    """Structured evaluation result for a validation run."""

    epoch: int
    metrics: Dict[str, float]
    predictions_path: Optional[Path]


class Evaluator:
    """Evaluate a model with COCO-style metrics."""

    def __init__(
        self,
        coco_gt: COCO,
        output_dir: Path,
        split: str = "val",
        score_threshold: float = 0.05,
        iou_type: str = "bbox",
        save_predictions: bool = True,
    ) -> None:
        """Initialize the evaluator with dataset and output settings."""
        self.coco_gt = coco_gt
        self.output_dir = output_dir
        self.split = split
        self.score_threshold = score_threshold
        self.iou_type = iou_type
        self.save_predictions = save_predictions

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(
        self, model: torch.nn.Module, loader: DataLoader, epoch: int
    ) -> EvaluationResult:
        """Run evaluation and return metrics."""
        model.eval()
        predictions = self._collect_predictions(model, loader)

        predictions_path = None
        if self.save_predictions:
            predictions_path = self.output_dir / f"predictions_{epoch}.json"
            with predictions_path.open("w", encoding="utf-8") as handle:
                json.dump(predictions, handle)

        coco_dt = self.coco_gt.loadRes(predictions) if predictions else []
        coco_eval = COCOeval(self.coco_gt, coco_dt, self.iou_type)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        metrics = {
            "AP": float(coco_eval.stats[0]),
            "AP50": float(coco_eval.stats[1]),
            "AP75": float(coco_eval.stats[2]),
            "AP_small": float(coco_eval.stats[3]),
            "AP_medium": float(coco_eval.stats[4]),
            "AP_large": float(coco_eval.stats[5]),
        }
        return EvaluationResult(epoch, metrics, predictions_path)

    def _collect_predictions(
        self, model: torch.nn.Module, loader: DataLoader
    ) -> List[Dict]:
        """Collect model predictions in COCO format."""
        results: List[Dict] = []
        device = next(model.parameters()).device

        total = len(loader.dataset) if hasattr(loader, "dataset") else None
        desc = f"Evaluating {self.split} ({self.iou_type})"
        with torch.no_grad():
            for images, targets in tqdm(
                loader, total=total, desc=desc, unit="img"
            ):
                images = [image.to(device) for image in images]
                outputs = model(images)

                for target, output in zip(targets, outputs):
                    image_id = int(target["image_id"].item())
                    boxes = output["boxes"].cpu().tolist()
                    scores = output["scores"].cpu().tolist()
                    labels = output["labels"].cpu().tolist()

                    for box, score, label in zip(boxes, scores, labels):
                        if score < self.score_threshold:
                            continue
                        x1, y1, x2, y2 = box
                        results.append(
                            {
                                "image_id": image_id,
                                "category_id": int(label),
                                "bbox": [x1, y1, x2 - x1, y2 - y1],
                                "score": float(score),
                            }
                        )
        return results
