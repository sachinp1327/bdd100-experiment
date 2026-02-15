"""Diagnostics pipeline for model evaluation on BDD100K."""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

import ijson
from .clustering import ClusterResult, kmeans, pca_2d, standardize
from .visuals import render_image_with_boxes


@dataclass
class DiagnosticsConfig:
    """Configuration for evaluation diagnostics."""

    predictions_path: Path
    coco_path: Path
    labels_path: Path
    images_root: Path
    output_dir: Path
    score_threshold: float = 0.05
    iou_threshold: float = 0.5
    top_k: int = 5
    cluster_k: int = 5
    analysis_summary: Path | None = None


def _load_predictions(path: Path) -> List[Dict[str, Any]]:
    """Load COCO-style predictions from disk."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_label_metadata(labels_path: Path) -> Dict[str, Dict[str, str]]:
    """Load image-level metadata (weather, scene, timeofday) from labels."""
    metadata: Dict[str, Dict[str, str]] = {}
    with labels_path.open("rb") as handle:
        for item in ijson.items(handle, "item"):
            name = item.get("name")
            if not name:
                continue
            attributes = item.get("attributes") or {}
            metadata[name] = {
                "timeofday": attributes.get("timeofday", "unknown"),
                "weather": attributes.get("weather", "unknown"),
                "scene": attributes.get("scene", "unknown"),
            }
    return metadata


def _bbox_iou(box_a: Iterable[float], box_b: Iterable[float]) -> float:
    """Compute IoU between two COCO bboxes [x, y, w, h]."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = aw * ah
    area_b = bw * bh
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def _match_class(
    preds: List[Dict[str, Any]],
    gts: List[Dict[str, Any]],
    iou_threshold: float,
) -> Tuple[int, int, int, float]:
    """Greedy match predictions to GT for one class."""
    preds_sorted = sorted(
        preds, key=lambda item: float(item.get("score", 0.0)), reverse=True
    )
    matched_gt: set[int] = set()
    tp = fp = 0
    iou_sum = 0.0

    for pred in preds_sorted:
        best_iou = 0.0
        best_idx = None
        for idx, gt in enumerate(gts):
            if idx in matched_gt:
                continue
            iou = _bbox_iou(pred["bbox"], gt["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_iou >= iou_threshold and best_idx is not None:
            matched_gt.add(best_idx)
            tp += 1
            iou_sum += best_iou
        else:
            fp += 1

    fn = len(gts) - tp
    mean_iou = iou_sum / tp if tp else 0.0
    return tp, fp, fn, mean_iou


def _precision_recall(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """Compute precision, recall, and f1."""
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def _compute_coco_metrics(
    coco_gt: COCO,
    predictions: List[Dict[str, Any]],
    iou_type: str = "bbox",
) -> Tuple[Dict[str, float], Dict[int, Dict[str, float]]]:
    """Compute COCO metrics and per-class AP/AR."""
    coco_dt = coco_gt.loadRes(predictions) if predictions else []
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    overall = {
        "AP": float(coco_eval.stats[0]),
        "AP50": float(coco_eval.stats[1]),
        "AP75": float(coco_eval.stats[2]),
        "AP_small": float(coco_eval.stats[3]),
        "AP_medium": float(coco_eval.stats[4]),
        "AP_large": float(coco_eval.stats[5]),
        "AR_1": float(coco_eval.stats[6]),
        "AR_10": float(coco_eval.stats[7]),
        "AR_100": float(coco_eval.stats[8]),
        "AR_small": float(coco_eval.stats[9]),
        "AR_medium": float(coco_eval.stats[10]),
        "AR_large": float(coco_eval.stats[11]),
    }

    precision = coco_eval.eval["precision"]
    recall = coco_eval.eval["recall"]
    cat_ids = coco_eval.params.catIds
    iou_thrs = coco_eval.params.iouThrs
    max_dets = coco_eval.params.maxDets
    det_index = max_dets.index(100) if 100 in max_dets else -1

    def _mean(arr: np.ndarray) -> float:
        valid = arr[arr > -1]
        return float(valid.mean()) if valid.size else 0.0

    iou_50_idx = int(np.where(np.isclose(iou_thrs, 0.5))[0][0])
    iou_75_idx = int(np.where(np.isclose(iou_thrs, 0.75))[0][0])

    per_class: Dict[int, Dict[str, float]] = {}
    for idx, cat_id in enumerate(cat_ids):
        per_class[cat_id] = {
            "AP": _mean(precision[:, :, idx, 0, det_index]),
            "AP50": _mean(precision[iou_50_idx, :, idx, 0, det_index]),
            "AP75": _mean(precision[iou_75_idx, :, idx, 0, det_index]),
            "AP_small": _mean(precision[:, :, idx, 1, det_index]),
            "AP_medium": _mean(precision[:, :, idx, 2, det_index]),
            "AP_large": _mean(precision[:, :, idx, 3, det_index]),
            "AR": _mean(recall[:, idx, 0, det_index]),
        }
    return overall, per_class


class EvaluationDiagnostics:
    """Generate evaluation diagnostics and artifacts for a predictions file."""

    def __init__(self, config: DiagnosticsConfig) -> None:
        self.config = config
        self.output_dir = config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> Dict[str, Any]:
        """Run the diagnostics pipeline and write artifacts to disk."""
        predictions = _load_predictions(self.config.predictions_path)
        coco_gt = COCO(str(self.config.coco_path))

        image_map = {img_id: meta["file_name"] for img_id, meta in coco_gt.imgs.items()}
        class_map = {cat["id"]: cat["name"] for cat in coco_gt.loadCats(coco_gt.getCatIds())}

        metadata = _load_label_metadata(self.config.labels_path)

        overall_metrics, per_class_ap = _compute_coco_metrics(coco_gt, predictions)

        preds_by_image: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for pred in predictions:
            if pred.get("score", 0.0) >= self.config.score_threshold:
                preds_by_image[int(pred["image_id"])].append(pred)

        class_ids = coco_gt.getCatIds()
        per_class_counts = {cat_id: {"tp": 0, "fp": 0, "fn": 0} for cat_id in class_ids}
        per_class_samples: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        attribute_counts: Dict[str, Dict[int, Dict[str, Dict[str, int]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "images": 0}))
        )

        image_metrics: List[Dict[str, Any]] = []

        img_ids = coco_gt.getImgIds()
        for image_id in tqdm(img_ids, desc="Scoring images"):
            file_name = image_map.get(image_id, "")
            attrs = metadata.get(
                file_name,
                {"timeofday": "unknown", "weather": "unknown", "scene": "unknown"},
            )
            gt_anns = coco_gt.imgToAnns.get(image_id, [])
            preds = preds_by_image.get(image_id, [])

            image_tp = image_fp = image_fn = 0
            image_iou_sum = 0.0
            image_match_count = 0

            for cat_id in class_ids:
                gt_class = [ann for ann in gt_anns if ann["category_id"] == cat_id]
                pred_class = [pred for pred in preds if pred["category_id"] == cat_id]
                tp, fp, fn, mean_iou = _match_class(
                    pred_class, gt_class, self.config.iou_threshold
                )
                per_class_counts[cat_id]["tp"] += tp
                per_class_counts[cat_id]["fp"] += fp
                per_class_counts[cat_id]["fn"] += fn

                if gt_class or pred_class:
                    attribute_counts["timeofday"][cat_id][attrs["timeofday"]]["tp"] += tp
                    attribute_counts["timeofday"][cat_id][attrs["timeofday"]]["fp"] += fp
                    attribute_counts["timeofday"][cat_id][attrs["timeofday"]]["fn"] += fn
                    attribute_counts["timeofday"][cat_id][attrs["timeofday"]]["images"] += 1

                    attribute_counts["weather"][cat_id][attrs["weather"]]["tp"] += tp
                    attribute_counts["weather"][cat_id][attrs["weather"]]["fp"] += fp
                    attribute_counts["weather"][cat_id][attrs["weather"]]["fn"] += fn
                    attribute_counts["weather"][cat_id][attrs["weather"]]["images"] += 1

                    attribute_counts["scene"][cat_id][attrs["scene"]]["tp"] += tp
                    attribute_counts["scene"][cat_id][attrs["scene"]]["fp"] += fp
                    attribute_counts["scene"][cat_id][attrs["scene"]]["fn"] += fn
                    attribute_counts["scene"][cat_id][attrs["scene"]]["images"] += 1

                image_tp += tp
                image_fp += fp
                image_fn += fn
                if tp:
                    image_iou_sum += mean_iou * tp
                    image_match_count += tp

                if gt_class:
                    precision, recall, f1 = _precision_recall(tp, fp, fn)
                    per_class_samples[cat_id].append(
                        {
                            "image_id": image_id,
                            "file_name": file_name,
                            "tp": tp,
                            "fp": fp,
                            "fn": fn,
                            "precision": precision,
                            "recall": recall,
                            "f1": f1,
                            "attributes": attrs,
                        }
                    )

            mean_iou = image_iou_sum / image_match_count if image_match_count else 0.0
            precision, recall, f1 = _precision_recall(image_tp, image_fp, image_fn)
            image_metrics.append(
                {
                    "image_id": image_id,
                    "file_name": file_name,
                    "tp": image_tp,
                    "fp": image_fp,
                    "fn": image_fn,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "mean_iou": mean_iou,
                    "num_gt": len(gt_anns),
                    "num_pred": len(preds),
                    "attributes": attrs,
                }
            )

        per_class_metrics = {}
        for cat_id, counts in per_class_counts.items():
            precision, recall, f1 = _precision_recall(
                counts["tp"], counts["fp"], counts["fn"]
            )
            per_class_metrics[cat_id] = {
                "tp": counts["tp"],
                "fp": counts["fp"],
                "fn": counts["fn"],
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
            per_class_metrics[cat_id].update(per_class_ap.get(cat_id, {}))

        attribute_metrics = self._finalize_attribute_metrics(
            attribute_counts, class_map
        )

        samples = self._select_samples(per_class_samples, class_map, image_map, preds_by_image, coco_gt)

        clusters = self._cluster_images(image_metrics)

        analysis_links = self._build_analysis_links(per_class_metrics, class_map)

        self._write_json("overall_metrics.json", overall_metrics)
        self._write_json(
            "per_class_metrics.json",
            {class_map[k]: v for k, v in per_class_metrics.items()},
        )
        self._write_json("attribute_metrics.json", attribute_metrics)
        self._write_json("samples.json", samples)
        self._write_json("clusters.json", clusters)
        if analysis_links:
            self._write_json("analysis_links.json", analysis_links)

        self._write_jsonl("image_metrics.jsonl", image_metrics)

        return {
            "overall": overall_metrics,
            "per_class": per_class_metrics,
            "attributes": attribute_metrics,
        }

    def _finalize_attribute_metrics(
        self,
        counts: Dict[str, Any],
        class_map: Dict[int, str],
    ) -> Dict[str, Any]:
        """Convert attribute counts into metrics."""
        metrics: Dict[str, Any] = {}
        for attribute, per_class in counts.items():
            metrics[attribute] = {}
            for cat_id, values in per_class.items():
                class_name = class_map.get(cat_id, str(cat_id))
                metrics[attribute][class_name] = {}
                for value, entry in values.items():
                    precision, recall, f1 = _precision_recall(
                        entry["tp"], entry["fp"], entry["fn"]
                    )
                    metrics[attribute][class_name][value] = {
                        "tp": entry["tp"],
                        "fp": entry["fp"],
                        "fn": entry["fn"],
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "images": entry["images"],
                    }
        return metrics

    def _select_samples(
        self,
        per_class_samples: Dict[int, List[Dict[str, Any]]],
        class_map: Dict[int, str],
        image_map: Dict[int, str],
        preds_by_image: Dict[int, List[Dict[str, Any]]],
        coco_gt: COCO,
    ) -> Dict[str, Any]:
        """Select and render best/worst samples per class."""
        rendered_dir = self.output_dir / "renders"
        samples_out: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

        for cat_id, samples in per_class_samples.items():
            if not samples:
                continue
            class_name = class_map.get(cat_id, str(cat_id))
            sorted_samples = sorted(samples, key=lambda item: item["f1"])
            worst = sorted_samples[: self.config.top_k]
            best = sorted_samples[-self.config.top_k :][::-1]

            def _render(items: List[Dict[str, Any]], tag: str) -> List[Dict[str, Any]]:
                rendered_items = []
                for item in items:
                    image_id = item["image_id"]
                    file_name = image_map.get(image_id, item["file_name"])
                    if not file_name:
                        continue
                    image_path = self.config.images_root / file_name
                    if not image_path.exists():
                        continue
                    gt_boxes = [
                        ann
                        for ann in coco_gt.imgToAnns.get(image_id, [])
                        if ann["category_id"] == cat_id
                    ]
                    pred_boxes = [
                        pred
                        for pred in preds_by_image.get(image_id, [])
                        if pred["category_id"] == cat_id
                    ]
                    output_path = (
                        rendered_dir / class_name / tag / file_name
                    )
                    render_image_with_boxes(
                        image_path,
                        gt_boxes,
                        pred_boxes,
                        output_path,
                        class_map,
                    )
                    entry = dict(item)
                    entry["rendered_path"] = str(output_path)
                    rendered_items.append(entry)
                return rendered_items

            samples_out[class_name] = {
                "best": _render(best, "best"),
                "worst": _render(worst, "worst"),
            }
        return samples_out

    def _cluster_images(self, image_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Cluster images using error attributes and metadata."""
        if not image_metrics:
            return {}

        time_values = sorted(
            {item["attributes"]["timeofday"] for item in image_metrics}
        )
        weather_values = sorted(
            {item["attributes"]["weather"] for item in image_metrics}
        )
        scene_values = sorted(
            {item["attributes"]["scene"] for item in image_metrics}
        )

        feature_names = [
            "tp",
            "fp",
            "fn",
            "precision",
            "recall",
            "mean_iou",
            "num_gt",
            "num_pred",
        ]
        feature_names += [f"time_{v}" for v in time_values]
        feature_names += [f"weather_{v}" for v in weather_values]
        feature_names += [f"scene_{v}" for v in scene_values]

        features = []
        for item in image_metrics:
            row = [
                item["tp"],
                item["fp"],
                item["fn"],
                item["precision"],
                item["recall"],
                item["mean_iou"],
                item["num_gt"],
                item["num_pred"],
            ]
            row += [1 if item["attributes"]["timeofday"] == v else 0 for v in time_values]
            row += [1 if item["attributes"]["weather"] == v else 0 for v in weather_values]
            row += [1 if item["attributes"]["scene"] == v else 0 for v in scene_values]
            features.append(row)

        feature_array = np.array(features, dtype=float)
        scaled, _, _ = standardize(feature_array)
        cluster_result: ClusterResult = kmeans(
            scaled, k=self.config.cluster_k
        )
        coords = pca_2d(scaled)

        images_out = []
        for idx, item in enumerate(image_metrics):
            images_out.append(
                {
                    "image_id": item["image_id"],
                    "file_name": item["file_name"],
                    "cluster": int(cluster_result.labels[idx]) if cluster_result.labels.size else 0,
                    "coords": [float(coords[idx, 0]), float(coords[idx, 1])],
                    "tp": item["tp"],
                    "fp": item["fp"],
                    "fn": item["fn"],
                    "attributes": item["attributes"],
                }
            )

        cluster_summary = []
        for cluster_id in range(int(cluster_result.labels.max()) + 1 if cluster_result.labels.size else 0):
            members = [img for img in images_out if img["cluster"] == cluster_id]
            if not members:
                continue
            avg_error = float(np.mean([m["fp"] + m["fn"] for m in members]))
            cluster_summary.append(
                {
                    "cluster": cluster_id,
                    "count": len(members),
                    "avg_error": avg_error,
                    "sample_images": members[: min(5, len(members))],
                }
            )

        return {
            "k": self.config.cluster_k,
            "features": feature_names,
            "centroids": cluster_result.centroids.tolist(),
            "images": images_out,
            "summary": cluster_summary,
        }

    def _build_analysis_links(
        self,
        per_class_metrics: Dict[int, Dict[str, Any]],
        class_map: Dict[int, str],
    ) -> List[Dict[str, Any]]:
        """Join evaluation metrics with analysis summary stats."""
        if not self.config.analysis_summary:
            return []
        summary_path = self.config.analysis_summary
        if not summary_path.exists():
            return []
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        classes = summary.get("classes", {})
        links = []
        for class_name, stats in classes.items():
            match_id = None
            for cat_id, name in class_map.items():
                if name == class_name:
                    match_id = cat_id
                    break
            if match_id is None:
                continue
            metrics = per_class_metrics.get(match_id, {})
            links.append(
                {
                    "class": class_name,
                    "box_count": stats.get("box_count", 0),
                    "image_count": stats.get("image_count", 0),
                    "area_mean": stats.get("area", {}).get("mean", 0.0),
                    "aspect_mean": stats.get("aspect", {}).get("mean", 0.0),
                    "AP": metrics.get("AP", 0.0),
                    "AP50": metrics.get("AP50", 0.0),
                    "precision": metrics.get("precision", 0.0),
                    "recall": metrics.get("recall", 0.0),
                }
            )
        return links

    def _write_json(self, name: str, payload: Any) -> None:
        """Write JSON payload to output directory."""
        path = self.output_dir / name
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def _write_jsonl(self, name: str, rows: List[Dict[str, Any]]) -> None:
        """Write JSON Lines output to disk."""
        path = self.output_dir / name
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
