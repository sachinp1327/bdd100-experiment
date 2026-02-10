"""Core analysis logic for BDD100K detection labels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image

from .stats import Histogram, OnlineStats, TopK
from .streaming import iter_label_items

AREA_BINS = [0.0] + np.logspace(-4, 0, 20).tolist()
ASPECT_BINS = np.linspace(0, 5, 21).tolist() + [10.0]


@dataclass
class ClassAggregator:
    """Aggregate stats for a single detection class."""

    name: str
    top_k: int = 10

    def __post_init__(self) -> None:
        self.box_count = 0
        self.image_count = 0
        self.area_stats = OnlineStats()
        self.aspect_stats = OnlineStats()
        self.area_hist = Histogram(AREA_BINS)
        self.aspect_hist = Histogram(ASPECT_BINS)
        self.largest_area = TopK(self.top_k, largest=True)
        self.smallest_area = TopK(self.top_k, largest=False)
        self.widest = TopK(self.top_k, largest=True)
        self.tallest = TopK(self.top_k, largest=False)

    def add_box(
        self,
        box: Dict[str, float],
        image_name: str,
        image_size: Tuple[int, int],
    ) -> None:
        """Update class-level stats for a single box."""
        width, height = image_size
        box_w = max(float(box["x2"]) - float(box["x1"]), 0.0)
        box_h = max(float(box["y2"]) - float(box["y1"]), 0.0)
        if box_w <= 0 or box_h <= 0:
            return

        self.box_count += 1
        aspect = box_w / box_h
        self.aspect_stats.add(aspect)
        self.aspect_hist.add(aspect)

        payload = {
            "image": image_name,
            "box": {k: float(v) for k, v in box.items()},
        }
        self.widest.add(aspect, payload)
        self.tallest.add(aspect, payload)

        if width <= 0 or height <= 0:
            return

        area_norm = (box_w * box_h) / (width * height)
        self.area_stats.add(area_norm)
        self.area_hist.add(area_norm)
        self.largest_area.add(area_norm, payload)
        self.smallest_area.add(area_norm, payload)

    def increment_image_count(self) -> None:
        """Increment the number of images containing this class."""
        self.image_count += 1

    def to_dict(self) -> Dict[str, Any]:
        """Serialize class-level stats to a dictionary."""
        return {
            "box_count": self.box_count,
            "image_count": self.image_count,
            "area": {
                **self.area_stats.to_dict(),
                "hist": self.area_hist.to_dict(),
                "largest": self.largest_area.to_list(),
                "smallest": self.smallest_area.to_list(),
            },
            "aspect": {
                **self.aspect_stats.to_dict(),
                "hist": self.aspect_hist.to_dict(),
                "widest": self.widest.to_list(),
                "tallest": self.tallest.to_list(),
            },
        }


class SplitAggregator:
    """Aggregate dataset-level stats for a single split."""

    def __init__(
        self, split: str, top_k: int = 10, images_root: Path | None = None
    ) -> None:
        """Initialize split-level aggregation state."""
        self.split = split
        self.image_count = 0
        self.images_with_boxes = 0
        self.total_boxes = 0
        self.class_aggs: Dict[str, ClassAggregator] = {}
        self.object_count_hist: Dict[str, int] = {}
        self.class_combo_counts: Dict[str, int] = {}
        self.class_combo_examples: Dict[str, str] = {}
        self.top_object_count_images = TopK(top_k, largest=True)
        self.images_root = images_root
        self._size_cache: Dict[str, Tuple[int, int]] = {}

    def _get_image_size(self, image_name: str) -> Tuple[int, int]:
        """Return image size from disk, cached per filename."""
        if image_name in self._size_cache:
            return self._size_cache[image_name]
        if self.images_root is None:
            return (0, 0)
        image_path = self.images_root / image_name
        try:
            with Image.open(image_path) as image:
                size = image.size  # (width, height)
        except FileNotFoundError:
            size = (0, 0)
        self._size_cache[image_name] = size
        return size

    def _get_class_agg(self, name: str) -> ClassAggregator:
        """Return (and create if needed) a class aggregator."""
        if name not in self.class_aggs:
            self.class_aggs[name] = ClassAggregator(name)
        return self.class_aggs[name]

    def update(self, item: Dict[str, Any]) -> None:
        """Update split-level stats from a single label item."""
        self.image_count += 1
        image_name = item.get("name", "unknown")
        width = int(item.get("width") or 0)
        height = int(item.get("height") or 0)
        if width <= 0 or height <= 0:
            width, height = self._get_image_size(image_name)
        labels = item.get("labels") or []

        classes_in_image = set()
        box_count = 0

        for label in labels:
            if "box2d" not in label:
                continue
            category = label.get("category")
            if not category:
                continue
            box = label["box2d"]
            class_agg = self._get_class_agg(category)
            class_agg.add_box(box, image_name, (width, height))
            classes_in_image.add(category)
            box_count += 1

        if box_count > 0:
            self.images_with_boxes += 1
        self.total_boxes += box_count
        self._update_object_hist(box_count)

        if classes_in_image:
            combo_key = "|".join(sorted(classes_in_image))
            self.class_combo_counts[combo_key] = (
                self.class_combo_counts.get(combo_key, 0) + 1
            )
            if combo_key not in self.class_combo_examples:
                self.class_combo_examples[combo_key] = image_name

        for category in classes_in_image:
            self.class_aggs[category].increment_image_count()

        if box_count > 0:
            payload = {"image": image_name, "count": box_count}
            self.top_object_count_images.add(
                float(box_count), payload
            )

    def _update_object_hist(self, box_count: int) -> None:
        """Update the objects-per-image histogram."""
        bucket = str(box_count) if box_count < 20 else "20+"
        self.object_count_hist[bucket] = (
            self.object_count_hist.get(bucket, 0) + 1
        )

    def to_summary(self) -> Dict[str, Any]:
        """Serialize split-level stats to a dictionary."""
        class_stats = {
            name: agg.to_dict() for name, agg in self.class_aggs.items()
        }

        rare_combos = sorted(
            self.class_combo_counts.items(), key=lambda x: x[1]
        )[:10]
        rare_combo_samples = [
            {
                "combo": combo,
                "count": count,
                "example_image": self.class_combo_examples.get(combo, ""),
            }
            for combo, count in rare_combos
        ]

        return {
            "split": self.split,
            "image_count": self.image_count,
            "images_with_boxes": self.images_with_boxes,
            "total_boxes": self.total_boxes,
            "classes": class_stats,
            "object_count_hist": self.object_count_hist,
            "class_combo_counts": self.class_combo_counts,
            "interesting_samples": {
                "rare_combos": rare_combo_samples,
                "high_object_count": self.top_object_count_images.to_list(),
            },
        }


def analyze_split(
    labels_path: str | Path,
    split: str,
    images_root: str | Path | None = None,
) -> Dict[str, Any]:
    """Analyze a BDD100K split and return summary stats."""
    images_root_path = Path(images_root) if images_root else None
    aggregator = SplitAggregator(
        split=split, top_k=15, images_root=images_root_path
    )
    for item in iter_label_items(labels_path):
        aggregator.update(item)
    return aggregator.to_summary()
