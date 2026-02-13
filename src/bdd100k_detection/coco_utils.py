"""Utilities for converting BDD100K labels to COCO format."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

from PIL import Image
from tqdm import tqdm

from bdd100k_analysis.streaming import iter_label_items


@dataclass
class CocoCategory:
    """COCO category descriptor."""

    id: int
    name: str


class CocoWriter:
    """Stream COCO images and annotations to disk."""

    def __init__(
        self, output_path: Path, categories: List[CocoCategory]
    ) -> None:
        self.output_path = output_path
        self.categories = categories
        self.images_path = output_path.with_suffix(".images.jsonl")
        self.annotations_path = output_path.with_suffix(".ann.jsonl")

        self._images_handle = self.images_path.open("w", encoding="utf-8")
        self._ann_handle = self.annotations_path.open("w", encoding="utf-8")

    def add_image(self, image: Dict) -> None:
        """Append a COCO image record."""
        self._images_handle.write(json.dumps(image) + "\n")

    def add_annotation(self, annotation: Dict) -> None:
        """Append a COCO annotation record."""
        self._ann_handle.write(json.dumps(annotation) + "\n")

    def close(self) -> None:
        """Finalize and close file handles."""
        self._images_handle.close()
        self._ann_handle.close()

    def materialize(self) -> Path:
        """Create the final COCO JSON file and return its path."""
        with self.output_path.open("w", encoding="utf-8") as handle:
            handle.write('{"images":[')
            self._write_jsonl_array(self.images_path, handle)
            handle.write('],"annotations":[')
            self._write_jsonl_array(self.annotations_path, handle)
            handle.write('],"categories":')
            json.dump([cat.__dict__ for cat in self.categories], handle)
            handle.write("}")
        return self.output_path

    @staticmethod
    def _write_jsonl_array(jsonl_path: Path, handle) -> None:
        """Write JSONL lines as a JSON array into an open handle."""
        first = True
        with jsonl_path.open("r", encoding="utf-8") as reader:
            for line in reader:
                if not line.strip():
                    continue
                if not first:
                    handle.write(",")
                handle.write(line.strip())
                first = False


def _get_image_size(images_root: Path, image_name: str) -> tuple[int, int]:
    """Read image dimensions from disk."""
    image_path = images_root / image_name
    with Image.open(image_path) as image:
        return image.size


def convert_bdd_to_coco(
    labels_path: Path,
    images_root: Path,
    output_path: Path,
    classes: Iterable[str],
) -> Path:
    """Convert BDD100K labels to COCO format and return output path."""
    if output_path.exists():
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    categories = [
        CocoCategory(id=index + 1, name=name)
        for index, name in enumerate(classes)
    ]
    category_map = {cat.name: cat.id for cat in categories}

    writer = CocoWriter(output_path, categories)
    image_id = 1
    annotation_id = 1

    for item in tqdm(
        iter_label_items(labels_path), desc=f"Convert {labels_path.name}"
    ):
        image_name = item.get("name", "")
        width = int(item.get("width") or 0)
        height = int(item.get("height") or 0)
        if width <= 0 or height <= 0:
            width, height = _get_image_size(images_root, image_name)

        writer.add_image(
            {
                "id": image_id,
                "file_name": image_name,
                "width": width,
                "height": height,
            }
        )

        labels = item.get("labels") or []
        for label in labels:
            if "box2d" not in label:
                continue
            category = label.get("category")
            if category not in category_map:
                continue
            box = label["box2d"]
            x1 = float(box.get("x1", 0.0))
            y1 = float(box.get("y1", 0.0))
            x2 = float(box.get("x2", 0.0))
            y2 = float(box.get("y2", 0.0))
            box_w = max(x2 - x1, 0.0)
            box_h = max(y2 - y1, 0.0)
            if box_w <= 0 or box_h <= 0:
                continue

            writer.add_annotation(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_map[category],
                    "bbox": [x1, y1, box_w, box_h],
                    "area": box_w * box_h,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

        image_id += 1

    writer.close()
    return writer.materialize()
