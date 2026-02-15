"""Dataset utilities for BDD100K detection training."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image
from pycocotools.coco import COCO

from .transforms import Compose


class CocoDetectionDataset(torch.utils.data.Dataset):
    """COCO-format dataset wrapper for detection training."""

    def __init__(
        self,
        images_root: Path,
        annotation_file: Path,
        transforms: Compose | None = None,
    ) -> None:
        """Initialize the COCO detection dataset wrapper."""
        self.images_root = images_root
        self.coco = COCO(str(annotation_file))
        self.image_ids = sorted(self.coco.imgs.keys())
        self.transforms = transforms

    def __len__(self) -> int:
        """Return number of images in the dataset."""
        return len(self.image_ids)

    def __getitem__(self, index: int):
        """Return image and target dict for an index."""
        image_id = self.image_ids[index]
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = self.images_root / image_info["file_name"]
        image = Image.open(image_path).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=[image_id])
        anns = self.coco.loadAnns(ann_ids)

        boxes: List[List[float]] = []
        labels: List[int] = []
        areas: List[float] = []
        iscrowd: List[int] = []

        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(int(ann["category_id"]))
            areas.append(float(ann.get("area", w * h)))
            iscrowd.append(int(ann.get("iscrowd", 0)))

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([image_id]),
            "area": torch.tensor(areas, dtype=torch.float32),
            "iscrowd": torch.tensor(iscrowd, dtype=torch.int64),
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target


def collate_fn(batch: List[Tuple[torch.Tensor, Dict]]) -> Tuple[List, List]:
    """Custom collate function for detection batches."""
    images, targets = zip(*batch)
    return list(images), list(targets)
