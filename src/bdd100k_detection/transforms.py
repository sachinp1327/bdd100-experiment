"""Custom transforms for detection training."""
from __future__ import annotations

from typing import Callable, Dict, Tuple

import torch
from torchvision.transforms import functional as F


class Compose:
    """Compose multiple transforms for detection tasks."""

    def __init__(self, transforms: Tuple[Callable, ...]) -> None:
        """Create a composed transform."""
        self.transforms = transforms

    def __call__(self, image, target):
        """Apply each transform in sequence."""
        for transform in self.transforms:
            image, target = transform(image, target)
        return image, target


class ToTensor:
    """Convert PIL image to float tensor."""

    def __call__(self, image, target):
        """Convert image to tensor and return target unchanged."""
        image = F.to_tensor(image)
        return image, target


class RandomHorizontalFlip:
    """Randomly flip the image and boxes horizontally."""

    def __init__(self, prob: float = 0.5) -> None:
        """Create a random horizontal flip transform."""
        self.prob = prob

    def __call__(self, image, target: Dict):
        """Randomly flip image and adjust boxes."""
        if torch.rand(1).item() >= self.prob:
            return image, target

        image = F.hflip(image)
        width = image.shape[-1]
        boxes = target["boxes"].clone()
        boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
        target["boxes"] = boxes
        return image, target


def build_transforms(train: bool) -> Compose:
    """Create transforms for train or validation split."""
    transforms = [ToTensor()]
    if train:
        transforms.append(RandomHorizontalFlip(0.5))
    return Compose(tuple(transforms))
