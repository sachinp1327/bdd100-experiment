"""Model builders for detection training."""
from __future__ import annotations

from typing import Optional

import torch
from torchvision.models import ResNet101_Weights
from torchvision.models.detection import (
    FasterRCNN,
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from .config import ModelConfig


def _resolve_weights(weights: Optional[str]):
    """Resolve a weights identifier into torchvision weights enum."""
    if weights is None:
        return None
    if weights == "DEFAULT":
        return FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    return FasterRCNN_ResNet50_FPN_Weights.DEFAULT


def build_model(config: ModelConfig) -> torch.nn.Module:
    """Build a Faster R-CNN model based on the config."""
    if config.backbone == "resnet50":
        weights = _resolve_weights(config.weights)
        model = fasterrcnn_resnet50_fpn(
            weights=weights,
            trainable_backbone_layers=config.trainable_backbone_layers,
            min_size=config.min_size,
            max_size=config.max_size,
        )
    elif config.backbone == "resnet101":
        if config.weights in ("DEFAULT", "IMAGENET"):
            backbone_weights = ResNet101_Weights.IMAGENET1K_V1
        else:
            backbone_weights = None
        backbone = resnet_fpn_backbone(
            backbone_name="resnet101",
            weights=backbone_weights,
            trainable_layers=config.trainable_backbone_layers,
        )
        model = FasterRCNN(
            backbone,
            num_classes=config.num_classes,
            min_size=config.min_size,
            max_size=config.max_size,
        )
    else:
        raise ValueError(f"Unsupported backbone: {config.backbone}")

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features, config.num_classes
    )
    return model
