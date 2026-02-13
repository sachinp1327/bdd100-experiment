"""Configuration models for the detection training pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from bdd100k_analysis.constants import BDD100K_DET_CLASSES


@dataclass
class DatasetConfig:
    """Dataset configuration for a single split."""

    split: str
    labels_path: Path
    images_root: Path
    coco_output: Path
    classes: List[str] = field(
        default_factory=lambda: BDD100K_DET_CLASSES.copy()
    )


@dataclass
class ModelConfig:
    """Model configuration for Faster R-CNN."""

    num_classes: int
    backbone: str = "resnet50"
    weights: str | None = "DEFAULT"
    trainable_backbone_layers: int = 3
    min_size: int = 800
    max_size: int = 1333


@dataclass
class TrainingConfig:
    """Training hyperparameters and runtime settings."""

    output_dir: Path
    epochs: int = 12
    batch_size: int = 2
    num_workers: int = 4
    lr: float = 0.005
    momentum: float = 0.9
    weight_decay: float = 0.0005
    step_size: int = 8
    gamma: float = 0.1
    seed: int = 42
    device: str = "cuda"
    amp: bool = False
    pin_memory: bool = True
    persistent_workers: bool = True
    log_every: int = 50
    save_every: int = 1
    resume_checkpoint: Optional[Path] = None


@dataclass
class EvaluationConfig:
    """Evaluation configuration settings."""

    score_threshold: float = 0.05
    iou_type: str = "bbox"
    save_predictions: bool = True


@dataclass
class PipelineConfig:
    """Unified configuration for the training pipeline."""

    train: DatasetConfig
    val: DatasetConfig
    model: ModelConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
