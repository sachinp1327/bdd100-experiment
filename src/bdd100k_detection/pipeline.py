"""End-to-end pipeline for training and evaluation."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

from .config import PipelineConfig
from .coco_utils import convert_bdd_to_coco
from .datasets import CocoDetectionDataset, collate_fn
from .evaluator import Evaluator
from .model import build_model
from .trainer import Trainer
from .transforms import build_transforms


class DetectionPipeline:
    """Orchestrates dataset conversion, training, and evaluation."""

    def __init__(self, config: PipelineConfig) -> None:
        """Initialize the pipeline with a configuration."""
        self.config = config

    def run(self) -> Dict[str, float]:
        """Execute conversion, training, and evaluation."""
        train_coco = convert_bdd_to_coco(
            self.config.train.labels_path,
            self.config.train.images_root,
            self.config.train.coco_output,
            self.config.train.classes,
        )
        val_coco = convert_bdd_to_coco(
            self.config.val.labels_path,
            self.config.val.images_root,
            self.config.val.coco_output,
            self.config.val.classes,
        )

        train_dataset = CocoDetectionDataset(
            self.config.train.images_root,
            train_coco,
            transforms=build_transforms(train=True),
        )
        val_dataset = CocoDetectionDataset(
            self.config.val.images_root,
            val_coco,
            transforms=build_transforms(train=False),
        )

        num_workers = self.config.training.num_workers
        pin_memory = (
            self.config.training.pin_memory
            and self.config.training.device.startswith("cuda")
        )
        persistent_workers = (
            self.config.training.persistent_workers and num_workers > 0
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

        model = build_model(self.config.model)
        device = torch.device(self.config.training.device)

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(
            params,
            lr=self.config.training.lr,
            momentum=self.config.training.momentum,
            weight_decay=self.config.training.weight_decay,
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.config.training.step_size,
            gamma=self.config.training.gamma,
        )

        start_epoch = 1
        resume_path = self.config.training.resume_checkpoint
        if resume_path:
            checkpoint = torch.load(resume_path, map_location="cpu")
            model_state = checkpoint.get("model_state")
            optimizer_state = checkpoint.get("optimizer_state")
            scheduler_state = checkpoint.get("scheduler_state")
            if model_state is None or optimizer_state is None:
                raise ValueError(
                    "Resume checkpoint is missing model/optimizer state."
                )
            if scheduler_state is None:
                raise ValueError(
                    "Resume checkpoint is missing scheduler state."
                )
            model.load_state_dict(model_state, strict=True)
            optimizer.load_state_dict(optimizer_state)
            lr_scheduler.load_state_dict(scheduler_state)
            start_epoch = int(checkpoint.get("epoch", 0)) + 1
            for state in optimizer.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value):
                        state[key] = value.to(device)

        evaluator = Evaluator(
            val_dataset.coco,
            self.config.training.output_dir / "eval",
            split=self.config.val.split,
            score_threshold=self.config.evaluation.score_threshold,
            iou_type=self.config.evaluation.iou_type,
            save_predictions=self.config.evaluation.save_predictions,
        )

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            device=device,
            output_dir=self.config.training.output_dir,
            epochs=self.config.training.epochs,
            log_every=self.config.training.log_every,
            save_every=self.config.training.save_every,
            amp=self.config.training.amp,
            start_epoch=start_epoch,
            evaluator=evaluator,
            val_loader=val_loader,
        )
        trainer.train()

        return asdict(self.config)

    def summarize(self) -> Dict:
        """Return configuration for reproducibility."""
        return asdict(self.config)
