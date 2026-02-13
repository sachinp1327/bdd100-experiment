"""Training loop for Faster R-CNN."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from .evaluator import EvaluationResult, Evaluator


@dataclass
class TrainState:
    """Track state for training progress."""

    epoch: int
    step: int
    loss: float


class Trainer:
    """Train a detection model with optional evaluation."""

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
        device: torch.device,
        output_dir: Path,
        epochs: int,
        log_every: int,
        save_every: int,
        amp: bool,
        start_epoch: int = 1,
        evaluator: Optional[Evaluator] = None,
        val_loader: Optional[DataLoader] = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.output_dir = output_dir
        self.epochs = epochs
        self.log_every = log_every
        self.save_every = save_every
        self.amp = amp
        self.start_epoch = start_epoch
        self.evaluator = evaluator
        self.val_loader = val_loader
        self.scaler = GradScaler(enabled=amp)

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train(self) -> None:
        """Run the full training loop."""
        self.model.to(self.device)
        for epoch in range(self.start_epoch, self.epochs + 1):
            self._train_one_epoch(epoch)
            self.lr_scheduler.step()

            if self.evaluator and self.val_loader:
                result = self.evaluator.evaluate(
                    self.model, self.val_loader, epoch
                )
                self._log_eval_result(result)

            if epoch % self.save_every == 0:
                self._save_checkpoint(epoch)

    def _train_one_epoch(self, epoch: int) -> None:
        """Train the model for a single epoch."""
        self.model.train()
        running_loss = 0.0

        progress = tqdm(
            enumerate(self.train_loader, start=1),
            total=len(self.train_loader),
            desc=f"Train epoch {epoch}/{self.epochs}",
        )
        for step, (images, targets) in progress:
            images = [image.to(self.device) for image in images]
            targets = [
                {k: v.to(self.device) for k, v in target.items()}
                for target in targets
            ]

            with autocast(enabled=self.amp):
                loss_dict = self.model(images, targets)
                loss = sum(loss_dict.values())

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += float(loss.item())
            image_ids = self._format_image_ids(targets)
            if step % self.log_every == 0 or step == 1:
                avg_loss = running_loss / step
                progress.set_postfix(loss=f"{avg_loss:.4f}", ids=image_ids)
                print(
                    f"Epoch {epoch} Step {step} "
                    f"Loss {avg_loss:.4f} "
                    f"Image IDs {image_ids}"
                )
            else:
                progress.set_postfix(ids=image_ids)

    @staticmethod
    def _format_image_ids(targets: list[Dict]) -> str:
        """Format image IDs from a batch of targets for logging."""
        ids = []
        for target in targets:
            value = target.get("image_id")
            if value is None:
                continue
            if torch.is_tensor(value):
                if value.numel() == 1:
                    ids.append(int(value.item()))
                else:
                    ids.extend(int(v) for v in value.flatten().tolist())
            else:
                ids.append(int(value))
        if not ids:
            return "n/a"
        if len(ids) <= 3:
            return ",".join(str(item) for item in ids)
        return f"{min(ids)}..{max(ids)} ({len(ids)})"

    def _save_checkpoint(self, epoch: int) -> None:
        """Save model checkpoint to disk."""
        checkpoint = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.lr_scheduler.state_dict(),
        }
        path = self.output_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, path)

    def _log_eval_result(self, result: EvaluationResult) -> None:
        """Log and persist evaluation metrics for an epoch."""
        summary = (
            f"Eval epoch {result.epoch} "
            f"AP={result.metrics.get('AP', 0.0):.4f} "
            f"AP50={result.metrics.get('AP50', 0.0):.4f} "
            f"AP75={result.metrics.get('AP75', 0.0):.4f}"
        )
        print(summary)
        metrics_path = self.output_dir / "eval_metrics.jsonl"
        payload = {
            "epoch": result.epoch,
            "metrics": result.metrics,
            "predictions_path": str(result.predictions_path)
            if result.predictions_path
            else None,
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")
