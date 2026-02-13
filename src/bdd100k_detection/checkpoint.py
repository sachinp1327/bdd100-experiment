"""Checkpoint loading utilities for detection models."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch


def _normalize_state_dict(state_dict: Dict) -> Dict:
    """Normalize common prefixes in checkpoint state dictionaries."""
    normalized = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]
        if key.startswith("model."):
            key = key[len("model."):]
        normalized[key] = value
    return normalized


def load_model_checkpoint(
    model: torch.nn.Module, checkpoint_path: Path
) -> Dict[str, int]:
    """Load model weights from a checkpoint path.

    Supports checkpoints saved either as raw state dicts or as dictionaries
    containing a "model_state" or "model" entry.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get(
            "model_state", checkpoint.get("model", checkpoint)
        )
    else:
        state_dict = checkpoint
    state_dict = _normalize_state_dict(state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())
    return {
        "model_keys": len(model_keys),
        "checkpoint_keys": len(ckpt_keys),
        "matched_keys": len(model_keys & ckpt_keys),
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
    }
