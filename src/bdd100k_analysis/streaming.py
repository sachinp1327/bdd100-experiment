"""Streaming IO helpers for large BDD100K label files."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import ijson


def iter_label_items(labels_path: str | Path) -> Iterable[Dict]:
    """Yield label items from a BDD100K labels JSON file."""
    labels_path = Path(labels_path)
    with labels_path.open("rb") as handle:
        for item in ijson.items(handle, "item"):
            yield item
