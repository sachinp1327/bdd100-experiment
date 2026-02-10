"""Streaming statistics and utility aggregators."""
from __future__ import annotations

from dataclasses import dataclass, field
import bisect
import heapq
from typing import Any, Dict, List, Tuple


@dataclass
class OnlineStats:
    """Track mean/variance/min/max in one pass."""

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    min_value: float | None = None
    max_value: float | None = None

    def add(self, value: float) -> None:
        self.count += 1
        if self.min_value is None or value < self.min_value:
            self.min_value = value
        if self.max_value is None or value > self.max_value:
            self.max_value = value
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2

    @property
    def variance(self) -> float:
        if self.count < 2:
            return 0.0
        return self.m2 / (self.count - 1)

    @property
    def std(self) -> float:
        return self.variance ** 0.5

    def to_dict(self) -> Dict[str, float]:
        return {
            "count": self.count,
            "mean": self.mean,
            "std": self.std,
            "min": self.min_value if self.min_value is not None else 0.0,
            "max": self.max_value if self.max_value is not None else 0.0,
        }


@dataclass
class Histogram:
    """Fixed-bin histogram with overflow handling."""

    bins: List[float]
    counts: List[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.bins) < 2:
            raise ValueError("Histogram requires at least two bin edges")
        if not self.counts:
            self.counts = [0 for _ in range(len(self.bins) - 1)]

    def add(self, value: float) -> None:
        idx = bisect.bisect_right(self.bins, value) - 1
        if idx < 0:
            idx = 0
        if idx >= len(self.counts):
            idx = len(self.counts) - 1
        self.counts[idx] += 1

    def to_dict(self) -> Dict[str, List[float]]:
        return {"bins": self.bins, "counts": self.counts}


class TopK:
    """Keep top-k elements by value."""

    def __init__(self, k: int, largest: bool = True) -> None:
        self.k = k
        self.largest = largest
        self._heap: List[Tuple[float, int, Dict[str, Any]]] = []
        self._counter = 0

    def add(self, value: float, payload: Dict[str, Any]) -> None:
        key = value if self.largest else -value
        self._counter += 1
        entry = (key, self._counter, payload)
        if len(self._heap) < self.k:
            heapq.heappush(self._heap, entry)
            return
        if key > self._heap[0][0]:
            heapq.heapreplace(self._heap, entry)

    def to_list(self) -> List[Dict[str, Any]]:
        sorted_items = sorted(self._heap, key=lambda x: x[0], reverse=True)
        results = []
        for key, _counter, payload in sorted_items:
            value = key if self.largest else -key
            entry = dict(payload)
            entry["value"] = value
            results.append(entry)
        return results
