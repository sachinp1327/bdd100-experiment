"""Simple clustering helpers for evaluation diagnostics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class ClusterResult:
    """Result of a clustering run."""

    labels: np.ndarray
    centroids: np.ndarray


def standardize(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize features to zero mean, unit variance."""
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std == 0] = 1.0
    normalized = (features - mean) / std
    return normalized, mean, std


def kmeans(
    features: np.ndarray,
    k: int,
    max_iter: int = 50,
    seed: int = 42,
) -> ClusterResult:
    """Cluster features with a simple k-means implementation."""
    if features.shape[0] == 0:
        return ClusterResult(labels=np.array([]), centroids=np.zeros((0, 0)))

    rng = np.random.default_rng(seed)
    indices = rng.choice(features.shape[0], size=min(k, features.shape[0]), replace=False)
    centroids = features[indices]
    labels = np.zeros(features.shape[0], dtype=np.int64)

    for _ in range(max_iter):
        distances = np.linalg.norm(
            features[:, None, :] - centroids[None, :, :], axis=2
        )
        new_labels = distances.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for idx in range(centroids.shape[0]):
            members = features[labels == idx]
            if members.size:
                centroids[idx] = members.mean(axis=0)
    return ClusterResult(labels=labels, centroids=centroids)


def pca_2d(features: np.ndarray) -> np.ndarray:
    """Project features to 2D using PCA via SVD."""
    if features.shape[0] == 0:
        return np.zeros((0, 2))
    centered = features - features.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2].T
    return centered @ components
