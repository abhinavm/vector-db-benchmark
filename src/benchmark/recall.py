"""Exact (brute-force) nearest-neighbor search, used as the recall ground
truth that every backend's approximate results get compared against.
"""
from __future__ import annotations

import numpy as np


def brute_force_neighbors(vectors: np.ndarray, query: np.ndarray, k: int) -> np.ndarray:
    """Exact top-k neighbor indices for one query, by cosine similarity."""
    norms = np.linalg.norm(vectors, axis=1) * np.linalg.norm(query) + 1e-12
    sims = vectors @ query / norms
    return np.argsort(-sims)[:k]


def recall_at_k(ground_truth: np.ndarray, retrieved: np.ndarray) -> float:
    """Fraction of the ground-truth top-k ids present in the retrieved ids."""
    if len(ground_truth) == 0:
        return 1.0
    gt_set = set(ground_truth.tolist())
    ret_set = set(retrieved.tolist())
    return len(gt_set & ret_set) / len(gt_set)
