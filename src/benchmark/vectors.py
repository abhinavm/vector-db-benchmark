"""Synthetic embedding generation for benchmarking, with reproducible seeds.

Real embeddings (e.g. from sentence-transformers) would make the recall
numbers reflect actual semantic structure; synthetic vectors are enough to
benchmark latency/throughput/recall@k mechanics, which is what this project
is actually about. Swapping in a real encoder later doesn't change anything
downstream of this module.
"""
from __future__ import annotations

import numpy as np


def generate_vectors(n: int, dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(n, dim)).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


def generate_queries(n: int, dim: int, seed: int = 1) -> np.ndarray:
    return generate_vectors(n, dim, seed=seed)
