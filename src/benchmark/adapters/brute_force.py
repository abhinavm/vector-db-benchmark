"""Reference adapter: exact brute-force search.

Used two ways: as the recall ground truth (recall.py calls the same
function directly), and as a working default backend so the benchmark
harness runs end to end without any external vector database running.
"""
from __future__ import annotations

import numpy as np

from benchmark.recall import brute_force_neighbors


class BruteForceAdapter:
    def __init__(self) -> None:
        self._vectors: np.ndarray | None = None

    def build(self, vectors: np.ndarray) -> None:
        self._vectors = vectors

    def query(self, vector: np.ndarray, k: int) -> np.ndarray:
        if self._vectors is None:
            raise RuntimeError("call build() before query()")
        return brute_force_neighbors(self._vectors, vector, k)
