"""Adapter interface every vector store backend implements, so harness.py
can benchmark any of them identically."""
from __future__ import annotations

from typing import Protocol

import numpy as np


class VectorIndexAdapter(Protocol):
    def build(self, vectors: np.ndarray) -> None:
        """Load vectors into the backend, building whatever index it uses."""
        ...

    def query(self, vector: np.ndarray, k: int) -> np.ndarray:
        """Return the indices of the top-k nearest vectors."""
        ...
