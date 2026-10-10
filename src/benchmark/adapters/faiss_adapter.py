"""FAISS adapter -- stub.

TODO: build a FAISS index (start with IndexHNSWFlat for a fair comparison
against pgvector/Qdrant's HNSW-based indexes) and implement query() via
index.search(). Needs the faiss-cpu package.
"""
from __future__ import annotations

import numpy as np


class FaissAdapter:
    def build(self, vectors: np.ndarray) -> None:
        raise NotImplementedError("FAISS adapter not yet implemented")

    def query(self, vector: np.ndarray, k: int) -> np.ndarray:
        raise NotImplementedError
