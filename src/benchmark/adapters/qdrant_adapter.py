"""Qdrant adapter -- stub.

TODO: connect to a local Qdrant instance (docker-compose service), create
a collection, upsert vectors, and implement query() via the search API.
Needs the qdrant-client package.
"""
from __future__ import annotations

import numpy as np


class QdrantAdapter:
    def build(self, vectors: np.ndarray) -> None:
        raise NotImplementedError("Qdrant adapter not yet implemented -- needs a running Qdrant instance")

    def query(self, vector: np.ndarray, k: int) -> np.ndarray:
        raise NotImplementedError
