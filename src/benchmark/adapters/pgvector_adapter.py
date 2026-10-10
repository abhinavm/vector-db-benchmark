"""pgvector adapter -- stub.

TODO: connect to a Postgres instance with the pgvector extension (docker-
compose service), create a table with a `vector` column, build an index
(HNSW or IVFFlat), and implement query() as an ORDER BY <-> ... LIMIT k
query. Needs psycopg2/psycopg + the pgvector extension installed.
"""
from __future__ import annotations

import numpy as np


class PgVectorAdapter:
    def build(self, vectors: np.ndarray) -> None:
        raise NotImplementedError("pgvector adapter not yet implemented -- needs a running Postgres+pgvector instance")

    def query(self, vector: np.ndarray, k: int) -> np.ndarray:
        raise NotImplementedError
