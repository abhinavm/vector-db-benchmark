"""Benchmark harness: measures query latency and recall@k for a given
adapter against a synthetic vector set.

Works against any adapter implementing build()/query() (see adapters/base.py).
Only BruteForceAdapter is wired up end to end right now -- pgvector/Qdrant/
FAISS adapters are stubs pending Docker services (see README).
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from benchmark.adapters.base import VectorIndexAdapter
from benchmark.recall import brute_force_neighbors, recall_at_k


@dataclass
class BenchmarkResult:
    backend: str
    n_vectors: int
    dim: int
    k: int
    mean_latency_ms: float
    p95_latency_ms: float
    mean_recall: float


def run_benchmark(
    adapter: VectorIndexAdapter,
    vectors: np.ndarray,
    queries: np.ndarray,
    k: int = 10,
    backend_name: str = "unknown",
) -> BenchmarkResult:
    adapter.build(vectors)

    latencies = []
    recalls = []
    for q in queries:
        ground_truth = brute_force_neighbors(vectors, q, k)

        start = time.perf_counter()
        retrieved = adapter.query(q, k)
        latencies.append((time.perf_counter() - start) * 1000)

        recalls.append(recall_at_k(ground_truth, retrieved))

    latencies_arr = np.array(latencies)
    return BenchmarkResult(
        backend=backend_name,
        n_vectors=len(vectors),
        dim=vectors.shape[1],
        k=k,
        mean_latency_ms=round(float(latencies_arr.mean()), 4),
        p95_latency_ms=round(float(np.percentile(latencies_arr, 95)), 4),
        mean_recall=round(float(np.mean(recalls)), 4),
    )
