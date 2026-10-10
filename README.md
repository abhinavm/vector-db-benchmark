# vector-db-benchmark

A benchmarking harness for vector search backends (pgvector, Qdrant, FAISS) that measures query latency and recall@k, with an exact brute-force reference implementation used both as the recall ground truth and as a working default backend.

## Why this exists

Every vector database claims to be fast. The interesting engineering question is the latency/recall tradeoff each one makes under its approximate-search index — and whether that tradeoff holds up under concurrent load, not just a single-threaded loop. This project measures it directly instead of taking vendor benchmarks at face value.

## Architecture

```
synthetic vectors + queries (numpy, seeded)
          │
          ▼
   [adapter.build()]      load vectors into the backend under test
          │
          ▼
   [adapter.query()]      top-k search, timed per query
          │
          ▼
  [recall.py]             compare against brute-force ground truth
          │
          ▼
  [harness.py]             BenchmarkResult: mean/p95 latency, mean recall@k
```

Every backend implements the same two-method interface (`adapters/base.py`), so `harness.py` benchmarks any of them identically.

## Status

| Component | Status |
|---|---|
| `vectors.py` | implemented — synthetic seeded embeddings, unit tested |
| `recall.py` | implemented — brute-force ground truth + recall@k, unit tested |
| `adapters/brute_force.py` | implemented — reference backend |
| `harness.py` | implemented — latency + recall measurement, tested end-to-end against brute force |
| `adapters/pgvector_adapter.py` | stub — needs a running Postgres+pgvector instance (docker-compose) |
| `adapters/qdrant_adapter.py` | stub — needs a running Qdrant instance |
| `adapters/faiss_adapter.py` | stub — in-process, no service needed, just not written yet |
| Concurrent load testing (Locust) | not started (stretch goal) |
| Results dashboard / report | not started (stretch goal) |

The harness and recall measurement are real and tested today — what's missing is the actual backends to compare. That's the next step: get pgvector and Qdrant running in docker-compose and wire in two adapters, since a benchmark with one backend isn't a benchmark yet.

## Running

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest
python -c "
from benchmark.adapters.brute_force import BruteForceAdapter
from benchmark.harness import run_benchmark
from benchmark.vectors import generate_vectors, generate_queries

vectors = generate_vectors(10_000, 128)
queries = generate_queries(100, 128)
print(run_benchmark(BruteForceAdapter(), vectors, queries, k=10, backend_name='brute_force'))
"
```

## Benchmark results

Not yet meaningful with only one backend (brute force is always recall@k = 1.0 by construction — it's the ground truth). Real numbers go here once pgvector/Qdrant adapters exist:

| Backend | n_vectors | dim | mean latency (ms) | p95 latency (ms) | mean recall@10 |
|---|---|---|---|---|---|
| brute_force | — | — | — | — | 1.0 (by definition) |
| pgvector | — | — | — | — | — |
| qdrant | — | — | — | — | — |
