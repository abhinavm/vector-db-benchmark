from benchmark.adapters.brute_force import BruteForceAdapter
from benchmark.harness import run_benchmark
from benchmark.vectors import generate_queries, generate_vectors


def test_brute_force_backend_has_perfect_recall():
    # Recall is measured against brute-force ground truth, and the
    # backend under test here IS brute force -- so recall must be exactly
    # 1.0. This is a check that the harness plumbing (not the algorithm)
    # is correct; a real recall/latency tradeoff only shows up once an
    # approximate backend (pgvector/Qdrant/FAISS) is wired in.
    vectors = generate_vectors(200, 32, seed=0)
    queries = generate_queries(20, 32, seed=1)

    result = run_benchmark(BruteForceAdapter(), vectors, queries, k=5, backend_name="brute_force")

    assert result.mean_recall == 1.0
    assert result.n_vectors == 200
    assert result.dim == 32
    assert result.mean_latency_ms >= 0
