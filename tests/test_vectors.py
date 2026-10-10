import numpy as np

from benchmark.vectors import generate_queries, generate_vectors


def test_shape_and_normalization():
    vecs = generate_vectors(100, 32, seed=0)
    assert vecs.shape == (100, 32)
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_same_seed_is_reproducible():
    a = generate_vectors(50, 16, seed=42)
    b = generate_vectors(50, 16, seed=42)
    assert np.array_equal(a, b)


def test_vectors_and_queries_differ_by_default_seed():
    vecs = generate_vectors(50, 16)
    queries = generate_queries(50, 16)
    assert not np.array_equal(vecs, queries)
