import numpy as np

from benchmark.recall import brute_force_neighbors, recall_at_k


def test_brute_force_finds_exact_match_first():
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]], dtype=np.float32)
    query = np.array([1.0, 0.0], dtype=np.float32)
    neighbors = brute_force_neighbors(vectors, query, k=1)
    assert neighbors[0] == 0


def test_recall_full_overlap_is_one():
    gt = np.array([1, 2, 3])
    retrieved = np.array([3, 2, 1])
    assert recall_at_k(gt, retrieved) == 1.0


def test_recall_no_overlap_is_zero():
    gt = np.array([1, 2, 3])
    retrieved = np.array([4, 5, 6])
    assert recall_at_k(gt, retrieved) == 0.0


def test_recall_partial_overlap():
    gt = np.array([1, 2, 3, 4])
    retrieved = np.array([1, 2, 9, 9])
    assert recall_at_k(gt, retrieved) == 0.5
