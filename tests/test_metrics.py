"""
Unit tests for src/metrics.py. Worth having precisely because of how this
file's bug history went: we initially had a `precision_at_k` that did
`if not top_k` on a list slice and it worked fine in isolation but raised
"truth value of an array is ambiguous" once a numpy array got passed
through instead of a plain list (see experiments/exp_log.md context around
the metrics fix). These tests pin down the exact contract so a regression
like that fails loudly instead of silently shipping in submission.csv.
"""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.metrics import average_precision, dcg_at_k, ndcg_at_k, precision_at_k


def test_dcg_at_k_basic():
    # rel=[3,2,1], k=3 -> 3/log2(2) + 2/log2(3) + 1/log2(4) = 3 + 1.2619 + 0.5
    expected = 3 / math.log2(2) + 2 / math.log2(3) + 1 / math.log2(4)
    assert math.isclose(dcg_at_k([3, 2, 1], 3), expected)


def test_dcg_at_k_truncates_to_k():
    assert dcg_at_k([3, 2, 1, 4, 4], 2) == dcg_at_k([3, 2], 2)


def test_ndcg_at_k_perfect_order_is_one():
    assert math.isclose(ndcg_at_k([4, 3, 2, 1, 0], 5), 1.0)


def test_ndcg_at_k_reversed_order_is_less_than_one():
    assert ndcg_at_k([0, 1, 2, 3, 4], 5) < 1.0


def test_ndcg_at_k_all_zero_relevance_returns_zero_not_nan():
    assert ndcg_at_k([0, 0, 0], 3) == 0.0


def test_precision_at_k_basic():
    assert precision_at_k([4, 3, 0, 0, 2], k=5, relevance_threshold=3) == 0.4


def test_precision_at_k_empty_list_returns_zero():
    assert precision_at_k([], k=10) == 0.0


def test_precision_at_k_accepts_numpy_array():
    # this is the exact input shape that broke the original implementation
    arr = np.array([4, 3, 0, 0, 2])
    assert precision_at_k(arr, k=5, relevance_threshold=3) == 0.4


def test_average_precision_basic():
    # relevant at rank 1 and rank 3: precisions = [1/1, 2/3] -> mean = 0.8333
    ap = average_precision([4, 0, 3, 0], relevance_threshold=3)
    assert math.isclose(ap, (1 / 1 + 2 / 3) / 2)


def test_average_precision_no_relevant_items_returns_zero():
    assert average_precision([0, 0, 0], relevance_threshold=3) == 0.0


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
