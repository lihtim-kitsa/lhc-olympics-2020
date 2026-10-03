import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src.evaluation.auc_bootstrap import (
    paired_auc_bootstrap,
    prepare_ranking,
    weighted_auc,
)


def test_weighted_ties_match_sklearn():
    labels = np.array([0, 1, 0, 1, 0, 1])
    scores = np.array([0.2, 0.2, 0.5, 0.5, 0.8, 0.9])
    weights = np.array([2, 1, 4, 3, 1, 2], dtype=float)
    assert weighted_auc(prepare_ranking(scores), labels, weights) == pytest.approx(
        roc_auc_score(labels, scores, sample_weight=weights)
    )


def test_paired_identical_and_perfect_rankings():
    labels = np.array([0] * 10 + [1] * 10)
    scores = labels.astype(float)
    draws = paired_auc_bootstrap(labels, {"a": scores, "b": scores}, 20, 42)
    np.testing.assert_array_equal(draws["a"], np.ones(20))
    np.testing.assert_array_equal(draws["a"] - draws["b"], np.zeros(20))
    constant = paired_auc_bootstrap(labels, {"a": np.ones(20)}, 20, 42)
    np.testing.assert_array_equal(constant["a"], np.full(20, 0.5))


def test_invalid_bootstrap_input():
    with pytest.raises(ValueError, match="two classes"):
        paired_auc_bootstrap(np.zeros(4), {"a": np.arange(4)}, 10, 42)
    with pytest.raises(ValueError, match="align"):
        paired_auc_bootstrap(np.array([0, 1]), {"a": np.arange(4)}, 10, 42)
