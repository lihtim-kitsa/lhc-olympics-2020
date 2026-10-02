"""Tie-aware fixed-efficiency score cuts.

When a model emits tied scores at the operating boundary, a scalar threshold
cannot attain the requested efficiency. These helpers report an acceptance
probability for boundary ties; downstream weighted yields are the expectation
under randomized tie-breaking.
"""
import numpy as np


def score_cut(reference_scores, efficiency):
    scores = np.asarray(reference_scores, dtype=float).reshape(-1)
    if not len(scores) or not np.all(np.isfinite(scores)):
        raise ValueError("reference_scores must be a non-empty finite vector")
    if not 0 < efficiency < 1:
        raise ValueError("efficiency must lie in (0, 1)")
    n_target = efficiency * len(scores)
    order = np.sort(scores)
    rank = max(0, len(order) - int(np.ceil(n_target)))
    threshold = float(order[rank])
    n_above = int(np.sum(scores > threshold))
    n_tied = int(np.sum(scores == threshold))
    tie_probability = float(np.clip((n_target - n_above) / n_tied, 0., 1.))
    return threshold, tie_probability


def score_cut_weights(scores, threshold, tie_probability):
    scores = np.asarray(scores, dtype=float)
    if not 0 <= tie_probability <= 1:
        raise ValueError("tie_probability must lie in [0, 1]")
    return np.where(scores > threshold, 1., np.where(scores == threshold, tie_probability, 0.))
