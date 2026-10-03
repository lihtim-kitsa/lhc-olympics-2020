"""Stratified paired bootstrap of fixed detector scores, with exact score ties."""

import numpy as np


def prepare_ranking(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Scores must be a finite vector")
    order = np.argsort(values, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(values[order])) + 1]
    return order, starts


def weighted_auc(
    ranking: tuple[np.ndarray, np.ndarray],
    labels: np.ndarray,
    weights: np.ndarray,
) -> float:
    order, starts = ranking
    signal = np.add.reduceat((weights * labels)[order], starts)
    background = np.add.reduceat((weights * (1 - labels))[order], starts)
    lower_background = np.cumsum(background) - background
    return float(
        np.sum(signal * (lower_background + 0.5 * background))
        / (signal.sum() * background.sum())
    )


def paired_auc_bootstrap(
    labels: np.ndarray, scores: dict[str, np.ndarray], trials: int, seed: int
) -> dict[str, np.ndarray]:
    """Resample each class with replacement; share event weights across methods."""
    labels = np.asarray(labels, dtype=int)
    if labels.ndim != 1 or not np.isin(labels, [0, 1]).all():
        raise ValueError("Labels must be a binary vector")
    groups = [np.flatnonzero(labels == label) for label in (0, 1)]
    if trials < 2 or any(len(group) == 0 for group in groups):
        raise ValueError("Need two classes and at least two bootstrap trials")
    if any(np.asarray(value).shape != labels.shape for value in scores.values()):
        raise ValueError("Every score vector must align with labels")
    rankings = {name: prepare_ranking(value) for name, value in scores.items()}
    draws = {name: np.empty(trials) for name in scores}
    rng = np.random.default_rng(seed)
    for trial in range(trials):
        weights = np.zeros(len(labels), dtype=float)
        for group in groups:
            counts = rng.multinomial(len(group), np.full(len(group), 1 / len(group)))
            weights[group] = counts
        for name, ranking in rankings.items():
            draws[name][trial] = weighted_auc(ranking, labels, weights)
    return draws
