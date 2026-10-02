"""Held-out diagnostics and score calibration for ANODE/CATHODE density proxies.

These routines report predictive calibration of a real-vs-generated classifier
and held-out log density under a fitted GMM. Split-conformal transforms can
calibrate anomaly-score p-values for an exchangeable background population;
they do not establish that either method recovers an absolute background density.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.linear_model import LogisticRegression
from scipy.special import logsumexp, ndtr
from scipy.stats import kstest


def _finite_1d(values, name):
    values = np.asarray(values, dtype=float).reshape(-1)
    if not len(values) or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be a non-empty finite vector")
    return values


def gmm_heldout_nll(gmm, X, *, n_features=5):
    """Summarize held-out negative log likelihood for a fitted sklearn GMM.

    X must use exactly the preprocessing and feature order used for fitting.
    NLL is in nats per event and is comparative only for the same data space.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] < n_features:
        raise ValueError(f"X must have shape (n, >= {n_features}) with n > 0")
    log_density = np.asarray(gmm.score_samples(X[:, :n_features]), dtype=float)
    if not np.all(np.isfinite(log_density)):
        raise ValueError("GMM returned non-finite held-out log densities")
    nll = -log_density
    return {
        "diagnostic": "heldout_gmm_nll",
        "n": int(len(nll)),
        "mean_nll_nats": float(np.mean(nll)),
        "std_nll_nats": float(np.std(nll)),
        "median_nll_nats": float(np.median(nll)),
        "q90_nll_nats": float(np.quantile(nll, 0.90)),
        "q99_nll_nats": float(np.quantile(nll, 0.99)),
    }


def gmm_rosenblatt_pit(gmm, X, *, n_features=5):
    """Compute ordered Rosenblatt PIT values for a fitted full-covariance GMM.

    Uniform marginal PITs on an independent sample are a calibration diagnostic
    for the fitted joint density in the same feature space. The result depends
    on feature order and does not validate extrapolation into a different region.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] < n_features:
        raise ValueError(f"X must have shape (n, >= {n_features}) with n > 0")
    if not np.all(np.isfinite(X[:, :n_features])):
        raise ValueError("X contains non-finite values")
    if getattr(gmm, "covariance_type", None) != "full":
        raise ValueError("Rosenblatt PIT currently requires a full-covariance GaussianMixture")
    values = X[:, :n_features]
    n, d = values.shape
    k_count = len(gmm.weights_)
    pit = np.empty((n, d), dtype=float)
    for j in range(d):
        conditional_means = np.empty((n, k_count), dtype=float)
        conditional_vars = np.empty(k_count, dtype=float)
        log_component_weights = np.tile(np.log(gmm.weights_), (n, 1))
        for k in range(k_count):
            mu = gmm.means_[k]
            cov = gmm.covariances_[k]
            if j == 0:
                conditional_means[:, k] = mu[j]
                conditional_vars[k] = cov[j, j]
            else:
                cov_pp = cov[:j, :j]
                cov_jp = cov[j, :j]
                inv_pp = np.linalg.pinv(cov_pp, rcond=1e-12)
                delta = values[:, :j] - mu[:j]
                sign, logdet = np.linalg.slogdet(cov_pp)
                if sign <= 0:
                    raise ValueError("GMM covariance has non-positive determinant")
                quad = np.einsum("ni,ij,nj->n", delta, inv_pp, delta)
                log_component_weights[:, k] += -.5 * (j * np.log(2 * np.pi) + logdet + quad)
                conditional_means[:, k] = mu[j] + delta @ (inv_pp @ cov_jp)
                conditional_vars[k] = cov[j, j] - cov_jp @ inv_pp @ cov[j, :j]
        posterior = np.exp(log_component_weights - logsumexp(log_component_weights, axis=1, keepdims=True))
        z = (values[:, j, None] - conditional_means) / np.sqrt(np.maximum(conditional_vars, 1e-12))[None, :]
        pit[:, j] = np.sum(posterior * ndtr(z), axis=1)
    return pit


def gmm_pit_summary(gmm, X, *, n_features=5, n_bins=10):
    """Summarize marginal uniformity of held-out Rosenblatt PIT coordinates."""
    pit = gmm_rosenblatt_pit(gmm, X, n_features=n_features)
    edges = np.linspace(0, 1, n_bins + 1)
    result = []
    for j in range(pit.shape[1]):
        values = pit[:, j]
        counts, _ = np.histogram(values, bins=edges)
        stat = kstest(values, "uniform")
        result.append({"feature_index": j, "mean": float(values.mean()),
                       "std": float(values.std()), "ks_statistic": float(stat.statistic),
                       "ks_pvalue": float(stat.pvalue), "uniform_bin_counts": counts.tolist()})
    return {"diagnostic": "heldout_gmm_rosenblatt_pit", "n": int(len(pit)),
            "feature_order_sensitive": True, "marginals": result}


def conformal_density_score_calibration(calibration_nll, evaluation_nll,
                                        *, alphas=(0.01, 0.05, 0.10), n_bins=10):
    """Calibrate upper-tail GMM NLL scores using an independent background set.

    The resulting finite-sample conformal p-values are super-uniform under
    exchangeability. This calibrates the anomaly-score tail for the calibration
    population; it does not make the GMM an absolute or conditional SR density.
    """
    cal = _finite_1d(calibration_nll, "calibration_nll")
    test = _finite_1d(evaluation_nll, "evaluation_nll")
    ordered = np.sort(cal)
    first_ge = np.searchsorted(ordered, test, side="left")
    p_values = (len(ordered) - first_ge + 1) / (len(ordered) + 1)
    coverage = {}
    for alpha in alphas:
        alpha = float(alpha)
        if not 0 < alpha < 1:
            raise ValueError("alphas must lie in (0, 1)")
        coverage[f"{alpha:g}"] = {
            "n_flagged": int(np.sum(p_values <= alpha)),
            "observed_rate": float(np.mean(p_values <= alpha)),
            "nominal_rate": alpha,
        }
    counts, _ = np.histogram(p_values, bins=np.linspace(0, 1, n_bins + 1))
    ks = kstest(p_values, "uniform")
    return {
        "diagnostic": "split_conformal_upper_tail_density_score",
        "calibration_events": int(len(cal)),
        "evaluation_events": int(len(test)),
        "calibration_method": "finite-sample upper-tail rank p-value with ties counted conservatively",
        "p_value_mean": float(np.mean(p_values)),
        "p_value_quantiles": {str(q): float(np.quantile(p_values, q)) for q in (.01, .05, .1, .5, .9, .95, .99)},
        "uniform_ks_statistic": float(ks.statistic),
        "uniform_ks_pvalue": float(ks.pvalue),
        "uniform_bin_counts": counts.tolist(),
        "tail_coverage": coverage,
        "interpretation": "calibrated anomaly-score p-values for exchangeable held-out sideband background; not an absolute density or a signal-region conditional density",
    }


def classifier_calibration(y_real, probabilities, *, n_bins=10, n_real_train=None,
                           n_generated_train=None):
    """Evaluate held-out real-vs-generated classifier probabilities.

    Labels: 1=held-out real SR event, 0=independent generated pseudo-background.
    Classifier probabilities should be raw sigmoid outputs, before any calibration.
    The optional training counts correct the classifier odds for unequal training
    class priors when forming its implied real/generated density-ratio proxy.
    This is not a background-density calibration unless generated events represent
    the target background distribution and all sampling corrections are known.
    """
    y = _finite_1d(y_real, "y_real").astype(int)
    p = _finite_1d(probabilities, "probabilities")
    if len(y) != len(p) or not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
        raise ValueError("y_real and probabilities must align and contain both binary classes")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("probabilities must lie in [0, 1]")
    if n_bins < 1:
        raise ValueError("n_bins must be positive")

    eps = np.finfo(float).eps
    pclip = np.clip(p, eps, 1 - eps)
    # Quantile bins have comparable occupancy; duplicate edges are removed.
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    bins = []
    ece = 0.0
    intervals = [(float(edges[0]), float(edges[0]), np.ones(len(p), dtype=bool))] if len(edges) == 1 else [
        (lo, hi, (p >= lo) & ((p <= hi) if i == len(edges) - 2 else (p < hi)))
        for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:]))
    ]
    for lo, hi, mask in intervals:
        if not np.any(mask):
            continue
        confidence = float(np.mean(p[mask]))
        frequency = float(np.mean(y[mask]))
        ece += float(np.mean(mask)) * abs(confidence - frequency)
        bins.append({"n": int(mask.sum()), "mean_probability": confidence,
                     "observed_real_fraction": frequency, "lower_edge": float(lo),
                     "upper_edge": float(hi)})

    # Calibration slope/intercept from a logistic recalibration fit on the held-out
    # sample. These are descriptive; do not refit the classifier using this sample.
    logits = np.log(pclip / (1 - pclip)).reshape(-1, 1)
    recal = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(logits, y)
    result = {
        "diagnostic": "heldout_real_vs_generated_classifier_calibration",
        "n": int(len(y)),
        "n_real": int(y.sum()),
        "n_generated": int(len(y) - y.sum()),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, pclip, labels=[0, 1])),
        "ece_quantile_bins": float(ece),
        "calibration_intercept": float(recal.intercept_[0]),
        "calibration_slope": float(recal.coef_[0, 0]),
        "reliability_bins": bins,
        "interpretation": "classifier calibration for the held-out real/generated sampling mixture; not absolute background-density calibration",
    }
    if (n_real_train is None) != (n_generated_train is None):
        raise ValueError("provide both training class counts or neither")
    if n_real_train is not None:
        if n_real_train <= 0 or n_generated_train <= 0:
            raise ValueError("training class counts must be positive")
        # Equal-prior posterior odds equal the class-conditional density ratio.
        log_ratio = np.log(pclip / (1 - pclip)) + np.log(n_generated_train / n_real_train)
        result["implied_log_real_to_generated_density_ratio"] = {
            "mean": float(np.mean(log_ratio)),
            "median": float(np.median(log_ratio)),
            "q90": float(np.quantile(log_ratio, .90)),
        }
        result["density_ratio_caveat"] = "prior-corrected classifier proxy; generated distribution must match target background for a background likelihood ratio interpretation"
    return result
