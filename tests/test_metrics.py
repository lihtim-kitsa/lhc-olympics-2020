import numpy as np
import pytest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.evaluation.metrics import get_core_metrics, calculate_js_divergence

def test_get_core_metrics():
    # Perfect classifier
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    
    metrics = get_core_metrics(y_true, y_score)
    assert np.isclose(metrics['roc_auc'], 1.0)
    # At eS=1.0, eB is 0 (or smallest possible). Rejection is inf.
    assert np.isinf(metrics['rej_10']) or metrics['rej_10'] > 0
    
def test_js_divergence():
    h1 = np.array([0.5, 0.5])
    h2 = np.array([0.5, 0.5])
    assert np.isclose(calculate_js_divergence(h1, h2), 0.0)
    
    h3 = np.array([1.0, 0.0])
    h4 = np.array([0.0, 1.0])
    # Max JS divergence with log base e is ln(2)
    # Scipy jensenshannon uses base e by default
    assert calculate_js_divergence(h3, h4) > 0.5
