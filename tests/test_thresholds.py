import numpy as np
import pytest

from src.evaluation.working_points import score_cut, score_cut_weights


@pytest.mark.parametrize('scores', [
    np.ones(103), np.arange(103), np.repeat([0., 1., 2.], [70, 20, 13])])
@pytest.mark.parametrize('efficiency', [0.01, 0.1, 0.5])
def test_expected_validation_acceptance_including_ties(scores, efficiency):
    threshold, probability = score_cut(scores, efficiency)
    weights = score_cut_weights(scores, threshold, probability)
    assert weights.mean() == pytest.approx(efficiency)
    assert np.all((weights >= 0) & (weights <= 1))


def test_frozen_cut_applies_without_test_recalibration():
    cut = score_cut(np.arange(100), 0.1)
    np.testing.assert_array_equal(
        score_cut_weights([0, 90, 99, 100], *cut), [0, 1, 1, 1])


@pytest.mark.parametrize('scores,efficiency', [
    ([], .1), ([np.nan], .1), ([np.inf], .1), ([1], 0), ([1], 1)])
def test_invalid_calibration_inputs(scores, efficiency):
    with pytest.raises(ValueError):
        score_cut(scores, efficiency)
