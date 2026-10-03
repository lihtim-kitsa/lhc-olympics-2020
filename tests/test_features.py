import fastjet
import numpy as np
import pytest

from src.features.clustering import compute_n_subjettiness, extract_event_features


def test_subjettiness_single_particle_limit_and_two_particle_geometry():
    particles = [fastjet.PseudoJet(100., 0., 0., 100.),
                 fastjet.PseudoJet(100 * np.cos(.2), 100 * np.sin(.2), 0., 100.)]
    assert compute_n_subjettiness(particles[:1], 1) == 0
    assert compute_n_subjettiness(particles, 2) == 0
    assert compute_n_subjettiness(particles, 1) == pytest.approx(.1)


def test_dijet_mass_and_angular_features():
    row = np.zeros((700, 3))
    row[:4] = [[100, 0, -.1], [100, 0, .1],
               [100, 0, np.pi - .1], [100, 0, -np.pi + .1]]
    result = extract_event_features(row.ravel())
    assert result['mJJ'] == pytest.approx(400)
    assert result['mJ1'] == pytest.approx(200 * np.sin(.1))
    assert result['dmJ'] == pytest.approx(0, abs=1e-10)
    assert result['dRJJ'] == pytest.approx(np.pi)
    assert result['tau21_J1'] == pytest.approx(0)
    assert result['tau21_J2'] == pytest.approx(0)


@pytest.mark.parametrize('row', [
    np.zeros(2100), np.zeros(3), np.full(2100, np.nan)])
def test_invalid_events_are_rejected(row):
    assert extract_event_features(row) is None
