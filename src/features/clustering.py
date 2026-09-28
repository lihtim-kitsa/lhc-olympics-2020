"""Low-level LHCO event feature extraction using FastJet."""

import numpy as np
import fastjet


def compute_n_subjettiness(constituents, n, beta=1.0, radius=1.0):
    """Compute normalized tau_N using exclusive-kT axes and angular exponent beta.

    This is an explicitly documented exclusive-kT-axis definition; it is not
    the FastJet-contrib one-pass/minimization implementation used by Zenodo's
    supplied feature files. Those are suitable as a cross-check, not exact
    equality tests for these tau values.
    """
    if n < 1 or not constituents:
        return np.nan
    sum_pt = sum(p.pt() for p in constituents)
    denom = sum_pt * radius**beta
    if not np.isfinite(denom) or denom <= 0:
        return np.nan
    if len(constituents) <= n:
        return 0.0
    seq = fastjet.ClusterSequence(
        list(constituents), fastjet.JetDefinition(fastjet.kt_algorithm, radius)
    )
    axes = seq.exclusive_jets(n)
    if len(axes) < n:
        return np.nan
    numerator = 0.0
    for p in constituents:
        dr = min(p.delta_R(axis) for axis in axes)
        numerator += p.pt() * dr**beta
    return float(numerator / denom)


def _tau21(jet):
    consts = list(jet.constituents())
    tau1 = compute_n_subjettiness(consts, 1)
    tau2 = compute_n_subjettiness(consts, 2)
    if not np.isfinite(tau1) or tau1 <= 0 or not np.isfinite(tau2):
        return np.nan
    return tau2 / tau1


def extract_event_features(event_row):
    """Return five detector inputs and mJJ from one 2100-value pt/eta/phi row."""
    row = np.asarray(event_row, dtype=np.float64)
    if row.size != 2100 or not np.isfinite(row).all():
        return None
    particles = row.reshape(700, 3)
    mask = (particles[:, 0] > 0) & np.isfinite(particles).all(axis=1)
    particles = particles[mask]
    if len(particles) < 2:
        return None

    inputs = []
    for pt, eta, phi in particles:
        px, py, pz = pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)
        energy = np.sqrt(px * px + py * py + pz * pz)
        inputs.append(fastjet.PseudoJet(float(px), float(py), float(pz), float(energy)))
    jet_def = fastjet.JetDefinition(fastjet.antikt_algorithm, 1.0)
    cluster = fastjet.ClusterSequence(inputs, jet_def)
    jets = sorted(cluster.inclusive_jets(), key=lambda jet: jet.pt(), reverse=True)
    if len(jets) < 2:
        return None
    j1, j2 = jets[:2]
    tau21_1, tau21_2 = _tau21(j1), _tau21(j2)
    dijet = j1 + j2
    deta = j1.eta() - j2.eta()
    dphi = (j1.phi() - j2.phi() + np.pi) % (2 * np.pi) - np.pi
    vals = np.asarray([
        j1.m(), abs(j1.m() - j2.m()), tau21_1, tau21_2,
        np.hypot(deta, dphi), dijet.m()
    ], dtype=np.float64)
    if not np.isfinite(vals).all():
        return None
    return dict(zip(('mJ1', 'dmJ', 'tau21_J1', 'tau21_J2', 'dRJJ', 'mJJ'), vals))
