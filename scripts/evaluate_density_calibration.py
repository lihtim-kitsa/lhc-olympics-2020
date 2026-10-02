"""Run held-out predictive diagnostics for ANODE/CATHODE density proxies."""
import argparse
import json
import os
import pickle
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.evaluate import load_eval_arrays, load_model
from src.evaluation.density_calibration import (
    classifier_calibration, conformal_density_score_calibration,
    gmm_heldout_nll, gmm_pit_summary,
)


def run(config, output="reports/density_calibration", seed=2026):
    name = config["model"]["name"]
    if name not in ("M8_ANODE", "M11_CATHODE_CVAE"):
        raise ValueError("Only M8_ANODE and M11_CATHODE_CVAE expose density-score proxies")
    data = config["data"]
    variant = f"f{float(data.get('f_train', 0)):g}_k{int(data.get('k_labels', 0))}"
    model_seed = int(config["training"]["seed"])
    with open(os.path.join("models", f"scaler_{name}_{variant}_{model_seed}.pkl"), "rb") as f:
        scaler = pickle.load(f)
    Xb, _, masses, _ = load_eval_arrays(
        "data/splits", "data/processed/events_v2_features.h5", config, scaler, "test")
    n_bkg = len(Xb)
    background_mass = np.asarray(masses[:n_bkg], dtype=float)
    sr = (background_mass >= 3300) & (background_mass <= 3700)
    sideband = ((background_mass >= 2500) & (background_mass < 3300)) | (
        (background_mass > 3700) & (background_mass <= 4500))
    if not np.any(sr) or not np.any(sideband):
        raise ValueError("Held-out background is missing the configured signal window or sidebands")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(name, variant, model_seed, Xb.shape[1], config, device)
    if name == "M8_ANODE":
        heldout = Xb[sideband, :5]
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(heldout))
        n_cal = len(order) // 2
        calibration = heldout[order[:n_cal]]
        evaluation = heldout[order[n_cal:]]
        calibration_nll = -model.bkg_gmm.score_samples(calibration)
        evaluation_nll = -model.bkg_gmm.score_samples(evaluation)
        result = gmm_heldout_nll(model.bkg_gmm, evaluation)
        result["conformal_score_calibration"] = conformal_density_score_calibration(
            calibration_nll, evaluation_nll)
        result["rosenblatt_pit"] = gmm_pit_summary(model.bkg_gmm, evaluation)
        result["heldout_region"] = "independent background test events in the 2.5-3.3 and 3.7-4.5 TeV sidebands"
        result["calibration_split"] = "independent deterministic half-split of held-out sideband events; one half calibrates the NLL tail, the other evaluates it"
        result["model_semantics"] = "held-out calibration diagnostics for sideband GMM density; not conditional SR background density"
    else:
        torch.manual_seed(seed)
        model.cvae.to(device).eval()
        model.clf.to(device).eval()
        x_real = torch.as_tensor(Xb[sr, :5], dtype=torch.float32, device=device)
        mass_col = 7 if data.get("use_extended", False) else 5
        # CATHODE's conditional input is the scaled mJJ column from the model input.
        if Xb.shape[1] <= mass_col:
            raise ValueError("CATHODE calibration requires scaled mJJ among model inputs")
        c = torch.as_tensor(Xb[sr, mass_col:mass_col + 1], dtype=torch.float32, device=device)
        with torch.no_grad():
            z = torch.randn(len(x_real), 4, device=device)
            x_generated = model.cvae.decode(z, c)
            p_real = torch.sigmoid(model.clf(x_real)).squeeze(-1).cpu().numpy()
            p_generated = torch.sigmoid(model.clf(x_generated)).squeeze(-1).cpu().numpy()
        probabilities = np.r_[p_real, p_generated]
        labels = np.r_[np.ones(len(p_real), dtype=int), np.zeros(len(p_generated), dtype=int)]
        result = classifier_calibration(labels, probabilities, n_real_train=1, n_generated_train=1)
        result["heldout_region"] = "background test events in 3.3-3.7 TeV window versus independent CVAE draws at their masses"
        result["model_semantics"] = "real-vs-generated classifier calibration; not absolute or calibrated background density"

    result.update({"model": name, "seed": model_seed, "pseudo_random_seed": seed,
                   "test_background_events_in_window": int(sr.sum()),
                   "test_background_events_for_density_diagnostic": int(sideband.sum() // 2 if name == "M8_ANODE" else sr.sum()),
                   "density_calibration_claim": False})
    os.makedirs(output, exist_ok=True)
    path = os.path.join(output, f"{name}_{variant}_{model_seed}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="reports/density_calibration")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    print(run(cfg, args.output, args.seed))
