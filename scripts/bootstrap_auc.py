"""Report finite-test AUC uncertainty without retraining or selecting models."""

import argparse
import itertools
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate import load_eval_arrays, load_model, score  # noqa: E402
from src.evaluation.auc_bootstrap import paired_auc_bootstrap  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261003)
    args = parser.parse_args()
    torch.set_num_threads(1)
    scores = {}
    labels = None
    device = torch.device("cpu")
    for stem in (
        "m1_autoencoder",
        "m2_isolation_forest",
        "m3_deep_svdd",
        "m4_deep_sad",
        "m5_supervised",
        "m6_mass_aware",
    ):
        config = yaml.safe_load(Path(f"configs/{stem}.yaml").read_text())
        name = config["model"]["name"]
        data = config["data"]
        variant = f"f{float(data.get('f_train', 0)):g}_k{data.get('k_labels', 0)}"
        with Path(f"models/scaler_{name}_{variant}_42.pkl").open("rb") as handle:
            scaler = pickle.load(handle)
        xb, xs, _, current_labels = load_eval_arrays(
            "data/splits",
            "data/processed/events_v2_features.h5",
            config,
            scaler,
            "test",
        )
        if labels is not None:
            np.testing.assert_array_equal(labels, current_labels)
        labels = current_labels
        model = load_model(name, variant, 42, xb.shape[1], config, device)
        scores[name] = np.r_[
            score(model, xb, name, device), score(model, xs, name, device)
        ]
    draws = paired_auc_bootstrap(labels, scores, args.trials, args.seed)
    output = {
        "method": "stratified paired event bootstrap; percentile 95% intervals",
        "scope": "frozen seed-42 headline detectors, 2-prong test AUC only",
        "trials": args.trials,
        "bootstrap_seed": args.seed,
        "n_background": int((labels == 0).sum()),
        "n_signal": int((labels == 1).sum()),
        "limitations": "Conditional on these trained models; separate from seed variance. No intervals for other metrics or global discovery claims.",
        "models": {
            name: {
                "auc": float(roc_auc_score(labels, scores[name])),
                "ci95": np.quantile(value, [0.025, 0.975]).tolist(),
            }
            for name, value in draws.items()
        },
        "paired_differences": {
            f"{a} minus {b}": {
                "difference": float(
                    roc_auc_score(labels, scores[a]) - roc_auc_score(labels, scores[b])
                ),
                "ci95": np.quantile(draws[a] - draws[b], [0.025, 0.975]).tolist(),
            }
            for a, b in itertools.combinations(draws, 2)
        },
    }
    target = Path("reports/tables/auc_bootstrap.json")
    target.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.trials} paired bootstrap trials to {target}")


if __name__ == "__main__":
    main()
