"""Evaluate extra raw particle-level signal hypotheses with a saved Deep Sets model."""
import argparse
import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_particle_deepsets import (
    ParticleDeepSetsAutoencoder, raw_dataset, split_indices, valid_class_indices,
    score_indices,
)
from src.models.particle_deepsets import PARTICLE_FEATURES
from src.evaluation.working_points import score_cut_weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw2", help="original 2-prong raw HDF5 used by the saved checkpoint")
    parser.add_argument("checkpoint", help="particle_deepsets.pt")
    parser.add_argument("--hypotheses", required=True, help="comma-separated hypothesis IDs")
    parser.add_argument("--external-dir", default="data/external_signals")
    parser.add_argument("--hypothesis-path", action="append", default=[], metavar="ID=PATH")
    parser.add_argument("--split-dir", default="data/splits")
    parser.add_argument("--output", default="reports/particle_hypotheses.json")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-background-events", type=int, default=20000,
                        help="seeded held-out background cap for faster comparison; default 20k")
    parser.add_argument("--max-events", type=int, help="seeded hypothesis event cap")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = checkpoint.get("model_config", {})
    model = ParticleDeepSetsAutoencoder(
        input_dim=config.get("input_dim", len(PARTICLE_FEATURES)),
        hidden_dim=config.get("hidden_dim", 64),
        latent_dim=config.get("latent_dim", 32),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    background = raw_dataset(args.raw2)
    split, _, split_source = split_indices(background, args.seed, args.split_dir)
    background_ids = split["test_background"]
    background_available = len(background_ids)
    if args.max_background_events is not None and background_available > args.max_background_events:
        background_ids = np.sort(np.random.default_rng(args.seed + 1).choice(
            background_ids, args.max_background_events, replace=False))
    _, bg_scores = score_indices(model, background, background_ids, args.batch_size, device)
    threshold = float(checkpoint["threshold"])
    tie_probability = float(checkpoint.get("tie_acceptance_probability", 1.0))

    paths = {key.strip(): os.path.join(args.external_dir, key.strip(), "events.h5")
             for key in args.hypotheses.split(",") if key.strip()}
    for item in args.hypothesis_path:
        if "=" not in item:
            parser.error(f"--hypothesis-path must be ID=PATH, got {item!r}")
        key, value = item.split("=", 1)
        paths[key] = value

    results = {
        "protocol": "frozen background-only Deep Sets model and validation-background threshold",
        "checkpoint": os.path.relpath(args.checkpoint),
        "split_source": split_source,
        "test_background_events_available": int(background_available),
        "test_background_events": int(len(bg_scores)),
        "test_background_false_positive_fraction": float(score_cut_weights(bg_scores, threshold, tie_probability).mean()),
        "threshold": threshold,
        "threshold_tie_acceptance_probability": tie_probability,
        "signal_hypotheses": {},
    }
    for hypothesis_id, path in paths.items():
        if not os.path.isfile(path):
            results["signal_hypotheses"][hypothesis_id] = {"status": "not_available", "path": path}
            continue
        dataset = raw_dataset(path)
        classes = valid_class_indices(dataset)
        ids = classes[1] if len(classes[1]) else np.concatenate(list(classes.values()))
        available = len(ids)
        if args.max_events is not None and len(ids) > args.max_events:
            ids = np.sort(np.random.default_rng(args.seed).choice(ids, args.max_events, replace=False))
        _, scores = score_indices(model, dataset, ids, args.batch_size, device)
        labels = np.concatenate((np.zeros(len(bg_scores)), np.ones(len(scores))))
        mixed_scores = np.concatenate((bg_scores, scores))
        results["signal_hypotheses"][hypothesis_id] = {
            "status": "evaluated", "source": os.path.relpath(path),
            "events_scored": int(len(scores)), "events_available": int(available),
            "signal_efficiency_at_threshold": float(score_cut_weights(scores, threshold, tie_probability).mean()),
            "roc_auc_vs_test_background": float(roc_auc_score(labels, mixed_scores)) if len(scores) else None,
        }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as output:
        json.dump(results, output, indent=2)
    print(json.dumps({"output": args.output, "signal_hypotheses": results["signal_hypotheses"]}, indent=2))


if __name__ == "__main__":
    main()
