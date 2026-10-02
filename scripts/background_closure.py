"""Background-only closure with a frozen detector and validation-selected cuts.

Example:
  python scripts/background_closure.py --config configs/m4_deep_sad.yaml --trials 500

Each pseudoexperiment is a Poisson(1) event bootstrap of the held-out test
background. The detector and score thresholds remain frozen in every trial.
"""
import argparse
import json
import os
import pickle
import sys

import h5py
import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.evaluate import load_model, score, threshold_at_efficiency
from src.evaluation.bump_hunt import BumpHunter
from src.evaluation.working_points import score_cut_weights


def _inputs(config, feature_file=None, split_dir=None):
    data = config['data']
    feature_file = feature_file or ('data/processed/events_v2_extended_features.h5'
                                   if data.get('use_extended') else 'data/processed/events_v2_features.h5')
    split_dir = split_dir or 'data/splits'
    variant = f"f{float(data.get('f_train', 0)):g}_k{int(data.get('k_labels', 0))}"
    seed = int(config['training']['seed'])
    name = config['model']['name']
    with open(os.path.join('models', f'scaler_{name}_{variant}_{seed}.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    cols = ([0, 1, 2, 3, 4, 5, 6, 7] if data.get('use_extended') else
            [0, 1, 2, 3, 4, 5]) if data.get('use_mjj') else (
            [0, 1, 2, 3, 4, 5, 6] if data.get('use_extended') else [0, 1, 2, 3, 4])
    mass_col = 7 if data.get('use_extended') else 5
    with h5py.File(feature_file, 'r') as f:
        matrix = f['features'][:]
    indices = {s: np.load(os.path.join(split_dir, f'background_{s}.npy')) for s in ('val', 'test')}
    X = {s: scaler.transform(matrix[idx][:, cols]).astype(np.float32) for s, idx in indices.items()}
    masses = {s: np.asarray(matrix[idx, mass_col], dtype=float) for s, idx in indices.items()}
    return X, masses, name, variant, seed, scaler


def run(config, trials=500, efficiencies=(.10, .01), seed=None, output='reports/closure'):
    X, masses, name, variant, model_seed, _ = _inputs(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(name, variant, model_seed, X['test'].shape[1], config, device)
    sv = score(model, X['val'], name, device)
    st = score(model, X['test'], name, device)
    thresholds = {f'{int(round(e * 100))}pct': threshold_at_efficiency(sv, e) for e in efficiencies}
    rng = np.random.default_rng(model_seed if seed is None else seed)
    hunter = BumpHunter()

    def diagnostics(mass, weights=None):
        n, b, z, sigma = hunter.fit_background(mass, weights=weights)
        return {'observed': float(n), 'expected_background': float(b),
                'local_significance': float(z), 'expected_background_uncertainty': float(sigma)}

    records = []
    # Include the unresampled held-out sample as a reference row.
    for trial in range(-1, trials):
        bootstrap = np.ones(len(st), dtype=float) if trial == -1 else rng.poisson(1., size=len(st))
        stages = [('pre_cut', np.ones(len(st), dtype=float))]
        stages.extend((f'post_{key}', score_cut_weights(st, *cut)) for key, cut in thresholds.items())
        for stage, acceptance in stages:
            # Tied boundary events are randomized independently per null trial;
            # the unresampled reference uses the fractional expected selection.
            realized_acceptance = (acceptance if trial == -1 else
                                   rng.binomial(1, np.clip(acceptance, 0., 1.)))
            weights = bootstrap * realized_acceptance
            d = diagnostics(masses['test'], weights=weights)
            records.append({'trial': trial, 'stage': stage, 'n_events': float(weights.sum()),
                            'threshold': np.nan if stage == 'pre_cut' else thresholds[stage[5:]][0],
                            'tie_acceptance': np.nan if stage == 'pre_cut' else thresholds[stage[5:]][1], **d})
    frame = pd.DataFrame(records)
    os.makedirs(output, exist_ok=True)
    stem = f'{name}_{variant}_{model_seed}'
    csv_path = os.path.join(output, f'{stem}_trials.csv')
    json_path = os.path.join(output, f'{stem}_summary.json')
    frame.to_csv(csv_path, index=False)
    summary = {'model': name, 'variant': variant, 'model_seed': model_seed,
               'pseudoexperiment_seed': model_seed if seed is None else seed,
               'n_validation_background': len(sv), 'n_test_background': len(st), 'trials': trials,
               'thresholds': thresholds,
               'validation_efficiency': {key: float(score_cut_weights(sv, *value).mean()) for key, value in thresholds.items()},
               'stages': {}}
    for stage, group in frame[frame.trial >= 0].groupby('stage'):
        zs = pd.to_numeric(group.local_significance, errors='coerce').dropna().to_numpy()
        observed = pd.to_numeric(frame[(frame.trial == -1) & (frame.stage == stage)].local_significance,
                                 errors='coerce').dropna().to_numpy()
        observed_z = float(observed[0]) if len(observed) else None
        summary['stages'][stage] = {'valid_trials': int(len(zs)),
            'median_local_significance': float(np.median(zs)) if len(zs) else None,
            'q95_local_significance': float(np.quantile(zs, .95)) if len(zs) else None,
            'fraction_z_ge_3': float(np.mean(zs >= 3)) if len(zs) else None,
            'fraction_z_ge_5': float(np.mean(zs >= 5)) if len(zs) else None,
            'observed_test_local_significance': observed_z,
            'bootstrap_tail_probability': (float((1 + np.sum(zs >= observed_z)) / (len(zs) + 1))
                                          if len(zs) and observed_z is not None else None)}
    summary['reference_observed_test'] = frame[frame.trial == -1].drop(columns='trial').to_dict('records')
    def json_safe(value):
        if isinstance(value, dict):
            return {k: json_safe(v) for k, v in value.items()}
        if isinstance(value, list):
            return [json_safe(v) for v in value]
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            return None
        if isinstance(value, (np.integer,)):
            return int(value)
        return value
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_safe(summary), f, indent=2, allow_nan=False)
    return csv_path, json_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--trials', type=int, default=500)
    parser.add_argument('--efficiencies', type=float, nargs='+', default=[.10, .01])
    parser.add_argument('--seed', type=int)
    parser.add_argument('--output', default='reports/closure')
    args = parser.parse_args()
    if args.trials < 1 or any(not 0 < e < 1 for e in args.efficiencies):
        parser.error('--trials must be positive and efficiencies must lie in (0,1)')
    with open(args.config, encoding='utf-8') as f:
        config = yaml.safe_load(f)
    paths = run(config, args.trials, args.efficiencies, args.seed, args.output)
    print('\n'.join(paths))
