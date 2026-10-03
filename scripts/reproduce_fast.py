"""Seeded cached-feature demo; never writes canonical benchmark results."""
import argparse
import json
import platform
import time
from importlib.metadata import version
from pathlib import Path

import h5py
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


def run_demo(root, output, limit=2000, seed=42):
    started = time.perf_counter()
    root, output = Path(root), Path(output)
    if limit < 100:
        raise ValueError('limit must be at least 100')
    rng = np.random.default_rng(seed)
    samples = {}
    with h5py.File(root / 'data/processed/events_v2_features.h5', 'r') as source:
        for name in ('background_train', 'background_val', 'background_test',
                     'signal2_test'):
            indices = np.load(root / 'data/splits' / f'{name}.npy')
            chosen = np.sort(rng.choice(indices, min(limit, len(indices)), replace=False))
            samples[name] = source['features'][chosen, :5]
    if not all(len(x) and np.isfinite(x).all() for x in samples.values()):
        raise ValueError('Demo requires non-empty finite cached splits')
    scaler = StandardScaler().fit(samples['background_train'])
    detector = IsolationForest(n_estimators=100, random_state=seed, n_jobs=1)
    detector.fit(scaler.transform(samples['background_train']))
    # Import after resolving the repository root for direct script execution.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.evaluation.working_points import score_cut, score_cut_weights

    scores = {name: -detector.decision_function(scaler.transform(values))
              for name, values in samples.items() if name != 'background_train'}
    cut = score_cut(scores['background_val'], .1)
    background, signal = scores['background_test'], scores['signal2_test']
    result = {
        'scope': 'cached-feature demo; not the full benchmark or independent reproduction',
        'seed': seed, 'sample_counts': {k: len(v) for k, v in samples.items()},
        'roc_auc': float(roc_auc_score(
            np.r_[np.zeros(len(background)), np.ones(len(signal))],
            np.r_[background, signal])),
        'validation_background_acceptance': float(
            score_cut_weights(scores['background_val'], *cut).mean()),
        'test_background_acceptance': float(score_cut_weights(background, *cut).mean()),
        'test_signal_acceptance': float(score_cut_weights(signal, *cut).mean()),
        'threshold': cut[0], 'threshold_tie_probability': cut[1],
        'python': platform.python_version(),
        'versions': {name: version(name) for name in ('numpy', 'h5py', 'scikit-learn')},
        'elapsed_seconds': time.perf_counter() - started,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    np.savez_compressed(output / 'scores.npz', **scores)
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, default=Path('output/reproduce_fast'))
    parser.add_argument('--limit', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    run_demo(args.root, args.output, args.limit, args.seed)
