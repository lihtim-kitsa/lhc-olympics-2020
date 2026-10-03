"""Train/evaluate a raw-event Deep Sets anomaly baseline without large intermediates.

Example:
  python scripts/train_particle_deepsets.py data/raw/events_anomalydetection_v2.h5 \
    --hypotheses lhco_3prong,xtowrto3w --output-dir output/particle_deepsets

Fits only 2-prong background (60% split); chooses a score threshold from validation
background (default 99th percentile); reports untouched test-background, 2-prong
  signal hypotheses. Reads/encodes bounded batches. Hypothesis IDs resolve to
  <external-dir>/<id>/events.h5; use --hypothesis-path ID=PATH to override.
"""
import argparse
import json
import os
import sys

import h5py
import hdf5plugin  # noqa: F401; registers Blosc filters for the official raw files
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.particle_deepsets import PARTICLE_FEATURES, ParticleDeepSetsAutoencoder, encode_particles
from src.evaluation.working_points import score_cut, score_cut_weights


RAW_KEYS = ("df/block0_values", "table", "events")
RAW_KEY = "df"


class RawEventTable:
    """Pandas/PyTables-backed reader for the released Blosc HDFStore matrix."""
    def __init__(self, path):
        self.path = path
        with pd.HDFStore(path, mode="r") as store:
            available = [key.lstrip("/") for key in store.keys()]
            if RAW_KEY in available:
                self.key = RAW_KEY
            elif "Particles" in available:
                self.key = "Particles"
            elif len(available) == 1:
                self.key = available[0]
            else:
                raise ValueError(f"Could not identify particle table in {path}; HDF keys: {available}")
            storer = store.get_storer(self.key)
            if np.isscalar(storer.shape):
                self.shape = (int(storer.nrows), int(storer.ncols))
            else:
                self.shape = tuple(int(x) for x in storer.shape)
        if self.shape[1] != 2101:
            raise ValueError(f"Expected raw LHCO N x 2101 matrix, got {self.shape}")

    def __len__(self):
        return self.shape[0]

    def _range(self, start, stop):
        return pd.read_hdf(self.path, key=self.key, start=int(start), stop=int(stop)).to_numpy()

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            block = self._range(start, stop)
            return block if step == 1 else block[::step]
        ids = np.asarray(key, dtype=np.int64).reshape(-1)
        out = np.empty((len(ids), self.shape[1]), dtype=np.float32)
        groups = {}
        block_size = 2048
        for position, index in enumerate(ids):
            groups.setdefault(int(index) // block_size, []).append((position, int(index)))
        for block, rows in groups.items():
            start = block * block_size
            data = self._range(start, min(start + block_size, len(self)))
            for position, index in rows:
                out[position] = data[index - start]
        return out


def raw_dataset(path):
    return RawEventTable(path)


def valid_class_indices(ds, chunk_size=8192):
    """Read compact labels and event-valid flags, never materializing event tensors."""
    by_class = {0: [], 1: []}
    for start in range(0, len(ds), chunk_size):
        rows = ds[start:min(start + chunk_size, len(ds))]
        triples = rows[:, :-1].reshape(len(rows), 700, 3)
        valid = (np.isfinite(triples).all(axis=2) & (triples[:, :, 0] > 0)).any(axis=1)
        labels = rows[:, -1]
        for label in (0, 1):
            local = np.flatnonzero(valid & (labels == label)) + start
            by_class[label].append(local.astype(np.int64))
    return {k: np.concatenate(v) if v else np.empty(0, dtype=np.int64) for k, v in by_class.items()}


def validate_split_file(ds, path, expected_label, validation_sample=16, seed=42):
    ids = np.asarray(np.load(path), dtype=np.int64).reshape(-1)
    if len(np.unique(ids)) != len(ids):
        raise ValueError(f"Duplicate event indices in {path}")
    if len(ids) and ((ids < 0).any() or (ids >= len(ds)).any()):
        raise ValueError(f"Out-of-range event index in {path} for raw file with {len(ds)} rows")
    if len(ids):
        # The shipped split arrays are produced by make_dataset.py from the
        # canonical feature table, and random HDF5 gathers of every 8 KB event
        # are extremely costly. Audit a deterministic sample here; the training
        # batches still read the selected rows from the raw file directly.
        sample = ids if len(ids) <= validation_sample else np.sort(
            np.random.default_rng(seed).choice(ids, size=validation_sample, replace=False))
        rows = ds[sample]
        particles = rows[:, :-1].reshape(len(rows), 700, 3)
        valid = (np.isfinite(particles).all(axis=2) & (particles[:, :, 0] > 0)).any(axis=1)
        if not valid.all():
            raise ValueError(f"{path} contains {int((~valid).sum())} invalid/empty raw events")
        labels = rows[:, -1]
        if not np.all(labels == expected_label):
            raise ValueError(f"{path} contains event(s) not labeled {expected_label} in the raw file")
    return ids


def split_indices(ds, seed, split_dir="data/splits", validation_sample=16):
    shared_names = {
        "train": ("background_train.npy", 0),
        "validation": ("background_val.npy", 0),
        "test_background": ("background_test.npy", 0),
        "test_signal2": ("signal2_test.npy", 1),
    }
    shared_paths = {key: os.path.join(split_dir, name) for key, (name, _) in shared_names.items()}
    present = [os.path.isfile(path) for path in shared_paths.values()]
    if any(present) and not all(present):
        missing = [os.path.basename(path) for path, exists in zip(shared_paths.values(), present) if not exists]
        raise ValueError(f"Partial shared split set in {split_dir}; missing {missing}")
    if all(present):
        split = {key: validate_split_file(ds, path, shared_names[key][1], validation_sample, seed + i)
                 for i, (key, path) in enumerate(shared_paths.items())}
        bg_pools = [split[k] for k in ("train", "validation", "test_background")]
        if any(np.intersect1d(bg_pools[i], bg_pools[j]).size
               for i in range(3) for j in range(i + 1, 3)):
            raise ValueError("Shared background split files overlap")
        counts = {k: int(len(v)) for k, v in split.items()}
        return split, counts, "existing_shared_split_files"

    classes = valid_class_indices(ds)
    rng = np.random.default_rng(seed)
    split = {}
    for label, name in ((0, "background"), (1, "signal2")):
        ids = classes[label].copy()
        rng.shuffle(ids)
        n_train, n_val = int(.6 * len(ids)), int(.2 * len(ids))
        if label == 0:
            split["train"] = ids[:n_train]
            split["validation"] = ids[n_train:n_train + n_val]
            split["test_background"] = ids[n_train + n_val:]
        else:
            # Signals are untouched until final test; discard development portion.
            split["test_signal2"] = ids[n_train + n_val:]
    counts = {k: int(len(v)) for k, v in split.items()}
    return split, counts, "generated_seeded_60_20_20"


def encode_batch(rows):
    x = np.zeros((len(rows), 700, len(PARTICLE_FEATURES)), dtype=np.float32)
    mask = np.zeros((len(rows), 700), dtype=np.bool_)
    for i, row in enumerate(rows):
        x[i], mask[i] = encode_particles(row[:-1])
    return torch.from_numpy(x), torch.from_numpy(mask)


def batches(ds, indices, batch_size, shuffle=False, seed=0):
    ids = np.asarray(indices, dtype=np.int64).copy()
    # The released HDFStore is compressed in row blocks; read each contiguous
    # slab once, then batch only the requested rows from that slab.
    block_size = 8192
    grouped = {}
    for index in ids:
        grouped.setdefault(int(index) // block_size, []).append(int(index))
    block_ids = np.asarray(list(grouped), dtype=np.int64)
    rng = np.random.default_rng(seed)
    if shuffle:
        rng.shuffle(block_ids)
    for block in block_ids:
        take = np.asarray(grouped[int(block)], dtype=np.int64)
        if shuffle:
            rng.shuffle(take)
        row_start = int(block) * block_size
        block_rows = ds[row_start:min(row_start + block_size, len(ds))]
        for start in range(0, len(take), batch_size):
            selected = take[start:start + batch_size]
            yield selected, encode_batch(block_rows[selected - row_start])


@torch.no_grad()
def score_indices(model, ds, indices, batch_size, device):
    model.eval()
    ids_out, scores = [], []
    for ids, (x, mask) in batches(ds, indices, batch_size):
        score = model.anomaly_score(x.to(device), mask.to(device)).cpu().numpy()
        ids_out.append(ids)
        scores.append(score)
    return (np.concatenate(ids_out) if ids_out else np.empty(0, dtype=np.int64),
            np.concatenate(scores) if scores else np.empty(0, dtype=np.float32))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("raw2", help="2-prong LHCO raw HDF5 (background + development signal)")
    p.add_argument("--hypotheses", default="", help="comma-separated IDs to evaluate; default none")
    p.add_argument("--external-dir", default="data/external_signals", help="root containing ID/events.h5")
    p.add_argument("--signal3", help="optional explicit held-out 3-prong raw HDF5 path")
    p.add_argument("--hypothesis-path", action="append", default=[], metavar="ID=PATH",
                   help="explicit raw HDF5 path for an evaluation hypothesis; repeatable")
    p.add_argument("--output-dir", default="output/particle_deepsets")
    p.add_argument("--split-dir", default="data/splits", help="existing baseline split directory")
    p.add_argument("--max-train-events", type=int, help="seeded cap on background training events only")
    p.add_argument("--max-eval-events", type=int,
                   help="seeded cap per validation/test pool for a pilot run; omit for full evaluation")
    p.add_argument("--max-hypothesis-events", type=int,
                   help="seeded cap per extra signal sample; omit to score every available event")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--threshold-quantile", type=float, default=.99)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or not 0 < args.threshold_quantile < 1:
        p.error("epochs/batch-size must be positive and threshold-quantile must lie in (0,1)")
    if args.max_train_events is not None and args.max_train_events < 1:
        p.error("max-train-events must be positive")
    if args.max_eval_events is not None and args.max_eval_events < 1:
        p.error("max-eval-events must be positive")
    if args.max_hypothesis_events is not None and args.max_hypothesis_events < 1:
        p.error("max-hypothesis-events must be positive")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.output_dir, "particle_deepsets.pt")
    metrics_path = os.path.join(args.output_dir, "particle_deepsets_metrics.json")

    ds = raw_dataset(args.raw2)
    with pd.HDFStore(args.raw2, mode="r"):
        split, split_counts, split_source = split_indices(ds, args.seed, args.split_dir)
        full_train_count = len(split["train"])
        if args.max_train_events is not None and len(split["train"]) > args.max_train_events:
            rng = np.random.default_rng(args.seed)
            split["train"] = np.sort(rng.choice(split["train"], size=args.max_train_events, replace=False))
        if args.max_eval_events is not None:
            rng = np.random.default_rng(args.seed + 1)
            for key in ("validation", "test_background", "test_signal2"):
                if len(split[key]) > args.max_eval_events:
                    split[key] = np.sort(rng.choice(split[key], size=args.max_eval_events, replace=False))
        for key, ids in split.items():
            split_counts[f"{key}_used"] = int(len(ids))
        split_counts["background_train_before_cap"] = int(full_train_count)
        split_counts["background_train_used"] = int(len(split["train"]))
        if not len(split["train"]) or not len(split["validation"]) or not len(split["test_background"]):
            raise ValueError(f"Insufficient valid 2-prong background for split: {split_counts}")
        model = ParticleDeepSetsAutoencoder().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        history, best_val = [], float("inf")
        best_state = None
        for epoch in range(args.epochs):
            model.train()
            train_losses = []
            for _, (x, mask) in batches(ds, split["train"], args.batch_size, True, args.seed + epoch):
                x, mask = x.to(device), mask.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = model.anomaly_score(x, mask).mean()
                loss.backward()
                optimizer.step()
                train_losses.append(float(loss.detach().cpu()))
            model.eval()
            val_loss = []
            with torch.no_grad():
                for _, (x, mask) in batches(ds, split["validation"], args.batch_size):
                    val_loss.extend(model.anomaly_score(x.to(device), mask.to(device)).cpu().tolist())
            mean_train = float(np.mean(train_losses))
            mean_val = float(np.mean(val_loss))
            history.append({"epoch": epoch + 1, "train_background_mse": mean_train,
                            "validation_background_mse": mean_val})
            print(f"epoch {epoch + 1}/{args.epochs}: train={mean_train:.6g} val={mean_val:.6g}", flush=True)
            if mean_val < best_val:
                best_val = mean_val
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        val_ids, val_scores = score_indices(model, ds, split["validation"], args.batch_size, device)
        threshold, tie_acceptance = score_cut(val_scores, 1.0 - args.threshold_quantile)
        bg_ids, bg_scores = score_indices(model, ds, split["test_background"], args.batch_size, device)
        s2_ids, s2_scores = score_indices(model, ds, split["test_signal2"], args.batch_size, device)
        test2_scores = np.concatenate((bg_scores, s2_scores))
        test2_labels = np.concatenate((np.zeros(len(bg_scores)), np.ones(len(s2_scores))))
        metrics = {
            "protocol": "background-only Deep Sets autoencoder; event-level 60/20/20 class split, signal train/validation portions unused",
            "seed": args.seed, "threshold_quantile": args.threshold_quantile,
            "threshold_from_validation_background": threshold,
            "tie_acceptance_probability": tie_acceptance,
            "split_source": split_source,
            "max_train_events": args.max_train_events,
            "max_eval_events_per_pool": args.max_eval_events,
            "max_hypothesis_events": args.max_hypothesis_events,
            "split_counts": split_counts,
            "validation_background_exceedance_fraction": float(score_cut_weights(val_scores, threshold, tie_acceptance).mean()),
            "test_background_false_positive_fraction": float(score_cut_weights(bg_scores, threshold, tie_acceptance).mean()),
            "test_2prong_signal_efficiency_at_threshold": float(score_cut_weights(s2_scores, threshold, tie_acceptance).mean()) if len(s2_scores) else None,
            "test_2prong_roc_auc": float(roc_auc_score(test2_labels, test2_scores)) if len(s2_scores) and len(bg_scores) else None,
            "best_validation_background_mse": best_val,
            "training_history": history,
        }
        hypotheses = {}
        for spec in filter(None, (part.strip() for part in args.hypotheses.split(","))):
            if spec == "lhco_3prong":
                path = os.path.join("data", "raw", "events_anomalydetection_Z_XY_qqq.h5")
            else:
                path = os.path.join(args.external_dir, spec, "events.h5")
            hypotheses[spec] = path
        for spec in args.hypothesis_path:
            if "=" not in spec:
                p.error(f"--hypothesis-path must be ID=PATH, got {spec!r}")
            hypothesis_id, path = spec.split("=", 1)
            hypotheses[hypothesis_id] = path
        # Backward-compatible named input for the original held-out 3-prong set.
        if getattr(args, "signal3", None):
            hypotheses["lhco_3prong"] = args.signal3
        metrics["signal_hypotheses"] = {}
        for hypothesis_id, hypothesis_path in hypotheses.items():
            if not os.path.isfile(hypothesis_path):
                metrics["signal_hypotheses"][hypothesis_id] = {"status": "not_available", "path": hypothesis_path}
                continue
            hyp_ds = raw_dataset(hypothesis_path)
            with pd.HDFStore(hypothesis_path, mode="r"):
                hyp_classes = valid_class_indices(hyp_ds)
                # Prefer explicit signal labels; if the file is a signal-only
                # sample with unset labels, evaluate every valid event.
                signal_ids = hyp_classes[1] if len(hyp_classes[1]) else np.concatenate(list(hyp_classes.values()))
                n_available = len(signal_ids)
                if args.max_hypothesis_events is not None and n_available > args.max_hypothesis_events:
                    rng = np.random.default_rng(args.seed + len(hypotheses))
                    signal_ids = np.sort(rng.choice(signal_ids, size=args.max_hypothesis_events, replace=False))
                _, hyp_scores = score_indices(model, hyp_ds, signal_ids, args.batch_size, device)
                metrics["signal_hypotheses"][hypothesis_id] = {
                    "status": "evaluated", "events": int(len(hyp_scores)), "available_events": int(n_available),
                    "source": os.path.relpath(hypothesis_path),
                    "signal_efficiency_at_validation_background_threshold": float(score_cut_weights(hyp_scores, threshold, tie_acceptance).mean()) if len(hyp_scores) else None,
                    "roc_auc_vs_2prong_test_background": float(roc_auc_score(
                        np.concatenate((np.zeros(len(bg_scores)), np.ones(len(hyp_scores)))),
                        np.concatenate((bg_scores, hyp_scores)))) if len(hyp_scores) and len(bg_scores) else None,
                }
        torch.save({"model_state_dict": model.state_dict(), "model_config": {"input_dim": len(PARTICLE_FEATURES), "hidden_dim": 64, "latent_dim": 32},
                    "seed": args.seed, "threshold_quantile": args.threshold_quantile, "threshold": threshold,
                    "tie_acceptance_probability": tie_acceptance,
                    "training_source": os.path.basename(args.raw2)}, checkpoint_path)
    metrics["checkpoint"] = os.path.basename(checkpoint_path)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps({"checkpoint": checkpoint_path, "metrics": metrics_path,
                      "threshold": metrics["threshold_from_validation_background"]}, indent=2))


if __name__ == "__main__":
    main()
