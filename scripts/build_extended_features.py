"""Add tau32 observables to the canonical LHCO feature table."""
import os

import h5py
import numpy as np
import pandas as pd


BASE_FEATURES = "data/processed/events_v2_features.h5"
HIGH_LEVEL = "data/raw/events_anomalydetection_v2.features.h5"
OUTPUT = "data/processed/events_v2_extended_features.h5"


def build_extended_features(base_path=BASE_FEATURES, high_path=HIGH_LEVEL, output_path=OUTPUT):
    """Write [five baseline inputs, tau32_J1, tau32_J2, mJJ, label]."""
    frame = pd.read_hdf(high_path, key="df")
    required = {"tau2j1", "tau3j1", "tau2j2", "tau3j2"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing subjettiness columns: {sorted(missing)}")

    with h5py.File(base_path, "r") as source:
        base = source["features"][:]
        names = source["features"].attrs.get("feature_names", "")
    if base.ndim != 2 or base.shape[1] != 7:
        raise ValueError(f"Expected canonical N x 7 feature table, got {base.shape}")
    if len(frame) != len(base):
        raise ValueError(f"High-level/base row counts differ: {len(frame)} vs {len(base)}")
    if names and names != "mJ1,dmJ,tau21_J1,tau21_J2,dRJJ,mJJ,label":
        raise ValueError(f"Unexpected canonical feature order: {names}")

    def ratio(numerator, denominator):
        num = frame[numerator].to_numpy(dtype=np.float64)
        den = frame[denominator].to_numpy(dtype=np.float64)
        out = np.full(len(frame), np.nan)
        valid = np.isfinite(num) & np.isfinite(den)
        out[valid & (den == 0) & (num == 0)] = 0.0
        np.divide(num, den, out=out, where=valid & (den > 0))
        return out

    tau32_1 = ratio("tau3j1", "tau2j1")
    tau32_2 = ratio("tau3j2", "tau2j2")
    # Preserve canonical ordering expected by train.py/evaluate.py: mJJ remains col 7.
    extended = np.column_stack((base[:, :5], tau32_1, tau32_2, base[:, 5:])).astype(np.float32)
    bad = ~np.isfinite(extended[:, :8]).all(axis=1)
    extended[bad, :8] = np.nan

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    temp_path = output_path + ".tmp"
    with h5py.File(temp_path, "w") as target:
        dataset = target.create_dataset(
            "features", data=extended, dtype="f4",
            chunks=(min(10000, len(extended)), extended.shape[1]),
            compression="gzip", compression_opts=1,
        )
        dataset.attrs["feature_names"] = "mJ1,dmJ,tau21_J1,tau21_J2,dRJJ,tau32_J1,tau32_J2,mJJ,label"
        dataset.attrs["feature_schema_version"] = "2.0.0"
        dataset.attrs["source"] = "canonical LHCO features plus released high-level tau3/tau2 ratios"
        dataset.attrs["invalid_rows"] = int(bad.sum())
    os.replace(temp_path, output_path)
    return {"rows": len(extended), "invalid_rows": int(bad.sum()), "output": output_path}


if __name__ == "__main__":
    print(build_extended_features())
