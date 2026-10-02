"""Build a higher-dimensional, masked particle-level LHCO representation.

Example: python scripts/build_particle_events.py data/raw/events_anomalydetection_v2.h5 data/processed/events_v2_particles.h5
Output datasets: features [N,700,6], mask [N,700], labels [N], event_valid [N].
"""
import argparse
import os
import sys

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.particle_deepsets import PARTICLE_FEATURES, encode_particles


def build(input_path, output_path, chunk_size=2048):
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    temporary = output_path + ".tmp"
    with pd.HDFStore(input_path, mode="r") as source:
        storer = source.get_storer("df")
        shape = tuple(int(x) for x in storer.shape)
        if len(shape) != 2 or shape[1] != 2101:
            raise ValueError(f"Expected LHCO raw matrix N x 2101, got {shape}")
        n = shape[0]
        try:
            with h5py.File(temporary, "w") as out:
                n_features = len(PARTICLE_FEATURES)
                xds = out.create_dataset("features", shape=(n, 700, n_features), dtype="f4",
                                         chunks=(min(chunk_size, max(1, n)), 700, n_features),
                                         compression="gzip", compression_opts=1)
                mds = out.create_dataset("mask", shape=(n, 700), dtype="?",
                                         chunks=(min(chunk_size, max(1, n)), 700),
                                         compression="gzip", compression_opts=1)
                lds = out.create_dataset("labels", shape=(n,), dtype="f4", chunks=True)
                vds = out.create_dataset("event_valid", shape=(n,), dtype="?", chunks=True)
                for start in tqdm(range(0, n, chunk_size), desc="Encoding LHCO particles"):
                    end = min(start + chunk_size, n)
                    rows = pd.read_hdf(input_path, key="df", start=start, stop=end).to_numpy()
                    x = np.zeros((len(rows), 700, 5), dtype=np.float32)
                    mask = np.zeros((len(rows), 700), dtype=np.bool_)
                    for i, row in enumerate(rows):
                        x[i], mask[i] = encode_particles(row[:-1])
                    xds[start:end] = x
                    mds[start:end] = mask
                    lds[start:end] = rows[:, -1]
                    vds[start:end] = mask.any(axis=1)
                xds.attrs["feature_names"] = ",".join(PARTICLE_FEATURES)
                xds.attrs["schema_version"] = "1.0.0"
                xds.attrs["input_representation"] = "700 padded (pt, eta, phi) constituent triples"
                xds.attrs["transforms"] = "pt/sum(pt); log(pt/max(pt)); log1p(pt [GeV]); clip(eta,-6,6)/6; sin(phi); cos(phi)"
                xds.attrs["permutation_invariant_model"] = "src.models.particle_deepsets.ParticleDeepSetsAutoencoder"
                mds.attrs["semantics"] = "true for finite triples with pt > 0; false for padding/malformed entries"
                out.attrs["source_file"] = os.path.basename(input_path)
                out.attrs["source_dataset"] = "df"
                out.attrs["label_semantics"] = "raw final column; 0 background, 1 signal per LHCO release"
            os.replace(temporary, output_path)
        finally:
            if os.path.exists(temporary):
                os.remove(temporary)
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="LHCO raw HDF5 file")
    parser.add_argument("output", help="output particle HDF5 file")
    parser.add_argument("--chunk-size", type=int, default=2048)
    args = parser.parse_args()
    print(build(args.input, args.output, args.chunk_size))


if __name__ == "__main__":
    main()
