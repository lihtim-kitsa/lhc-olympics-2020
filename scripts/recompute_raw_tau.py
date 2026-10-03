import numpy as np
import h5py
import hdf5plugin  # noqa: F401; registers Blosc filters for the official raw files
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.features.clustering import extract_event_features

def main():
    raw_file = 'data/raw/events_anomalydetection_v2.h5'
    if not os.path.exists(raw_file):
        print("Raw h5 file not found. Skipping raw recomputation.")
        return

    print("Loading raw events...")
    with h5py.File(raw_file, 'r') as f:
        events = f['events'][:1000] # Just testing a sample due to compute limits
    
    print("Recomputing features for a small subset...")
    new_features = []
    for idx, row in enumerate(events):
        feats = extract_event_features(row)
        if feats is not None:
            new_features.append([feats['mJ1'], feats['dmJ'], feats['tau21_J1'], feats['tau21_J2'], feats['dRJJ'], feats['mJJ']])

    print(f"Successfully recomputed {len(new_features)} events.")

if __name__ == '__main__':
    main()
