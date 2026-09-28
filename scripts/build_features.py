import os
import h5py
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.features.clustering import extract_event_features

# PRD Rule: mJJ is not included in the main detector input vector M1-M5
FEATURES = ['mJ1', 'dmJ', 'tau21_J1', 'tau21_J2', 'dRJJ', 'mJJ', 'label']

def process_file(input_path, output_path, chunk_size=10000):
    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        return
        
    print(f"Processing {input_path} -> {output_path}")
    
    with h5py.File(input_path, 'r') as f_in, h5py.File(output_path, 'w') as f_out:
        # Assuming the dataset is named 'events' or we just pick the first dataset
        dset_key = list(f_in.keys())[0] 
        dset = f_in[dset_key]
        
        n_events = dset.shape[0]
        print(f"Found {n_events} events in {dset_key}")
        
        # Create output dataset
        out_dset = f_out.create_dataset('features', shape=(n_events, len(FEATURES)), dtype='f4')
        
        for i in tqdm(range(0, n_events, chunk_size)):
            end = min(i + chunk_size, n_events)
            chunk = dset[i:end]
            
            # The last column is the label if shape is (N, 2101)
            # if shape is (N, 2100) then there's no label (unlikely based on PRD)
            has_label = chunk.shape[1] == 2101
            
            features_chunk = np.zeros((end - i, len(FEATURES)), dtype='f4')
            
            for j, event in enumerate(chunk):
                if has_label:
                    event_data = event[:-1]
                    label = event[-1]
                else:
                    event_data = event
                    label = 0.0
                    
                feats = extract_event_features(event_data)
                
                if feats is not None:
                    features_chunk[j, 0] = feats['mJ1']
                    features_chunk[j, 1] = feats['dmJ']
                    features_chunk[j, 2] = feats['tau21_J1']
                    features_chunk[j, 3] = feats['tau21_J2']
                    features_chunk[j, 4] = feats['dRJJ']
                    features_chunk[j, 5] = feats['mJJ']
                    features_chunk[j, 6] = label
                else:
                    # If event doesn't have 2 jets, fill with nan
                    features_chunk[j, :] = np.nan
                    
            out_dset[i:end] = features_chunk
            
def main():
    raw_dir = os.path.join('data', 'raw')
    proc_dir = os.path.join('data', 'processed')
    os.makedirs(proc_dir, exist_ok=True)
    
    files_to_process = [
        ('events_anomalydetection_v2.h5', 'events_v2_features.h5'),
        ('events_anomalydetection_Z_XY_qqq.h5', 'events_Z_XY_qqq_features.h5')
    ]
    
    for in_name, out_name in files_to_process:
        in_path = os.path.join(raw_dir, in_name)
        out_path = os.path.join(proc_dir, out_name)
        process_file(in_path, out_path)

if __name__ == '__main__':
    main()
