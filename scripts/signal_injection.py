import numpy as np
import h5py
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.evaluation.bump_hunt import perform_bump_hunt, bootstrap_null_significances

def run_injection_study():
    split_dir = 'data/splits'
    features_file = 'data/processed/events_v2_features.h5'
    
    if not os.path.exists(features_file):
        print(f"Features file {features_file} not found. Ensure the dataset is built.")
        return
        
    print("Loading data...")
    with h5py.File(features_file, 'r') as f:
        bidx = np.load(os.path.join(split_dir, 'background_test.npy'))
        sidx = np.load(os.path.join(split_dir, 'signal2_test.npy'))
        mjj_bkg = f['features'][bidx, 5]
        mjj_sig = f['features'][sidx, 5]

    print("Running 1,000 null trials on pure background...")
    null_res = perform_bump_hunt(mjj_bkg, prefix='null_')
    print("Observed null Z:", null_res['null_local_significance'])
    
    bootstrap_res = bootstrap_null_significances(
        mjj_bkg, n_trials=1000, seed=42, observed_z=null_res['null_local_significance']
    )
    print("Bootstrap results (1,000 trials):", bootstrap_res)

    print("\nStarting signal injection scan (0.1% to 1.0%)...")
    rng = np.random.default_rng(42)
    for frac in np.linspace(0.001, 0.01, 10):
        n_inj = int(len(mjj_bkg) * frac)
        chosen = rng.choice(len(mjj_sig), n_inj, replace=False)
        mjj_mix = np.r_[mjj_bkg, mjj_sig[chosen]]
        
        inj_res = perform_bump_hunt(mjj_mix, prefix=f'inj_{frac:.3f}_')
        print(f"Frac {frac*100:.1f}% ({n_inj} injected): Z = {inj_res[f'inj_{frac:.3f}_local_significance']:.3f}, p = {inj_res[f'inj_{frac:.3f}_local_pvalue']:.2e}")

if __name__ == '__main__':
    run_injection_study()
