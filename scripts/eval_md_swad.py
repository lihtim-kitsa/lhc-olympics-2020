import os
import h5py
import numpy as np
import torch
import pickle
from sklearn.metrics import roc_auc_score
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.models.m9_md_swad import MDSWAD

def evaluate():
    split_dir = os.path.join('data', 'splits')
    features_file = os.path.join('data', 'processed', 'events_v2_features.h5')
    
    # Load test indices
    bkg_test_idx = np.load(os.path.join(split_dir, 'background_test.npy'))
    sig_test_idx = np.load(os.path.join(split_dir, 'signal2_test.npy'))
    test_idx = np.concatenate([bkg_test_idx, sig_test_idx])
    y_test = np.concatenate([np.zeros(len(bkg_test_idx)), np.ones(len(sig_test_idx))])
    
    cols = [0, 1, 2, 3, 4]
    with h5py.File(features_file, 'r') as f:
        all_features = f['features'][:]
        
    X_unscaled = all_features[test_idx][:, cols]
        
        # Ensure y_test matches this concatenation (it already does: bkg then sig)
        
    scaler_path = os.path.join('models', 'scaler_M9_MDSWAD_f0.5_k100_42.pkl')
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
        
    X_test = scaler.transform(X_unscaled).astype(np.float32)
    X_tensor = torch.from_numpy(X_test)
    
    # Load model
    model = MDSWAD(input_dim=5, lam=1.0)
    model.load_state_dict(torch.load(os.path.join('models', 'M9_MDSWAD_f0.5_k100_42.pt'), weights_only=False)['model_state_dict'])
    model.eval()
    
    # Get anomaly scores (distance to origin)
    with torch.no_grad():
        scores = model.get_anomaly_score(X_tensor).numpy()
        
    auc = roc_auc_score(y_test, scores)
    print(f"MD-SWAD ROC AUC on Test Set: {auc:.4f}")

if __name__ == "__main__":
    evaluate()
