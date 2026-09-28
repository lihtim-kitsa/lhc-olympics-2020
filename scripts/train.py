import os
import argparse
import yaml
import numpy as np
import h5py
import torch
from sklearn.preprocessing import StandardScaler
import mlflow
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.models.m1_autoencoder import Autoencoder
from src.models.m2_isolation_forest import IsolationForestAnomalyDetector
from src.models.m3_deep_svdd import DeepSVDD
from src.models.m4_deep_sad import DeepSAD
from src.models.m5_supervised import SupervisedMLP

def load_data(split_dir, features_file, config):
    f_train = config['data']['f_train']
    k_labels = config['data'].get('k_labels', 0)
    seed = config['training']['seed']
    np.random.seed(seed)
    
    # Load indices
    bkg_train_idx = np.load(os.path.join(split_dir, 'background_train.npy'))
    sig_train_idx = np.load(os.path.join(split_dir, 'signal2_train.npy'))
    
    # Inject contamination
    n_bkg = len(bkg_train_idx)
    n_sig_inject = int(n_bkg * (f_train / 100.0))
    
    if n_sig_inject > len(sig_train_idx):
        raise ValueError("Not enough signal train events for requested contamination.")
        
    injected_sig = sig_train_idx[:n_sig_inject]
    remaining_sig = sig_train_idx[n_sig_inject:]
    
    # Select labeled anomalies for Deep SAD from remaining signal pool
    labeled_sig = remaining_sig[:k_labels]
    
    # Final training pool: bkg + injected (unlabeled)
    unlabeled_train_idx = np.concatenate([bkg_train_idx, injected_sig])
    np.random.shuffle(unlabeled_train_idx)
    
    # Load features
    with h5py.File(features_file, 'r') as f:
        dset = f['features']
        
        # Determine feature columns
        use_mjj = config['data'].get('use_mjj', False)
        # M1-M5: mJ1, dmJ, tau21_J1, tau21_J2, dRJJ -> indices 0,1,2,3,4
        # mJJ -> index 5
        feat_cols = [0, 1, 2, 3, 4, 5] if use_mjj else [0, 1, 2, 3, 4]
        
        X_train_unlabeled = dset[np.sort(unlabeled_train_idx)][:, feat_cols]
        # Reorder to match shuffled indices
        # (This can be optimized but is fine for now)
        X_train_unlabeled = np.array([dset[i, feat_cols] for i in unlabeled_train_idx])
        y_train_unlabeled = np.zeros(len(X_train_unlabeled)) # All treated as 0
        
        X_train_labeled = np.array([]).reshape(0, len(feat_cols))
        y_train_labeled = np.array([])
        
        if k_labels > 0:
            X_train_labeled = np.array([dset[i, feat_cols] for i in labeled_sig])
            y_train_labeled = np.ones(len(X_train_labeled))
            
    # Scale features
    scaler = StandardScaler()
    X_train_unlabeled = scaler.fit_transform(X_train_unlabeled)
    
    if k_labels > 0:
        X_train_labeled = scaler.transform(X_train_labeled)
        
    return X_train_unlabeled, y_train_unlabeled, X_train_labeled, y_train_labeled, scaler

def train_model(config):
    mlflow.set_experiment(config['mlflow']['experiment_name'])
    
    with mlflow.start_run():
        mlflow.log_params(config['training'])
        mlflow.log_params(config['data'])
        mlflow.log_param('model', config['model']['name'])
        
        split_dir = os.path.join('data', 'splits')
        features_file = os.path.join('data', 'processed', 'events_v2_features.h5')
        
        X_u, y_u, X_l, y_l, scaler = load_data(split_dir, features_file, config)
        
        model_name = config['model']['name']
        
        if model_name == 'M2_IsolationForest':
            model = IsolationForestAnomalyDetector(
                n_estimators=config['model'].get('n_estimators', 100),
                random_state=config['training']['seed']
            )
            model.fit(X_u) # IF only trains on unlabeled data
            
        elif model_name == 'M1_Autoencoder':
            model = Autoencoder(input_dim=X_u.shape[1])
            # PyTorch training loop for AE...
            # This is a stub for the actual training loop
            print("Training Autoencoder...")
            
        elif model_name == 'M3_DeepSVDD':
            model = DeepSVDD(input_dim=X_u.shape[1])
            print("Training Deep SVDD...")
            
        elif model_name == 'M4_DeepSAD':
            model = DeepSAD(input_dim=X_u.shape[1], eta=config['model'].get('eta', 1.0))
            print("Training Deep SAD...")
            
        elif model_name == 'M5_Supervised':
            model = SupervisedMLP(input_dim=X_u.shape[1])
            print("Training Supervised MLP...")
            
        # Save model and scaler (using torch.save or pickle)
        os.makedirs('models', exist_ok=True)
        # ... saving code ...

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    train_model(config)
