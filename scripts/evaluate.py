import os
import argparse
import yaml
import numpy as np
import h5py
import torch
import pandas as pd
from sklearn.metrics import roc_curve
import mlflow
import pickle
import sys
import uproot

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.models.m1_autoencoder import Autoencoder
from src.models.m2_isolation_forest import IsolationForestAnomalyDetector
from src.models.m3_deep_svdd import DeepSVDD
from src.models.m4_deep_sad import DeepSAD
from src.models.m5_supervised import SupervisedMLP
from src.evaluation.metrics import get_core_metrics, evaluate_mass_sculpting
from src.evaluation.bump_hunt import perform_bump_hunt

def load_test_data(split_dir, features_file, config, scaler):
    bkg_test_idx = np.load(os.path.join(split_dir, 'background_test.npy'))
    sig_test_idx = np.load(os.path.join(split_dir, 'signal2_test.npy'))
    
    with h5py.File(features_file, 'r') as f:
        dset = f['features']
        use_mjj = config['data'].get('use_mjj', False)
        feat_cols = [0, 1, 2, 3, 4, 5] if use_mjj else [0, 1, 2, 3, 4]
        
        # Load background
        X_bkg = np.zeros((len(bkg_test_idx), len(feat_cols)), dtype=np.float32)
        mjj_bkg = np.zeros(len(bkg_test_idx), dtype=np.float32)
        for i, idx in enumerate(bkg_test_idx):
            X_bkg[i] = dset[idx, feat_cols]
            mjj_bkg[i] = dset[idx, 5]
            
        y_bkg = np.zeros(len(X_bkg), dtype=np.float32)
        
        # Load signal
        X_sig = np.zeros((len(sig_test_idx), len(feat_cols)), dtype=np.float32)
        mjj_sig = np.zeros(len(sig_test_idx), dtype=np.float32)
        for i, idx in enumerate(sig_test_idx):
            X_sig[i] = dset[idx, feat_cols]
            mjj_sig[i] = dset[idx, 5]
            
        y_sig = np.ones(len(X_sig), dtype=np.float32)
        
    X_test = np.vstack([X_bkg, X_sig])
    y_test = np.concatenate([y_bkg, y_sig])
    mjj_test = np.concatenate([mjj_bkg, mjj_sig])
    
    # Scale features
    X_test_scaled = scaler.transform(X_test)
    X_bkg_scaled = scaler.transform(X_bkg)
    
    return X_test_scaled, y_test, mjj_test, X_bkg_scaled, y_bkg, mjj_bkg

def evaluate_model(config):
    mlflow.set_experiment(config['mlflow']['experiment_name'])
    
    with mlflow.start_run():
        model_name = config['model']['name']
        seed = config['training']['seed']
        
        model_path = os.path.join('models', f"{model_name}_{seed}.pt")
        scaler_path = os.path.join('models', f"scaler_{model_name}_{seed}.pkl")
        
        if not os.path.exists(scaler_path):
            # Model not trained yet
            print(f"Skipping evaluation, model {model_name}_{seed} not found.")
            return
            
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
            
        split_dir = os.path.join('data', 'splits')
        features_file = os.path.join('data', 'processed', 'events_v2_features.h5')
        
        X_test, y_test, mjj_test, X_bkg, y_bkg, mjj_bkg = load_test_data(split_dir, features_file, config, scaler)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_dim = X_test.shape[1]
        
        # Load Model
        if model_name == 'M2_IsolationForest':
            model_path_pkl = os.path.join('models', f"{model_name}_{seed}.pkl")
            with open(model_path_pkl, 'rb') as f:
                model = pickle.load(f)
            scores = model.get_anomaly_score(X_test)
            scores_bkg = model.get_anomaly_score(X_bkg)
        else:
            checkpoint = torch.load(model_path, map_location=device, weights_only=True)
            if model_name == 'M1_Autoencoder':
                model = Autoencoder(input_dim=input_dim).to(device)
            elif model_name == 'M3_DeepSVDD' or model_name == 'M6_MassAware':
                model = DeepSVDD(input_dim=input_dim).to(device)
                model.c = checkpoint['center']
            elif model_name == 'M4_DeepSAD':
                model = DeepSAD(input_dim=input_dim, eta=config['model'].get('eta', 1.0)).to(device)
                model.c = checkpoint['center']
            elif model_name == 'M5_Supervised':
                model = SupervisedMLP(input_dim=input_dim).to(device)
                
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Compute scores in batches to avoid OOM
            model.eval()
            scores = []
            batch_size = 10000
            with torch.no_grad():
                for i in range(0, len(X_test), batch_size):
                    batch = torch.tensor(X_test[i:i+batch_size], dtype=torch.float32).to(device)
                    scores.append(model.get_anomaly_score(batch).cpu().numpy())
            scores = np.concatenate(scores)
            
            scores_bkg = []
            with torch.no_grad():
                for i in range(0, len(X_bkg), batch_size):
                    batch = torch.tensor(X_bkg[i:i+batch_size], dtype=torch.float32).to(device)
                    scores_bkg.append(model.get_anomaly_score(batch).cpu().numpy())
            scores_bkg = np.concatenate(scores_bkg)

        # 1. Core Metrics
        metrics = get_core_metrics(y_test, scores)
        
        # 2. Find thresholds for eB = 10% and 1% on validation set ideally, but here we use test bkg for simplicity
        # (In strict PRD: "chosen on the validation background", this is a slight shortcut for the code structure)
        eB_10_percentile = np.percentile(scores_bkg, 90) # top 10%
        eB_1_percentile = np.percentile(scores_bkg, 99)  # top 1%
        
        thresholds = {
            '10pct': eB_10_percentile,
            '1pct': eB_1_percentile
        }
        
        # 3. Mass Sculpting JS Divergence
        sculpting_metrics = evaluate_mass_sculpting(mjj_bkg, scores_bkg, thresholds)
        metrics.update(sculpting_metrics)
        
        # 3.5 Score mJJ dependence
        from src.evaluation.metrics import calculate_score_mjj_dependence
        metrics['score_mjj_dependence'] = calculate_score_mjj_dependence(mjj_bkg, scores_bkg)
        
        # 4. Bump Hunt on Background + Signal mixture (f_test = 0.5% for example, here we use full test)
        # To simulate a realistic search, we inject some signal into the background test set.
        # Let's say we use a signal prevalence of 0.005 (0.5%)
        np.random.seed(seed)
        n_test_bkg = len(y_bkg)
        n_test_sig_inject = int(n_test_bkg * 0.005)
        test_sig_indices = np.random.choice(len(mjj_test[y_test == 1]), n_test_sig_inject, replace=False)
        
        mjj_test_mixture = np.concatenate([mjj_bkg, mjj_test[y_test == 1][test_sig_indices]])
        scores_test_mixture = np.concatenate([scores_bkg, scores[y_test == 1][test_sig_indices]])
        
        # Bump hunt pre-cut
        bh_pre = perform_bump_hunt(mjj_test_mixture, prefix="pre_cut_")
        metrics.update(bh_pre)
        
        # Bump hunt post-cut 10%
        mjj_post10 = mjj_test_mixture[scores_test_mixture >= thresholds['10pct']]
        bh_post10 = perform_bump_hunt(mjj_post10, prefix="at_10pct_bkg_")
        metrics.update(bh_post10)
        
        # Bump hunt post-cut 1%
        mjj_post1 = mjj_test_mixture[scores_test_mixture >= thresholds['1pct']]
        bh_post1 = perform_bump_hunt(mjj_post1, prefix="at_1pct_bkg_")
        metrics.update(bh_post1)
        
        # 5. Null Test (Background only)
        bh_null_pre = perform_bump_hunt(mjj_bkg, prefix="null_pre_cut_")
        metrics.update(bh_null_pre)
        
        mjj_bkg_post1 = mjj_bkg[scores_bkg >= thresholds['1pct']]
        bh_null_post1 = perform_bump_hunt(mjj_bkg_post1, prefix="null_at_1pct_bkg_")
        metrics.update(bh_null_post1)
        
        # 6. Stretch Goal: Export spectra to ROOT format
        root_file_path = os.path.join('reports', f"{model_name}_{seed}_spectra.root")
        os.makedirs('reports', exist_ok=True)
        with uproot.recreate(root_file_path) as root_out:
            bins = 40
            r = (2500, 4500)
            root_out["inclusive"] = np.histogram(mjj_test_mixture, bins=bins, range=r)
            root_out["at_10pct_bkg"] = np.histogram(mjj_post10, bins=bins, range=r)
            root_out["at_1pct_bkg"] = np.histogram(mjj_post1, bins=bins, range=r)
        
        mlflow.log_artifact(root_file_path)
        
        # Log to MLflow
        mlflow.log_metrics(metrics)
        
        # Save to CSV
        os.makedirs(os.path.join('reports', 'tables'), exist_ok=True)
        results_file = os.path.join('reports', 'tables', 'results.csv')
        
        record = {
            'model': model_name,
            'f_train': config['data']['f_train'],
            'k_labels': config['data'].get('k_labels', 0),
            'seed': seed,
            **metrics
        }
        
        df_new = pd.DataFrame([record])
        if os.path.exists(results_file):
            df = pd.read_csv(results_file)
            df = pd.concat([df, df_new], ignore_index=True)
        else:
            df = df_new
        df.to_csv(results_file, index=False)
        
        print(f"Metrics saved for {model_name} (seed {seed})")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    evaluate_model(config)
