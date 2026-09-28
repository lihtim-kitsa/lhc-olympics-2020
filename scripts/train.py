"""Train one configured LHCO detector with training-only scaling and validation stopping.
f_train: Percentage (e.g. 0.0 to 100.0) of the background training set size to inject as signal.
"""
import argparse
import os
import hashlib
import json
import pickle
import sys

import h5py
import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.models.m1_autoencoder import Autoencoder
from src.models.m2_isolation_forest import IsolationForestAnomalyDetector
from src.models.m3_deep_svdd import DeepSVDD
from src.models.m4_deep_sad import DeepSAD
from src.models.m5_supervised import SupervisedMLP


def _read_rows(dset, indices, cols):
    """Read unique rows through h5py's sorted-index interface, preserving order."""
    indices = np.asarray(indices, dtype=np.int64)
    if not len(indices):
        return np.empty((0, len(cols)), dtype=np.float32)
    order = np.argsort(indices)
    sorted_rows = dset[indices[order], :]
    out = np.empty((len(indices), len(cols)), dtype=np.float32)
    out[order] = sorted_rows[:, cols]
    return out


def load_data(split_dir, features_file, config):
    data = config['data']
    seed = int(config['training']['seed'])
    rng = np.random.default_rng(seed)
    model_name = config['model']['name']
    
    use_3p = config['data'].get('use_3prong', False)
    sig_train_file = 'signal3_train.npy' if use_3p else 'signal2_train.npy'
    sig_val_file = 'signal3_val.npy' if use_3p else 'signal2_val.npy'
    
    bkg_idx = np.load(os.path.join(split_dir, 'background_train.npy'))
    sig_idx = np.load(os.path.join(split_dir, sig_train_file))
    bkg_val_idx = np.load(os.path.join(split_dir, 'background_val.npy'))
    sig_val_idx = np.load(os.path.join(split_dir, sig_val_file))
    cols = [0, 1, 2, 3, 4, 5] if data.get('use_mjj', False) else [0, 1, 2, 3, 4]

    if model_name == 'M5_Supervised':
        train_idx = np.concatenate([bkg_idx, sig_idx])
        y_train = np.concatenate([np.zeros(len(bkg_idx)), np.ones(len(sig_idx))]).astype(np.float32)
        unlabeled_idx = train_idx
        labeled_idx = np.array([], dtype=np.int64)
    else:
        n_injected = int(len(bkg_idx) * float(data.get('f_train', 0.0)) / 100.0)
        k = int(data.get('k_labels', 0)) if model_name == 'M4_DeepSAD' else 0
        if n_injected + k > len(sig_idx):
            raise ValueError(f"Need {n_injected + k} distinct signal train events, have {len(sig_idx)}")
        perm = rng.permutation(sig_idx)
        injected_idx, labeled_idx = perm[:n_injected], perm[n_injected:n_injected + k]
        unlabeled_idx = np.concatenate([bkg_idx, injected_idx])
        y_train = np.zeros(len(unlabeled_idx), dtype=np.float32)

    val_idx = np.concatenate([bkg_val_idx, sig_val_idx]) if model_name == 'M5_Supervised' else bkg_val_idx
    y_val = (np.isin(val_idx, sig_val_idx)).astype(np.float32) if model_name == 'M5_Supervised' else np.zeros(len(val_idx), dtype=np.float32)
    with h5py.File(features_file, 'r') as f:
        all_features=f['features'][:]
    X_unscaled=all_features[unlabeled_idx][:,cols]
    X_val_unscaled=all_features[val_idx][:,cols]
    X_labeled_unscaled=all_features[labeled_idx][:,cols]
    if model_name == 'M5_Supervised':
        scaler_fit = X_unscaled
    else:
        # A common background-only transform keeps feature statistics independent of injected signals.
        scaler_fit=all_features[bkg_idx][:,cols]
    scaler = StandardScaler().fit(scaler_fit)
    X_u = scaler.transform(X_unscaled).astype(np.float32)
    X_val = scaler.transform(X_val_unscaled).astype(np.float32)
    X_l = scaler.transform(X_labeled_unscaled).astype(np.float32) if len(X_labeled_unscaled) else X_labeled_unscaled.astype(np.float32)
    if not all(np.isfinite(a).all() for a in (X_u, X_val, X_l)):
        raise ValueError('Non-finite detector inputs; rebuild features and exclude invalid rows in the saved splits.')
    y_u = y_train
    y_l = np.ones(len(X_l), dtype=np.float32)
    return X_u, y_u, X_l, y_l, X_val, y_val, scaler, unlabeled_idx, labeled_idx


def train_model(config):
    mlflow.set_tracking_uri('sqlite:///mlflow_reproduction.db')
    mlflow.set_experiment(config['mlflow']['experiment_name'])
    with mlflow.start_run():
        mlflow.log_params(config['training'])
        mlflow.log_params(config['data'])
        mlflow.log_param('model', config['model']['name'])
        split_dir = os.path.join('data', 'splits')
        features_file = os.path.join('data', 'processed', 'events_v2_features.h5')
        X_u, y_u, X_l, y_l, X_val, y_val, scaler, unlabeled_idx, labeled_idx = load_data(split_dir, features_file, config)
        seed = int(config['training']['seed'])
        torch.manual_seed(seed)
        np.random.seed(seed)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        name = config['model']['name']
        epochs = int(config['training'].get('epochs', 50))
        batch = int(config['training'].get('batch_size', 256))
        lr = float(config['training'].get('learning_rate', 1e-3))
        wd = float(config['training'].get('weight_decay', 1e-5))
        os.makedirs('models', exist_ok=True)
        variant=f"f{float(config['data'].get('f_train',0)):g}_k{int(config['data'].get('k_labels',0))}"
        model_path = os.path.join('models', f'{name}_{variant}_{seed}.pt')
        scaler_path = os.path.join('models', f'scaler_{name}_{variant}_{seed}.pkl')
        with open(scaler_path, 'wb') as out:
            pickle.dump(scaler, out)
        mlflow.log_param('unlabeled_train_count', len(unlabeled_idx))
        mlflow.log_param('labeled_signal_count', len(labeled_idx) if name == 'M4_DeepSAD' else int(y_u.sum()) if name == 'M5_Supervised' else 0)
        if name == 'M5_Supervised':
            # All signal training examples carry labels in the supervised reference.
            use_3p = config['data'].get('use_3prong', False)
            sig_train_file = 'signal3_train.npy' if use_3p else 'signal2_train.npy'
            labeled_signal_idx=np.load(os.path.join(split_dir, sig_train_file))
            injected_signal_idx=np.empty(0,dtype=np.int64)
        else:
            labeled_signal_idx=labeled_idx
            train_bkg=np.load(os.path.join(split_dir,'background_train.npy'))
            injected_signal_idx=np.setdiff1d(unlabeled_idx,train_bkg,assume_unique=False)
        sample_dir='reports/tables/experiment_samples'; os.makedirs(sample_dir,exist_ok=True)
        sample_path=os.path.join(sample_dir,f'{name}_{variant}_{seed}.npz')
        np.savez_compressed(sample_path,injected_signal_indices=injected_signal_idx,labeled_signal_indices=labeled_signal_idx)
        bkg_hash=hashlib.sha256(np.asarray(np.load(os.path.join(split_dir,'background_train.npy')),dtype=np.int64).tobytes()).hexdigest()
        with open(os.path.join(sample_dir,f'{name}_{variant}_{seed}.json'),'w',encoding='utf-8') as f:
            json.dump({'model':name,'f_train_percent':config['data'].get('f_train',0),'k_labels':config['data'].get('k_labels',0),'seed':seed,'background_train_count':int(len(np.load(os.path.join(split_dir,'background_train.npy')))),'background_train_indices_sha256':bkg_hash,'injected_signal_count':int(len(injected_signal_idx)),'labeled_signal_count':int(len(labeled_signal_idx)),'signal_index_file':os.path.basename(sample_path)},f,indent=2)

from src.models.m0_tau_cut import TauCutBaseline

        if name == 'M2_IsolationForest':
            model = IsolationForestAnomalyDetector(config['model'].get('n_estimators', 100), random_state=seed)
            model.fit(X_u)
            with open(model_path, 'wb') as out:
                pickle.dump(model, out)
        elif name == 'M0_TauCut':
            model = TauCutBaseline()
            with open(model_path, 'wb') as out:
                pickle.dump(model, out)
        elif name == 'M8_ANODE':
            from src.models.m8_anode import ANODEBaseline
            mjj_unscaled = scaler.inverse_transform(X_u)[:, 5]
            model = ANODEBaseline()
            model.fit(X_u, mjj_unscaled)
            with open(model_path, 'wb') as out:
                pickle.dump(model, out)
        else:
            X_train = np.concatenate([X_u, X_l])
            y_train = np.concatenate([y_u, y_l])
            
            if name == 'M7_CWoLa':
                # CWoLa specific logic: redefine y_train based on mJJ regions
                mjj_unscaled = scaler.inverse_transform(X_train)[:, 5]
                # Signal region: 3.3 to 3.7 TeV
                sr_mask = (mjj_unscaled >= 3300) & (mjj_unscaled <= 3700)
                # Sidebands: 2.5-3.3 and 3.7-4.5
                sb_mask = ((mjj_unscaled >= 2500) & (mjj_unscaled < 3300)) | ((mjj_unscaled > 3700) & (mjj_unscaled <= 4500))
                
                valid_mask = sr_mask | sb_mask
                X_train = X_train[valid_mask]
                # Label SR as 1, SB as 0
                y_train = sr_mask[valid_mask].astype(np.float32)

            ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
            loader = DataLoader(ds, batch_size=batch, shuffle=True, generator=torch.Generator().manual_seed(seed))
            input_dim = X_train.shape[1]
            if name == 'M7_CWoLa':
                from src.models.m7_cwola import CWoLaClassifier
                model = CWoLaClassifier(input_dim=5).to(device)
            elif name == 'M1_Autoencoder': model = Autoencoder(input_dim).to(device)
            elif name in ('M3_DeepSVDD', 'M6_MassAware'): model = DeepSVDD(input_dim).to(device)
            elif name == 'M4_DeepSAD': model = DeepSAD(input_dim, eta=config['model'].get('eta', 1.0)).to(device)
            elif name == 'M5_Supervised': model = SupervisedMLP(input_dim).to(device)
            else: raise ValueError(f'Unknown model: {name}')
            if name in ('M3_DeepSVDD', 'M4_DeepSAD', 'M6_MassAware'):
                center_data=TensorDataset(torch.from_numpy(X_u),torch.from_numpy(y_u))
                center_loader=DataLoader(center_data,batch_size=batch,shuffle=False)
                model.init_center(center_loader, device=device)
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
            criterion = nn.BCEWithLogitsLoss()
            
            if name == 'M7_CWoLa':
                mjj_val_unscaled = scaler.inverse_transform(X_val)[:, 5]
                sr_val = (mjj_val_unscaled >= 3300) & (mjj_val_unscaled <= 3700)
                sb_val = ((mjj_val_unscaled >= 2500) & (mjj_val_unscaled < 3300)) | ((mjj_val_unscaled > 3700) & (mjj_val_unscaled <= 4500))
                val_mask = sr_val | sb_val
                Xv = torch.from_numpy(X_val[val_mask]).to(device)
                yv = torch.from_numpy(sr_val[val_mask].astype(np.float32)).to(device)
            else:
                Xv = torch.from_numpy(X_val).to(device)
                yv = torch.from_numpy(y_val).to(device)
                
            best, best_state, patience = float('inf'), None, int(config['training'].get('patience', 8))
            stale = 0
            for epoch in range(epochs):
                model.train(); losses=[]
                for xb, yb in loader:
                    xb, yb = xb.to(device), yb.to(device)
                    optimizer.zero_grad()
                    if name == 'M1_Autoencoder': loss = ((model(xb) - xb) ** 2).mean()
                    elif name in ('M3_DeepSVDD', 'M6_MassAware'):
                        loss = ((model.net(xb) - model.c) ** 2).sum(dim=1).mean()
                    elif name == 'M4_DeepSAD': loss = model.sad_loss(model.net(xb), yb)
                    else: loss = criterion(model(xb).squeeze(-1), yb)
                    loss.backward(); optimizer.step(); losses.append(float(loss.detach().cpu()))
                model.eval()
                with torch.no_grad():
                    if name == 'M1_Autoencoder': val_loss = ((model(Xv) - Xv) ** 2).mean().item()
                    elif name in ('M3_DeepSVDD', 'M4_DeepSAD', 'M6_MassAware'):
                        val_loss = ((model.net(Xv) - model.c) ** 2).sum(dim=1).mean().item()
                    else: val_loss = criterion(model(Xv).squeeze(-1), yv).item()
                mlflow.log_metric('train_loss', float(np.mean(losses)), step=epoch)
                mlflow.log_metric('validation_objective', val_loss, step=epoch)
                if val_loss < best:
                    best, stale = val_loss, 0
                    best_state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
                else: stale += 1
                if stale >= patience: break
            if best_state is not None: model.load_state_dict(best_state)
            torch.save({'model_state_dict': model.state_dict(), 'center': getattr(model, 'c', None), 'best_validation_objective': best}, model_path)
        mlflow.log_artifact(model_path); mlflow.log_artifact(scaler_path)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--use-3prong', action='store_true')
    args=parser.parse_args()
    with open(args.config,encoding='utf-8') as f: 
        config = yaml.safe_load(f)
    if args.use_3prong:
        config['data']['use_3prong'] = True
    train_model(config)
