import os
import argparse
import yaml
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import mlflow
import sys
import pickle

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
    
    bkg_train_idx = np.load(os.path.join(split_dir, 'background_train.npy'))
    sig_train_idx = np.load(os.path.join(split_dir, 'signal2_train.npy'))
    
    n_bkg = len(bkg_train_idx)
    n_sig_inject = int(n_bkg * (f_train / 100.0))
    
    if n_sig_inject > len(sig_train_idx):
        raise ValueError("Not enough signal train events for requested contamination.")
        
    injected_sig = sig_train_idx[:n_sig_inject]
    remaining_sig = sig_train_idx[n_sig_inject:]
    
    labeled_sig = remaining_sig[:k_labels]
    
    unlabeled_train_idx = np.concatenate([bkg_train_idx, injected_sig])
    np.random.shuffle(unlabeled_train_idx)
    
    with h5py.File(features_file, 'r') as f:
        dset = f['features']
        use_mjj = config['data'].get('use_mjj', False)
        feat_cols = [0, 1, 2, 3, 4, 5] if use_mjj else [0, 1, 2, 3, 4]
        
        # Load efficiently
        X_train_unlabeled = np.zeros((len(unlabeled_train_idx), len(feat_cols)), dtype=np.float32)
        for i, idx in enumerate(unlabeled_train_idx):
            X_train_unlabeled[i] = dset[idx, feat_cols]
            
        y_train_unlabeled = np.zeros(len(X_train_unlabeled), dtype=np.float32)
        
        X_train_labeled = np.zeros((len(labeled_sig), len(feat_cols)), dtype=np.float32)
        for i, idx in enumerate(labeled_sig):
            X_train_labeled[i] = dset[idx, feat_cols]
            
        y_train_labeled = np.ones(len(X_train_labeled), dtype=np.float32)
            
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
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_name = config['model']['name']
        epochs = config['training'].get('epochs', 50)
        batch_size = config['training'].get('batch_size', 256)
        lr = config['training'].get('learning_rate', 1e-3)
        weight_decay = config['training'].get('weight_decay', 1e-5)
        
        os.makedirs('models', exist_ok=True)
        model_path = os.path.join('models', f"{model_name}_{config['training']['seed']}.pt")
        scaler_path = os.path.join('models', f"scaler_{model_name}_{config['training']['seed']}.pkl")
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        if model_name == 'M2_IsolationForest':
            model = IsolationForestAnomalyDetector(
                n_estimators=config['model'].get('n_estimators', 100),
                random_state=config['training']['seed']
            )
            model.fit(X_u)
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            mlflow.log_artifact(model_path)
            mlflow.log_artifact(scaler_path)
            return

        # Prepare dataloader for PyTorch models
        X_train_t = torch.tensor(np.vstack([X_u, X_l]), dtype=torch.float32)
        y_train_t = torch.tensor(np.concatenate([y_u, y_l]), dtype=torch.float32)
        dataset = TensorDataset(X_train_t, y_train_t)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        input_dim = X_train_t.shape[1]
        
        if model_name == 'M1_Autoencoder':
            model = Autoencoder(input_dim=input_dim).to(device)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            
            for epoch in range(epochs):
                model.train()
                total_loss = 0
                for X_batch, _ in dataloader:
                    X_batch = X_batch.to(device)
                    optimizer.zero_grad()
                    x_rec = model(X_batch)
                    loss = criterion(x_rec, X_batch)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                mlflow.log_metric('train_loss', total_loss / len(dataloader), step=epoch)
                
        elif model_name == 'M3_DeepSVDD' or model_name == 'M6_MassAware':
            model = DeepSVDD(input_dim=input_dim).to(device)
            model.init_center(dataloader, device=device)
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            
            for epoch in range(epochs):
                model.train()
                total_loss = 0
                for X_batch, _ in dataloader:
                    X_batch = X_batch.to(device)
                    optimizer.zero_grad()
                    outputs = model(X_batch)
                    dist = torch.sum((outputs - model.c) ** 2, dim=1)
                    loss = torch.mean(dist)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                mlflow.log_metric('train_loss', total_loss / len(dataloader), step=epoch)
                
        elif model_name == 'M4_DeepSAD':
            eta = config['model'].get('eta', 1.0)
            model = DeepSAD(input_dim=input_dim, eta=eta).to(device)
            model.init_center(dataloader, device=device)
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            
            for epoch in range(epochs):
                model.train()
                total_loss = 0
                for X_batch, y_batch in dataloader:
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)
                    optimizer.zero_grad()
                    outputs = model(X_batch)
                    loss = model.sad_loss(outputs, y_batch)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                mlflow.log_metric('train_loss', total_loss / len(dataloader), step=epoch)
                
        elif model_name == 'M5_Supervised':
            model = SupervisedMLP(input_dim=input_dim).to(device)
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            
            for epoch in range(epochs):
                model.train()
                total_loss = 0
                for X_batch, y_batch in dataloader:
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)
                    optimizer.zero_grad()
                    outputs = model(X_batch).squeeze(-1)
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                mlflow.log_metric('train_loss', total_loss / len(dataloader), step=epoch)
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'center': getattr(model, 'c', None)
        }, model_path)
        
        mlflow.log_artifact(model_path)
        mlflow.log_artifact(scaler_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    train_model(config)
