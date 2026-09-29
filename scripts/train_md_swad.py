"""Train MD-SWAD detector with mass decorrelation."""
import argparse
import os
import pickle
import numpy as np
import torch
import torch.optim as optim
import yaml
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.models.m9_md_swad import MDSWAD, distance_correlation
import h5py

def load_data_with_mass(split_dir, features_file, config):
    seed = int(config['training']['seed'])
    rng = np.random.default_rng(seed)
    
    bkg_idx = np.load(os.path.join(split_dir, 'background_train.npy'))
    bkg_val_idx = np.load(os.path.join(split_dir, 'background_val.npy'))
    sig_train_idx = np.load(os.path.join(split_dir, 'signal2_train.npy'))
    
    f_train = float(config['data'].get('f_train', 0.0))
    k_labels = int(config['data'].get('k_labels', 0))
    
    n_injected = int(len(bkg_idx) * f_train / 100.0)
    perm = rng.permutation(sig_train_idx)
    injected_idx = perm[:n_injected]
    labeled_idx = perm[n_injected:n_injected + k_labels]
    
    unlabeled_idx = np.concatenate([bkg_idx, injected_idx])
    
    cols = [0, 1, 2, 3, 4]
    mass_col = 5
    
    with h5py.File(features_file, 'r') as f:
        all_features = f['features'][:]
        
    X_unscaled = all_features[unlabeled_idx][:, cols]
    mjj_train = all_features[unlabeled_idx][:, mass_col]
    
    X_labeled_unscaled = all_features[labeled_idx][:, cols]
    
    X_val_unscaled = all_features[bkg_val_idx][:, cols]
    mjj_val = all_features[bkg_val_idx][:, mass_col]
    
    # Fit scaler ONLY on background
    scaler = StandardScaler().fit(all_features[bkg_idx][:, cols])
    
    X_train = scaler.transform(X_unscaled).astype(np.float32)
    X_labeled = scaler.transform(X_labeled_unscaled).astype(np.float32) if len(labeled_idx) > 0 else np.empty((0, 5), dtype=np.float32)
    
    X_val = scaler.transform(X_val_unscaled).astype(np.float32)
    
    return X_train, mjj_train, X_labeled, X_val, mjj_val, scaler

def train_md_swad(config):
    split_dir = os.path.join('data', 'splits')
    features_file = os.path.join('data', 'processed', 'events_v2_features.h5')
    
    print("Loading data...")
    X_train, mjj_train, X_labeled, X_val, mjj_val, scaler = load_data_with_mass(split_dir, features_file, config)
    
    seed = int(config['training']['seed'])
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    epochs = int(config['training'].get('epochs', 50))
    batch = int(config['training'].get('batch_size', 256))
    lr = float(config['training'].get('learning_rate', 1e-3))
    wd = float(config['training'].get('weight_decay', 1e-5))
    lam = float(config['model'].get('lam', 1.0))
    
    ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(mjj_train.astype(np.float32)))
    loader = DataLoader(ds, batch_size=batch, shuffle=True, generator=torch.Generator().manual_seed(seed))
    
    Xv = torch.from_numpy(X_val).to(device)
    Mv = torch.from_numpy(mjj_val.astype(np.float32)).to(device)
    
    model = MDSWAD(input_dim=5, lam=lam).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    
    os.makedirs('models', exist_ok=True)
    f_train = float(config['data'].get('f_train', 0.0))
    k_labels = int(config['data'].get('k_labels', 0))
    variant = f"f{f_train:g}_k{k_labels}"
    model_path = os.path.join('models', f"M9_MDSWAD_{variant}_{seed}.pt")
    scaler_path = os.path.join('models', f"scaler_M9_MDSWAD_{variant}_{seed}.pkl")
    with open(scaler_path, 'wb') as out:
        pickle.dump(scaler, out)
        
    best = float('inf')
    best_state = None
    patience = int(config['training'].get('patience', 8))
    stale = 0
    
    has_labeled = len(X_labeled) > 0
    if has_labeled:
        X_labeled_tensor = torch.from_numpy(X_labeled).to(device)
        
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        losses = []
        for xb, mb in loader:
            xb, mb = xb.to(device), mb.to(device)
            optimizer.zero_grad()
            
            z = model.net(xb)
            swd_loss = model.sliced_wasserstein_distance(z)
            disco_loss = distance_correlation(z, mb.unsqueeze(-1))
            
            loss = swd_loss + model.lam * disco_loss
            
            # Semi-supervised SAD loss on labeled anomalies
            if has_labeled:
                z_labeled = model.net(X_labeled_tensor)
                loss += model.sad_loss(z_labeled, eta=1.0)
                
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
        model.eval()
        val_losses = []
        val_swds = []
        val_discos = []
        val_ds = TensorDataset(Xv, Mv.unsqueeze(-1))
        val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False)
        
        with torch.no_grad():
            for xvb, mvb in val_loader:
                zv = model.net(xvb)
                v_swd = model.sliced_wasserstein_distance(zv)
                v_disco = distance_correlation(zv, mvb)
                v_loss = v_swd + model.lam * v_disco
                
                val_losses.append(v_loss.item())
                val_swds.append(v_swd.item())
                val_discos.append(v_disco.item())
                
        val_loss = np.mean(val_losses)
        val_swd_mean = np.mean(val_swds)
        val_disco_mean = np.mean(val_discos)
            
        print(f"Epoch {epoch}: Train Loss {np.mean(losses):.4f}, Val Loss {val_loss:.4f} (SWD: {val_swd_mean:.4f}, DisCo: {val_disco_mean:.4f})")
        
        if val_loss < best:
            best = val_loss
            stale = 0
            best_state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            stale += 1
            
        if stale >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
            
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save({'model_state_dict': model.state_dict(), 'best_validation_objective': best}, model_path)
    print(f"Saved model to {model_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    with open(args.config, encoding='utf-8') as f: 
        config = yaml.safe_load(f)
    train_md_swad(config)
