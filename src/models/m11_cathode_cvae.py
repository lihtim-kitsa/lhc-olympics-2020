import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

class CVAE(nn.Module):
    def __init__(self, input_dim=5, cond_dim=1, latent_dim=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + cond_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )
        
    def encode(self, x, c):
        h = self.encoder(torch.cat([x, c], dim=1))
        return self.fc_mu(h), self.fc_logvar(h)
        
    def decode(self, z, c):
        return self.decoder(torch.cat([z, c], dim=1))
        
    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return self.decode(z, c), mu, logvar

class Classifier(nn.Module):
    def __init__(self, input_dim=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

class CATHODEBaseline:
    def __init__(self, random_state=42, epochs_cvae=5, epochs_clf=5):
        self.cvae = CVAE()
        self.clf = Classifier()
        self.epochs_cvae = epochs_cvae
        self.epochs_clf = epochs_clf
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cvae.to(self.device)
        self.clf.to(self.device)
        
    def fit(self, X_scaled, mjj_unscaled):
        # 1. Train CVAE on Sidebands
        sb_mask = ((mjj_unscaled >= 2500) & (mjj_unscaled < 3300)) | ((mjj_unscaled > 3700) & (mjj_unscaled <= 4500))
        sr_mask = (mjj_unscaled >= 3300) & (mjj_unscaled <= 3700)
        
        X_sb = torch.tensor(X_scaled[sb_mask][:, :5], dtype=torch.float32, device=self.device)
        m_sb = torch.tensor(X_scaled[sb_mask][:, 5:6], dtype=torch.float32, device=self.device) # use scaled mjj as condition
        
        optimizer = optim.Adam(self.cvae.parameters(), lr=1e-3)
        ds_sb = TensorDataset(X_sb, m_sb)
        loader_sb = DataLoader(ds_sb, batch_size=256, shuffle=True)
        
        for _ in range(self.epochs_cvae):
            for x, c in loader_sb:
                optimizer.zero_grad()
                recon, mu, logvar = self.cvae(x, c)
                recon_loss = nn.functional.mse_loss(recon, x, reduction='sum')
                kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = (recon_loss + kld_loss) / x.size(0)
                loss.backward()
                optimizer.step()
                
        # 2. Generate Pseudo-Background in SR
        X_sr_real = torch.tensor(X_scaled[sr_mask][:, :5], dtype=torch.float32, device=self.device)
        m_sr_real = torch.tensor(X_scaled[sr_mask][:, 5:6], dtype=torch.float32, device=self.device)
        
        self.cvae.eval()
        with torch.no_grad():
            z = torch.randn(m_sr_real.size(0), 4, device=self.device)
            X_sr_fake = self.cvae.decode(z, m_sr_real)
            
        # 3. Train Classifier (1 for real SR data, 0 for fake SR background)
        X_clf = torch.cat([X_sr_real, X_sr_fake], dim=0)
        y_clf = torch.cat([torch.ones(X_sr_real.size(0)), torch.zeros(X_sr_fake.size(0))], dim=0).unsqueeze(1).to(self.device)
        
        ds_clf = TensorDataset(X_clf, y_clf)
        loader_clf = DataLoader(ds_clf, batch_size=256, shuffle=True)
        optimizer_clf = optim.Adam(self.clf.parameters(), lr=1e-3)
        criterion = nn.BCEWithLogitsLoss()
        
        for _ in range(self.epochs_clf):
            for x, y in loader_clf:
                optimizer_clf.zero_grad()
                loss = criterion(self.clf(x), y)
                loss.backward()
                optimizer_clf.step()

    def get_anomaly_score(self, X):
        if not torch.is_tensor(X):
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
        else:
            X = X.to(self.device)
        self.clf.eval()
        with torch.no_grad():
            out = self.clf(X[:, :5])
        return torch.sigmoid(out).squeeze(-1).cpu().numpy()
