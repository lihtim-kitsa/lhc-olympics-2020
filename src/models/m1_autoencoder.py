import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim=5, hidden_dims=[64, 32], latent_dim=16):
        super().__init__()
        
        # Encoder
        enc_layers = []
        in_d = input_dim
        for h in hidden_dims:
            enc_layers.append(nn.Linear(in_d, h))
            enc_layers.append(nn.LeakyReLU(0.1))
            in_d = h
        enc_layers.append(nn.Linear(in_d, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)
        
        # Decoder
        dec_layers = []
        in_d = latent_dim
        for h in reversed(hidden_dims):
            dec_layers.append(nn.Linear(in_d, h))
            dec_layers.append(nn.LeakyReLU(0.1))
            in_d = h
        dec_layers.append(nn.Linear(in_d, input_dim))
        self.decoder = nn.Sequential(*dec_layers)
        
    def forward(self, x):
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return x_rec

    def get_anomaly_score(self, x):
        """
        Returns reconstruction error as anomaly score
        """
        self.eval()
        with torch.no_grad():
            x_rec = self(x)
            # MSE per event
            score = torch.mean((x - x_rec)**2, dim=1)
        return score
