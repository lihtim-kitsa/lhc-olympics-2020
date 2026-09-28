import torch
import torch.nn as nn
import numpy as np

class BiasFreeLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # Deep SVDD requires bias-free networks to prevent hypersphere collapse
        self.linear = nn.Linear(in_features, out_features, bias=False)
        
    def forward(self, x):
        return self.linear(x)

class DeepSVDDNet(nn.Module):
    def __init__(self, input_dim=5, hidden_dims=[64, 32], latent_dim=16):
        super().__init__()
        
        layers = []
        in_d = input_dim
        for h in hidden_dims:
            layers.append(BiasFreeLinear(in_d, h))
            layers.append(nn.LeakyReLU(0.1)) # LeakyReLU works better without biases
            in_d = h
            
        layers.append(BiasFreeLinear(in_d, latent_dim))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class DeepSVDD(nn.Module):
    def __init__(self, input_dim=5, hidden_dims=[64, 32], latent_dim=16):
        super().__init__()
        self.net = DeepSVDDNet(input_dim, hidden_dims, latent_dim)
        self.c = None # Hypersphere center
        
    def init_center(self, dataloader, eps=0.1, device='cpu'):
        """
        Initialize hypersphere center c as the mean from an initial forward pass on the data.
        eps prevents a zero center.
        """
        n_samples = 0
        c = torch.zeros(self.net.network[-1].linear.out_features, device=device)

        self.net.eval()
        with torch.no_grad():
            for x, _ in dataloader:
                x = x.to(device)
                outputs = self.net(x)
                n_samples += outputs.shape[0]
                c += torch.sum(outputs, dim=0)
                
        c /= n_samples
        
        # If c is too close to zero, set to eps
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps
        
        self.c = c
        
    def get_anomaly_score(self, x):
        """
        Anomaly score is the distance to the center.
        """
        self.net.eval()
        with torch.no_grad():
            outputs = self.net(x)
            dist = torch.sum((outputs - self.c) ** 2, dim=1)
        return dist
