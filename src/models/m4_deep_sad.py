import torch
import torch.nn as nn
from .m3_deep_svdd import DeepSVDD, DeepSVDDNet

class DeepSAD(DeepSVDD):
    def __init__(self, input_dim=5, hidden_dims=[64, 32], latent_dim=16, eta=1.0):
        super().__init__(input_dim, hidden_dims, latent_dim)
        self.eta = eta # Weight for the labeled term
        
    def sad_loss(self, outputs, labels):
        """
        Deep SAD loss:
        For normal samples (label 0): minimize distance to c
        For anomaly samples (label 1): maximize distance to c (minimize inverse distance)
        Wait, in Deep SAD, normal is y=1, anomaly is y=-1, unlabeled is y=0.
        Here we'll use:
        unlabeled = 0 (treat as normal)
        labeled anomaly = 1
        labeled normal = 2 (if any, typically we just have unlabeled background and few labeled anomalies)
        
        So:
        if label == 0 (unlabeled background): loss = dist
        if label == 1 (labeled anomaly): loss = eta * (dist ** -1) # or margin based
        """
        dist = torch.sum((outputs - self.c) ** 2, dim=1)
        
        # Deep SAD formulation:
        # L = 1/n sum_{unlabeled} dist + eta/m sum_{labeled} loss_labeled
        # If label is -1 (anomaly): eta * ((dist + eps) ** -1)
        # Using labels: 0=unlabeled, 1=anomaly
        
        eps = 1e-6
        unlabeled_mask = (labels == 0)
        anomaly_mask = (labels == 1)
        
        loss_u = dist[unlabeled_mask].mean() if unlabeled_mask.sum() > 0 else 0.0
        loss_a = (self.eta / (dist[anomaly_mask] + eps)).mean() if anomaly_mask.sum() > 0 else 0.0
        
        return loss_u + loss_a

