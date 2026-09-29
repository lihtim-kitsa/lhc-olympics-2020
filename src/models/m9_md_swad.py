import torch
import torch.nn as nn

def distance_covariance(x, y):
    """
    Compute the distance covariance between two batches of tensors x and y.
    x: (N, d1)
    y: (N, d2)
    """
    n = x.shape[0]
    if n == 1:
        return torch.tensor(0.0, device=x.device)
        
    x_dist = torch.cdist(x, x)
    y_dist = torch.cdist(y, y)
    
    x_mean_row = x_dist.mean(dim=1, keepdim=True)
    x_mean_col = x_dist.mean(dim=0, keepdim=True)
    x_mean_all = x_dist.mean()
    
    y_mean_row = y_dist.mean(dim=1, keepdim=True)
    y_mean_col = y_dist.mean(dim=0, keepdim=True)
    y_mean_all = y_dist.mean()
    
    # Doubly centered distance matrices
    A = x_dist - x_mean_row - x_mean_col + x_mean_all
    B = y_dist - y_mean_row - y_mean_col + y_mean_all
    
    dcov2 = (A * B).sum() / (n * n)
    
    return torch.clamp(dcov2, min=0.0)

def distance_correlation(x, y):
    """
    Compute the distance correlation between two batches of tensors x and y.
    """
    dcov2_xy = distance_covariance(x, y)
    dcov2_xx = distance_covariance(x, x)
    dcov2_yy = distance_covariance(y, y)
    
    if dcov2_xx == 0 or dcov2_yy == 0:
        return torch.tensor(0.0, device=x.device)
        
    return torch.sqrt(dcov2_xy / torch.sqrt(dcov2_xx * dcov2_yy))

class BiasFreeLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        
    def forward(self, x):
        return self.linear(x)

class MDSWADNet(nn.Module):
    """
    Mass-Decorrelated Sliced Wasserstein Anomaly Detector (MD-SWAD)
    Uses a standard MLP architecture but without bias restrictions if target distribution handles translation.
    """
    def __init__(self, input_dim=5, hidden_dims=[64, 32], latent_dim=16):
        super().__init__()
        
        layers = []
        in_d = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_d, h))
            layers.append(nn.ReLU())
            in_d = h
            
        layers.append(nn.Linear(in_d, latent_dim))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class MDSWAD(nn.Module):
    def __init__(self, input_dim=5, hidden_dims=[64, 32], latent_dim=16, lam=1.0):
        super().__init__()
        self.net = MDSWADNet(input_dim, hidden_dims, latent_dim)
        self.lam = lam  # Mass decorrelation weight
        
    def sliced_wasserstein_distance(self, z, num_projections=50):
        """
        Compute the Sliced Wasserstein Distance to a target standard normal distribution.
        z: (N, latent_dim)
        """
        n, d = z.shape
        device = z.device
        
        # Target samples from standard normal
        target = torch.randn_like(z)
        
        # Random projections on unit sphere
        projections = torch.randn(d, num_projections, device=device)
        projections = projections / torch.norm(projections, dim=0, keepdim=True)
        
        # Project data and target
        z_proj = torch.matmul(z, projections)  # (N, num_projections)
        target_proj = torch.matmul(target, projections)  # (N, num_projections)
        
        # Sort to compute 1D Wasserstein distance
        z_proj_sorted, _ = torch.sort(z_proj, dim=0)
        target_proj_sorted, _ = torch.sort(target_proj, dim=0)
        
        # W2 distance on slices
        w2_distances = torch.mean((z_proj_sorted - target_proj_sorted)**2, dim=0)
        
        return torch.mean(w2_distances)

    def sad_loss(self, z_s, eta=1.0, eps=1e-6):
        """
        Semi-supervised repulsion loss for labeled anomalies.
        z_s: (N, latent_dim) for signal events.
        """
        # Repulse from the origin (which is the center of the normal distribution)
        dist = torch.sum(z_s ** 2, dim=1)
        return eta * torch.mean(1.0 / (dist + eps))

    def get_anomaly_score(self, x):
        """
        Anomaly score is the Euclidean norm in latent space (since target is N(0,I))
        or we could compute the SWD of the single point.
        Norm is the maximum likelihood estimator for distance to N(0,I).
        """
        self.net.eval()
        with torch.no_grad():
            outputs = self.net(x)
            dist = torch.sum(outputs ** 2, dim=1)
        return dist
