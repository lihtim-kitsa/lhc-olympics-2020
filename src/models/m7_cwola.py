import torch
import torch.nn as nn
import numpy as np

class CWoLaClassifier(nn.Module):
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
        # x may contain mJJ as the 6th feature, we only want the first 5
        return self.net(x[:, :5])
        
    def get_anomaly_score(self, x):
        self.eval()
        with torch.no_grad():
            outputs = self(x)
        return torch.sigmoid(outputs).squeeze(-1)
