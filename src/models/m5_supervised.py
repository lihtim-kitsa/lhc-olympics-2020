import torch
import torch.nn as nn

class SupervisedMLP(nn.Module):
    def __init__(self, input_dim=5, hidden_dims=[64, 32]):
        super().__init__()
        
        layers = []
        in_d = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_d, h))
            layers.append(nn.ReLU())
            in_d = h
            
        layers.append(nn.Linear(in_d, 1))
        # Sigmoid for probability, or just use BCEWithLogitsLoss
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

    def get_anomaly_score(self, x):
        """
        Anomaly score is the predicted probability of being an anomaly (signal).
        """
        self.eval()
        with torch.no_grad():
            logits = self(x)
            probs = torch.sigmoid(logits)
        return probs.squeeze(-1)
