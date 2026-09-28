import numpy as np

class TauCutBaseline:
    def __init__(self, **kwargs):
        pass

    def fit(self, X):
        pass

    def get_anomaly_score(self, x):
        """
        Anomaly score based on tau_21.
        Small tau_21 indicates a multi-prong structure (anomaly).
        So we score as -(tau21_1 + tau21_2).
        Features are scaled, but linear scaling preserves the sum-based ordering.
        tau21_1 is index 2, tau21_2 is index 3.
        """
        if hasattr(x, 'cpu'):
            x = x.detach().cpu().numpy()
        return -(x[:, 2] + x[:, 3])
