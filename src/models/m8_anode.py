import numpy as np
from sklearn.mixture import GaussianMixture

class ANODEBaseline:
    def __init__(self, n_components=5):
        self.bkg_gmm = GaussianMixture(n_components=n_components, random_state=42)
        self.sr_mask = None
        
    def fit(self, X_scaled, mjj_unscaled):
        """
        Expects scaled features, but unscaled mJJ to define the sidebands.
        """
        sb_mask = ((mjj_unscaled >= 2500) & (mjj_unscaled < 3300)) | ((mjj_unscaled > 3700) & (mjj_unscaled <= 4500))
        sb_features = X_scaled[sb_mask][:, :5]
        self.bkg_gmm.fit(sb_features)

    def get_anomaly_score(self, X):
        if hasattr(X, 'cpu'):
            X = X.detach().cpu().numpy()
        scores = -self.bkg_gmm.score_samples(X[:, :5])
        return scores
