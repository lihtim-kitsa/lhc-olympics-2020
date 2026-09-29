import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

class CWoLaTreesBaseline:
    def __init__(self, random_state=42):
        self.clf = HistGradientBoostingClassifier(
            max_iter=100, 
            learning_rate=0.1, 
            random_state=random_state, 
            early_stopping=True, 
            validation_fraction=0.1
        )
        
    def fit(self, X_scaled, y_cwola):
        """
        X_scaled: scaled features (N, 5)
        y_cwola: 1 for SR, 0 for SB
        """
        self.clf.fit(X_scaled[:, :5], y_cwola)

    def get_anomaly_score(self, X):
        if hasattr(X, 'cpu'):
            X = X.detach().cpu().numpy()
        # Return probability of being in the SR
        return self.clf.predict_proba(X[:, :5])[:, 1]
