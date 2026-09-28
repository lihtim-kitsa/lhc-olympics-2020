from sklearn.ensemble import IsolationForest
import numpy as np

class IsolationForestAnomalyDetector:
    def __init__(self, n_estimators=100, contamination='auto', random_state=42):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1
        )
        
    def fit(self, X):
        self.model.fit(X)
        
    def get_anomaly_score(self, X):
        """
        scikit-learn IF returns anomaly score as negative for anomalies.
        Decision function: lower is more anomalous. 
        We want a score where higher = more anomalous.
        So we negate the decision_function.
        """
        score = -self.model.decision_function(X)
        return score
