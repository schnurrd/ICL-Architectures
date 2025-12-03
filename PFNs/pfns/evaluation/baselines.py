"""
Baseline Models: Random Forest and XGBoost
"""

from __future__ import annotations
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


class RandomForestBaseline:
    """Random Forest baseline with sensible defaults."""
    
    name = "RandomForest"
    
    def __init__(self, n_estimators: int = 500, n_jobs: int = 4, random_state: int = 42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=n_jobs,
            random_state=random_state,
        )
        self.classes_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)


class XGBoostBaseline:
    """XGBoost baseline with default params."""
    
    name = "XGBoost"
    
    def __init__(self, n_estimators: int = 500, n_jobs: int = 4, random_state: int = 42):        
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.model = None
        self.classes_ = None
        self._label_map = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        # Remap labels to contiguous 0-indexed integers for XGBoost
        self.classes_ = np.unique(y)
        self._label_map = {c: i for i, c in enumerate(self.classes_)}
        y_mapped = np.array([self._label_map[c] for c in y])
        
        self.model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbosity=0,
            eval_metric="logloss",
        )
        self.model.fit(X, y_mapped)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        y_pred = self.model.predict(X)
        return np.array([self.classes_[i] for i in y_pred])
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)


def get_baselines(n_jobs: int = 4, random_state: int = 42):
    """Get baselines: Random Forest and XGBoost."""
    return [
        RandomForestBaseline(n_jobs=n_jobs, random_state=random_state),
        XGBoostBaseline(n_jobs=n_jobs, random_state=random_state),
    ]
