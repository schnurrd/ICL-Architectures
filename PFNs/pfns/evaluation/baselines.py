"""
Baseline Models: Random Forest, XGBoost, and CatBoost
Ordinal-encodes categorical columns via ColumnTransformer for all models.
"""

from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


def _cat_list(categorical_feats) -> list[int]:
    return [int(i) for i in categorical_feats] if categorical_feats else []


def _encode_labels(y):
    classes, y_mapped = np.unique(y, return_inverse=True)
    return classes, y_mapped.astype(np.int64)


def _preprocessor(cat: list[int]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat),
        ],
        remainder="passthrough",
    )


class RandomForestBaseline:
    name = "RandomForest"

    def __init__(self, n_estimators: int = 500, n_jobs: int = 4, random_state: int = 42):
        self.n_estimators, self.n_jobs, self.random_state = n_estimators, n_jobs, random_state
        self.model: Pipeline | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, categorical_feats=None):
        cat = _cat_list(categorical_feats)
        self.model = Pipeline(
            steps=[
                ("preprocess", _preprocessor(cat)),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=self.n_estimators,
                        max_features="sqrt",
                        class_weight="balanced",
                        n_jobs=self.n_jobs,
                        random_state=self.random_state,
                    ),
                ),
            ]
        )
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Call fit() first.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Call fit() first.")
        return self.model.predict_proba(X)


class XGBoostBaseline:
    name = "XGBoost"

    def __init__(self, n_estimators: int = 500, n_jobs: int = 4, random_state: int = 42):
        self.n_estimators, self.n_jobs, self.random_state = n_estimators, n_jobs, random_state
        self.model: Pipeline | None = None
        self.classes_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, categorical_feats=None):
        cat = _cat_list(categorical_feats)
        self.classes_, y_mapped = _encode_labels(y)

        self.model = Pipeline(
            steps=[
                ("preprocess", _preprocessor(cat)),
                (
                    "clf",
                    XGBClassifier(
                        n_estimators=self.n_estimators,
                        max_depth=6,
                        learning_rate=0.1,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        n_jobs=self.n_jobs,
                        random_state=self.random_state,
                        verbosity=0,
                        eval_metric="logloss",
                        tree_method="hist",
                    ),
                ),
            ]
        )
        self.model.fit(X, y_mapped)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None or self.classes_ is None:
            raise RuntimeError("Call fit() first.")
        y_pred = self.model.predict(X).astype(np.int64)
        return self.classes_[y_pred]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Call fit() first.")
        return self.model.predict_proba(X)


class CatBoostBaseline:
    name = "CatBoost"

    def __init__(self, n_estimators: int = 500, n_jobs: int = 4, random_state: int = 42):
        self.n_estimators, self.n_jobs, self.random_state = n_estimators, n_jobs, random_state
        self.model: Pipeline | None = None
        self.classes_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, categorical_feats=None):
        cat = _cat_list(categorical_feats)
        self.classes_, y_mapped = _encode_labels(y)

        self.model = Pipeline(
            steps=[
                ("preprocess", _preprocessor(cat)),
                (
                    "clf",
                    CatBoostClassifier(
                        iterations=self.n_estimators,
                        depth=6,
                        learning_rate=0.1,
                        loss_function="MultiClass",
                        verbose=False,
                        thread_count=self.n_jobs,
                        random_seed=self.random_state,
                        allow_writing_files=False,
                    ),
                ),
            ]
        )
        self.model.fit(X, y_mapped)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None or self.classes_ is None:
            raise RuntimeError("Call fit() first.")
        y_pred = np.asarray(self.model.predict(X)).ravel().astype(np.int64)
        return self.classes_[y_pred]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Call fit() first.")
        return np.asarray(self.model.predict_proba(X))


def get_baselines(n_jobs: int = 4, random_state: int = 42):
    return [
        RandomForestBaseline(n_jobs=n_jobs, random_state=random_state),
        XGBoostBaseline(n_jobs=n_jobs, random_state=random_state),
        CatBoostBaseline(n_jobs=n_jobs, random_state=random_state),
    ]