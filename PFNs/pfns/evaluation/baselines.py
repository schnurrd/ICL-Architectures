"""
Baseline Models: Random Forest, XGBoost, and CatBoost
Ordinal-encodes categorical columns via ColumnTransformer for all models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import warnings
from contextlib import contextmanager
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from tabicl import TabICLClassifier
from ticl.prediction.tabflex import TabFlex
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion



def _cat_list(categorical_feats) -> list[int]:
    if not categorical_feats:
        return []
    return list(sorted({int(i) for i in categorical_feats}))


def _encode_labels(y):
    classes, y_mapped = np.unique(y, return_inverse=True)
    return classes, y_mapped.astype(np.int64)

def to_dataframe(X: np.ndarray, categorical_features: list[int]) -> pd.DataFrame:
    """
    Convert numpy array X into a pandas DataFrame suitable for e.g. TabICL.
    Categorical columns are explicitly cast to pandas 'category' dtype.
    """
    df = pd.DataFrame(X)
    cat = sorted({int(i) for i in categorical_features or []})

    for j in cat:
        col = df.iloc[:, j]

        if pd.api.types.is_float_dtype(col):
            s = col.where(col.isna(), col.round().astype("Int64"))
            df[df.columns[j]] = s.astype("category")
        else:
            df[df.columns[j]] = col.astype("category")

    return df


def _preprocessor(cat: list[int], n_features: int) -> ColumnTransformer:
    cat_set = set(cat)
    num = [i for i in range(n_features) if i not in cat_set]

    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="mean")),
    ])
    return ColumnTransformer(
        transformers=[
            ("cat", cat_pipe, cat),
            ("num", num_pipe, num),
        ],
        remainder="drop",
    )


@contextmanager
def _ignore_sklearn_futurewarnings():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="`BaseEstimator._validate_data` is deprecated",
            category=FutureWarning,
            module="sklearn.base",
        )
        yield


class RandomForestBaseline:
    name = "RandomForest"

    def __init__(self, n_estimators: int = 500, n_jobs: int = 4, random_state: int = 42):
        self.n_estimators, self.n_jobs, self.random_state = n_estimators, n_jobs, random_state
        self.model: Pipeline | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, categorical_feats=None):
        cat = _cat_list(categorical_feats)
        self.model = Pipeline(
            steps=[
                ("preprocess", _preprocessor(cat, X.shape[1])),
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
                ("preprocess", _preprocessor(cat, X.shape[1])),
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
                ("preprocess", _preprocessor(cat, X.shape[1])),
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

class TabICLBaseline:
    name = "TabICL"

    def __init__(self, random_state: int = 42, **tabicl_kwargs):
        self.random_state = random_state
        self.tabicl_kwargs = tabicl_kwargs
        self.model = TabICLClassifier(random_state=random_state, **tabicl_kwargs)
        self.classes_: np.ndarray | None = None
        self.cat_: list[int] | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, categorical_feats=None):
        self.classes_, y_mapped = _encode_labels(y)
        self.cat_ = _cat_list(categorical_feats)

        X_df = to_dataframe(X, self.cat_)
        with _ignore_sklearn_futurewarnings():
            self.model.fit(X_df, y_mapped)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.classes_ is None or self.cat_ is None:
            raise RuntimeError("Call fit() first.")
        X_df = to_dataframe(X, self.cat_)
        with _ignore_sklearn_futurewarnings():
            y_pred = np.asarray(self.model.predict(X_df)).astype(np.int64)
        return self.classes_[y_pred]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.cat_ is None:
            raise RuntimeError("Call fit() first.")
        X_df = to_dataframe(X, self.cat_)
        with _ignore_sklearn_futurewarnings():
            return np.asarray(self.model.predict_proba(X_df))
        
        
class TabPFNV2_5Baseline:
    name = "TabPFNv2.5"
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = None
    
    def fit(self, X: np.ndarray, y: np.ndarray, categorical_feats=None):
        self.model = TabPFNClassifier(categorical_features_indices=categorical_feats)
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

class TabFlexBaseline:
    name = "TabFlex"
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = None
        self.classes_: np.ndarray | None = None
        self.cat_: list[int] | None = None
    
    def fit(self, X: np.ndarray, y: np.ndarray, categorical_feats=None):
        self.classes_, y_mapped = _encode_labels(y)
        self.cat_ = _cat_list(categorical_feats)
        X_df = to_dataframe(X, self.cat_)
        self.model = TabFlex()
        self.model.fit(X_df, y_mapped)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None or self.classes_ is None or self.cat_ is None:
            raise RuntimeError("Call fit() first.")
        X_df = to_dataframe(X, self.cat_)
        y_pred = np.asarray(self.model.predict(X_df)).astype(np.int64)
        return self.classes_[y_pred]
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None or self.cat_ is None:
            raise RuntimeError("Call fit() first.")
        X_df = to_dataframe(X, self.cat_)
        return np.asarray(self.model.predict_proba(X_df))


def get_baselines(n_jobs: int = 4, random_state: int = 42):
    return [
        RandomForestBaseline(n_jobs=n_jobs, random_state=random_state),
        XGBoostBaseline(n_jobs=n_jobs, random_state=random_state),
        CatBoostBaseline(n_jobs=n_jobs, random_state=random_state),
        TabICLBaseline(random_state=random_state),
        TabPFNV2_5Baseline(random_state=random_state),
        TabFlexBaseline(random_state=random_state),
    ]
