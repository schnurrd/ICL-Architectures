"""
Baseline Models: Random Forest, XGBoost, and CatBoost
RandomForest uses sklearn preprocessing; XGBoost and CatBoost rely on
model-native missing-value handling. CatBoost uses native categorical features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import warnings
import io
from contextlib import contextmanager
from contextlib import redirect_stderr, redirect_stdout
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from tabicl import TabICLClassifier
from tabpfn import TabPFNClassifier

try:
    from ticl.prediction.tabflex import TabFlex
except ImportError:
    TabFlex = None


def _cat_list(categorical_feats) -> list[int]:
    if not categorical_feats:
        return []
    return list(sorted({int(i) for i in categorical_feats}))

def _encode_labels(y):
    classes, y_mapped = np.unique(y, return_inverse=True)
    return classes, y_mapped.astype(np.int64)

def to_dataframe(
    X: np.ndarray,
    categorical_features: list[int],
    *,
    categorical_mode: str = "category",
) -> pd.DataFrame:
    """
    Convert numpy array X into a pandas DataFrame.
    - categorical_mode="category": pandas categorical dtype (for TabICL/TabFlex).
    - categorical_mode="string": string tokens for categorical columns (for CatBoost).
    """
    if categorical_mode not in {"category", "string"}:
        raise ValueError(
            f"Unknown categorical_mode={categorical_mode!r}. "
            "Expected 'category' or 'string'."
        )

    df = pd.DataFrame(X)
    cat = sorted({int(i) for i in categorical_features or []})

    for j in cat:
        col = df.iloc[:, j]
        if categorical_mode == "string":
            col_obj = col.astype(object)
            col_obj = col_obj.where(~pd.isna(col_obj), "__MISSING__")
            df[df.columns[j]] = col_obj.map(
                lambda v: v if v == "__MISSING__" else str(v)
            )
        else:
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
    transformers = [
        ("cat", cat_pipe, cat),
        ("num", num_pipe, num),
    ]

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.0,
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
        warnings.filterwarnings(
            "ignore",
            message="'force_all_finite' was renamed to 'ensure_all_finite' in 1.6 and will be removed in 1.8\\.",
            category=FutureWarning,
            module="sklearn.utils.deprecation",
        )
        yield


@contextmanager
def _ignore_tabflex_warnings():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*autocast.*deprecated.*",
            category=FutureWarning,
        )
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            module="ticl\\.prediction\\..*",
        )
        yield


@contextmanager
def _suppress_output():
    buf = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(buf):
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
        self.model: XGBClassifier | None = None
        self.classes_: np.ndarray | None = None
        self.cat_: list[int] = []
        self.cat_categories_: dict[int, pd.Index] = {}

    def _prepare_xgb_input(self, X: np.ndarray, *, fit: bool) -> np.ndarray | pd.DataFrame:
        X_np = np.asarray(X, dtype=np.float32)
        if not self.cat_:
            return X_np

        X_df = to_dataframe(X_np, self.cat_, categorical_mode="string")
        if fit:
            self.cat_categories_ = {}
            for col_idx in self.cat_:
                col_name = X_df.columns[col_idx]
                cat_values = pd.Categorical(X_df[col_name])
                X_df[col_name] = cat_values
                self.cat_categories_[col_idx] = cat_values.categories
            return X_df

        for col_idx in self.cat_:
            col_name = X_df.columns[col_idx]
            X_df[col_name] = pd.Categorical(
                X_df[col_name],
                categories=self.cat_categories_.get(col_idx),
            )
        return X_df

    def fit(self, X: np.ndarray, y: np.ndarray, categorical_feats=None):
        self.cat_ = _cat_list(categorical_feats)
        self.classes_, y_mapped = _encode_labels(y)
        X_fit = self._prepare_xgb_input(X, fit=True)

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
            tree_method="hist",
            missing=np.nan,
            enable_categorical=bool(self.cat_),
        )
        self.model.fit(X_fit, y_mapped)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None or self.classes_ is None:
            raise RuntimeError("Call fit() first.")
        X_pred = self._prepare_xgb_input(X, fit=False)
        y_pred = self.model.predict(X_pred).astype(np.int64)
        return self.classes_[y_pred]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Call fit() first.")
        X_pred = self._prepare_xgb_input(X, fit=False)
        return self.model.predict_proba(X_pred)


class CatBoostBaseline:
    name = "CatBoost"

    def __init__(self, n_estimators: int = 500, n_jobs: int = 4, random_state: int = 42):
        self.n_estimators, self.n_jobs, self.random_state = n_estimators, n_jobs, random_state
        self.model: CatBoostClassifier | None = None
        self.classes_: np.ndarray | None = None
        self.cat_: list[int] = []

    def fit(self, X: np.ndarray, y: np.ndarray, categorical_feats=None):
        cat = _cat_list(categorical_feats)
        self.classes_, y_mapped = _encode_labels(y)
        self.cat_ = cat

        self.model = CatBoostClassifier(
            iterations=self.n_estimators,
            depth=6,
            learning_rate=0.1,
            loss_function="MultiClass",
            verbose=False,
            thread_count=self.n_jobs,
            random_seed=self.random_state,
            allow_writing_files=False,
        )
        X_fit = (
            to_dataframe(X, cat, categorical_mode="string")
            if cat
            else np.asarray(X, dtype=np.float32)
        )
        self.model.fit(X_fit, y_mapped, cat_features=cat if cat else None)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None or self.classes_ is None:
            raise RuntimeError("Call fit() first.")
        X_pred = (
            to_dataframe(X, self.cat_, categorical_mode="string")
            if self.cat_
            else np.asarray(X, dtype=np.float32)
        )
        y_pred = np.asarray(self.model.predict(X_pred)).ravel().astype(np.int64)
        return self.classes_[y_pred]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Call fit() first.")
        X_pred = (
            to_dataframe(X, self.cat_, categorical_mode="string")
            if self.cat_
            else np.asarray(X, dtype=np.float32)
        )
        return np.asarray(self.model.predict_proba(X_pred))

class TabICLBaseline:
    name = "TabICLv2"

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

    def fit(self, X: np.ndarray, y: np.ndarray, categorical_feats=None):
        self.classes_, y_mapped = _encode_labels(y)
        X_df = pd.DataFrame(np.asarray(X, dtype=np.float32))
        self.model = TabFlex()
        with _ignore_tabflex_warnings(), _ignore_sklearn_futurewarnings(), _suppress_output():
            self.model.fit(X_df, y_mapped)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None or self.classes_ is None:
            raise RuntimeError("Call fit() first.")
        X_df = pd.DataFrame(np.asarray(X, dtype=np.float32))
        with _ignore_tabflex_warnings(), _ignore_sklearn_futurewarnings(), _suppress_output():
            y_pred = np.asarray(self.model.predict(X_df)).astype(np.int64)
        return self.classes_[y_pred]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Call fit() first.")
        X_df = pd.DataFrame(np.asarray(X, dtype=np.float32))
        with _ignore_tabflex_warnings(), _ignore_sklearn_futurewarnings(), _suppress_output():
            return np.asarray(self.model.model.predict_proba(X_df))


def get_baselines(n_jobs: int = 4, random_state: int = 42):
    baselines = [
        RandomForestBaseline(n_jobs=n_jobs, random_state=random_state),
        XGBoostBaseline(n_jobs=n_jobs, random_state=random_state),
        CatBoostBaseline(n_jobs=n_jobs, random_state=random_state),
        TabICLBaseline(random_state=random_state),
        TabPFNV2_5Baseline(random_state=random_state),
    ]
    if TabFlex is not None:
        baselines.append(TabFlexBaseline(random_state=random_state))
    return baselines
