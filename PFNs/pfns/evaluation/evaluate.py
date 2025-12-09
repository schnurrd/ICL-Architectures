"""
Simple Evaluation Utilities - reuses tabular_datasets.py for OpenML loading
"""

from __future__ import annotations
import time
import numpy as np
import pandas as pd
from typing import List, Dict
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss

from pfns.datasets.tabular_datasets import load_openml_list


def evaluate_model(
    model,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict[str, float]:
    """Evaluate a model with cross-validation."""
    X = np.nan_to_num(np.asarray(X, dtype=np.float32), nan=0.0)
    y = np.asarray(y, dtype=np.int64)
    
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    results = []
    
    for train_idx, test_idx in cv.split(X, y):
        start = time.time()
        model.fit(X[train_idx], y[train_idx])
        fit_time = time.time() - start
        
        start = time.time()
        y_pred, y_proba = model.predict(X[test_idx], return_prediction_probs=True)
        predict_time = time.time() - start
        
        acc = accuracy_score(y[test_idx], y_pred)
        n_classes = len(np.unique(y[test_idx]))
        auc = roc_auc_score(y[test_idx], y_proba[:, 1]) if n_classes == 2 else \
              roc_auc_score(y[test_idx], y_proba, multi_class="ovr", average="weighted")
        ll = log_loss(y[test_idx], y_proba)
        
        results.append({"accuracy": acc, "roc_auc": auc, "log_loss": ll, 
                        "fit_time": fit_time, "predict_time": predict_time})
    
    return {k: np.mean([r[k] for r in results]) for k in results[0]}


def compare_models(
    models: List,
    model_names: List[str],
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
) -> pd.DataFrame:
    """Compare multiple models on a single dataset."""
    results = []
    for model, name in zip(models, model_names):
        print(f"Evaluating {name}...")
        result = evaluate_model(model, X, y, n_splits=n_splits)
        result["model"] = name
        results.append(result)
        print(f"  Accuracy: {result['accuracy']:.4f}")
    return pd.DataFrame(results)[["model", "accuracy", "roc_auc", "log_loss", "fit_time"]]


def evaluate_on_openml(
    models: List,
    model_names: List[str],
    dataset_ids: List[int],
    max_samples: int = 1024,
    max_features: int = 100,
    max_classes: int = 10,
    n_splits: int = 5,
) -> pd.DataFrame:
    """Evaluate models on OpenML datasets using tabular_datasets.py loader."""
    datasets, _ = load_openml_list(
        dataset_ids, 
        max_samples=max_samples,
        num_feats=max_features,
        max_num_classes=max_classes,
        return_capped=True,
        filter_for_nan=True,
    )
    
    all_results = []
    for name, X, y, _, _, _ in datasets:
        print(f"\n{'='*75}")
        print(f"{name}: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"{'='*75}")
        print(f"{'Model':<20} {'Accuracy':>10} {'ROC-AUC':>10} {'Fit (s)':>10} {'Pred (s)':>10}")
        print(f"{'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
        for model, model_name in zip(models, model_names):
            try:
                result = evaluate_model(model, X.numpy(), y.numpy(), n_splits=n_splits)
                result.update({"model": model_name, "dataset": name})
                all_results.append(result)
                print(f"{model_name:<20} {result['accuracy']:>10.4f} {result['roc_auc']:>10.4f} {result['fit_time']:>10.2f} {result['predict_time']:>10.2f}")
            except Exception as e:
                print(f"{model_name:<20} {'Error':>10} - {e}")
    
    return pd.DataFrame(all_results)[["dataset", "model", "accuracy", "roc_auc", "log_loss", "fit_time", "predict_time"]] if all_results else pd.DataFrame()
