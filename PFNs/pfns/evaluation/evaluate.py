from __future__ import annotations
import time
import numpy as np
import pandas as pd
from typing import List, Any
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from tqdm.auto import tqdm

from pfns.datasets.tabular_datasets import load_openml_list
from pfns.evaluation.metrics import expected_calibration_error


def evaluate_model(
    model,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
    categorical_feats: list[int] | tuple[int, ...] | None = None,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """Evaluate a model with cross-validation. Returns per-split metrics (no aggregation)."""
    X = np.nan_to_num(np.asarray(X, dtype=np.float32), nan=0.0)
    y = np.asarray(y, dtype=np.int64)

    if categorical_feats is not None and hasattr(model, "categorical_feats"):
        model.categorical_feats = tuple(categorical_feats)
    
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    results: list[dict[str, Any]] = []
    
    for train_idx, test_idx in cv.split(X, y):
        start = time.time()
        fit_kwargs = {}
        if categorical_feats is not None:
            fit_kwargs["categorical_feats"] = categorical_feats
        model.fit(X[train_idx], y[train_idx], **fit_kwargs)
        fit_time = time.time() - start
        
        start = time.time()
        if model.__class__.__name__ == "TabPFNClassifier":
            y_pred, y_proba = model.predict(X[test_idx], return_prediction_probs=True)
        else:
            y_pred = model.predict(X[test_idx])
            y_proba = model.predict_proba(X[test_idx])
        predict_time = time.time() - start
        
        acc = accuracy_score(y[test_idx], y_pred)
        n_classes = len(np.unique(y[test_idx]))
        
        y_proba = y_proba.astype(np.float32) # Renorm to float32 as with fp16 auc calculation is unstable (probs. deviate from 1.0)
        if verbose and not np.isfinite(y_proba).all():
            print(
                f"Non-finite probabilities from {model.__class__.__name__}:"
            )
        y_proba /= y_proba.sum(axis=1, keepdims=True)
        
        auc = roc_auc_score(y[test_idx], y_proba[:, 1]) if n_classes == 2 else \
              roc_auc_score(y[test_idx], y_proba, multi_class="ovr", average="weighted")
        ll = log_loss(y[test_idx], y_proba)
        ece = expected_calibration_error(y[test_idx], y_proba)
        
        results.append(
            {
                "split": len(results),
                "accuracy": acc,
                "roc_auc": auc,
                "log_loss": ll,
                "ece": ece,
                "fit_time": fit_time,
                "predict_time": predict_time,
            }
        )
    return results


def compare_models(
    models: List,
    model_names: List[str],
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    categorical_feats: list[int] | tuple[int, ...] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Compare multiple models on a single dataset (returns per-split rows)."""
    results: list[dict[str, Any]] = []
    for model, name in zip(models, model_names):
        if verbose:
            print(f"Evaluating {name}...")
        split_results = evaluate_model(
            model,
            X,
            y,
            n_splits=n_splits,
            categorical_feats=categorical_feats,
            verbose=verbose,
        )
        for row in split_results:
            row.update({"model": name})
        results.extend(split_results)

        mean_acc = float(np.mean([r["accuracy"] for r in split_results])) if split_results else float("nan")
        if verbose:
            print(f"  Mean accuracy over splits: {mean_acc:.4f}")

    df = pd.DataFrame(results)
    desired_cols = [
        "model",
        "split",
        "accuracy",
        "roc_auc",
        "log_loss",
        "ece",
        "fit_time",
        "predict_time",
    ]
    cols = [c for c in desired_cols if c in df.columns] + [c for c in df.columns if c not in desired_cols]
    return df[cols]


def evaluate_on_openml(
    models: List,
    model_names: List[str],
    dataset_ids: List[int],
    max_samples: int = 1024,
    max_features: int = 100,
    max_classes: int = 10,
    n_splits: int = 5,
    verbose: bool = True,
) -> pd.DataFrame:
    """Evaluate models on OpenML datasets using tabular_datasets.py loader."""
    datasets, _ = load_openml_list(
        dataset_ids, 
        max_samples=max_samples,
        num_feats=max_features,
        max_num_classes=max_classes,
        return_capped=True,
        filter_for_nan=False,
        verbose=verbose,
    )
    
    all_results: list[dict[str, Any]] = []
    for name, X, y, categorical_feats, _, _ in tqdm(datasets, desc="Overall progress over datasets"):
        header = (
            f"{'Model':<18} {'Accuracy':>10} {'ROC-AUC':>10} {'LogLoss':>10} "
            f"{'ECE':>10} {'Fit (s)':>10} {'Pred (s)':>10}"
        )
        bar_len = len(header)
        if verbose:
            print(f"\n{'='*bar_len}")
            print(f"{name}: {X.shape[0]} samples, {X.shape[1]} features")
            print(f"{'='*bar_len}")
            print(header)
            print("-" * bar_len)
        for model, model_name in zip(models, model_names):
            try:
                split_results = evaluate_model(
                    model,
                    X.numpy(),
                    y.numpy(),
                    n_splits=n_splits,
                    categorical_feats=categorical_feats,
                    verbose=verbose,
                )
                for row in split_results:
                    row.update({"model": model_name, "dataset": name})
                all_results.extend(split_results)

                mean_acc = float(np.mean([r["accuracy"] for r in split_results])) if split_results else float("nan")
                mean_auc = float(np.mean([r["roc_auc"] for r in split_results])) if split_results else float("nan")
                mean_ll = float(np.mean([r["log_loss"] for r in split_results])) if split_results else float("nan")
                mean_ece = float(np.mean([r["ece"] for r in split_results])) if split_results else float("nan")
                mean_fit = float(np.mean([r["fit_time"] for r in split_results])) if split_results else float("nan")
                mean_pred = float(np.mean([r["predict_time"] for r in split_results])) if split_results else float("nan")
                if verbose:
                    print(
                        f"{model_name:<18} {mean_acc:>10.4f} {mean_auc:>10.4f} "
                        f"{mean_ll:>10.4f} {mean_ece:>10.4f} {mean_fit:>10.2f} "
                        f"{mean_pred:>10.2f}"
                    )
            except Exception as e:
                print(f"{model_name:<20} {'Error':>10} - {e}")
    
    if not all_results:
        return pd.DataFrame()

    return pd.DataFrame(all_results)
