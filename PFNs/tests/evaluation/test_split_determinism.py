from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import pytest
import pfns.evaluation.evaluate as eval_module


class _RecordingDeterministicModel:
    def __init__(self, name: str = "RecordingDeterministic") -> None:
        self.name = name
        self.majority_class = 0
        self.fit_history: list[tuple[int, ...]] = []
        self.predict_history: list[tuple[int, ...]] = []

    @staticmethod
    def _sample_ids(X: np.ndarray) -> tuple[int, ...]:
        return tuple(np.asarray(X[:, 0], dtype=np.int64).tolist())

    def fit(self, X: np.ndarray, y: np.ndarray, categorical_feats=None):
        self.fit_history.append(self._sample_ids(X))
        y = np.asarray(y, dtype=np.int64)
        self.majority_class = int(np.bincount(y).argmax())
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.predict_history.append(self._sample_ids(X))
        return np.full(X.shape[0], self.majority_class, dtype=np.int64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probs = np.zeros((X.shape[0], 2), dtype=np.float32)
        probs[:, self.majority_class] = 1.0
        return probs


def _build_dataset(name: str, base_id: int):
    n_samples = 12
    sample_ids = np.arange(base_id, base_id + n_samples, dtype=np.float32)
    X = np.stack([sample_ids, sample_ids % 5], axis=1)
    y = np.array([0, 1] * (n_samples // 2), dtype=np.int64)
    return [name, torch.tensor(X), torch.tensor(y), [], ["id", "feat"], {}]


def test_deterministic_evaluation(monkeypatch):
    datasets = [
        _build_dataset("dataset_alpha", 100),
        _build_dataset("dataset_beta", 200),
    ]
    load_call_count = {"count": 0}

    def fake_load_openml_list(*args, **kwargs):
        load_call_count["count"] += 1
        return datasets, pd.DataFrame()

    monkeypatch.setattr(eval_module, "load_openml_list", fake_load_openml_list)

    model_a_first = _RecordingDeterministicModel(name="ModelA")
    model_b_first = _RecordingDeterministicModel(name="ModelB")
    model_a_second = _RecordingDeterministicModel(name="ModelA")
    model_b_second = _RecordingDeterministicModel(name="ModelB")

    run_kwargs = dict(
        dataset_ids=[1, 2],
        max_samples=128,
        max_features=8,
        max_classes=2,
        n_splits=3,
        verbose=False,
    )

    first_results = eval_module.evaluate_on_openml(
        models=[model_a_first, model_b_first],
        model_names=[model_a_first.name, model_b_first.name],
        **run_kwargs,
    )
    second_results = eval_module.evaluate_on_openml(
        models=[model_a_second, model_b_second],
        model_names=[model_a_second.name, model_b_second.name],
        **run_kwargs,
    )

    assert load_call_count["count"] == 2
    assert (
        first_results["dataset"].drop_duplicates().tolist()
        == second_results["dataset"].drop_duplicates().tolist()
    )
    assert first_results["dataset"].drop_duplicates().tolist() == [
        "dataset_alpha",
        "dataset_beta",
    ]

    pd.testing.assert_frame_equal(
        first_results[["dataset", "model", "split"]].reset_index(drop=True),
        second_results[["dataset", "model", "split"]].reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(
        first_results[
            ["dataset", "model", "split", "accuracy", "roc_auc", "log_loss", "ece"]
        ].reset_index(drop=True),
        second_results[
            ["dataset", "model", "split", "accuracy", "roc_auc", "log_loss", "ece"]
        ].reset_index(drop=True),
        check_exact=True,
    )
    # Test that models compared in the same run must see identical splits.
    assert model_a_first.fit_history == model_b_first.fit_history
    assert model_a_first.predict_history == model_b_first.predict_history
    assert model_a_second.fit_history == model_b_second.fit_history
    assert model_a_second.predict_history == model_b_second.predict_history

    # Test stability check across repeated evaluations.
    assert model_a_first.fit_history == model_a_second.fit_history
    assert model_a_first.predict_history == model_a_second.predict_history