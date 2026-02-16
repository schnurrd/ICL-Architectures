from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def eval_cli():
    return pytest.importorskip("pfns.run_evaluation_cli")


class _FakeTabPFNClassifier:
    models_in_memory = {"stale": object()}

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.name = "FakeTabPFN"


class _FakeBaseline:
    def __init__(self, name: str):
        self.name = name


def test_run_evaluation_tabpfn_single_model(monkeypatch, eval_cli):
    captured: dict[str, object] = {}

    def fake_evaluate_on_openml(**kwargs):
        captured.update(kwargs)
        return pd.DataFrame([{"model": "FakeTabPFN"}])

    monkeypatch.setattr(eval_cli, "TabPFNClassifier", _FakeTabPFNClassifier)
    monkeypatch.setattr(eval_cli, "evaluate_on_openml", fake_evaluate_on_openml)
    monkeypatch.setattr(eval_cli, "get_baselines", lambda **kwargs: [_FakeBaseline("B1")])

    results = eval_cli.run_evaluation(
        runner="tabpfn",
        model_config={"base_path": ".", "checkpoint_name": "checkpoint.pt"},
        benchmark="test",
        verbose=False,
    )

    assert isinstance(results, pd.DataFrame)
    assert captured["dataset_ids"] == eval_cli.TEST_BENCHMARK
    assert captured["model_names"] == ["FakeTabPFN"]
    assert len(captured["models"]) == 1
    assert eval_cli.TabPFNClassifier.models_in_memory == {}


def test_run_evaluation_baseline_runner(monkeypatch, eval_cli):
    captured: dict[str, object] = {}

    def fake_evaluate_on_openml(**kwargs):
        captured.update(kwargs)
        return pd.DataFrame([{"model": "BaselineA"}])

    monkeypatch.setattr(eval_cli, "evaluate_on_openml", fake_evaluate_on_openml)
    monkeypatch.setattr(
        eval_cli,
        "get_baselines",
        lambda **kwargs: [_FakeBaseline("BaselineA"), _FakeBaseline("BaselineB")],
    )

    eval_cli.run_evaluation(
        runner="evaluation_cli_baseline",
        model_config={"baseline_name": "BaselineA"},
        verbose=False,
    )

    assert captured["model_names"] == ["BaselineA"]
    assert len(captured["models"]) == 1
    assert captured["models"][0].name == "BaselineA"


def test_run_evaluation_infers_baseline_runner(monkeypatch, eval_cli):
    captured: dict[str, object] = {}

    def fake_evaluate_on_openml(**kwargs):
        captured.update(kwargs)
        return pd.DataFrame([{"model": "BaselineA"}])

    monkeypatch.setattr(eval_cli, "evaluate_on_openml", fake_evaluate_on_openml)
    monkeypatch.setattr(
        eval_cli,
        "get_baselines",
        lambda **kwargs: [_FakeBaseline("BaselineA"), _FakeBaseline("BaselineB")],
    )

    eval_cli.run_evaluation(
        model_config={"baseline_name": "BaselineA"},
        verbose=False,
    )

    assert captured["model_names"] == ["BaselineA"]
    assert len(captured["models"]) == 1
    assert captured["models"][0].name == "BaselineA"


def test_run_evaluation_invalid_runner_raises(eval_cli):
    with pytest.raises(ValueError, match="Unknown runner"):
        eval_cli.run_evaluation(
            runner="unknown",
            model_config={},
            verbose=False,
        )


def test_run_evaluation_baseline_name_missing_raises(monkeypatch, eval_cli):
    monkeypatch.setattr(eval_cli, "get_baselines", lambda **kwargs: [_FakeBaseline("BaselineA")])
    with pytest.raises(KeyError, match="Missing required key 'baseline_name'"):
        eval_cli.run_evaluation(
            runner="evaluation_cli_baseline",
            model_config={},
            verbose=False,
        )


def test_run_evaluation_unknown_baseline_raises(monkeypatch, eval_cli):
    monkeypatch.setattr(eval_cli, "get_baselines", lambda **kwargs: [_FakeBaseline("BaselineA")])
    with pytest.raises(KeyError, match="Unknown baseline"):
        eval_cli.run_evaluation(
            runner="evaluation_cli_baseline",
            model_config={"baseline_name": "Nope"},
            verbose=False,
        )


def test_run_real_world_model_from_config_delegates(monkeypatch, eval_cli):
    captured: dict[str, object] = {}

    def fake_run_evaluation(**kwargs):
        captured.update(kwargs)
        return pd.DataFrame([{"model": "x"}])

    monkeypatch.setattr(eval_cli, "run_evaluation", fake_run_evaluation)
    experiment = {
        "benchmark": "opencc",
        "max_samples": 1000,
        "max_features": 20,
        "max_classes": 10,
        "n_splits": 5,
        "batch_size_inference": 32,
        "n_ensemble_configurations": 16,
        "preprocess_transforms": ["none"],
        "sample_order_permutation": False,
        "fla_cache_chunk_size": None,
    }

    eval_cli.run_real_world_model_from_config(
        model_config={"baseline_name": "BaselineA"},
        experiment=experiment,
        baseline_n_jobs=8,
        baseline_random_state=123,
        verbose=False,
    )

    assert captured["n_jobs"] == 8
    assert captured["random_state"] == 123

    captured.clear()
    eval_cli.run_real_world_model_from_config(
        model_config={"checkpoint_name": "checkpoint.pt"},
        experiment=experiment,
        baseline_n_jobs=8,
        baseline_random_state=123,
        verbose=False,
    )


def test_run_training_cli_uses_unified_evaluation_entrypoint():
    source = (
        Path(__file__).resolve().parents[2] / "pfns" / "run_training_cli.py"
    ).read_text(encoding="utf-8")
    assert "run_evaluation(" in source
    assert "run_tabpfn_evaluation(" not in source
