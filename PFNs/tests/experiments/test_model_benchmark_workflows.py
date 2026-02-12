from __future__ import annotations

import pandas as pd
import pytest

from pfns.experiments.model_benchmarks.workflows import (
    aggregate_real_world_results_from_bundles,
    build_real_world_run_metadata,
    build_seq_len_run_metadata,
    merge_seq_len_model_results,
    real_world_bundle_is_compatible,
    seq_len_bundle_is_compatible,
    single_model_seq_len_result_from_bundle,
)


def test_build_seq_len_run_metadata():
    experiment = {
        "seqlen_list": [128, 256],
        "num_features": 10,
        "num_classes": 5,
        "num_test_samples": 100,
        "num_repetitions": 7,
        "data_generation_seed": 42,
    }
    metadata = build_seq_len_run_metadata(experiment=experiment, device="cpu")
    assert metadata == {
        "seqlen_list": [128, 256],
        "num_features": 10,
        "num_classes": 5,
        "number_of_test_samples": 100,
        "number_of_repetitions": 7,
        "device": "cpu",
        "data_generation_seed": 42,
    }


def test_build_real_world_run_metadata_sorts_transforms():
    experiment = {
        "benchmark": "opencc",
        "max_samples": 1000,
        "max_features": 20,
        "max_classes": 10,
        "n_splits": 5,
        "batch_size_inference": 32,
        "n_ensemble_configurations": 16,
        "preprocess_transforms": ["robust", "none", "power"],
        "sample_order_permutation": True,
        "fla_cache_chunk_size": None,
    }
    metadata = build_real_world_run_metadata(experiment=experiment, device="cuda:0")
    assert metadata["preprocess_transforms"] == ["none", "power", "robust"]
    assert metadata["device"] == "cuda:0"


def test_seq_len_bundle_compatibility_and_extract_single_model():
    bundle = {
        "metric_table": {"M1": {"acc": {128: [0.8]}}},
        "timing_table": {"M1": {"forward_time_ms": {128: [1.0]}}},
        "memory_table": {"M1": {"context_size_mb": {128: [10.0]}}},
        "oom_errors": {"M1": [1024]},
        "metadata": {"device": "cpu"},
        "bundle_metadata": {"schema_version": "1.0"},
    }
    expected = {"device": "cpu"}

    assert seq_len_bundle_is_compatible(bundle, model_name="M1", expected_metadata=expected)
    assert not seq_len_bundle_is_compatible(bundle, model_name="M2", expected_metadata=expected)
    assert not seq_len_bundle_is_compatible(
        bundle,
        model_name="M1",
        expected_metadata={"device": "cuda"},
    )

    single = single_model_seq_len_result_from_bundle(bundle, model_name="M1")
    assert list(single["metric_table"].keys()) == ["M1"]
    assert single["metadata"]["device"] == "cpu"


def test_merge_seq_len_model_results_materializes_dataframes():
    model_result = {
        "schema_version": "1.0",
        "metric_table": {"M1": {"acc": {128: [0.8]}}},
        "timing_table": {"M1": {"forward_time_ms": {128: [1.0]}}},
        "memory_table": {"M1": {"context_size_mb": {128: [10.0]}}},
        "oom_errors": {"M1": []},
        "metadata": {"device": "cpu"},
    }
    merged = merge_seq_len_model_results(
        results_by_model={"M1": model_result},
        expected_run_metadata={"device": "cpu"},
        model_names=["M1"],
        experiment={"name": "exp"},
        model_bundle_paths={"M1": "bundle/path"},
    )

    assert not merged["metric_df"].empty
    assert not merged["timing_df"].empty
    assert not merged["memory_df"].empty
    assert merged["bundle_metadata"]["per_model_bundle_paths"]["M1"] == "bundle/path"


def test_real_world_bundle_compatibility():
    results_df = pd.DataFrame({"model": ["M1"], "dataset": ["d1"], "split": [0]})
    bundle = {
        "dataframes": {"results": results_df},
        "metadata": {"benchmark": "opencc"},
    }
    assert real_world_bundle_is_compatible(
        bundle,
        model_name="M1",
        expected_metadata={"benchmark": "opencc"},
    )
    assert not real_world_bundle_is_compatible(
        bundle,
        model_name="M2",
        expected_metadata={"benchmark": "opencc"},
    )


def test_aggregate_real_world_results_from_bundles():
    m1 = pd.DataFrame(
        {
            "model": ["M1", "M1"],
            "dataset": ["d1", "d1"],
            "split": [0, 1],
            "accuracy": [0.8, 0.9],
        }
    )
    m2 = pd.DataFrame(
        {
            "model": ["M2", "M2"],
            "dataset": ["d1", "d1"],
            "split": [0, 1],
            "accuracy": [0.7, 0.75],
        }
    )
    all_results, results_by_model = aggregate_real_world_results_from_bundles(
        {
            "M1": {"bundle": {"dataframes": {"results": m1}}},
            "M2": {"bundle": {"dataframes": {"results": m2}}},
        },
        expected_splits=2,
    )
    assert len(all_results) == 4
    assert set(results_by_model) == {"M1", "M2"}


def test_aggregate_real_world_results_split_mismatch_raises():
    bad = pd.DataFrame(
        {
            "model": ["M1", "M1"],
            "dataset": ["d1", "d1"],
            "split": [0, 0],
        }
    )
    with pytest.raises(RuntimeError, match="Split count mismatch"):
        aggregate_real_world_results_from_bundles(
            {"M1": {"bundle": {"dataframes": {"results": bad}}}},
            expected_splits=2,
        )
