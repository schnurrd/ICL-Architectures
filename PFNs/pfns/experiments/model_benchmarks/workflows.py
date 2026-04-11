from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .analysis import nested_metric_table_to_long_df
from .io import merge_model_results, run_metadata_matches


def build_seq_len_run_metadata(
    *,
    experiment: dict[str, Any],
    device: str,
) -> dict[str, Any]:
    """Canonical seq-len metadata used for cache compatibility checks."""
    metadata = {
        "seqlen_list": list(experiment["seqlen_list"]),
        "num_features": experiment["num_features"],
        "num_classes": experiment["num_classes"],
        "number_of_test_samples": experiment["num_test_samples"],
        "number_of_repetitions": experiment["num_repetitions"],
        "device": device,
        "data_generation_seed": experiment["data_generation_seed"],
    }
    if experiment.get("only_numerical_features", False):
        metadata["only_numerical_features"] = True
    return metadata


def build_real_world_run_metadata(
    *,
    experiment: dict[str, Any],
    device: str,
) -> dict[str, Any]:
    """Canonical real-world metadata used for cache compatibility checks."""
    return {
        "benchmark": experiment["benchmark"],
        "max_samples": experiment["max_samples"],
        "max_features": experiment["max_features"],
        "max_classes": experiment["max_classes"],
        "n_splits": experiment["n_splits"],
        "device": device,
        "batch_size_inference": experiment["batch_size_inference"],
        "n_ensemble_configurations": experiment["n_ensemble_configurations"],
        "preprocess_transforms": sorted(experiment["preprocess_transforms"]),
        "sample_order_permutation": experiment["sample_order_permutation"],
        "fla_cache_chunk_size": experiment["fla_cache_chunk_size"],
    }


def seq_len_bundle_is_compatible(
    bundle: dict[str, Any],
    *,
    model_name: str,
    expected_metadata: dict[str, Any],
) -> bool:
    """Check if a cached seq-len bundle can be reused for the given model."""
    has_model = (
        model_name in bundle.get("metric_table", {})
        and model_name in bundle.get("timing_table", {})
    )
    bundle_metadata = bundle.get("metadata", {})
    bundle_only_numerical = bool(
        bundle_metadata.get(
            "only_numerical_features",
            bundle_metadata.get("task_kwargs", {}).get("only_numerical_features", False),
        )
    )
    expected_only_numerical = bool(expected_metadata.get("only_numerical_features", False))
    if bundle_only_numerical != expected_only_numerical:
        return False

    metadata_ok = run_metadata_matches(
        bundle_metadata,
        expected=expected_metadata,
        keys=tuple(k for k in expected_metadata if k != "only_numerical_features"),
    )
    return bool(has_model and metadata_ok)


def real_world_bundle_is_compatible(
    bundle: dict[str, Any],
    *,
    model_name: str,
    expected_metadata: dict[str, Any],
) -> bool:
    """Check if a cached real-world dataframe bundle can be reused."""
    results_df = bundle.get("dataframes", {}).get("results")
    if results_df is None or results_df.empty or "model" not in results_df.columns:
        return False

    has_model = model_name in set(results_df["model"].astype(str))
    metadata_ok = run_metadata_matches(
        bundle.get("metadata", {}),
        expected=expected_metadata,
        keys=tuple(expected_metadata.keys()),
    )
    return bool(has_model and metadata_ok)


def alias_single_model_seq_len_bundle(
    bundle: dict[str, Any],
    *,
    target_model_name: str,
) -> tuple[dict[str, Any], str | None]:
    """Remap one-model seq-len bundle keys to `target_model_name` when possible."""
    metric_table = bundle.get("metric_table", {})
    timing_table = bundle.get("timing_table", {})
    if target_model_name in metric_table and target_model_name in timing_table:
        return bundle, None

    source_candidates = sorted(set(metric_table) & set(timing_table))
    if len(source_candidates) != 1:
        return bundle, None

    source_name = source_candidates[0]
    aliased_bundle = dict(bundle)
    aliased_bundle["metric_table"] = {
        target_model_name: metric_table[source_name],
    }
    aliased_bundle["timing_table"] = {
        target_model_name: timing_table[source_name],
    }
    aliased_bundle["memory_table"] = {
        target_model_name: bundle.get("memory_table", {}).get(source_name, {}),
    }
    aliased_bundle["oom_errors"] = {
        target_model_name: bundle.get("oom_errors", {}).get(source_name, []),
    }
    return aliased_bundle, source_name


def alias_real_world_dataframe_bundle(
    bundle: dict[str, Any],
    *,
    target_model_name: str,
) -> tuple[dict[str, Any], set[str]]:
    """Copy dataframe bundle and rewrite any `model` columns to `target_model_name`."""
    dataframes = bundle.get("dataframes", {})
    aliased_dataframes: dict[str, pd.DataFrame] = {}
    source_labels: set[str] = set()

    for key, frame in dataframes.items():
        frame_copy = frame.copy()
        if "model" in frame_copy.columns:
            source_labels.update(set(frame_copy["model"].astype(str).unique()))
            frame_copy["model"] = target_model_name
        aliased_dataframes[key] = frame_copy

    aliased_bundle = {
        **bundle,
        "dataframes": aliased_dataframes,
    }

    if source_labels == {target_model_name}:
        source_labels = set()
    return aliased_bundle, source_labels


def single_model_seq_len_result_from_bundle(
    bundle: dict[str, Any],
    *,
    model_name: str,
) -> dict[str, Any]:
    """Extract a one-model results payload from a loaded seq-len bundle."""
    return {
        "schema_version": bundle.get("bundle_metadata", {}).get("schema_version"),
        "metric_table": {model_name: bundle["metric_table"][model_name]},
        "timing_table": {model_name: bundle["timing_table"][model_name]},
        "memory_table": {model_name: bundle.get("memory_table", {}).get(model_name, {})},
        "oom_errors": {model_name: bundle.get("oom_errors", {}).get(model_name, [])},
        "metadata": bundle.get("metadata", {}),
    }


def merge_seq_len_model_results(
    *,
    results_by_model: dict[str, dict[str, Any]],
    expected_run_metadata: dict[str, Any],
    model_names: list[str],
    experiment: dict[str, Any],
    model_bundle_paths: dict[str, Path | str] | None = None,
) -> dict[str, Any]:
    """Merge per-model seq-len results and materialize long-form dataframes."""
    results = merge_model_results(
        results_by_model,
        merged_metadata={
            **expected_run_metadata,
            "models": list(model_names),
        },
    )

    metric_df = nested_metric_table_to_long_df(results["metric_table"])
    timing_df = nested_metric_table_to_long_df(results["timing_table"])
    memory_df = nested_metric_table_to_long_df(results["memory_table"])
    bundle_paths = model_bundle_paths or {}

    bundle_metadata = {
        "schema_version": results.get("schema_version"),
        "experiment": experiment,
        "run_metadata": results.get("metadata", {}),
        "row_counts": {
            "metric": int(len(metric_df)),
            "timing": int(len(timing_df)),
            "memory": int(len(memory_df)),
        },
        "per_model_bundle_paths": {name: str(path) for name, path in bundle_paths.items()},
    }

    return {
        "results": results,
        "metric_df": metric_df,
        "timing_df": timing_df,
        "memory_df": memory_df,
        "bundle_metadata": bundle_metadata,
    }


def aggregate_real_world_results_from_bundles(
    bundles_by_model: dict[str, dict[str, Any]],
    *,
    expected_splits: int,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Collect + validate per-model real-world results from loaded bundles."""
    result_frames: list[pd.DataFrame] = []
    results_by_model: dict[str, pd.DataFrame] = {}

    for model_name, model_bundle in bundles_by_model.items():
        bundle = model_bundle.get("bundle", model_bundle)
        model_results = bundle.get("dataframes", {}).get("results", pd.DataFrame()).copy()
        if model_results.empty:
            continue
        if "model" not in model_results.columns:
            raise RuntimeError(
                "Missing required column 'model' in aggregated results."
            )
        model_results = model_results[model_results["model"].astype(str) == model_name]
        if model_results.empty:
            continue
        result_frames.append(model_results)
        results_by_model[model_name] = model_results

    if not result_frames:
        raise RuntimeError("Compatible bundles were found, but no result rows matched the selected models.")

    all_results = pd.concat(result_frames, ignore_index=True)
    if all_results.empty:
        raise RuntimeError("Aggregated result dataframe is empty.")

    required_cols = {"model", "dataset", "split"}
    missing_cols = required_cols - set(all_results.columns)
    if missing_cols:
        raise RuntimeError(f"Missing required columns in aggregated results: {sorted(missing_cols)}")

    datasets_by_model = {
        model_name: set(df["dataset"].astype(str).unique())
        for model_name, df in results_by_model.items()
    }
    if not datasets_by_model:
        raise RuntimeError("No per-model dataset coverage was found in aggregated results.")

    all_datasets = set().union(*datasets_by_model.values())
    coverage_mismatch = {
        model_name: sorted(all_datasets - datasets)
        for model_name, datasets in datasets_by_model.items()
        if datasets != all_datasets
    }
    if coverage_mismatch:
        details = ", ".join(
            [f"{model}: missing {missing}" for model, missing in sorted(coverage_mismatch.items())]
        )
        raise RuntimeError(
            "Dataset coverage mismatch across models; all models must be evaluated on the same "
            f"dataset set. Details: {details}"
        )

    split_counts = all_results.groupby(["model", "dataset"])["split"].nunique()
    bad_split_counts = split_counts[split_counts != int(expected_splits)]
    if not bad_split_counts.empty:
        details = ", ".join(
            [f"{model}/{dataset}: {int(count)}" for (model, dataset), count in bad_split_counts.items()]
        )
        raise RuntimeError(
            f"Split count mismatch detected (expected {int(expected_splits)} per model/dataset): {details}"
        )

    return all_results, results_by_model
