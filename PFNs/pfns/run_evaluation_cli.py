#!/usr/bin/env python
"""
Evaluate TabPFN against baselines on OpenML benchmarks.

Usage:
    python run_evaluation.py --model_path <path> --benchmark opencc
"""

import argparse
from typing import Any

import torch

from pfns.scripts.tabpfn_interface import TabPFNClassifier
from pfns.evaluation.baselines import get_baselines
from pfns.evaluation.evaluate import evaluate_on_openml
from pfns.datasets.tabular_datasets import open_cc_dids as OPENCC_BENCHMARK
from pfns.datasets.tabular_datasets import test_dids_classification as TEST_BENCHMARK
from pfns.datasets.tabular_datasets import get_benchmark_suite_dids
from pfns.experiments.model_benchmarks.plotting import resolve_display_name_map
from pfns.utils import get_default_device
SUMMARY_METRIC_DEFAULTS: dict[str, dict[str, Any]] = {
    "accuracy": {"label": "Accuracy", "direction": "up", "precision": 4, "show_std": True},
    "roc_auc": {"label": "ROC-AUC", "direction": "up", "precision": 4, "show_std": True},
    "log_loss": {"label": "CE", "direction": "down", "precision": 4, "show_std": True},
    "ece": {"label": "ECE", "direction": "down", "precision": 4, "show_std": True},
    "fit_time": {"label": "Fit (s)", "direction": "down", "precision": 2, "show_std": False},
    "predict_time": {"label": "Pred (s)", "direction": "down", "precision": 2, "show_std": False},
}
TIMING_METRICS = {"fit_time", "predict_time"}
BENCHMARK_CHOICES = [
    "opencc",
    "test",
    "openml_large_dataset",
    "tabarena_full",
    "tabarena_medium",
]


def _is_oom_error(exc: BaseException) -> bool:
    if isinstance(exc, torch.OutOfMemoryError | torch.cuda.OutOfMemoryError):
        return True
    return "out of memory" in str(exc).lower()


def _resolve_summary_metrics(
    metric_specs: list[str | dict[str, Any]] | tuple[str | dict[str, Any], ...] | None,
) -> list[dict[str, Any]]:
    if metric_specs is None:
        metric_specs = list(SUMMARY_METRIC_DEFAULTS)

    resolved_specs: list[dict[str, Any]] = []
    for metric_spec in metric_specs:
        if isinstance(metric_spec, str):
            metric_name = metric_spec
            spec = {"metric": metric_name}
        else:
            spec = dict(metric_spec)
            metric_name = str(spec["metric"])

        if metric_name not in SUMMARY_METRIC_DEFAULTS:
            available_metrics = ", ".join(sorted(SUMMARY_METRIC_DEFAULTS))
            raise KeyError(
                f"Unknown summary metric '{metric_name}'. Available metrics: {available_metrics}"
            )

        resolved = {**SUMMARY_METRIC_DEFAULTS[metric_name], **spec, "metric": metric_name}
        if metric_name in TIMING_METRICS:
            resolved["show_std"] = False
        resolved_specs.append(resolved)

    return resolved_specs


def run_evaluation(
    *,
    runner: str | None = None,
    model_config: dict[str, Any],
    device: str | None = None,
    benchmark: str = "opencc",
    max_samples: int = 1000,
    max_features: int = 20,
    max_classes: int = 10,
    n_splits: int = 5,
    n_jobs: int = 4,
    random_state: int = 42,
    batch_size_inference: int = 16,
    n_ensemble_configurations: int = 10,
    preprocess_transforms: list[str] | tuple[str, ...] = ("none", "power"),
    sample_order_permutation: bool = False,
    fla_cache_chunk_size: int | None = None,
    verbose: bool = True,
) -> Any:
    """Run one OpenML evaluation entry point for TabPFN or baseline models."""
    if device is None:
        device = get_default_device()

    if benchmark == "opencc":
        dataset_ids = OPENCC_BENCHMARK
    elif benchmark == "test":
        dataset_ids = TEST_BENCHMARK
    elif benchmark == "openml_large_dataset":
        dataset_ids = [1461]
    elif benchmark == "tabarena_full":
        dataset_ids = get_benchmark_suite_dids(
            suite_id=457, # TabArena suite
            max_features=None,
        )
    elif benchmark == "tabarena_medium":
        dataset_ids = get_benchmark_suite_dids(
            suite_id=457, # TabArena suite
            min_samples=10_000,
            max_samples=None,
            max_features=None,
        )
    else:
        supported = ", ".join(BENCHMARK_CHOICES)
        raise ValueError(f"Benchmark must be one of: {supported}.")

    resolved_runner = runner or model_config.get("runner")
    if resolved_runner is None:
        resolved_runner = "baseline" if "baseline_name" in model_config else "tabpfn"
        print(f"No runner specified, inferred runner='{resolved_runner}' from model_config keys / defaulting.")
    if resolved_runner == "baseline":
        available_baselines = {
            model.name: model
            for model in get_baselines(n_jobs=n_jobs, random_state=random_state)
        }
        available = ", ".join(sorted(available_baselines))
        baseline_name = model_config.get("baseline_name")
        if baseline_name is None:
            raise KeyError(
                "Missing required key 'baseline_name' for baseline runner. "
                f"Available baselines: {available}"
            )
        if baseline_name not in available_baselines:
            raise KeyError(
                f"Unknown baseline '{baseline_name}'. Available baselines: {available}"
            )
        return evaluate_on_openml(
            models=[available_baselines[baseline_name]],
            model_names=[baseline_name],
            dataset_ids=dataset_ids,
            max_samples=max_samples,
            max_features=max_features,
            max_classes=max_classes,
            n_splits=n_splits,
            random_state=random_state,
            verbose=verbose,
        )

    if resolved_runner != "tabpfn":
        raise ValueError(
            "Unknown runner "
            f"{resolved_runner!r}. Expected 'tabpfn' or 'baseline'."
        )

    TabPFNClassifier.models_in_memory.clear()
    try:
        tabpfn = TabPFNClassifier(
            base_path=str(model_config.get("base_path") or "."),
            device=device,
            model_string=str(model_config.get("checkpoint_name") or "checkpoint.pt"),
            wandb_run_id=model_config.get("wandb_run_id"),
            autocast_dtype=model_config.get("eval_autocast_dtype"),
            high_cardinality_categorical_threshold=model_config.get(
                "high_cardinality_categorical_threshold"
            ),
            N_ensemble_configurations=n_ensemble_configurations,
            preprocess_transforms=list(preprocess_transforms),
            batch_size_inference=batch_size_inference,
            sample_order_permutation=sample_order_permutation,
            fla_cache_chunk_size=fla_cache_chunk_size,
            seed=random_state,
        )
        model_name = str(model_config.get("name") or tabpfn.name)
        return evaluate_on_openml(
            models=[tabpfn],
            model_names=[model_name],
            dataset_ids=dataset_ids,
            max_samples=max_samples,
            max_features=max_features,
            max_classes=max_classes,
            n_splits=n_splits,
            random_state=random_state,
            verbose=verbose,
        )
    finally:
        TabPFNClassifier.models_in_memory.clear()


def get_available_baseline_names(
    *,
    n_jobs: int = 4,
    random_state: int = 42,
) -> list[str]:
    return [model.name for model in get_baselines(n_jobs=n_jobs, random_state=random_state)]


def build_available_baseline_model_configs(
    *,
    candidates: dict[str, dict[str, Any]],
    n_jobs: int = 4,
    random_state: int = 42,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Filter candidate baseline configs to those available in the current env."""
    available_baseline_names = set(
        get_available_baseline_names(n_jobs=n_jobs, random_state=random_state)
    )

    selected: dict[str, dict[str, Any]] = {}
    skipped: list[str] = []
    for model_name, model_config in candidates.items():
        baseline_name = str(model_config.get("baseline_name") or model_name)
        if baseline_name not in available_baseline_names:
            skipped.append(model_name)
            continue

        cfg = model_config.copy()
        cfg["runner"] = "baseline"
        cfg["baseline_name"] = baseline_name
        cfg["n_jobs"] = int(n_jobs)
        cfg["random_state"] = int(random_state)
        selected[model_name] = cfg

    return selected, sorted(skipped)


def run_real_world_model_from_config(
    *,
    model_config: dict[str, Any],
    experiment: dict[str, Any],
    device: str | None = None,
    baseline_n_jobs: int = 4,
    random_state: int = 42,
    verbose: bool = True,
):
    """Run one real-world model entry from the notebook model-config structure."""
    exp = {
        key: experiment[key]
        for key in (
            "benchmark",
            "max_samples",
            "max_features",
            "max_classes",
            "n_splits",
            "batch_size_inference",
            "n_ensemble_configurations",
            "preprocess_transforms",
            "sample_order_permutation",
            "fla_cache_chunk_size",
        )
    }
    run_kwargs = {
        "model_config": model_config,
        "device": device,
        "n_jobs": int(model_config.get("n_jobs", baseline_n_jobs)),
        "random_state": int(model_config.get("random_state", random_state)),
        "verbose": verbose,
        **exp,
    }

    try:
        return run_evaluation(**run_kwargs)
    except Exception as exc:
        is_tabpfn_runner = str(model_config.get("runner") or "tabpfn") == "tabpfn"
        if not is_tabpfn_runner or exp["batch_size_inference"] <= 1 or not _is_oom_error(exc):
            raise

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(
            "Evaluation hit an OOM error. Retrying once with "
            "batch_size_inference=1."
        )
        retry_kwargs = dict(run_kwargs)
        retry_kwargs["batch_size_inference"] = 1
        return run_evaluation(**retry_kwargs)


def print_results_summary(
    results,
    title: str = "Aggregated Results Across All Datasets",
    metrics: list[str | dict[str, Any]] | tuple[str | dict[str, Any], ...] | None = None,
):
    if results.empty:
        print("Evaluation produced no results.")
        return

    print("\n" + "=" * 111)
    print(f"SUMMARY: {title}")
    print("=" * 111)

    summary = summarize_results(results)
    display_name_map = resolve_display_name_map(results)
    model_display_names = [display_name_map.get(str(model), str(model)) for model in summary.index]
    model_col_width = max(20, max((len(name) for name in model_display_names), default=0))
    metric_specs = _resolve_summary_metrics(metrics)
    arrow = {
        "up": {"latex": "$\\uparrow$", "text": "↑"},
        "down": {"latex": "$\\downarrow$", "text": "↓"},
    }

    def format_metric(row, spec: dict[str, Any], *, latex: bool) -> str:
        metric = str(spec["metric"])
        precision = int(spec["precision"])
        value = f"{float(row[f'{metric}_mean']):.{precision}f}"
        if not spec["show_std"]:
            return value
        std = f"{float(row[f'{metric}_std']):.{precision}f}"
        separator = " $\\pm$ " if latex else " ± "
        return f"{value}{separator}{std}"

    print("\nLaTeX table:")
    print("\\begin{table}[ht]")
    print("\\centering")
    print(f"\\begin{{tabular}}{{l{'c' * len(metric_specs)}}}")
    print("\\hline")
    latex_header = " & ".join(
        [
            "Model",
            *[
                f"{spec['label']} {arrow[str(spec['direction'])]['latex']}"
                for spec in metric_specs
            ],
        ]
    )
    print(f"{latex_header} \\\\")
    print("\\hline")
    for model in summary.index:
        row = summary.loc[model]
        model_display = display_name_map.get(str(model), str(model))
        model_latex = model_display.replace("_", "\\_")
        formatted_values = [format_metric(row, spec, latex=True) for spec in metric_specs]
        print(
            " & ".join(
                [
                    model_latex,
                    *formatted_values,
                ]
            )
            + " \\\\"
        )
    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")

    print("\nFormatted Table:")
    formatted_rows: list[tuple[str, list[str]]] = []
    for model in summary.index:
        row = summary.loc[model]
        model_display = display_name_map.get(str(model), str(model))
        formatted_values = [format_metric(row, spec, latex=False) for spec in metric_specs]
        formatted_rows.append((model_display, formatted_values))

    column_headers = [
        f"{spec['label']} {arrow[str(spec['direction'])]['text']}"
        for spec in metric_specs
    ]
    column_subheaders = ["mean ± std" if spec["show_std"] else "mean" for spec in metric_specs]
    column_widths = [
        max(
            len(column_headers[idx]),
            len(column_subheaders[idx]),
            max((len(row_values[idx]) for _, row_values in formatted_rows), default=0),
        )
        for idx in range(len(metric_specs))
    ]

    header = (
        f"{'Model':<{model_col_width}}"
        + "".join(
            f" {column_headers[idx]:>{column_widths[idx]}}"
            for idx in range(len(column_headers))
        )
    )
    subheader = (
        f"{'':{model_col_width}}"
        + "".join(
            f" {column_subheaders[idx]:>{column_widths[idx]}}"
            for idx in range(len(column_subheaders))
        )
    )
    print(header)
    print(subheader)
    print("-" * len(header))
    for model_display, row_values in formatted_rows:
        print(
            f"{model_display:<{model_col_width}}"
            + "".join(
                f" {row_values[idx]:>{column_widths[idx]}}"
                for idx in range(len(row_values))
            )
        )
    print("=" * len(header))


def compute_per_dataset_stats(results):
    if results.empty:
        return None

    per_dataset = results.groupby(["model", "dataset"]).agg(
        {
            "accuracy": ["mean", "std"],
            "roc_auc": ["mean", "std"],
            "log_loss": ["mean", "std"],
            "ece": ["mean", "std"],
            "fit_time": ["mean"],
            "predict_time": ["mean"],
        }
    )
    per_dataset.columns = ["_".join(col).strip() for col in per_dataset.columns.values]
    per_dataset = per_dataset.reset_index()
    return per_dataset

def summarize_results(results):
    if results.empty:
        return None

    per_dataset = compute_per_dataset_stats(results)
    summary = per_dataset.groupby("model").agg(
        {
            "accuracy_mean": "mean",
            "accuracy_std": "mean",
            "roc_auc_mean": "mean",
            "roc_auc_std": "mean",
            "log_loss_mean": "mean",
            "log_loss_std": "mean",
            "ece_mean": "mean",
            "ece_std": "mean",
            "fit_time_mean": "mean",
            "predict_time_mean": "mean",
        }
    ).round(4)
    return summary.sort_values("accuracy_mean", ascending=False)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--checkpoint_name", type=str, default="checkpoint.pt")
    parser.add_argument("--wandb_run_id", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--benchmark", type=str, default="opencc", choices=BENCHMARK_CHOICES,)
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument("--max_features", type=int, default=20)
    parser.add_argument("--max_classes", type=int, default=10)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--n_jobs", type=int, default=4, help="Number of CPU cores for baseline models (RF, XGBoost)")
    parser.add_argument("--batch_size_inference", type=int, default=32, help="Batch size for TabPFN inference. Lower values reduce memory usage without affecting accuracy")
    parser.add_argument("--n_ensemble_configurations", type=int, default=32, help="Number of ensemble configurations for TabPFN")
    parser.add_argument("--preprocess_transforms", type=str, nargs='+', default=["none", "power", "robust"], help="Preprocessing transforms to ensemble over for TabPFN")
    parser.add_argument("--sample_order_permutation", action="store_true", help="Permute training sample order for each ensemble configuration")
    parser.add_argument("--fla_cache_chunk_size", type=int, default=None, help="Chunk size for cache-backed inference when using an FLA backbone")
    args = parser.parse_args()

    if args.model_path is None and args.wandb_run_id is None:
        raise ValueError("Provide --model_path or --wandb_run_id.")

    results = run_evaluation(
        runner="tabpfn",
        model_config={
            "base_path": args.model_path if args.model_path is not None else ".",
            "checkpoint_name": args.checkpoint_name,
            "wandb_run_id": args.wandb_run_id,
        },
        device=args.device,
        benchmark=args.benchmark,
        max_samples=args.max_samples,
        max_features=args.max_features,
        max_classes=args.max_classes,
        n_splits=args.n_splits,
        n_jobs=args.n_jobs,
        batch_size_inference=args.batch_size_inference,
        n_ensemble_configurations=args.n_ensemble_configurations,
        preprocess_transforms=args.preprocess_transforms,
        sample_order_permutation=args.sample_order_permutation,
        fla_cache_chunk_size=args.fla_cache_chunk_size,
    )

    print_results_summary(results)

    if args.output and not results.empty:
        results.to_csv(args.output, index=False)
        print(f"\nDetailed results saved to: {args.output}")


if __name__ == "__main__":
    main()
