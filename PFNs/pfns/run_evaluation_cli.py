#!/usr/bin/env python
"""
Evaluate TabPFN against baselines on OpenML benchmarks.

Usage:
    python run_evaluation.py --model_path <path> --benchmark opencc
"""

import argparse
from typing import Any

from pfns.scripts.tabpfn_interface import TabPFNClassifier
from pfns.evaluation import (
    evaluate_on_openml,
    get_baselines,
)
from pfns.datasets.tabular_datasets import open_cc_dids as OPENCC_BENCHMARK
from pfns.datasets.tabular_datasets import test_dids_classification as TEST_BENCHMARK
from pfns.utils import get_default_device


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
    batch_size_inference: int = 32,
    n_ensemble_configurations: int = 32,
    preprocess_transforms: list[str] | tuple[str, ...] = ("none", "power", "robust"),
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
    else:
        raise ValueError("Benchmark must be 'opencc' or 'test'")

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
    return run_evaluation(
        model_config=model_config,
        device=device,
        n_jobs=int(model_config.get("n_jobs", baseline_n_jobs)),
        random_state=int(model_config.get("random_state", random_state)),
        verbose=verbose,
        **exp,
    )


def print_results_summary(results, title: str = "Aggregated Results Across All Datasets"):
    if results.empty:
        print("Evaluation produced no results.")
        return

    print("\n" + "=" * 111)
    print(f"SUMMARY: {title}")
    print("=" * 111)

    summary = summarize_results(results)

    print("\nLaTeX table rows:")
    for model in summary.index:
        row = summary.loc[model]
        acc_str = f"{row['accuracy_mean']:.4f} ± {row['accuracy_std']:.4f}"
        auc_str = f"{row['roc_auc_mean']:.4f} ± {row['roc_auc_std']:.4f}"
        ll_str = f"{row['log_loss_mean']:.4f} ± {row['log_loss_std']:.4f}"
        ece_str = f"{row['ece_mean']:.4f} ± {row['ece_std']:.4f}"
        fit_str = f"{row['fit_time_mean']:.2f}"
        pred_str = f"{row['predict_time_mean']:.2f}"
        print(
            f"{model} & {acc_str} & {auc_str} & {ll_str} & {ece_str} "
            f"& {fit_str} & {pred_str} \\\\"
        )

    print("\nFormatted Table:")
    header = (
        f"{'Model':<20} {'Accuracy':>18} {'ROC-AUC':>18} {'LogLoss':>18} "
        f"{'ECE':>18} {'Fit (s)':>14} {'Pred (s)':>14}"
    )
    subheader = (
        f"{'':20} {'mean ± std':>18} {'mean ± std':>18} {'mean ± std':>18} "
        f"{'mean ± std':>18} {'mean':>14} {'mean':>14}"
    )
    print(header)
    print(subheader)
    print("-" * len(header))
    for model in summary.index:
        row = summary.loc[model]
        acc_str = f"{row['accuracy_mean']:.4f} ± {row['accuracy_std']:.4f}"
        auc_str = f"{row['roc_auc_mean']:.4f} ± {row['roc_auc_std']:.4f}"
        ll_str = f"{row['log_loss_mean']:.4f} ± {row['log_loss_std']:.4f}"
        ece_str = f"{row['ece_mean']:.4f} ± {row['ece_std']:.4f}"
        fit_str = f"{row['fit_time_mean']:.2f}"
        pred_str = f"{row['predict_time_mean']:.2f}"
        print(
            f"{model:<20} {acc_str:>18} {auc_str:>18} {ll_str:>18} {ece_str:>18} "
            f"{fit_str:>14} {pred_str:>14}"
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
    parser.add_argument("--benchmark", type=str, default="opencc", choices=["opencc", "test"])
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
