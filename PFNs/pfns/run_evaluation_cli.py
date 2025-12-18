#!/usr/bin/env python
"""
Evaluate TabPFN against baselines on OpenML benchmarks.

Usage:
    python run_evaluation.py --model_path <path> --benchmark opencc
"""

import argparse

from pfns.scripts.tabpfn_interface import TabPFNClassifier
from pfns.evaluation import (
    evaluate_on_openml,
    get_baselines,
)
from pfns.datasets.tabular_datasets import open_cc_dids as OPENCC_BENCHMARK
from pfns.datasets.tabular_datasets import test_dids_classification as TEST_BENCHMARK
from pfns.utils import get_default_device


def run_tabpfn_evaluation(
    *,
    base_path: str,
    checkpoint_name: str = "checkpoint.pt",
    device: str | None = None,
    benchmark: str = "opencc",
    max_samples: int = 1000,
    max_features: int = 20,
    max_classes: int = 10,
    n_splits: int = 5,
    only_tabpfn: bool = False,
    n_jobs: int = 4,
    batch_size_inference: int = 32,
    n_ensemble_configurations: int = 32,
    preprocess_transforms: list[str] | tuple[str, ...] = ("none", "power", "robust"),
):
    """Run TabPFN (and optionally baselines) on the requested benchmark."""
    if device is None:
        device = get_default_device()

    assert benchmark in ["opencc", "test"], "Benchmark must be 'opencc' or 'test'"
    dataset_ids = OPENCC_BENCHMARK if benchmark == "opencc" else TEST_BENCHMARK

    tabpfn = TabPFNClassifier(
        base_path=base_path,
        device=device,
        model_string=checkpoint_name,
        N_ensemble_configurations=n_ensemble_configurations,
        preprocess_transforms=list(preprocess_transforms),
        batch_size_inference=batch_size_inference,
    )

    models = [tabpfn] if only_tabpfn else [
        tabpfn,
        *get_baselines(n_jobs=n_jobs)
    ]
    model_names = [model.name for model in models]

    results = evaluate_on_openml(
        models=models,
        model_names=model_names,
        dataset_ids=dataset_ids,
        max_samples=max_samples,
        max_features=max_features,
        max_classes=max_classes,
        n_splits=n_splits,
    )
    return results


def print_results_summary(results, title: str = "Aggregated Results Across All Datasets"):
    if results.empty:
        print("Evaluation produced no results.")
        return

    print("\n" + "=" * 95)
    print(f"SUMMARY: {title}")
    print("=" * 95)

    # Compute per-dataset mean and std, then aggregate across datasets for each model
    per_dataset = results.groupby(['model', 'dataset']).agg({
        'accuracy': ['mean', 'std'],
        'roc_auc': ['mean', 'std'],
        'fit_time': ['mean'],
        'predict_time': ['mean'],
    })
    per_dataset.columns = ['_'.join(col).strip() for col in per_dataset.columns.values]
    per_dataset = per_dataset.reset_index()

    summary = per_dataset.groupby('model').agg({
        'accuracy_mean': 'mean',
        'accuracy_std': 'mean',  # mean std over datasets
        'roc_auc_mean': 'mean',
        'roc_auc_std': 'mean',
        'fit_time_mean': 'mean',
        'predict_time_mean': 'mean',
    }).round(4)
    summary = summary.sort_values('accuracy_mean', ascending=False)

    print("\nLaTeX table rows:")
    for model in summary.index:
        row = summary.loc[model]
        acc_str = f"{row['accuracy_mean']:.4f} ± {row['accuracy_std']:.4f}"
        auc_str = f"{row['roc_auc_mean']:.4f} ± {row['roc_auc_std']:.4f}"
        fit_str = f"{row['fit_time_mean']:.2f}"
        pred_str = f"{row['predict_time_mean']:.2f}"
        print(f"{model} & {acc_str} & {auc_str} & {fit_str} & {pred_str} \\\\")

    print("\nFormatted Table:")
    print(f"{'Model':<20} {'Accuracy':>18} {'ROC-AUC':>18} {'Fit (s)':>14} {'Pred (s)':>14}")
    print(f"{'':20} {'mean ± std':>18} {'mean ± std':>18} {'mean':>14} {'mean':>14}")
    print("-" * 95)
    for model in summary.index:
        row = summary.loc[model]
        acc_str = f"{row['accuracy_mean']:.4f} ± {row['accuracy_std']:.4f}"
        auc_str = f"{row['roc_auc_mean']:.4f} ± {row['roc_auc_std']:.4f}"
        fit_str = f"{row['fit_time_mean']:.2f}"
        pred_str = f"{row['predict_time_mean']:.2f}"
        print(f"{model:<20} {acc_str:>18} {auc_str:>18} {fit_str:>14} {pred_str:>14}")
    print("=" * 95)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--checkpoint_name", type=str, default="checkpoint.pt")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--benchmark", type=str, default="opencc", choices=["opencc", "test"])
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument("--max_features", type=int, default=20)
    parser.add_argument("--max_classes", type=int, default=10)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--only_tabpfn", action="store_true", help="Evaluate only TabPFN")
    parser.add_argument("--n_jobs", type=int, default=4, help="Number of CPU cores for baseline models (RF, XGBoost)")
    parser.add_argument("--batch_size_inference", type=int, default=32, help="Batch size for TabPFN inference. Lower values reduce memory usage without affecting accuracy")
    parser.add_argument("--n_ensemble_configurations", type=int, default=32, help="Number of ensemble configurations for TabPFN")
    parser.add_argument("--preprocess_transforms", type=str, nargs='+', default=["none", "power", "robust"], help="Preprocessing transforms to ensemble over for TabPFN")
    args = parser.parse_args()
    
    results = run_tabpfn_evaluation(
        base_path=args.model_path,
        checkpoint_name=args.checkpoint_name,
        device=args.device,
        benchmark=args.benchmark,
        max_samples=args.max_samples,
        max_features=args.max_features,
        max_classes=args.max_classes,
        n_splits=args.n_splits,
        only_tabpfn=args.only_tabpfn,
        n_jobs=args.n_jobs,
        batch_size_inference=args.batch_size_inference,
        n_ensemble_configurations=args.n_ensemble_configurations,
        preprocess_transforms=args.preprocess_transforms,
    )

    print_results_summary(results)

    if args.output and not results.empty:
        results.to_csv(args.output, index=False)
        print(f"\nDetailed results saved to: {args.output}")


if __name__ == "__main__":
    main()
