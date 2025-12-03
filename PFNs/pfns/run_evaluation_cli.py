#!/usr/bin/env python
"""
Evaluate TabPFN against baselines on OpenML benchmarks.

Usage:
    python run_evaluation.py --model_path <path> --benchmark opencc
"""

import argparse

from pfns.scripts.tabpfn_interface import TabPFNClassifier
from pfns.evaluation import (
    RandomForestBaseline,
    XGBoostBaseline,
    evaluate_on_openml
)
from pfns.datasets.tabular_datasets import open_cc_dids as OPENCC_BENCHMARK
from pfns.datasets.tabular_datasets import test_dids_classification as TEST_BENCHMARK
from pfns.utils import get_default_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--checkpoint_name", type=str, default="checkpoint.pt")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--benchmark", type=str, default="opencc", choices=["opencc", "test"])
    parser.add_argument("--max_samples", type=int, default=1024)
    parser.add_argument("--max_features", type=int, default=25)
    parser.add_argument("--max_classes", type=int, default=10)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--only_tabpfn", action="store_true", help="Evaluate only TabPFN")
    parser.add_argument("--n_jobs", type=int, default=4, help="Number of CPU cores for baseline models (RF, XGBoost)")
    parser.add_argument("--batch_size_inference", type=int, default=32, help="Batch size for TabPFN inference. Lower values reduce memory usage without affecting accuracy")
    args = parser.parse_args()
    
    if args.device is None:
        args.device = get_default_device()
    assert args.benchmark in ["opencc", "test"], "Benchmark must be 'opencc' or 'test'"
    dataset_ids = OPENCC_BENCHMARK if args.benchmark == "opencc" else TEST_BENCHMARK
    
    tabpfn = TabPFNClassifier(
        base_path=args.model_path,
        device=args.device,
        model_string=args.checkpoint_name,
        N_ensemble_configurations=32,
        batch_size_inference=args.batch_size_inference,
    )
    
    models = [tabpfn] if args.only_tabpfn else [
        tabpfn,
        RandomForestBaseline(n_jobs=args.n_jobs),
        XGBoostBaseline(n_jobs=args.n_jobs),
    ]
    
    results = evaluate_on_openml(
        models=models,
        model_names=["TabPFN", "RandomForest", "XGBoost"],
        dataset_ids=dataset_ids,
        max_samples=args.max_samples,
        max_features=args.max_features,
        max_classes=args.max_classes,
        n_splits=args.n_splits,
    )
    
    if not results.empty:
        print("\n" + "=" * 95)
        print("SUMMARY: Aggregated Results Across All Datasets")
        print("=" * 95)
        
        summary = results.groupby('model').agg({
            'accuracy': ['mean', 'std'],
            'roc_auc': ['mean', 'std'],
            'fit_time': ['mean', 'sum'],
            'predict_time': ['mean', 'sum'],
        }).round(4)
        
        summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
        summary = summary.sort_values('accuracy_mean', ascending=False)
        
        print(f"{'Model':<20} {'Accuracy':>18} {'ROC-AUC':>18} {'Fit (s)':>14} {'Pred (s)':>14}")
        print(f"{'':20} {'mean ± std':>18} {'mean ± std':>18} {'mean (tot)':>14} {'mean (tot)':>14}")
        print("-" * 95)
        for model in summary.index:
            row = summary.loc[model]
            acc_str = f"{row['accuracy_mean']:.4f} ± {row['accuracy_std']:.4f}"
            auc_str = f"{row['roc_auc_mean']:.4f} ± {row['roc_auc_std']:.4f}"
            fit_str = f"{row['fit_time_mean']:.2f} ({row['fit_time_sum']:.1f})"
            pred_str = f"{row['predict_time_mean']:.2f} ({row['predict_time_sum']:.1f})"
            print(f"{model:<20} {acc_str:>18} {auc_str:>18} {fit_str:>14} {pred_str:>14}")
        print("=" * 95)
        
        if args.output:
            results.to_csv(args.output, index=False)
            print(f"\nDetailed results saved to: {args.output}")


if __name__ == "__main__":
    main()
