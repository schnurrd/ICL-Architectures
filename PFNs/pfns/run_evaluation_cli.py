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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--checkpoint_name", type=str, default="checkpoint.pt")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--benchmark", type=str, default="opencc", choices=["opencc", "test"])
    parser.add_argument("--max_samples", type=int, default=1024)
    parser.add_argument("--max_features", type=int, default=25)
    parser.add_argument("--max_classes", type=int, default=10)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--only_tabpfn", action="store_true", help="Evaluate only TabPFN")
    args = parser.parse_args()
    
    assert args.benchmark in ["opencc", "test"], "Benchmark must be 'opencc' or 'test'"
    dataset_ids = OPENCC_BENCHMARK if args.benchmark == "opencc" else TEST_BENCHMARK
    
    tabpfn = TabPFNClassifier(
        base_path=args.model_path,
        device=args.device,
        model_string=args.checkpoint_name,
        N_ensemble_configurations=32,
    )
    
    models = [tabpfn] if args.only_tabpfn else [tabpfn, RandomForestBaseline(), XGBoostBaseline()]
    
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
        print("\n" + "="*50)
        print(results.groupby('model')['accuracy'].agg(['mean', 'std']).sort_values('mean', ascending=False))
        if args.output:
            results.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
