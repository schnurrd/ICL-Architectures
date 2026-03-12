import argparse
import sys
from pathlib import Path
from typing import Any

# Ensure `pfns` imports resolve when this script is run directly from `notebooks/`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pfns.experiments.model_benchmarks.hashing import single_model_hash
from pfns.experiments.model_benchmarks.io import (
    REAL_WORLD_BUNDLE_KEYS,
    REAL_WORLD_REQUIRED_FILES,
    download_results_bundle_from_wandb,
    load_dataframe_bundle,
    make_bundle_path,
    make_model_artifact_name,
    sanitize_wandb_artifact_component,
    save_dataframe_bundle,
    upload_results_bundle_to_wandb,
)
from pfns.experiments.model_benchmarks.model_registry import (
    MODEL_FAMILIES,
    get_all_models,
    get_baseline_models,
    get_models_from_families,
    get_models_from_names,
)
from pfns.experiments.model_benchmarks.workflows import (
    alias_real_world_dataframe_bundle,
    build_real_world_run_metadata,
    real_world_bundle_is_compatible,
)
from pfns.run_evaluation_cli import (
    BENCHMARK_CHOICES,
    build_available_baseline_model_configs,
    compute_per_dataset_stats,
    run_real_world_model_from_config,
    summarize_results,
)
from pfns.utils import get_default_device

DEFAULT_EXPERIMENT: dict[str, Any] = {
    "name": "real_world_tabarena_comparison",# "real_world_openml_comparison",
    "benchmark": "tabarena_full",
    "max_samples": 1_000_000,
    "max_features": 20,
    "max_classes": 10,
    "n_splits": 5,
    "batch_size_inference": 32,
    "n_ensemble_configurations": 32,
    "preprocess_transforms": ["none", "power", "robust"],
    "sample_order_permutation": True,
    "fla_cache_chunk_size": None,
}

DEFAULT_BASELINE: dict[str, int] = {
    "n_jobs": 4,
    "random_state": 42,
}

DEFAULT_WANDB: dict[str, Any] = {
    "enabled": True,
    "overwrite": True,
    "artifact_name_real_eval": "real_eval_results",
    "entity": "icl_arch",
    "artifact_project": "real_world_tabarena_full_eval_artifacts", #  "real_world_eval_artifacts"
    "mode": "online",
}


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run real-world OpenML benchmark experiments with per-model sharding.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Optional exact model names to evaluate.",
    )
    parser.add_argument(
        "--families",
        nargs="+",
        default=None,
        help=(
            "Optional model family names to evaluate. "
            "Ignored when --models is set."
        ),
    )
    parser.add_argument(
        "--include-baselines",
        action="store_true",
        help="Include all available baseline models.",
    )
    parser.add_argument(
        "--baseline-models",
        nargs="+",
        default=None,
        help=(
            "Optional subset of baseline model names to include "
            "(e.g. RandomForest XGBoost CatBoost)."
        ),
    )

    parser.add_argument(
        "--benchmark",
        type=str,
        default=DEFAULT_EXPERIMENT["benchmark"],
        choices=BENCHMARK_CHOICES,
    )
    parser.add_argument("--experiment-name", type=str, default=DEFAULT_EXPERIMENT["name"])
    parser.add_argument("--max-samples", type=int, default=DEFAULT_EXPERIMENT["max_samples"])
    parser.add_argument("--max-features", type=int, default=DEFAULT_EXPERIMENT["max_features"])
    parser.add_argument("--max-classes", type=int, default=DEFAULT_EXPERIMENT["max_classes"])
    parser.add_argument("--n-splits", type=int, default=DEFAULT_EXPERIMENT["n_splits"])
    parser.add_argument(
        "--batch-size-inference",
        type=int,
        default=DEFAULT_EXPERIMENT["batch_size_inference"],
    )
    parser.add_argument(
        "--n-ensemble-configurations",
        type=int,
        default=DEFAULT_EXPERIMENT["n_ensemble_configurations"],
    )
    parser.add_argument(
        "--preprocess-transforms",
        nargs="+",
        default=DEFAULT_EXPERIMENT["preprocess_transforms"],
        help="TabPFN preprocessing transforms.",
    )
    parser.add_argument(
        "--sample-order-permutation",
        action="store_true",
        default=DEFAULT_EXPERIMENT["sample_order_permutation"],
        help="Permute sample order per TabPFN ensemble member.",
    )
    parser.add_argument(
        "--no-sample-order-permutation",
        action="store_false",
        dest="sample_order_permutation",
        help="Disable sample-order permutation for TabPFN.",
    )
    parser.add_argument("--fla-cache-chunk-size", type=int, default=DEFAULT_EXPERIMENT["fla_cache_chunk_size"])

    parser.add_argument("--baseline-n-jobs", type=int, default=DEFAULT_BASELINE["n_jobs"])
    parser.add_argument("--random-state", type=int, default=DEFAULT_BASELINE["random_state"])

    parser.add_argument(
        "--num-runs",
        "--num-nodes",
        dest="num_runs",
        type=int,
        default=1,
        help="Total number of parallel runs/shards.",
    )
    parser.add_argument(
        "--run-index",
        type=int,
        default=0,
        help="This run's shard index in [0, num_runs-1].",
    )

    parser.add_argument("--output-root", type=Path, default=Path.cwd().resolve() / "exp_outputs" / "real_world_eval")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--verbose", action="store_true", help="Verbose dataset-level logging.")

    parser.add_argument(
        "--wandb-enabled",
        action="store_true",
        dest="wandb_enabled",
        help="Enable W&B artifact cache/download/upload.",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_false",
        dest="wandb_enabled",
        help="Disable all W&B reads/writes.",
    )
    parser.set_defaults(wandb_enabled=DEFAULT_WANDB["enabled"])
    parser.add_argument("--wandb-overwrite", action="store_true", default=DEFAULT_WANDB["overwrite"])
    parser.add_argument(
        "--artifact-name-real-eval",
        type=str,
        default=DEFAULT_WANDB["artifact_name_real_eval"],
    )
    parser.add_argument("--wandb-entity", type=str, default=DEFAULT_WANDB["entity"])
    parser.add_argument("--wandb-artifact-project", type=str, default=DEFAULT_WANDB["artifact_project"])
    parser.add_argument("--wandb-mode", type=str, default=DEFAULT_WANDB["mode"])

    args, _ = parser.parse_known_args()

    if args.max_samples < 1:
        parser.error("--max-samples must be >= 1")
    if args.max_features < 1:
        parser.error("--max-features must be >= 1")
    if args.max_classes < 2:
        parser.error("--max-classes must be >= 2")
    if args.n_splits < 2:
        parser.error("--n-splits must be >= 2")
    if args.batch_size_inference < 1:
        parser.error("--batch-size-inference must be >= 1")
    if args.n_ensemble_configurations < 1:
        parser.error("--n-ensemble-configurations must be >= 1")
    if args.baseline_n_jobs < 1:
        parser.error("--baseline-n-jobs must be >= 1")
    if args.num_runs < 1:
        parser.error("--num-runs must be >= 1")
    if args.run_index < 0 or args.run_index >= args.num_runs:
        parser.error("--run-index must be in [0, num-runs-1]")

    return args


def _resolve_selected_models(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    if args.models:
        selected = get_models_from_names(args.models)
    elif args.families:
        selected = get_models_from_families(args.families)
    else:
        selected = get_all_models()

    baseline_candidates: dict[str, dict[str, Any]] = {}
    if args.include_baselines:
        baseline_candidates.update(get_baseline_models())
    if args.baseline_models:
        all_baselines = get_baseline_models()
        missing = sorted(set(args.baseline_models) - set(all_baselines))
        if missing:
            available = ", ".join(sorted(all_baselines))
            missing_str = ", ".join(missing)
            raise KeyError(
                f"Unknown baseline model(s): {missing_str}. Available: {available}"
            )
        baseline_candidates.update({name: all_baselines[name] for name in args.baseline_models})

    if baseline_candidates:
        available_baselines, skipped_baselines = build_available_baseline_model_configs(
            candidates=baseline_candidates,
            n_jobs=args.baseline_n_jobs,
            random_state=args.random_state,
        )
        if skipped_baselines:
            print(f"Skipping unavailable baselines in this environment: {skipped_baselines}")
        selected.update(available_baselines)

    return selected


def main() -> None:
    args = parse_cli_args()

    experiment = {
        "name": args.experiment_name,
        "benchmark": args.benchmark,
        "max_samples": args.max_samples,
        "max_features": args.max_features,
        "max_classes": args.max_classes,
        "n_splits": args.n_splits,
        "batch_size_inference": args.batch_size_inference,
        "n_ensemble_configurations": args.n_ensemble_configurations,
        "preprocess_transforms": list(args.preprocess_transforms),
        "sample_order_permutation": bool(args.sample_order_permutation),
        "fla_cache_chunk_size": args.fla_cache_chunk_size,
    }

    wandb_cfg = {
        "enabled": bool(args.wandb_enabled),
        "overwrite": bool(args.wandb_overwrite),
        "artifact_name_real_eval": args.artifact_name_real_eval,
        "entity": args.wandb_entity,
        "artifact_project": args.wandb_artifact_project,
        "mode": args.wandb_mode,
    }

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Results are stored in: {output_root}")
    print(f"Available model families: {list(MODEL_FAMILIES)}")
    print(f"Sharding config: num_runs={args.num_runs}, run_index={args.run_index}")

    all_models_to_compare = _resolve_selected_models(args)
    all_model_items = list(all_models_to_compare.items())
    models_to_compare = dict(all_model_items[args.run_index::args.num_runs])
    if not models_to_compare:
        print(
            f"No models assigned to run_index={args.run_index} "
            f"with num_runs={args.num_runs} (total models={len(all_model_items)})."
        )
        raise SystemExit(0)

    device = str(args.device or get_default_device())
    print(f"Using device: {device}")
    print(f"Models assigned to this run: {len(models_to_compare)} / {len(all_model_items)}")
    if args.models:
        print(f"Requested models: {args.models}")
    if args.families:
        print(f"Requested families: {args.families}")

    expected_real_metadata = build_real_world_run_metadata(
        experiment=experiment,
        device=device,
    )

    if wandb_cfg["enabled"] and wandb_cfg["overwrite"]:
        print("WANDB overwrite=True: skipping per-model download and forcing reruns.")

    completed = 0
    for model_name, model_config in models_to_compare.items():
        model_hash = single_model_hash(
            model_name=model_name,
            model_config=model_config,
            experiment_payload=experiment,
        )
        model_artifact_name = make_model_artifact_name(
            base_artifact_name=wandb_cfg["artifact_name_real_eval"],
            model_name=model_name,
            model_hash=model_hash,
        )

        reused_cached_result = False
        if wandb_cfg["enabled"] and not wandb_cfg["overwrite"]:
            cached_bundle_path = download_results_bundle_from_wandb(
                artifact_name=model_artifact_name,
                entity=wandb_cfg["entity"],
                project=wandb_cfg["artifact_project"],
                download_root=output_root / "wandb_model_cache" / "real_world",
                required_files=REAL_WORLD_REQUIRED_FILES,
            )
            print(f"Checked for cached W&B artifact for {model_name}: {cached_bundle_path}")

            if cached_bundle_path is not None:
                cached_bundle = load_dataframe_bundle(
                    cached_bundle_path,
                    expected_keys=REAL_WORLD_BUNDLE_KEYS,
                )
                cached_bundle_for_model, source_labels = alias_real_world_dataframe_bundle(
                    cached_bundle,
                    target_model_name=model_name,
                )
                cached_dataframes = cached_bundle_for_model["dataframes"]

                if real_world_bundle_is_compatible(
                    cached_bundle_for_model,
                    model_name=model_name,
                    expected_metadata=expected_real_metadata,
                ):
                    model_bundle_path = make_bundle_path(
                        output_root / "real_world",
                        f"{experiment['name']}_{sanitize_wandb_artifact_component(model_name)}",
                    )
                    save_dataframe_bundle(
                        dataframes=cached_dataframes,
                        bundle_dir=model_bundle_path,
                        experiment=experiment,
                        run_metadata=expected_real_metadata,
                    )
                    if source_labels:
                        print(
                            f"Reused cached real-world W&B result for {model_name} from stored labels "
                            f"{sorted(source_labels)}: {cached_bundle_path}. Saved local alias bundle: {model_bundle_path}"
                        )
                    else:
                        print(
                            f"Reused cached real-world W&B result for {model_name}: {cached_bundle_path}. "
                            f"Saved local alias bundle: {model_bundle_path}"
                        )
                    reused_cached_result = True
                    completed += 1

        if reused_cached_result:
            continue

        print(f"Running real-world benchmark for model: {model_name}")
        results = run_real_world_model_from_config(
            model_config=model_config,
            experiment=experiment,
            device=device,
            baseline_n_jobs=args.baseline_n_jobs,
            random_state=args.random_state,
            verbose=args.verbose,
        )

        if results.empty:
            print(f"Warning: No results for model {model_name}, skipping saving and upload.")
            continue

        results = results.copy()
        results["model"] = model_name

        summary = summarize_results(results)
        per_dataset = compute_per_dataset_stats(results)

        model_bundle_path = make_bundle_path(
            output_root / "real_world",
            f"{experiment['name']}_{sanitize_wandb_artifact_component(model_name)}",
        )
        save_dataframe_bundle(
            dataframes={
                "results": results,
                "summary": summary.reset_index() if summary is not None else None,
                "per_dataset": per_dataset,
            },
            bundle_dir=model_bundle_path,
            experiment=experiment,
            run_metadata=expected_real_metadata,
        )
        print(f"Saved real-world bundle for {model_name}: {model_bundle_path}")

        if wandb_cfg["enabled"]:
            artifact_ref = upload_results_bundle_to_wandb(
                model_bundle_path,
                artifact_name=model_artifact_name,
                entity=wandb_cfg["entity"],
                project=wandb_cfg["artifact_project"],
                run_name=(
                    f"{experiment['name']}_{sanitize_wandb_artifact_component(model_name)}_"
                    f"{model_hash}_artifact"
                ),
                metadata={
                    "experiment": experiment,
                    "model_name": model_name,
                    "model_config": model_config,
                    "model_hash": model_hash,
                    "run_metadata": expected_real_metadata,
                },
                run_mode=wandb_cfg["mode"],
                job_type="real_world_bundle_upload",
            )
            print(f"Uploaded real-world artifact for {model_name}: {artifact_ref}")

        completed += 1

    print("\nReal-world evaluation completed.")
    print(f"Finished models: {completed}/{len(models_to_compare)}")


if __name__ == "__main__":
    main()
