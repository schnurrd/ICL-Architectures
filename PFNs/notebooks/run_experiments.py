import argparse
from pathlib import Path

from pfns.utils import get_default_device
from notebook_utils import single_model_hash

from pfns.experiments.model_benchmarks.analysis import nested_metric_table_to_long_df
from pfns.experiments.model_benchmarks.evaluation import evaluate_models_over_seqlens
from pfns.experiments.model_benchmarks.io import (
    SEQ_LEN_REQUIRED_FILES,
    download_results_bundle_from_wandb,
    load_results_bundle,
    make_bundle_path,
    make_model_artifact_name,
    merge_model_results,
    run_metadata_matches,
    sanitize_wandb_artifact_component,
    save_results_bundle,
    upload_results_bundle_to_wandb,
)
from pfns.experiments.model_benchmarks.models import load_models_for_benchmark
from pfns.experiments.model_benchmarks.model_registry import get_all_models
from pfns.experiments.model_benchmarks.model_registry import (
    MODEL_FAMILIES,
    get_autocast_models_from_registry,
    get_forward_models_from_registry,
    get_models_from_families,
    get_models_from_names,
)

EXPERIMENT = {
    "name": "seq_len_comparison",
    "num_classes": 5,
    "num_features": 10,
    "num_test_samples": 100,
    "num_repetitions": 500,
    "data_generation_seed": 42,
    "use_warmup_iters": False,
    "print_timing": False,
    "seqlen_list": [250, 500, 750, 1_000, 2_000, 4_000, 8_000, 16_000, 32_000, 64_000, 128_000],
}

WANDB = {
    "enabled": True,
    "artifact_name": "seq_len_comparison",
    "overwrite": False,
    "entity": "icl_arch",
    "project": "seq_len_exp",
}


def parse_cli_args():
    parser = argparse.ArgumentParser(
        description="Run sequence length benchmark experiments."
    )
    parser.add_argument(
        "--num-repeats",
        dest="num_repetitions",
        type=int,
        default=EXPERIMENT["num_repetitions"],
        help=(
            "Number of repetitions used for evaluation "
            f"(default: {EXPERIMENT['num_repetitions']})."
        ),
    )
    args, _ = parser.parse_known_args()
    if args.num_repetitions < 1:
        parser.error("--num-repetitions must be >= 1")
    return args


CLI_ARGS = parse_cli_args()
EXPERIMENT["num_repetitions"] = CLI_ARGS.num_repetitions

OUTPUT_ROOT = Path.cwd().resolve() / "exp_outputs" / "seq_len"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

print(f"Results are stored in: {OUTPUT_ROOT}")
print(f"Available model families: {list(MODEL_FAMILIES)}")
print(f"Configured repetitions: {EXPERIMENT['num_repetitions']}")

# Example by family:
# models_to_compare = get_models_from_families(["transformer"])

# models_to_compare = get_models_from_names([
#     "Softmax_Transformer",
#     "KDA_causal",
#     "GLA_Cached",
#     "DeltaNet_Cached",
#     "Gated_DeltaNet_Cached_seq_len_10K",
#     "Rebased",
#     "Linear_Attention",
# ])

models_to_compare = get_all_models()


device = get_default_device()
print(f"Using device: {device}")

expected_run_metadata = {
    "seqlen_list": list(EXPERIMENT["seqlen_list"]),
    "num_features": EXPERIMENT["num_features"],
    "num_classes": EXPERIMENT["num_classes"],
    "number_of_test_samples": EXPERIMENT["num_test_samples"],
    "number_of_repetitions": EXPERIMENT["num_repetitions"],
    "device": device,
    "data_generation_seed": EXPERIMENT["data_generation_seed"],
}

results_by_model = {}
model_bundle_paths = {}

if WANDB["enabled"] and WANDB["overwrite"]:
    print("WANDB overwrite=True: skipping per-model download and forcing rerun.")

for model_name, model_config in models_to_compare.items():
    model_hash = single_model_hash(
        model_name=model_name,
        model_config=model_config,
        experiment_payload=EXPERIMENT,
    )
    model_artifact_name = make_model_artifact_name(
        base_artifact_name=WANDB["artifact_name"],
        model_name=model_name,
        model_hash=model_hash,
    )

    reused_cached_result = False
    if WANDB["enabled"] and not WANDB["overwrite"]:
        cached_bundle_path = download_results_bundle_from_wandb(
            artifact_name=model_artifact_name,
            entity=WANDB["entity"],
            project=WANDB["project"],
            download_root=OUTPUT_ROOT / "wandb_model_cache",
            required_files=SEQ_LEN_REQUIRED_FILES,
        )
        if cached_bundle_path is not None:
            cached_bundle = load_results_bundle(cached_bundle_path)
            has_model = (
                model_name in cached_bundle["metric_table"]
                and model_name in cached_bundle["timing_table"]
            )
            run_metadata = cached_bundle.get("metadata", {})
            metadata_ok = run_metadata_matches(
                run_metadata,
                expected=expected_run_metadata,
                keys=tuple(expected_run_metadata.keys()),
            )

            if has_model and metadata_ok:
                results_by_model[model_name] = {
                    "schema_version": cached_bundle["bundle_metadata"].get("schema_version"),
                    "metric_table": {model_name: cached_bundle["metric_table"][model_name]},
                    "timing_table": {model_name: cached_bundle["timing_table"][model_name]},
                    "memory_table": {
                        model_name: cached_bundle["memory_table"].get(model_name, {})
                    },
                    "oom_errors": {
                        model_name: cached_bundle["oom_errors"].get(model_name, [])
                    },
                    "metadata": run_metadata,
                }
                model_bundle_paths[model_name] = cached_bundle_path
                reused_cached_result = True
                print(f"Reused cached W&B result for {model_name}: {cached_bundle_path}")
            else:
                print(
                    f"Cached artifact for {model_name} is incompatible "
                    f"(has_model={has_model}, metadata_ok={metadata_ok}). Rerunning model."
                )

    if reused_cached_result:
        continue

    print(f"Running benchmark for model: {model_name}")
    models, configs = load_models_for_benchmark({model_name: model_config}, device=device)
    model_results = evaluate_models_over_seqlens(
        models=models,
        configs=configs,
        seqlen_list=EXPERIMENT["seqlen_list"],
        num_features=EXPERIMENT["num_features"],
        num_classes=EXPERIMENT["num_classes"],
        number_of_test_samples=EXPERIMENT["num_test_samples"],
        number_of_repetitions=EXPERIMENT["num_repetitions"],
        use_warmup_iters=EXPERIMENT["use_warmup_iters"],
        print_timing=EXPERIMENT["print_timing"],
        autocast_models=get_autocast_models_from_registry(
            {model_name: model_config},
            device=device,
        ),
        device=device,
        data_generation_seed=EXPERIMENT["data_generation_seed"],
        progress_desc=f"{model_name} progress",
        forward_models=get_forward_models_from_registry({model_name: model_config}),
    )
    results_by_model[model_name] = model_results

    model_bundle_path = make_bundle_path(
        OUTPUT_ROOT,
        f"{EXPERIMENT['name']}_{sanitize_wandb_artifact_component(model_name)}",
    )
    save_results_bundle(
        model_results,
        model_bundle_path,
        experiment=EXPERIMENT,
        include_raw_torch=True,
    )
    model_bundle_paths[model_name] = model_bundle_path
    print(f"Saved per-model bundle for {model_name}: {model_bundle_path}")

    if WANDB["enabled"]:
        artifact_ref = upload_results_bundle_to_wandb(
            model_bundle_path,
            artifact_name=model_artifact_name,
            entity=WANDB["entity"],
            project=WANDB["project"],
            run_name=(
                f"{EXPERIMENT['name']}_{sanitize_wandb_artifact_component(model_name)}_"
                f"{model_hash}"
            ),
            metadata={
                "experiment": EXPERIMENT,
                "model_name": model_name,
                "model_config": model_config,
                "model_hash": model_hash,
                "run_metadata": model_results.get("metadata", {}),
            },
        )
        print(f"Uploaded per-model artifact for {model_name}: {artifact_ref}")
