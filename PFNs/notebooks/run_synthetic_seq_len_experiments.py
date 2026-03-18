import argparse
from pathlib import Path

from pfns.utils import get_default_device

from pfns.experiments.model_benchmarks.evaluation import evaluate_models_over_seqlens
from pfns.experiments.model_benchmarks.fixed_batches import resolve_fixed_batches
from pfns.experiments.model_benchmarks.hashing import single_model_hash
from pfns.experiments.model_benchmarks.io import (
    SEQ_LEN_REQUIRED_FILES,
    download_results_bundle_from_wandb,
    load_results_bundle,
    make_bundle_path,
    make_model_artifact_name,
    sanitize_wandb_artifact_component,
    save_results_bundle,
    upload_results_bundle_to_wandb,
)
from pfns.experiments.model_benchmarks.path_utils import build_repo_output_root
from pfns.experiments.model_benchmarks.models import load_models_for_benchmark
from pfns.experiments.model_benchmarks.model_registry import get_all_models
from pfns.experiments.model_benchmarks.model_registry import (
    MODEL_FAMILIES,
    get_autocast_models_from_registry,
    get_forward_models_from_registry,
    get_models_from_families,
    get_models_from_names,
)

DEFAULT_MODEL_FAMILIES_FOR_RUN = MODEL_FAMILIES.keys()  # By default, run all families 

from pfns.experiments.model_benchmarks.workflows import (
    alias_single_model_seq_len_bundle,
    build_seq_len_run_metadata,
    merge_seq_len_model_results,
    seq_len_bundle_is_compatible,
    single_model_seq_len_result_from_bundle,
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
    "batch_artifact_name": "seq_len_batches",
    "batch_project": "seq_len_batch_cache",
}


def parse_cli_args():
    parser = argparse.ArgumentParser(
        description="Run sequence length benchmark experiments."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help=(
            "Optional exact model names to evaluate. "
            "Example: --models Softmax_Transformer Rebased"
        ),
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
    args, _ = parser.parse_known_args()
    if args.num_repetitions < 1:
        parser.error("--num-repetitions must be >= 1")
    if args.num_runs < 1:
        parser.error("--num-runs must be >= 1")
    if args.run_index < 0 or args.run_index >= args.num_runs:
        parser.error("--run-index must be in [0, num-runs-1]")
    return args


CLI_ARGS = parse_cli_args()
EXPERIMENT["num_repetitions"] = CLI_ARGS.num_repetitions

OUTPUT_ROOT = build_repo_output_root(__file__, "seq_len")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

print(f"Results are stored in: {OUTPUT_ROOT}")
print(f"Available model families: {list(MODEL_FAMILIES)}")
print(f"Configured repetitions: {EXPERIMENT['num_repetitions']}")
print(
    f"Sharding config: num_runs={CLI_ARGS.num_runs}, run_index={CLI_ARGS.run_index}"
)

# Example by family:
# models_to_compare = get_models_from_families(["transformer"])

# models_to_compare = get_models_from_names([
#     "Softmax_Transformer",
#     "Rebased",
#     "Linear_Attention",
# ])

if CLI_ARGS.models:
    all_models_to_compare = get_models_from_names(CLI_ARGS.models)
else:
    all_models_to_compare = get_all_models()
all_model_items = list(all_models_to_compare.items())
models_to_compare = dict(all_model_items[CLI_ARGS.run_index::CLI_ARGS.num_runs])
if not models_to_compare:
    print(
        f"No models assigned to run_index={CLI_ARGS.run_index} "
        f"with num_runs={CLI_ARGS.num_runs} (total models={len(all_model_items)})."
    )
    raise SystemExit(0)

device = str(get_default_device())
print(f"Using device: {device}")
print(f"Models assigned to this run: {len(models_to_compare)} / {len(all_model_items)}")
if CLI_ARGS.models:
    print(f"Requested models: {CLI_ARGS.models}")

expected_run_metadata = build_seq_len_run_metadata(experiment=EXPERIMENT, device=device)
precomputed_batches = resolve_fixed_batches(
    experiment=EXPERIMENT,
    output_root=OUTPUT_ROOT,
    default_device=device,
    wandb=WANDB,
)
print(f"Using {len(precomputed_batches)} fixed batches for this experiment.")

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
            cached_bundle_for_model, aliased_from = alias_single_model_seq_len_bundle(
                cached_bundle,
                target_model_name=model_name,
            )
            if seq_len_bundle_is_compatible(
                cached_bundle_for_model,
                model_name=model_name,
                expected_metadata=expected_run_metadata,
            ):
                results_by_model[model_name] = single_model_seq_len_result_from_bundle(
                    cached_bundle_for_model,
                    model_name=model_name,
                )
                model_bundle_paths[model_name] = cached_bundle_path
                reused_cached_result = True
                if aliased_from is not None and aliased_from != model_name:
                    print(
                        f"Reused cached W&B result for {model_name} from stored label "
                        f"{aliased_from}: {cached_bundle_path}"
                    )
                else:
                    print(f"Reused cached W&B result for {model_name}: {cached_bundle_path}")
            else:
                print(
                    f"Cached artifact for {model_name} is incompatible with "
                    "this run metadata. Rerunning model."
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
        subsample_dataset_size=model_config.get("subsample_dataset_size"),
        progress_desc=f"{model_name} progress",
        forward_models=get_forward_models_from_registry({model_name: model_config}),
        precomputed_batches=precomputed_batches,
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
