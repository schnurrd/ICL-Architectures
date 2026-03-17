from __future__ import annotations

from pathlib import Path
from typing import Any

from .benchmark_batch_generators import create_seq_len_batch_generator
from .hashing import experiment_payload_hash
from .io import (
    SEQ_LEN_BATCH_REQUIRED_FILES,
    download_results_bundle_from_wandb,
    load_seq_len_batches,
    sanitize_wandb_artifact_component,
    save_seq_len_batches,
    upload_results_bundle_to_wandb,
)


def make_fixed_batch_artifact_name(
    *,
    base_artifact_name: str,
    experiment: dict[str, Any],
) -> str:
    experiment_hash = experiment_payload_hash(experiment_payload=experiment)
    return (
        f"{sanitize_wandb_artifact_component(base_artifact_name)}_"
        f"{sanitize_wandb_artifact_component(experiment_hash)}"
    )


def resolve_fixed_batches(
    *,
    experiment: dict[str, Any],
    output_root: str | Path,
    default_device: str,
    wandb: dict[str, Any],
    task_variant: str = "tabular_prior",
    task_kwargs: dict[str, Any] | None = None,
) -> list[Any]:
    experiment_hash = experiment_payload_hash(experiment_payload=experiment)
    bundle_dir = (
        Path(output_root)
        / "fixed_batches"
        / f"{experiment['name']}_{sanitize_wandb_artifact_component(experiment_hash)}"
    )

    if (bundle_dir / SEQ_LEN_BATCH_REQUIRED_FILES[0]).exists():
        print(f"Loaded fixed batches from local cache: {bundle_dir}")
        return load_seq_len_batches(bundle_dir)

    if wandb.get("enabled"):
        artifact_name = make_fixed_batch_artifact_name(
            base_artifact_name=wandb["batch_artifact_name"],
            experiment=experiment,
        )
        downloaded_bundle = download_results_bundle_from_wandb(
            artifact_name=artifact_name,
            entity=wandb["entity"],
            project=wandb["batch_project"],
            download_root=Path(output_root) / "wandb_batch_cache",
            required_files=SEQ_LEN_BATCH_REQUIRED_FILES,
        )
        if downloaded_bundle is not None:
            batches = load_seq_len_batches(downloaded_bundle)
            save_seq_len_batches(batches, bundle_dir)
            print(f"Loaded fixed batches from W&B cache: {downloaded_bundle}")
            return batches

    print(f"Generating fixed batches for experiment hash {experiment_hash}")
    batches = list(
        create_seq_len_batch_generator(
            task_variant=task_variant,
            num_batches=experiment["num_repetitions"],
            smallest_seqlen=min(experiment["seqlen_list"]),
            largest_seqlen=max(experiment["seqlen_list"]),
            num_features=experiment["num_features"],
            num_classes=experiment["num_classes"],
            number_of_test_samples=experiment["num_test_samples"],
            default_device=default_device,
            task_kwargs=task_kwargs,
        )
    )
    save_seq_len_batches(batches, bundle_dir)
    print(f"Saved fixed batches locally: {bundle_dir}")

    if wandb.get("enabled"):
        artifact_ref = upload_results_bundle_to_wandb(
            bundle_dir,
            artifact_name=make_fixed_batch_artifact_name(
                base_artifact_name=wandb["batch_artifact_name"],
                experiment=experiment,
            ),
            entity=wandb["entity"],
            project=wandb["batch_project"],
            run_name=(
                f"{experiment['name']}_batches_"
                f"{sanitize_wandb_artifact_component(experiment_hash)}"
            ),
            metadata={
                "experiment": experiment,
                "experiment_hash": experiment_hash,
                "bundle_type": "fixed_seq_len_batches",
            },
            job_type="seq_len_batch_bundle_upload",
        )
        print(f"Uploaded fixed batch artifact: {artifact_ref}")

    return batches
