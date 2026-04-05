from __future__ import annotations

from typing import Any

import torch

from configs.config_utils import (
    normalize_optional_none_string,
    resolve_batch_size_stages,
    resolve_eval_pos_split_pct,
    resolve_prior_device,
)
from pfns.prior_defaults import (
    ASSOCIATIVE_RECALL_SETTINGS,
    TABPFN_PRIOR_DEFAULTS,
    build_prior_for_task,
    resolve_training_setup_for_task,
)
from pfns.model.backbones import BackboneConfig
from pfns.model.criterions import CrossEntropyConfig
from pfns.model.encoders import EncoderConfig
from pfns.run_logger import WandbConfig
from pfns.train import (
    BatchShapeSamplerConfig,
    MainConfig,
    ModelConfig,
    OptimizerConfig,
)

DEFAULT_BATCH_SIZE = 8
GLOBAL_TRAIN_MIXED_PRECISION = (
    torch.cuda.is_available()
    and torch.cuda.is_bf16_supported()
    and torch.cuda.get_device_capability()[0] >= 8
)
GLOBAL_TRAIN_MIXED_PRECISION_DTYPE = "bf16" if GLOBAL_TRAIN_MIXED_PRECISION else "fp32"
MAX_NUM_CLASSES = int(TABPFN_PRIOR_DEFAULTS["max_num_classes"])
MAX_NUM_FEATURES = int(TABPFN_PRIOR_DEFAULTS["max_num_features"])

TRAINING_PROFILES = {
    "debug": {"lr": 3.0e-5, "steps_per_epoch": 20, "epochs": 50, "aggregate_k_gradients": 1},
    "low": {"lr": 3.0e-5, "steps_per_epoch": 500, "epochs": 400, "aggregate_k_gradients": 2},
    "high": {"lr": 3.0e-5, "steps_per_epoch": 4000, "epochs": 200, "aggregate_k_gradients": 2},
    "ar": {"lr": 3.0e-5, "steps_per_epoch": 500, "epochs": 200, "aggregate_k_gradients": 1},
}


def build_hybrid_batch_delta_main_config(
    *,
    config_index: int,
    training_setup: str,
    task_variant: str,
    batch_size: int | None,
    max_seq_len: int | None,
    batch_size_stages: list[tuple[int, int]] | tuple[tuple[int, int], ...] | None,
    dynamic_batch_size_compensate_grad_accumulation: bool,
    eval_pos_split_pct: float | tuple[float, float] | list[float] | None,
    seq_len_stages: list[tuple[int | float | str, ...]] | tuple[tuple[int | float | str, ...], ...] | None,
    lr: float | None,
    aggregate_k_gradients: int | None,
    feature_positional_embedding: str | None,
    emsize: int,
    features_per_group: int,
    backbone: BackboneConfig,
    wandb_name_prefix: str,
    wandb_tags: list[str],
    extra_name_parts: list[str],
) -> MainConfig:
    feature_positional_embedding = normalize_optional_none_string(
        feature_positional_embedding
    )
    training_setup = training_setup.strip().lower()
    training_setup, is_associative_recall = resolve_training_setup_for_task(
        training_setup=training_setup,
        task_variant=task_variant,
    )
    if training_setup not in TRAINING_PROFILES:
        raise ValueError(
            f"Unknown training_setup {training_setup!r}. Available: {sorted(TRAINING_PROFILES)}"
        )
    profile = TRAINING_PROFILES[training_setup]

    resolved_lr = float(profile["lr"]) if lr is None else float(lr)
    resolved_batch_size = batch_size or DEFAULT_BATCH_SIZE
    resolved_max_seq_len = int(max_seq_len) if max_seq_len is not None else 1000
    resolved_batch_size_stages = resolve_batch_size_stages(batch_size_stages)
    resolved_dynamic_batch_size_compensate_grad_accumulation = bool(
        dynamic_batch_size_compensate_grad_accumulation
    )
    resolved_eval_pos_split_pct_min, resolved_eval_pos_split_pct_max = (
        resolve_eval_pos_split_pct(eval_pos_split_pct)
    )
    resolved_aggregate_k = (
        int(aggregate_k_gradients)
        if aggregate_k_gradients is not None
        else int(profile["aggregate_k_gradients"])
    )

    prior = build_prior_for_task(
        task_variant=task_variant,
        prior_device=resolve_prior_device(max_seq_len=resolved_max_seq_len),
        max_num_classes=MAX_NUM_CLASSES,
        max_num_features=MAX_NUM_FEATURES,
    )
    batch_shape = BatchShapeSamplerConfig(
        batch_size=resolved_batch_size,
        min_single_eval_pos=(
            ASSOCIATIVE_RECALL_SETTINGS["min_single_eval_pos"]
            if is_associative_recall
            else 64
        ),
        max_seq_len=resolved_max_seq_len,
        batch_size_stages=resolved_batch_size_stages,
        dynamic_batch_size_compensate_grad_accumulation=resolved_dynamic_batch_size_compensate_grad_accumulation,
        eval_pos_split_pct_min=resolved_eval_pos_split_pct_min,
        eval_pos_split_pct_max=resolved_eval_pos_split_pct_max,
        seq_len_stages=seq_len_stages,
        min_num_features=2,
        max_num_features=MAX_NUM_FEATURES,
        fixed_num_test_instances=None,
    )
    model = ModelConfig(
        criterion=CrossEntropyConfig(num_classes=MAX_NUM_CLASSES),
        encoder=EncoderConfig(
            variable_num_features_normalization=True,
            nan_handling=True,
            use_categorical_encoder=True,
            train_normalization=True,
        ),
        y_encoder=EncoderConfig(
            nan_handling=True,
            constant_normalization_mean=0.0,
            constant_normalization_std=1.0,
        ),
        emsize=emsize,
        backbone=backbone,
        features_per_group=features_per_group,
        attention_between_features=False,
        feature_positional_embedding=feature_positional_embedding,
        interleave_x_y_pairs=False,
    )
    optimizer = OptimizerConfig(
        optimizer="adamw",
        lr=resolved_lr,
        weight_decay=0.01,
    )

    extras = [
        *extra_name_parts,
        f"fpg{features_per_group}",
        f"bs{resolved_batch_size}" if batch_size is not None else None,
        f"seq{resolved_max_seq_len}" if max_seq_len is not None else None,
        f"bsstages{len(resolved_batch_size_stages)}" if resolved_batch_size_stages else None,
        (
            "dynbs_compagg"
            if resolved_dynamic_batch_size_compensate_grad_accumulation
            else None
        ),
        "evalsplit" if eval_pos_split_pct is not None else None,
        f"stages{len(seq_len_stages)}" if seq_len_stages else None,
        f"lr{resolved_lr:g}" if lr is not None else None,
        f"agg{resolved_aggregate_k}" if aggregate_k_gradients is not None else None,
        f"fpe_{feature_positional_embedding}",
    ]
    extras_str = "_".join(x for x in extras if x)
    wandb_name = (
        f"{wandb_name_prefix}_{training_setup}_{extras_str}_config_{config_index}_matched"
    )
    if is_associative_recall:
        wandb_name += "_ar"

    wandb_config = WandbConfig(
        entity="icl_arch",
        project=(
            ASSOCIATIVE_RECALL_SETTINGS["wandb_project"]
            if is_associative_recall
            else "transformer_custom_layers"
        ),
        name=wandb_name,
        tags=wandb_tags,
        mode="online",
        log_every_n_steps=10,
    )

    return MainConfig(
        priors=[prior],
        optimizer=optimizer,
        model=model,
        batch_shape_sampler=batch_shape,
        epochs=int(profile["epochs"]),
        steps_per_epoch=int(profile["steps_per_epoch"]),
        aggregate_k_gradients=resolved_aggregate_k,
        n_targets_per_input=1,
        train_mixed_precision=GLOBAL_TRAIN_MIXED_PRECISION,
        train_mixed_precision_dtype=GLOBAL_TRAIN_MIXED_PRECISION_DTYPE,
        skip_grad_norm_spike_factor=5.0,
        scheduler="cosine_decay",
        warmup_epochs=10,
        min_lr=2e-6,
        validation_period=10,
        test_steps_per_epoch=500,
        verbose=True,
        progress_bar=True,
        wandb=wandb_config,
    )
