#!/usr/bin/env python3
"""
Config for the two-axis (row/column) tabular backbone.
"""

from __future__ import annotations

import torch

from configs.config_utils import (
    TRAINING_PROFILES,
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
from pfns.priors.tabpfn_prior_adapter import TabPFNPriorConfig
from pfns.model.backbones import TwoAxisBackboneConfig
from pfns.model.mode_normalization import resolve_sequence_mode
from pfns.model.criterions import CrossEntropyConfig
from pfns.model.encoders import EncoderConfig
from pfns.run_logger import WandbConfig
from pfns.train import (
    BatchShapeSamplerConfig,
    MainConfig,
    OptimizerConfig,
    ModelConfig,
)

DEFAULT_BATCH_SIZE = 8
GLOBAL_TRAIN_MIXED_PRECISION = (
    torch.cuda.is_available()
    and torch.cuda.is_bf16_supported()
    and torch.cuda.get_device_capability()[0] >= 8
)
GLOBAL_TRAIN_MIXED_PRECISION_DTYPE = "bf16" if GLOBAL_TRAIN_MIXED_PRECISION else "fp32"
GLOBAL_AGGREGATE_K_GRADIENTS = 2
MAX_NUM_CLASSES = int(TABPFN_PRIOR_DEFAULTS["max_num_classes"])
MAX_NUM_FEATURES = int(TABPFN_PRIOR_DEFAULTS["max_num_features"])

MODEL_SETTINGS = {
    "emsize": 320,
    "nlayers": 12,
    "nhead": 4,
    "dim_feedforward": 320 * 2,
    "features_per_group": 4,
}


def get_config(
    config_index: int = 0,
    # Architecture
    row_attention: str = "deltanet",
    sequence_mode: str = "Comb_ST",
    hidden_size: int | None = None,
    nlayers: int | None = None,
    nhead: int | None = None,
    features_per_group: int | None = None,
    task_variant: str = "tabular_prior",
    # Training
    training_setup: str = "high",
    batch_size: int | None = None,
    max_seq_len: int | None = None,
    batch_size_stages: list[tuple[int, int]] | tuple[tuple[int, int], ...] | None = None,
    dynamic_batch_size_compensate_grad_accumulation: bool = False,
    eval_pos_split_pct: float | tuple[float, float] | list[float] | None = None,
    seq_len_stages: list[tuple[int | float | str, ...]] | tuple[tuple[int | float | str, ...], ...] | None = None,
    lr: float | None = None,
    steps_per_epoch: int | None = None,
    aggregate_k_gradients: int | None = None,
    recompute_layer: bool = True,
    use_categorical_features: bool = True,
    feature_positional_embedding: str | None = None,
) -> MainConfig:
    """Build a MainConfig for two-axis DeltaNet training."""
    feature_positional_embedding = normalize_optional_none_string(
        feature_positional_embedding
    )
    sequence_mode = resolve_sequence_mode(sequence_mode)
    if sequence_mode != "Comb_ST":
        raise ValueError(
            "The two-axis DeltaNet backbone supports only sequence_mode='Comb_ST', "
            f"got {sequence_mode!r}."
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
    resolved_steps_per_epoch = (
        int(steps_per_epoch)
        if steps_per_epoch is not None
        else int(profile["steps_per_epoch"])
    )
    resolved_max_seq_len = int(max_seq_len) if max_seq_len is not None else 1000
    resolved_batch_size_stages = resolve_batch_size_stages(batch_size_stages)
    resolved_eval_pos_split_pct_min, resolved_eval_pos_split_pct_max = (
        resolve_eval_pos_split_pct(eval_pos_split_pct)
    )
    resolved_aggregate_k = (
        int(aggregate_k_gradients)
        if aggregate_k_gradients is not None
        else 1 if is_associative_recall else GLOBAL_AGGREGATE_K_GRADIENTS
    )
    resolved_batch_size = batch_size or DEFAULT_BATCH_SIZE

    train_mixed_precision = GLOBAL_TRAIN_MIXED_PRECISION
    train_mixed_precision_dtype = GLOBAL_TRAIN_MIXED_PRECISION_DTYPE

    resolved_prior_device = resolve_prior_device(max_seq_len=resolved_max_seq_len)
    prior = build_prior_for_task(
        task_variant=task_variant,
        prior_device=resolved_prior_device,
        max_num_classes=MAX_NUM_CLASSES,
        max_num_features=MAX_NUM_FEATURES,
    )
    if not use_categorical_features and isinstance(prior, TabPFNPriorConfig):
        prior = TabPFNPriorConfig(
            **{**prior.__dict__, "return_categorical_mask": False}
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
        dynamic_batch_size_compensate_grad_accumulation=bool(
            dynamic_batch_size_compensate_grad_accumulation
        ),
        eval_pos_split_pct_min=resolved_eval_pos_split_pct_min,
        eval_pos_split_pct_max=resolved_eval_pos_split_pct_max,
        seq_len_stages=seq_len_stages,
        min_num_features=2,
        max_num_features=MAX_NUM_FEATURES,
        fixed_num_test_instances=None,
    )

    resolved_emsize = int(hidden_size) if hidden_size is not None else MODEL_SETTINGS["emsize"]
    resolved_nlayers = int(nlayers) if nlayers is not None else MODEL_SETTINGS["nlayers"]
    resolved_nhead = int(nhead) if nhead is not None else MODEL_SETTINGS["nhead"]
    resolved_features_per_group = (
        int(features_per_group)
        if features_per_group is not None
        else MODEL_SETTINGS["features_per_group"]
    )
    resolved_dim_feedforward = (
        MODEL_SETTINGS["dim_feedforward"]
        if hidden_size is None
        else resolved_emsize * 2
    )

    model = ModelConfig(
        criterion=CrossEntropyConfig(num_classes=MAX_NUM_CLASSES),
        encoder=EncoderConfig(
            variable_num_features_normalization=True,
            nan_handling=True,
            use_categorical_encoder=use_categorical_features,
            train_normalization=True,
        ),
        y_encoder=EncoderConfig(
            nan_handling=True,
            constant_normalization_mean=0.0,
            constant_normalization_std=1.0,
        ),
        emsize=resolved_emsize,
        backbone=TwoAxisBackboneConfig(
            nlayers=resolved_nlayers,
            nhead=resolved_nhead,
            dim_feedforward=resolved_dim_feedforward,
            row_attention=row_attention,
            recompute_layer=recompute_layer,
        ),
        features_per_group=resolved_features_per_group,
        attention_between_features=True,
        feature_positional_embedding=feature_positional_embedding,
        interleave_x_y_pairs=False,
    )

    optimizer = OptimizerConfig(
        optimizer="adamw",
        lr=resolved_lr,
        weight_decay=0.01,
    )

    extras = [
        f"emb{resolved_emsize}",
        f"layers{resolved_nlayers}",
        f"heads{resolved_nhead}",
        f"fpg{resolved_features_per_group}",
        f"row_{row_attention}",
        f"bs{resolved_batch_size}" if batch_size else None,
        f"seq{resolved_max_seq_len}" if max_seq_len else None,
        f"lr{resolved_lr:g}" if lr else None,
        f"agg{resolved_aggregate_k}" if aggregate_k_gradients else None,
        f"steps{resolved_steps_per_epoch}" if steps_per_epoch else None,
        "nocat" if not use_categorical_features else None,
        f"fpe_{feature_positional_embedding}",
    ]
    extras_str = "_".join(e for e in extras if e)
    wandb_name = (
        f"two_axis_{sequence_mode}_{training_setup}_{extras_str}"
        f"_config_{config_index}_matched"
    )
    if is_associative_recall:
        wandb_name += "_ar"

    wandb_config = WandbConfig(
        entity="icl_arch",
        project=(
            ASSOCIATIVE_RECALL_SETTINGS["wandb_project"]
            if is_associative_recall
            else "fla_models"
        ),
        name=wandb_name,
        tags=[
            "matched_high_config",
            "model_two_axis",
            f"row_attention_{row_attention}",
            "two_axis_attention",
            f"emb_{resolved_emsize}",
            f"layers_{resolved_nlayers}",
        ],
        mode="online",
        log_every_n_steps=10,
    )

    return MainConfig(
        priors=[prior],
        optimizer=optimizer,
        model=model,
        batch_shape_sampler=batch_shape,
        epochs=int(profile.get("epochs", 200)),
        warmup_epochs=10,
        steps_per_epoch=resolved_steps_per_epoch,
        n_targets_per_input=1,
        train_mixed_precision=train_mixed_precision,
        train_mixed_precision_dtype=train_mixed_precision_dtype,
        scheduler="cosine_decay",
        progress_bar=True,
        wandb=wandb_config,
        num_workers=8 if resolved_prior_device == "cpu" else 0,
        aggregate_k_gradients=resolved_aggregate_k,
        validation_period=10,
        test_steps_per_epoch=500,
    )
