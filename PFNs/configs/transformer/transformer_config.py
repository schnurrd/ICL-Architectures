#!/usr/bin/env python3
"""
Combined transformer config.
"""

from __future__ import annotations

import torch

from configs.config_utils import (
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
from pfns.model.backbones import TransformerBackboneConfig
from pfns.model.criterions import CrossEntropyConfig
from pfns.model.encoders import EncoderConfig
from pfns.run_logger import WandbConfig
from pfns.train import (
    BatchShapeSamplerConfig,
    MainConfig,
    OptimizerConfig,
    ModelConfig,
)

GLOBAL_TRAIN_MIXED_PRECISION = (
    torch.cuda.is_available()
    and torch.cuda.is_bf16_supported()
    and torch.cuda.get_device_capability()[0] >= 8
)
GLOBAL_TRAIN_MIXED_PRECISION_DTYPE = "bf16" if GLOBAL_TRAIN_MIXED_PRECISION else "fp32"

MAX_NUM_CLASSES = int(TABPFN_PRIOR_DEFAULTS["max_num_classes"])
MAX_NUM_FEATURES = int(TABPFN_PRIOR_DEFAULTS["max_num_features"])


BASE_PROFILE = {
    "nhead": 4,
    "nlayers": 12,
    "layer_kwargs": None,
}

TRAINING_PROFILES = {
    "debug": {
        **BASE_PROFILE,
        "emsize": 64,
        "nhid": 64 * 4,
        "lr": 1.5e-4,
        "steps_per_epoch": 100,
        "epochs": 200,
        "warmup_epochs": 10,
        "aggregate_k_gradients": 1,
        "attention_between_features": False,
        "features_per_group": MAX_NUM_FEATURES,
        "wandb_suffix": "_debug",
    },
    "low": {
        **BASE_PROFILE,
        "emsize": 256,
        "nhid": 256 * 4,
        "lr": 1.5e-4,
        "steps_per_epoch": 250,
        "epochs": 200,
        "warmup_epochs": 10,
        "aggregate_k_gradients": 1,
        "attention_between_features": False,
        "features_per_group": MAX_NUM_FEATURES,
        "wandb_suffix": "_low",
    },
    "high": {
        **BASE_PROFILE,
        "nlayers": 15,
        "emsize": 320,
        "nhid": 320 * 2,
        "lr": 3.0e-5,
        "steps_per_epoch": 4000,
        "epochs": 200,
        "warmup_epochs": 10,
        "aggregate_k_gradients": 2,
        "attention_between_features": False,
        "features_per_group": MAX_NUM_FEATURES,
        "wandb_suffix": "_high",
    },
    "high_feature_att": {
        **BASE_PROFILE,
        "nlayers": 10,
        "emsize": 320,
        "nhid": 320 * 2,
        "lr": 3.0e-5,
        "steps_per_epoch": 4000,
        "epochs": 200,
        "warmup_epochs": 10,
        "aggregate_k_gradients": 2,
        "attention_between_features": True,
        "features_per_group": 2,
        "wandb_suffix": "_high_feature_att",
    },
    "very_high": {
        **BASE_PROFILE,
        "emsize": 384,
        "nhid": 384 * 4,
        "lr": 3.0e-5,
        "steps_per_epoch": 4000,
        "epochs": 400,
        "warmup_epochs": 20,
        "aggregate_k_gradients": 2,
        "attention_between_features": False,
        "features_per_group": MAX_NUM_FEATURES,
        "wandb_suffix": "_very_high",
    },
    "ar": {
        **BASE_PROFILE,
        "nlayers": 15,
        "emsize": 320,
        "nhid": 320 * 2,
        "lr": 3.0e-5,
        "steps_per_epoch": 500,
        "epochs": 200,
        "warmup_epochs": 10,
        "aggregate_k_gradients": 1,
        "attention_between_features": False,
        "features_per_group": MAX_NUM_FEATURES,
        "wandb_suffix": "_ar",
    },
}

def get_config(
    config_index: int = 0,
    training_setup: str = "high",
    batch_size: int | None = None,
    max_seq_len: int | None = None,
    batch_size_stages: list[tuple[int, int]] | tuple[tuple[int, int], ...] | None = None,
    dynamic_batch_size_compensate_grad_accumulation: bool = False,
    eval_pos_split_pct: float | tuple[float, float] | list[float] | None = None,
    seq_len_stages: list[tuple[int | float | str, ...]] | tuple[tuple[int | float | str, ...], ...] | None = None,
    task_variant: str = "tabular_prior",
    interleave_x_y_pairs: bool = False,
    item_attention_use_rope: bool = False,
    item_attention_rope_base: float = 128_000.0,
    item_attention_rope_pairwise_positions: bool = False,
) -> MainConfig:
    """
    Build a config for training a TabPFN-style classifier on the synthetic
    tabpfn_prior data.
    """

    training_setup = training_setup.strip().lower()
    training_setup, is_associative_recall = resolve_training_setup_for_task(
        training_setup=training_setup,
        task_variant=task_variant,
    )
    if training_setup not in TRAINING_PROFILES:
        raise ValueError(
            f"Unknown training_setup {training_setup!r}. "
            f"Available: {sorted(TRAINING_PROFILES)}"
        )
    profile = TRAINING_PROFILES[training_setup]
    resolved_layer_kwargs = dict(profile["layer_kwargs"] or {})
    if item_attention_use_rope:
        resolved_layer_kwargs["item_attention_use_rope"] = True
        resolved_layer_kwargs["item_attention_rope_base"] = float(item_attention_rope_base)
    if item_attention_rope_pairwise_positions:
        resolved_layer_kwargs["item_attention_rope_pairwise_positions"] = True
    resolved_layer_kwargs = resolved_layer_kwargs or None

    resolved_max_seq_len = int(max_seq_len) if max_seq_len is not None else 1000
    resolved_batch_size = batch_size or 8
    resolved_batch_size_stages = resolve_batch_size_stages(batch_size_stages)
    resolved_dynamic_batch_size_compensate_grad_accumulation = bool(
        dynamic_batch_size_compensate_grad_accumulation
    )
    resolved_eval_pos_split_pct_min, resolved_eval_pos_split_pct_max = (
        resolve_eval_pos_split_pct(eval_pos_split_pct)
    )
    resolved_seq_len_stages = seq_len_stages
    resolved_epochs = profile.get("epochs", 200)
    resolved_steps_per_epoch = profile["steps_per_epoch"]

    resolved_prior_device = resolve_prior_device(max_seq_len=resolved_max_seq_len)

    prior = build_prior_for_task(
        task_variant=task_variant,
        prior_device=resolved_prior_device,
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
        seq_len_stages=resolved_seq_len_stages,
        min_num_features=2,
        max_num_features=MAX_NUM_FEATURES,
        fixed_num_test_instances=None,
    )

    model = ModelConfig(
        criterion=CrossEntropyConfig(num_classes=MAX_NUM_CLASSES),
        encoder=EncoderConfig(
            variable_num_features_normalization=True,
            nan_handling=True,  # currently only nan to mean imputation works
            use_categorical_encoder=True,
            train_normalization=True,
        ),
        y_encoder=EncoderConfig(
            nan_handling=True,
            constant_normalization_mean=0.0,
            constant_normalization_std=1.0,
        ),
        emsize=profile["emsize"],
        backbone=TransformerBackboneConfig(
            nhid=profile["nhid"],
            nlayers=profile["nlayers"],
            nhead=profile["nhead"],
            layer_kwargs=resolved_layer_kwargs,
        ),
        features_per_group=profile["features_per_group"],
        attention_between_features=profile["attention_between_features"], # was True before
        feature_positional_embedding=(
            "subspace" if profile["attention_between_features"] else None
        ),
        interleave_x_y_pairs=interleave_x_y_pairs,
    )

    optimizer = OptimizerConfig(
        optimizer="adamw",
        lr=profile["lr"],
        weight_decay=0.01,
    )

    wandb_name = f"transformer_1_gpu_v4{profile['wandb_suffix']}_{config_index}_matched"
    if interleave_x_y_pairs:
        wandb_name += "_interleaved"
    if max_seq_len is not None:
        wandb_name += f"_seq{resolved_max_seq_len}"
    if resolved_batch_size_stages:
        wandb_name += f"_bsstages{len(resolved_batch_size_stages)}"
    if resolved_dynamic_batch_size_compensate_grad_accumulation:
        wandb_name += "_dynbs_compagg"
    if eval_pos_split_pct is not None:
        wandb_name += "_evalsplit"
    if resolved_seq_len_stages:
        wandb_name += f"_stages{len(resolved_seq_len_stages)}"
    if item_attention_use_rope:
        wandb_name += "_item_rope"
        if item_attention_rope_pairwise_positions:
            wandb_name += "_pairwise"
    if is_associative_recall:
        wandb_name += "_ar"

    wandb_config = WandbConfig(
        entity="icl_arch",
        project=(
            ASSOCIATIVE_RECALL_SETTINGS["wandb_project"]
            if is_associative_recall
            else "tabpfn_transformer"
        ),
        name=wandb_name,
        tags=["matched_high_config"],
        mode="online",
        log_every_n_steps=10,
    )

    return MainConfig(
        priors=[prior],
        optimizer=optimizer,
        model=model,
        batch_shape_sampler=batch_shape,
        epochs=resolved_epochs,
        warmup_epochs=profile["warmup_epochs"],
        steps_per_epoch=resolved_steps_per_epoch,
        n_targets_per_input=1,
        train_mixed_precision=GLOBAL_TRAIN_MIXED_PRECISION,
        train_mixed_precision_dtype=GLOBAL_TRAIN_MIXED_PRECISION_DTYPE,
        scheduler="cosine_decay",
        progress_bar=True,
        wandb=wandb_config,
        num_workers=8 if resolved_prior_device == "cpu" else 0,
        aggregate_k_gradients=profile["aggregate_k_gradients"],
        validation_period=10,
        test_steps_per_epoch=500
    )
