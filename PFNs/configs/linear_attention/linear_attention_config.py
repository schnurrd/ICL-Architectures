#!/usr/bin/env python3
"""
Config selector for Linear Attention backbone training profiles.
"""

from __future__ import annotations

import torch

from pfns.model.backbones import LinearAttentionBackboneConfig
from pfns.model.criterions import CrossEntropyConfig
from pfns.model.encoders import EncoderConfig
from pfns.priors.tabpfn_prior_adapter import TabPFNPriorConfig
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

TRAINING_PROFILES = {
    "low": {
        "lr": 3.0e-5,
        "steps_per_epoch": 500,
        "epochs": 400,
        "aggregate_k_gradients": 2,
    },
    "high": {
        "lr": 3.0e-5,
        "steps_per_epoch": 4000,
        "epochs": 200,
        "aggregate_k_gradients": 2,
    },
}


def get_config(
    config_index: int = 0,
    training_setup: str = "high",
    batch_size: int | None = None,
    max_seq_len: int | None = None,
    lr: float | None = None,
    aggregate_k_gradients: int | None = None,
    interleave_x_y_pairs: bool = False,
    feature_positional_embedding: str | None = "subspace",
) -> MainConfig:
    """
    Build a config for training a TabPFN-style classifier on the synthetic
    tabpfn_prior data using a Linear Attention backbone.
    """

    max_num_classes = 10
    max_num_features = 20

    if feature_positional_embedding == "None":
        feature_positional_embedding = None

    training_setup = training_setup.strip().lower()
    if training_setup not in TRAINING_PROFILES:
        raise ValueError(
            f"Unknown training_setup {training_setup!r}. Available: {sorted(TRAINING_PROFILES)}"
        )
    profile = TRAINING_PROFILES[training_setup]

    resolved_lr = float(profile["lr"]) if lr is None else float(lr)
    resolved_batch_size = batch_size or DEFAULT_BATCH_SIZE
    resolved_max_seq_len = int(max_seq_len) if max_seq_len is not None else 1000
    resolved_aggregate_k = (
        aggregate_k_gradients
        if aggregate_k_gradients is not None
        else profile["aggregate_k_gradients"]
    )

    prior = TabPFNPriorConfig(
        prior_type="mlp",
        max_num_classes=max_num_classes,
        max_num_features=max_num_features,
        flexible=True,
        differentiable=True,
        nan_handling=True,
        return_categorical_mask=True,
    )

    batch_shape = BatchShapeSamplerConfig(
        batch_size=resolved_batch_size,
        min_single_eval_pos=24,
        max_seq_len=resolved_max_seq_len,
        min_num_features=2,
        max_num_features=max_num_features,
        fixed_num_test_instances=None,
    )

    model = ModelConfig(
        criterion=CrossEntropyConfig(num_classes=max_num_classes),
        encoder=EncoderConfig(
            variable_num_features_normalization=True,
            nan_handling=True,
            use_categorical_encoder=True,
        ),
        y_encoder=EncoderConfig(
            nan_handling=True,
            constant_normalization_mean=0.0,
            constant_normalization_std=1.0,
        ),
        emsize=512,
        backbone=LinearAttentionBackboneConfig(
            nlayers=12,
            nhead=4,
            mlp_hidden_dim=512 * 2,
            dropout=0.0,
            activation="relu",
            layer_kwargs={
                "feature_attention_softmax": False,
                #"feature_dim": 64,
            },
        ),
        features_per_group=20,
        attention_between_features=False,
        feature_positional_embedding=feature_positional_embedding,
        interleave_x_y_pairs=interleave_x_y_pairs,
    )

    optimizer = OptimizerConfig(
        optimizer="adamw",
        lr=resolved_lr,
        weight_decay=0.01,
    )

    wandb_extras = []
    if batch_size is not None:
        wandb_extras.append(f"bs{resolved_batch_size}")
    if max_seq_len is not None:
        wandb_extras.append(f"seq{resolved_max_seq_len}")
    if lr is not None:
        wandb_extras.append(f"lr{resolved_lr:g}")
    if aggregate_k_gradients is not None:
        wandb_extras.append(f"agg{resolved_aggregate_k}")
    if interleave_x_y_pairs:
        wandb_extras.append("interleaved")
    wandb_extras.append(f"fpe_{feature_positional_embedding}")
    wandb_suffix = f"_{'_'.join(wandb_extras)}" if wandb_extras else ""
    wandb_name = (
        f"linear_attention_{training_setup}"
        f"{wandb_suffix}"
        f"_config_{config_index}"
    )
    wandb_config = WandbConfig(
        entity="icl_arch",
        project="linear_attention",
        name=wandb_name,
        mode="online",
        log_every_n_steps=10,
    )

    return MainConfig(
        priors=[prior],
        optimizer=optimizer,
        model=model,
        batch_shape_sampler=batch_shape,
        epochs=profile["epochs"],
        warmup_epochs=10,
        steps_per_epoch=profile["steps_per_epoch"],
        n_targets_per_input=1,
        train_mixed_precision=GLOBAL_TRAIN_MIXED_PRECISION,
        train_mixed_precision_dtype=GLOBAL_TRAIN_MIXED_PRECISION_DTYPE,
        scheduler="cosine_decay",
        progress_bar=True,
        wandb=wandb_config,
        num_workers=8,
        aggregate_k_gradients=resolved_aggregate_k,
    )
