#!/usr/bin/env python3
"""
Config selector for Rebased backbone training profiles.
"""

from __future__ import annotations

import torch

from pfns.prior_defaults import (
    ASSOCIATIVE_RECALL_SETTINGS,
    TABPFN_PRIOR_DEFAULTS,
    build_prior_for_task,
    resolve_training_setup_for_task,
)
from pfns.model.backbones import RebasedBackboneConfig
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
MAX_NUM_CLASSES = int(TABPFN_PRIOR_DEFAULTS["max_num_classes"])
MAX_NUM_FEATURES = int(TABPFN_PRIOR_DEFAULTS["max_num_features"])

# Training speed on different gpus:
# -> In this model compiled is faster than non compiled (feature dim 32)
#    - RTX 5070 (bf16):   2.25it/s, 4.7GB
#    - RTX 2080Ti (fp32): 
#    - A5000:     (bf16):

# Based Model speed with 12.79M parameters
#    - RTX 5070 (bf16):   2.4it/s, 6.8 GB
#    - RTX 2080Ti (fp32): 
#    - A5000:     (bf16):

TRAINING_PROFILES = {
    "low": {
        "lr": 3.0e-5,
        "steps_per_epoch": 500,
        "epochs": 200,
        "aggregate_k_gradients": 2,
    },
    "high": {
        "lr": 3.0e-5,
        "steps_per_epoch": 4000,
        "epochs": 200,
        "aggregate_k_gradients": 2,
    },
    "ar": {
        "lr": 3.0e-5,
        "steps_per_epoch": 500,
        "aggregate_k_gradients": 1,
        "epochs": 200,
    },
}


def get_config(
    config_index: int = 0,
    training_setup: str = "high",
    task_variant: str = "tabular_prior",
    batch_size: int | None = None,
    max_seq_len: int | None = None,
    seq_len_choices: list[int] | tuple[int, ...] | None = None,
    seq_len_choice_weights: list[float] | tuple[float, ...] | None = None,
    seq_len_curriculum_start: int | None = None,
    seq_len_curriculum_warmup_epochs: int = 0,
    seq_len_choice_weight_exponent: float | None = None,
    dynamic_batch_size_power: int = 0,
    dynamic_batch_size_compensate_grad_accumulation: bool = False,
    lr: float | None = None,
    aggregate_k_gradients: int | None = None,
    interleave_x_y_pairs: bool = False,
    feature_positional_embedding: str | None = None,
    feature_map: str = "rebased",
    feature_dim: int | None = None,
    dense: bool = True,
    eps: float = 1e-5,
    gradient_checkpointing: bool = False,
    recompute_every_n_layers: int | None = None,
) -> MainConfig:
    """
    Build a config for training a TabPFN-style classifier on the synthetic
    tabpfn_prior data using a Rebased backbone.
    """

    if feature_positional_embedding == "None":
        feature_positional_embedding = None

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
    resolved_seq_len_choices = (
        [int(v) for v in seq_len_choices] if seq_len_choices is not None else None
    )
    resolved_seq_len_choice_weights = (
        [float(v) for v in seq_len_choice_weights]
        if seq_len_choice_weights is not None
        else None
    )
    resolved_seq_len_curriculum_start = (
        int(seq_len_curriculum_start)
        if seq_len_curriculum_start is not None
        else None
    )
    resolved_seq_len_curriculum_warmup_epochs = int(seq_len_curriculum_warmup_epochs)
    resolved_seq_len_choice_weight_exponent = (
        float(seq_len_choice_weight_exponent)
        if seq_len_choice_weight_exponent is not None
        else None
    )
    resolved_dynamic_batch_size_power = int(dynamic_batch_size_power)
    resolved_dynamic_batch_size_compensate_grad_accumulation = bool(
        dynamic_batch_size_compensate_grad_accumulation
    )
    resolved_epochs = profile.get("epochs", 200)
    resolved_steps_per_epoch = profile["steps_per_epoch"]
    resolved_aggregate_k = (
        aggregate_k_gradients
        if aggregate_k_gradients is not None
        else profile["aggregate_k_gradients"]
    )
    resolved_feature_dim = 32 if feature_dim is None else int(feature_dim)

    resolved_feature_map = feature_map.strip().lower().replace("-", "_")
    if resolved_feature_map not in {"rebased", "based"}:
        raise ValueError(
            f"Unknown feature_map {feature_map!r}. Available: ['rebased', 'based']"
        )
    resolved_dense = bool(dense)
    resolved_eps = float(eps)
    resolved_gradient_checkpointing = bool(gradient_checkpointing)
    resolved_recompute_every_n_layers = (
        None if recompute_every_n_layers is None else int(recompute_every_n_layers)
    )

    resolved_prior_device = "cuda" if torch.cuda.is_available() and resolved_max_seq_len > 2000 else "cpu" # use cuda only for very long sequences

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
        seq_len_choices=resolved_seq_len_choices,
        seq_len_choice_weights=resolved_seq_len_choice_weights,
        seq_len_curriculum_start=resolved_seq_len_curriculum_start,
        seq_len_curriculum_warmup_epochs=resolved_seq_len_curriculum_warmup_epochs,
        seq_len_choice_weight_exponent=resolved_seq_len_choice_weight_exponent,
        dynamic_batch_size_power=resolved_dynamic_batch_size_power,
        dynamic_batch_size_compensate_grad_accumulation=resolved_dynamic_batch_size_compensate_grad_accumulation,
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
        ),
        y_encoder=EncoderConfig(
            nan_handling=True,
            constant_normalization_mean=0.0,
            constant_normalization_std=1.0,
        ),
        # Model size 12.79M
        emsize=320,
        backbone=RebasedBackboneConfig(
            nlayers=18,
            mlp_hidden_dim=320 * 2,
            num_heads=4,
            activation="silu",
            dropout=0.0,
            recompute_layer=(
                resolved_gradient_checkpointing
                and resolved_recompute_every_n_layers is not None
            ),
            recompute_every_n_layers=resolved_recompute_every_n_layers,
            layer_kwargs={
                "feature_dim": resolved_feature_dim,
                "feature_map": resolved_feature_map,
                "dense": resolved_dense,
                "gradient_checkpointing": resolved_gradient_checkpointing,
                "use_gamma": True,
                "use_beta": True,
                "normalize": True,
                "eps": resolved_eps,
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
    wandb_extras.append(f"fm_{resolved_feature_map}")
    if feature_dim is not None:
        wandb_extras.append(f"fd_{resolved_feature_dim}")
    wandb_extras.append(f"dense_{int(resolved_dense)}")
    if eps != 1e-5:
        wandb_extras.append(f"eps_{resolved_eps:g}")
    if gradient_checkpointing:
        wandb_extras.append("gc_1")
    if (
        resolved_gradient_checkpointing
        and resolved_recompute_every_n_layers is not None
        and resolved_recompute_every_n_layers != 1
    ):
        wandb_extras.append(f"recompn_{resolved_recompute_every_n_layers}")
    wandb_extras.append(f"fpe_{feature_positional_embedding}")
    wandb_suffix = f"_{'_'.join(wandb_extras)}" if wandb_extras else ""
    wandb_name = (
        f"rebased_{training_setup}"
        f"{wandb_suffix}"
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
        warmup_epochs=10,
        steps_per_epoch=resolved_steps_per_epoch,
        n_targets_per_input=1,
        train_mixed_precision=GLOBAL_TRAIN_MIXED_PRECISION,
        train_mixed_precision_dtype=GLOBAL_TRAIN_MIXED_PRECISION_DTYPE,
        scheduler="cosine_decay",
        progress_bar=True,
        wandb=wandb_config,
        num_workers=8 if resolved_prior_device == "cpu" else 0,
        aggregate_k_gradients=resolved_aggregate_k,
        validation_period=10,
        test_steps_per_epoch=500
    )
