#!/usr/bin/env python3
"""
Config selector for Linear Attention backbone training profiles.
"""

from __future__ import annotations

import typing as tp

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
from pfns.priors.tabpfn_prior_adapter import TabPFNPriorConfig
from pfns.model.backbones import LinearAttentionBackboneConfig
from pfns.model.mode_normalization import normalize_mode_name
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
SUPPORTED_SEQUENCE_MODES = ("Comb_ST", "Comb_MT", "Non_Causal_Comb_ST")
GLOBAL_TRAIN_MIXED_PRECISION = (
    torch.cuda.is_available()
    and torch.cuda.is_bf16_supported()
    and torch.cuda.get_device_capability()[0] >= 8
)
GLOBAL_TRAIN_MIXED_PRECISION_DTYPE = "bf16" if GLOBAL_TRAIN_MIXED_PRECISION else "fp32"
MAX_NUM_CLASSES = int(TABPFN_PRIOR_DEFAULTS["max_num_classes"])
MAX_NUM_FEATURES = int(TABPFN_PRIOR_DEFAULTS["max_num_features"])


TRAINING_PROFILES = {
    "low": {
        "lr": 3.0e-5,
        "steps_per_epoch": 500,
        "epochs": 100,
        "aggregate_k_gradients": 1,
    },
    "mid": {
        "lr": 3.0e-5,
        "steps_per_epoch": 1000,
        "epochs": 100,
        "aggregate_k_gradients": 1,
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

# Model size 12.54 M
# Training speed on different gpus:
# -> In this model compiled is faster than non compiled 
#    - RTX 5070   (bf16): 20it/s, 2.3GB (uncompiled), 22it/s, 2.3GB (compiled)
#    - RTX 2080Ti (fp32): 
#    - A5000:     (bf16):


def _resolve_linear_attention_mode(
    sequence_mode: str,
) -> tuple[str, dict[str, bool]]:
    """Resolve the sequence mode and related layer kwargs for linear attention config."""
    resolved_mode = {
        "comb_st": "Comb_ST",
        "comb_mt": "Comb_MT",
        "non_causal_comb_st": "Non_Causal_Comb_ST",
    }.get(normalize_mode_name(sequence_mode))
    if resolved_mode not in SUPPORTED_SEQUENCE_MODES:
        raise ValueError(
            "Linear attention config only supports sequence_mode "
            f"{SUPPORTED_SEQUENCE_MODES}, got {sequence_mode!r}."
        )

    return resolved_mode, {
        "causal": resolved_mode == "Comb_MT",
        "causal_train_only": resolved_mode == "Comb_ST",
    }


def get_config(
    config_index: int = 0,
    training_setup: str = "high",
    task_variant: str = "tabular_prior",
    batch_size: int | None = None,
    max_seq_len: int | None = None,
    batch_size_stages: list[tuple[int, int]] | tuple[tuple[int, int], ...] | None = None,
    dynamic_batch_size_compensate_grad_accumulation: bool = False,
    eval_pos_split_pct: float | tuple[float, float] | list[float] | None = None,
    seq_len_stages: list[tuple[int | float | str, ...]] | tuple[tuple[int | float | str, ...], ...] | None = None,
    lr: float | None = None,
    aggregate_k_gradients: int | None = None,
    # Sequence mixing mode.
    sequence_mode: tp.Literal[
        "Comb_ST",
        "Comb_MT",
        "Non_Causal_Comb_ST",
    ] = "Non_Causal_Comb_ST",
    interleave_x_y_pairs: bool = False,
    nlayers: int | None = None,
    feature_positional_embedding: str | None = None,
    # Attention feature map and readout.
    normalize_q_sum: bool = False,
    normalize_k_sum: bool = False,
    scale_query_by_sqrt_dk: bool = False,
    use_k_sum_normalization: bool = True,
    use_attention_norm: bool = True,
    use_output_norm: bool = False,
    use_output_projection: bool = True,
    norm_type: str = "layernorm",
    use_mlp_norm: bool = True,
    state_renormalization: str | None = None,
    learnable_state_renorm_scale: bool = True,
    causal_chunk_size: int | None = None,
    eps: float = 1e-6,
) -> MainConfig:
    """
    Build a config for training a TabPFN-style classifier on the synthetic
    tabpfn_prior data using a Linear Attention backbone.
    """

    feature_positional_embedding = normalize_optional_none_string(
        feature_positional_embedding
    )
    state_renormalization = normalize_optional_none_string(state_renormalization)
    resolved_sequence_mode, layer_kwargs = _resolve_linear_attention_mode(sequence_mode)
    layer_kwargs = {
        **layer_kwargs,
        "causal_chunk_size": causal_chunk_size,
        "normalize_q_sum": normalize_q_sum,
        "normalize_k_sum": normalize_k_sum,
        "scale_query_by_sqrt_dk": scale_query_by_sqrt_dk,
        "use_k_sum_normalization": use_k_sum_normalization,
        "use_attention_norm": use_attention_norm,
        "norm_type": norm_type,
        "use_output_norm": use_output_norm,
        "use_output_projection": use_output_projection,
        "use_mlp_norm": use_mlp_norm,
        "state_renormalization": state_renormalization,
        "learnable_state_renorm_scale": learnable_state_renorm_scale,
        "eps": eps,
    }

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
    resolved_seq_len_stages = seq_len_stages
    resolved_nlayers = 15 if nlayers is None else int(nlayers)
    if resolved_nlayers <= 0:
        raise ValueError(f"nlayers must be >= 1, got {resolved_nlayers}.")
    resolved_epochs = profile.get("epochs", 200)
    resolved_steps_per_epoch = profile["steps_per_epoch"]
    resolved_aggregate_k = (
        aggregate_k_gradients
        if aggregate_k_gradients is not None
        else profile["aggregate_k_gradients"]
    )

    resolved_prior_device = resolve_prior_device(max_seq_len=resolved_max_seq_len)

    prior = build_prior_for_task(
        task_variant=task_variant,
        prior_device=resolved_prior_device,
        max_num_classes=MAX_NUM_CLASSES,
        max_num_features=MAX_NUM_FEATURES,
    )
    if not use_categorical_features and isinstance(prior, TabPFNPriorConfig):
        prior = TabPFNPriorConfig(
            **{
                **prior.__dict__,
                "return_categorical_mask": False,
            }
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
            nan_handling=True,
            use_categorical_encoder=use_categorical_features,
            train_normalization=True,
        ),
        y_encoder=EncoderConfig(
            nan_handling=True,
            constant_normalization_mean=0.0,
            constant_normalization_std=1.0,
        ),
        emsize=320,
        backbone=LinearAttentionBackboneConfig(
            nlayers=resolved_nlayers,
            nhead=4,
            mlp_hidden_dim=320 * 2,
            layer_kwargs=layer_kwargs,
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
    if resolved_batch_size_stages:
        wandb_extras.append(f"bsstages{len(resolved_batch_size_stages)}")
    if resolved_dynamic_batch_size_compensate_grad_accumulation:
        wandb_extras.append("dynbs_compagg")
    if eval_pos_split_pct is not None:
        wandb_extras.append("evalsplit")
    if resolved_seq_len_stages:
        wandb_extras.append(f"stages{len(resolved_seq_len_stages)}")
    if lr is not None:
        wandb_extras.append(f"lr{resolved_lr:g}")
    if aggregate_k_gradients is not None:
        wandb_extras.append(f"agg{resolved_aggregate_k}")
    if interleave_x_y_pairs:
        wandb_extras.append("interleaved")
    if resolved_sequence_mode is not None:
        wandb_extras.append(resolved_sequence_mode)
    if nlayers is not None:
        wandb_extras.append(f"layers{resolved_nlayers}")
    if not use_categorical_features:
        wandb_extras.append("nocat")
    wandb_extras.extend(
        f"{key}_{value}"
        for key, value in layer_kwargs.items()
        if key not in {"causal", "causal_train_only"}
        and value
        != {
            "use_attention_norm": True,
            "use_mlp_norm": True,
            "norm_type": "layernorm",
            "use_output_norm": False,
            "use_output_projection": True,
            "normalize_q_sum": False,
            "normalize_k_sum": False,
            "scale_query_by_sqrt_dk": False,
            "use_k_sum_normalization": True,
            "state_renormalization": None,
            "learnable_state_renorm_scale": True,
            "causal_chunk_size": None,
            "eps": 1e-6,
        }.get(key)
    )
    wandb_extras.append(f"fpe_{feature_positional_embedding}")
    wandb_suffix = f"_{'_'.join(wandb_extras)}" if wandb_extras else ""
    wandb_name = (
        f"linear_attention_{training_setup}"
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
            else "linear_attention"
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
