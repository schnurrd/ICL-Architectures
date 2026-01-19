#!/usr/bin/env python3
"""
Config selector for FLA backbones using CLI-provided axes.
"""

from __future__ import annotations

import torch

from pfns.model.backbones import FLABackboneConfig
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
GLOBAL_AGGREGATE_K_GRADIENTS = 2
SUPPORTED_SEQUENCE_MODES = {"cached", "causal", "teacher_forcing"}

TRAINING_PROFILES = {
    "low": (7.5e-5, 500),
    "high": (3.0e-5, 4000),
}

MODEL_SETTINGS = {
    "kda": {
        "emsize": 320,
        "config_kwargs": {
            "hidden_size": 320,
            "num_hidden_layers": 10,
            "num_heads": 4,
            "intermediate_size": 320 * 2,
            "hidden_act": "swish",
            "norm_eps": 1e-5,
            "use_cache": True,
        },
    },
    "gla": {
        "emsize": 320,
        "config_kwargs": {
            "hidden_size": 320,
            "num_hidden_layers": 12,
            "num_heads": 4,
            "intermediate_size": 320 * 2,
            "hidden_act": "swish",
            "norm_eps": 1e-5,
            "use_cache": True,
        },
    },
    "mamba2": {
        "emsize": 512,
        "config_kwargs": {
            "hidden_size": 512,
            "num_hidden_layers": 12,
            "state_size": 128,
            "conv_kernel": 4,
            "expand": 2,
            "head_dim": 64,
            "vocab_size": 1,
            "use_cache": True,
        },
    },
    "deltanet": {
        "emsize": 320,
        "config_kwargs": {
            "hidden_size": 320,
            "num_hidden_layers": 10,
            "num_heads": 4,
            "intermediate_size": 320 * 2,
            "hidden_act": "swish",
            "norm_eps": 1e-5,
            "use_cache": True,
        },
    },
    "gated_deltanet": {
        "emsize": 256,
        "config_kwargs": {
            "hidden_size": 256,
            "num_hidden_layers": 16,
            "num_heads": 4,
            "head_dim": 48,
            "intermediate_size": 256 * 2,
            "hidden_act": "swish",
            "norm_eps": 1e-5,
            "use_cache": True,
        },
    },
}

def _normalize_model_type(model_type: str) -> str:
    model_type = model_type.strip().lower().replace("-", "_").replace(" ", "_")
    if model_type == "mamba":
        return "mamba2"
    if model_type == "delta_net":
        return "deltanet"
    if model_type == "gated_delta_net":
        return "gated_deltanet"
    return model_type


def _normalize_sequence_mode(sequence_mode: str) -> str:
    sequence_mode = sequence_mode.strip().lower().replace("-", "_")
    return sequence_mode


def get_config(
    config_index: int = 0,
    model_type: str = "kda",
    sequence_mode: str = "cached",
    training_setup: str = "high",
    batch_size: int | None = None,
    cache_chunk_size: int | None = None,
    lr: float | None = None,
    aggregate_k_gradients: int | None = None,
) -> MainConfig:
    max_num_classes = 10
    max_num_features = 20

    model_type = _normalize_model_type(model_type)
    sequence_mode = _normalize_sequence_mode(sequence_mode)
    training_setup = training_setup.strip().lower()

    if model_type not in MODEL_SETTINGS:
        raise ValueError(
            f"Unknown model_type {model_type!r}. Available: {sorted(MODEL_SETTINGS)}"
        )
    if sequence_mode not in SUPPORTED_SEQUENCE_MODES:
        raise ValueError(
            f"Unknown sequence_mode {sequence_mode!r}. Available: ['cached', 'causal', 'teacher_forcing']"
        )
    if training_setup not in TRAINING_PROFILES:
        raise ValueError(
            f"Unknown training_setup {training_setup!r}. Available: {sorted(TRAINING_PROFILES)}"
        )
    profile_lr, steps_per_epoch = TRAINING_PROFILES[training_setup]
    resolved_lr = float(profile_lr) if lr is None else float(lr)
    resolved_aggregate_k = (
        aggregate_k_gradients
        if aggregate_k_gradients is not None
        else GLOBAL_AGGREGATE_K_GRADIENTS
    )
    resolved_batch_size = batch_size or DEFAULT_BATCH_SIZE
    train_mixed_precision = GLOBAL_TRAIN_MIXED_PRECISION 
    train_mixed_precision_dtype = GLOBAL_TRAIN_MIXED_PRECISION_DTYPE
    if model_type in {"deltanet"} and GLOBAL_TRAIN_MIXED_PRECISION_DTYPE == "fp32": # ChunkDeltaRuleFunction does not support fp32
        print(f"Enabling mixed precision with fp16 training for model_type {model_type!r}")
        GLOBAL_TRAIN_MIXED_PRECISION = True
        GLOBAL_TRAIN_MIXED_PRECISION_DTYPE = "fp16"

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
        max_seq_len=1000,
        min_num_features=2,
        max_num_features=max_num_features,
        fixed_num_test_instances=None,
    )

    backbone_kwargs = {
        "model_type": model_type,
        "config_kwargs": MODEL_SETTINGS[model_type]["config_kwargs"],
        "sequence_mode": sequence_mode,
    }
    if cache_chunk_size is not None:
        backbone_kwargs["cache_chunk_size"] = cache_chunk_size

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
        emsize=MODEL_SETTINGS[model_type]["emsize"],
        backbone=FLABackboneConfig(**backbone_kwargs),
        features_per_group=20,
        attention_between_features=False,
        feature_positional_embedding="subspace",
    )

    optimizer = OptimizerConfig(
        optimizer="adamw",
        lr=resolved_lr,
        weight_decay=0.01,
    )

    wandb_extras = []
    if batch_size is not None:
        wandb_extras.append(f"bs{resolved_batch_size}")
    if cache_chunk_size is not None:
        wandb_extras.append(f"cache{cache_chunk_size}")
    if lr is not None:
        wandb_extras.append(f"lr{resolved_lr:g}")
    if aggregate_k_gradients is not None:
        wandb_extras.append(f"agg{resolved_aggregate_k}")
    wandb_suffix = f"_{'_'.join(wandb_extras)}" if wandb_extras else ""
    wandb_name = (
        f"{model_type}_{sequence_mode}_{training_setup}"
        f"{wandb_suffix}"
        f"_config_{config_index}"
    )
    wandb_config = WandbConfig(
        entity="icl_arch",
        project="fla_models",
        name=wandb_name,
        mode="online",
        log_every_n_steps=10,
    )

    return MainConfig(
        priors=[prior],
        optimizer=optimizer,
        model=model,
        batch_shape_sampler=batch_shape,
        epochs=200,
        warmup_epochs=10,
        steps_per_epoch=int(steps_per_epoch),
        n_targets_per_input=1,
        train_mixed_precision=train_mixed_precision,
        train_mixed_precision_dtype=train_mixed_precision_dtype,
        scheduler="cosine_decay",
        progress_bar=True,
        wandb=wandb_config,
        num_workers=8,
        aggregate_k_gradients=resolved_aggregate_k,
    )
