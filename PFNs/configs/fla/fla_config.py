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
    "low": (6.0e-5, 1000),
    "high": (3.0e-5, 4000),
}

MODEL_SETTINGS = {
    # KDA Config: https://github.com/fla-org/flash-linear-attention/blob/3cf180339b8a1cbad823f553541cd531d18670ea/fla/models/kda/configuration_kda.py#L10
    # Model size: 12.09 M
    # Training speed on different gpus (uncompiled, single target): 
    #    - RTX 5070 (bf16):   16it/s, 3.4GiB (single target); 19it/s, 2.2GiB (multi target); 10it/s, 3.7GiB (multi target, interleaved)
    #    - RTX 2080Ti:        4it/s, 6.2GB (non-compiled), 
    #    - A5000:             6it/s (non-compiled), 
    "kda": {
        "emsize": 320,
        "config_kwargs": { # per default runs in chunked mode, has a max_position_embeddings set to 2048, supports attn dict
            "hidden_size": 320, # default 2048
            "use_short_conv": False, # typically true but we don't have temporal data
            "num_heads": 4, # default 16
            # "head_dim": 80, # currently 128
            "intermediate_size": 320 * 2, # default None -> 4*hidden_size
            "hidden_act": "swish",
            "num_hidden_layers": 8, # default 24
            "norm_eps": 1e-6, # default 1e-6
            "use_cache": True,
            "vocab_size": 1, # dummy value, not used default 32000
            # "cache_chunk_size": 16,  
        },
    },
    # GLA Config: https://github.com/fla-org/flash-linear-attention/blob/3cf180339b8a1cbad823f553541cd531d18670ea/fla/models/gla/configuration_gla.py#L12
    # Model size: 12.59 M
    # Training speed on different gpus (uncompiled, single target): 
    #    - RTX 5070:   22it/s, 1.8GiB (single target); 28it/s, 1.6GiB (multi target); 16it/s, 2.6GiB (multi target, interleaved)
    #    - RTX 2080Ti:  it/s
    #    - A5000:       it/s 
    "gla": {
        "emsize": 320,
        "config_kwargs": { # also has max_position_embeddings set to 2048, supports attn dict
            "hidden_size": 320, # default 2048
            "use_short_conv": False, 
            "num_heads": 4, # default 4
            "num_hidden_layers": 12, # default 24
            "intermediate_size": 320 * 2, # default None -> 4*hidden_size
            "hidden_act": "swish",
            "norm_eps": 1e-6, # default 1e-6
            "use_cache": True,
            "vocab_size": 1, # dummy value, not used default 32000
        },
    },
    # Mamba2 Config: https://github.com/fla-org/flash-linear-attention/blob/3cf180339b8a1cbad823f553541cd531d18670ea/fla/models/mamba2/configuration_mamba2.py#L21
    # Model size: 12.86 M
    # Training speed on different gpus (uncompiled): 
    #    - RTX 5070 (bf16):   7it/s, 4.3GB (single target); 5it/s (single target, interleaved); 15it/s, 2GB (multi target); 8it/s, 2.9GiB (multi target, interleaved)
    #    - RTX 2080Ti:  it/s
    #    - A5000:        it/s 
    "mamba2": { # cached currently not patched
        "emsize": 320,
        "config_kwargs": {
            "hidden_size": 320, # default 2048
            "num_hidden_layers": 18, # default 48
            "state_size": 128, # default 128
            "conv_kernel": 4, # default 4
            "expand": 2, # default 2, --> num_heads self.expand * hidden_size // head_dim
            "head_dim": 128, # default
            "vocab_size": 1, # dummy value, not used default 32000
            "use_cache": True,
            "cache_chunk_size": 16,  
        },
    },
    # DeltaNet Config: https://github.com/fla-org/flash-linear-attention/blob/3cf180339b8a1cbad823f553541cd531d18670ea/fla/models/delta_net/configuration_delta_net.py#L7
    # Model size: 10.46 M -> increased layers to 12 -> 12.52 M
    # Training speed on different gpus (uncompiled): 
    #    - RTX 5070 (bf16):   22it/s, 2.2GB (single target); 29it/s, 1.7GB (multi target); 16it/s, 2.8GiB (multi target, interleaved)
    #    - RTX 2080Ti:  it/s
    #    - A5000:        it/s 
    "deltanet": {
        "emsize": 320,
        "config_kwargs": {
            "hidden_size": 320,
            "num_hidden_layers": 12, # default 24
            "num_heads": 4, # default 16
            "intermediate_size": 320 * 2, # default None -> 4*hidden_size
            "hidden_act": "swish",
            "norm_eps": 1e-6, # default 1e-6
            "use_cache": True,
            "use_short_conv": False,
            "vocab_size": 1, # dummy value, not used default 32000
        },
    },
    # Gated DeltaNet Config: https://github.com/fla-org/flash-linear-attention/blob/3cf180339b8a1cbad823f553541cd531d18670ea/fla/models/gated_deltanet/configuration_gated_deltanet.py#L7
    # Model size: 12.93 M
    # Training speed on different gpus (uncompiled): 
    #    - RTX 5070 (bf16):   14it/s, 2.9GB (single target); 24it/s, 1.9GB (multi target); 15it/s, 3.5GiB (multi target, interleaved)
    #    - RTX 2080Ti:  it/s
    #    - A5000:        it/s 
    "gated_deltanet": {
        "emsize": 320,
        "config_kwargs": {
            "attn_mode": "chunk",
            "hidden_size": 320,
            "num_hidden_layers": 10, # default 21
            "num_heads": 4, # default 6
            "head_dim": 64, # default 256
            "intermediate_size": 320 * 2, # default None -> 4*hidden_size
            "hidden_act": "swish",
            "norm_eps": 1e-6, # default 1e-6
            "use_cache": True,
            "use_short_conv": False,
            "vocab_size": 1, # dummy value, not used default 32000
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
    # Architecture
    model_type: str = "kda",
    sequence_mode: str = "cached",
    # Training
    training_setup: str = "high",
    batch_size: int | None = None,
    max_seq_len: int | None = None,
    lr: float | None = None,
    steps_per_epoch: int | None = None,
    aggregate_k_gradients: int | None = None,
    # Model options
    cache_chunk_size: int | None = None,
    use_short_conv: bool | None = None,
    interleave_x_y_pairs: bool = False,
    feature_positional_embedding: str | None = "subspace",
    config_kwargs_override: dict[str, object] | None = None,
) -> MainConfig:
    """Build a MainConfig for FLA backbone training."""
    max_num_classes = 10
    max_num_features = 20
    
    if feature_positional_embedding == "None":
        feature_positional_embedding = None

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
    profile_lr, profile_steps_per_epoch = TRAINING_PROFILES[training_setup]
    resolved_lr = float(profile_lr) if lr is None else float(lr)
    resolved_steps_per_epoch = (
        int(steps_per_epoch)
        if steps_per_epoch is not None
        else int(profile_steps_per_epoch)
    )
    resolved_max_seq_len = int(max_seq_len) if max_seq_len is not None else 1000
    resolved_aggregate_k = (
        aggregate_k_gradients
        if aggregate_k_gradients is not None
        else GLOBAL_AGGREGATE_K_GRADIENTS
    )
    resolved_batch_size = batch_size or DEFAULT_BATCH_SIZE
    train_mixed_precision = GLOBAL_TRAIN_MIXED_PRECISION
    train_mixed_precision_dtype = GLOBAL_TRAIN_MIXED_PRECISION_DTYPE
    if model_type in {"deltanet"} and train_mixed_precision_dtype == "fp32":
        # ChunkDeltaRuleFunction does not support fp32.
        print(
            f"Enabling mixed precision with fp16 training for model_type {model_type!r}"
        )
        train_mixed_precision = True
        train_mixed_precision_dtype = "fp16"
        
    resolved_prior_device = "cuda" if torch.cuda.is_available() and resolved_max_seq_len > 2000 else "cpu" # use cuda only for very long sequences 

    prior = TabPFNPriorConfig(
        prior_type="mlp",
        max_num_classes=max_num_classes,
        max_num_features=max_num_features,
        flexible=True,
        differentiable=True,
        nan_handling=True,
        return_categorical_mask=True,
        device=resolved_prior_device,
    )

    batch_shape = BatchShapeSamplerConfig(
        batch_size=resolved_batch_size,
        min_single_eval_pos=24,
        max_seq_len=resolved_max_seq_len,
        min_num_features=2,
        max_num_features=max_num_features,
        fixed_num_test_instances=None,
    )

    resolved_config_kwargs = dict(MODEL_SETTINGS[model_type]["config_kwargs"])
    if use_short_conv is not None:
        if "use_short_conv" not in resolved_config_kwargs:
            raise ValueError(
                f"use_short_conv is not supported for model_type {model_type!r}."
            )
        resolved_config_kwargs["use_short_conv"] = use_short_conv
    if config_kwargs_override is not None:
        if not isinstance(config_kwargs_override, dict):
            raise ValueError(
                "config_kwargs_override must be a dict of config kwargs to override."
            )
        resolved_config_kwargs.update(config_kwargs_override)

    backbone_kwargs = {
        "model_type": model_type,
        "config_kwargs": resolved_config_kwargs,
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
        feature_positional_embedding=feature_positional_embedding,
        interleave_x_y_pairs=interleave_x_y_pairs,
    )

    optimizer = OptimizerConfig(
        optimizer="adamw",
        lr=resolved_lr,
        weight_decay=0.01,
    )

    # Build descriptive wandb run name
    extras = [
        f"bs{resolved_batch_size}" if batch_size else None,
        f"seq{resolved_max_seq_len}" if max_seq_len else None,
        f"cache{cache_chunk_size}" if cache_chunk_size else None,
        f"lr{resolved_lr:g}" if lr else None,
        f"agg{resolved_aggregate_k}" if aggregate_k_gradients else None,
        f"steps{resolved_steps_per_epoch}" if steps_per_epoch else None,
        "interleaved" if interleave_x_y_pairs else None,
        f"shortconv_{use_short_conv}" if use_short_conv is not None else None,
        f"fpe_{feature_positional_embedding}",
    ]
    extras_str = "_".join(e for e in extras if e)
    wandb_name = (
        f"{model_type}_{sequence_mode}_{training_setup}_{extras_str}_config_{config_index}_matched"
    )
    wandb_config = WandbConfig(
        entity="icl_arch",
        project="fla_models",
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
        epochs=200,
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
        test_steps_per_epoch=500
    )
