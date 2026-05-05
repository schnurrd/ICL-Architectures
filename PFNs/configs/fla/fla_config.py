#!/usr/bin/env python3
"""
Config selector for FLA backbones using CLI-provided arguments.
"""

from __future__ import annotations

import torch

from configs.config_utils import (
    normalize_optional_none_string,
    resolve_batch_size_stages,
    resolve_eval_pos_split_pct,
    resolve_prior_device,
)
from pfns.model.fla_mimetic_init import MimeticInitMode
from pfns.prior_defaults import (
    ASSOCIATIVE_RECALL_SETTINGS,
    TABPFN_PRIOR_DEFAULTS,
    build_prior_for_task,
    resolve_training_setup_for_task,
)
from pfns.priors.tabpfn_prior_adapter import TabPFNPriorConfig
from pfns.model.backbones import FLABackboneConfig
from pfns.model.mode_normalization import (
    CANONICAL_SEQUENCE_MODES,
    resolve_sequence_mode,
)
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
SUPPORTED_SEQUENCE_MODES = CANONICAL_SEQUENCE_MODES

TRAINING_PROFILES = {
    "debug": {"lr": 6.0e-5, "steps_per_epoch": 10, "epochs": 200},
    "low": {"lr": 6.0e-5, "steps_per_epoch": 1000, "epochs": 200},
    "high": {"lr": 3.0e-5, "steps_per_epoch": 4000, "epochs": 200},
    "ar": {"lr": 3.0e-5, "steps_per_epoch": 500, "epochs": 200},
}

MODEL_SETTINGS = {
    # KDA Config: https://github.com/fla-org/flash-linear-attention/blob/3cf180339b8a1cbad823f553541cd531d18670ea/fla/models/kda/configuration_kda.py#L10
    # Model size: 12.60 M full (12.42 M FLA backbone)
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
            "head_dim": 80, # currently 128
            "intermediate_size": 320 * 2, # default None -> 4*hidden_size
            "hidden_act": "swish",
            "num_hidden_layers": 11, # default 24
            "norm_eps": 1e-6, # default 1e-6
            "use_cache": True,
            "vocab_size": 1, # dummy value, not used default 32000
            # "cache_chunk_size": 16,  
        },
    },
    # GLA Config: https://github.com/fla-org/flash-linear-attention/blob/3cf180339b8a1cbad823f553541cd531d18670ea/fla/models/gla/configuration_gla.py#L12
    # Model size: 12.69 M full (12.52 M FLA backbone)
    # Training speed on different gpus (uncompiled, single target): 
    #    - RTX 5070:   22it/s, 1.8GiB (single target); 28it/s, 1.6GiB (multi target); 16it/s, 2.6GiB (multi target, interleaved)
    #    - RTX 2080Ti:  it/s
    #    - A5000:       it/s 
    "gla": {
        "emsize": 320,
        "config_kwargs": { # also has max_position_embeddings set to 2048, supports attn dict
            "hidden_size": 320, # default 2048
            "expand_k": 1.0, # equalizes recurrent state to total K/V width 320 x 320
            "expand_v": 1.0,
            "use_short_conv": False, 
            "num_heads": 4, # default 4
            "num_hidden_layers": 11, # default 24
            "intermediate_size": 320 * 2, # default None -> 4*hidden_size
            "hidden_act": "swish",
            "norm_eps": 1e-6, # default 1e-6
            "use_cache": True,
            "vocab_size": 1, # dummy value, not used default 32000
        },
    },
    # Mamba2 Config: https://github.com/fla-org/flash-linear-attention/blob/3cf180339b8a1cbad823f553541cd531d18670ea/fla/models/mamba2/configuration_mamba2.py#L21
    # Model size: 12.49 M full (12.32 M FLA backbone)
    # Training speed on different gpus (uncompiled): 
    #    - RTX 5070 (bf16):   7it/s, 4.3GB (single target); 5it/s (single target, interleaved); 15it/s, 2GB (multi target); 8it/s, 2.9GiB (multi target, interleaved)
    #    - RTX 2080Ti:  it/s
    #    - A5000:        it/s 
    "mamba2": { 
        "emsize": 320,
        "config_kwargs": {
            "hidden_size": 320, # default 2048
            "num_hidden_layers": 18, # default 48
            "state_size": 96, # default 128
            "expand": 2, # default 2, --> num_heads self.expand * hidden_size // head_dim
            "head_dim": 64, # default 64
            "norm_eps": 1e-6, # default 1e-5
            "vocab_size": 1, # dummy value, not used default 32000
            "use_cache": True,
            "cache_chunk_size": 16,  
        },
    },
    # DeltaNet Config: https://github.com/fla-org/flash-linear-attention/blob/3cf180339b8a1cbad823f553541cd531d18670ea/fla/models/delta_net/configuration_delta_net.py#L7
    # Model size: 12.49 M full (12.31 M FLA backbone)
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
    # Model size: 12.60 M full (12.43 M FLA backbone)
    # Training speed on different gpus (uncompiled): 
    #    - RTX 5070 (bf16):   14it/s, 2.9GB (single target); 24it/s, 1.9GB (multi target); 15it/s, 3.5GiB (multi target, interleaved)
    #    - RTX 2080Ti:  it/s
    #    - A5000:        it/s 
    "gated_deltanet": {
        "emsize": 320,
        "config_kwargs": {
            "attn_mode": "chunk",
            "hidden_size": 320,
            "num_hidden_layers": 11, # default 21
            "expand_v": 1.0, # default 2.0
            "num_heads": 4, # default 6
            "head_dim": 80, # default 256; equalizes recurrent state to total K/V width 320 x 320
            "intermediate_size": 320 * 2, # default None -> 4*hidden_size
            "hidden_act": "swish",
            "norm_eps": 1e-6, # default 1e-6
            "use_cache": True,
            "use_short_conv": False,
            "vocab_size": 1, # dummy value, not used default 32000
        },
    },
    # Linear Attention Config: https://github.com/fla-org/flash-linear-attention/blob/main/fla/models/linear_attn/configuration_linear_attn.py
    # Model size: 12.47 M full (12.30 M FLA backbone)
    "linear_attn": {
        "emsize": 320,
        "config_kwargs": {
            "attn_mode": "chunk",
            "hidden_size": 320,
            "num_hidden_layers": 12,
            "num_heads": 4,
            "intermediate_size": 320 * 2,
            "feature_map": "elu",
            "norm_q": False,
            "norm_k": False,
            "norm_feature_map": False,
            "hidden_act": "swish",
            "norm_eps": 1e-6,
            "use_cache": True,
            "vocab_size": 1, # dummy value, not used default 32000
        },
    },
    # MesaNet Config: https://github.com/fla-org/flash-linear-attention/blob/main/fla/models/mesa_net/configuration_mesa_net.py
    # Model size: 12.54 M full (12.36 M FLA backbone)
    "mesanet": {
        "emsize": 320,
        "config_kwargs": {
            "attn_mode": "chunk",
            "hidden_size": 320,
            "num_hidden_layers": 12,
            "num_heads": 4,
            "head_dim": 80,
            "intermediate_size": 320 * 2,
            "hidden_act": "swish",
            "norm_eps": 1e-6,
            "use_output_gate": False,
            "use_short_conv": False,
            "use_cache": True,
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
    if model_type in {"linear_attention", "linearattn"}:
        return "linear_attn"
    if model_type in {"mesa", "mesa_net"}:
        return "mesanet"
    return model_type


def get_config(
    config_index: int = 0,
    # Architecture
    model_type: str = "kda",
    hidden_size: int | None = None,
    sequence_mode: str = "Comb_ST",
    bidirectional: bool = False,
    bidirectional_share_weights: bool = True,
    bidirectional_state_fusion: str = "mean_output_mean_cache",
    state_passing: bool = False,
    state_passing_dropout: float = 0.1,
    include_self_term: bool = True,
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
    # Model options
    cache_chunk_size: int | None = None,
    mimetic_init: bool = False,
    mimetic_init_mode: MimeticInitMode = "gate_only",
    mimetic_init_layer_indices: tuple[int, ...] | list[int] | None = None,
    use_short_conv: bool | None = None,
    use_categorical_features: bool = True,
    feature_positional_embedding: str | None = None,
    config_kwargs_override: dict[str, object] | None = None,
) -> MainConfig:
    """Build a MainConfig for FLA backbone training."""
    # Resolve inputs and training profile.
    feature_positional_embedding = normalize_optional_none_string(
        feature_positional_embedding
    )
    model_type = _normalize_model_type(model_type)
    sequence_mode = resolve_sequence_mode(sequence_mode)
    training_setup = training_setup.strip().lower()
    training_setup, is_associative_recall = resolve_training_setup_for_task(
        training_setup=training_setup,
        task_variant=task_variant,
    )

    if model_type not in MODEL_SETTINGS:
        raise ValueError(
            f"Unknown model_type {model_type!r}. Available: {sorted(MODEL_SETTINGS)}"
        )
    if training_setup not in TRAINING_PROFILES:
        raise ValueError(
            f"Unknown training_setup {training_setup!r}. Available: {sorted(TRAINING_PROFILES)}"
        )

    model_settings = MODEL_SETTINGS[model_type]
    profile = TRAINING_PROFILES[training_setup]
    resolved_lr = float(profile["lr"]) if lr is None else float(lr)
    resolved_steps_per_epoch = (
        int(steps_per_epoch)
        if steps_per_epoch is not None
        else int(profile["steps_per_epoch"])
    )
    resolved_max_seq_len = int(max_seq_len) if max_seq_len is not None else 1000
    resolved_batch_size_stages = resolve_batch_size_stages(batch_size_stages)
    resolved_dynamic_batch_size_compensate_grad_accumulation = bool(
        dynamic_batch_size_compensate_grad_accumulation
    )
    resolved_eval_pos_split_pct_min, resolved_eval_pos_split_pct_max = (
        resolve_eval_pos_split_pct(eval_pos_split_pct)
    )
    resolved_seq_len_stages = seq_len_stages
    resolved_aggregate_k = (
        int(aggregate_k_gradients)
        if aggregate_k_gradients is not None
        else 1 if is_associative_recall else GLOBAL_AGGREGATE_K_GRADIENTS
    )
    resolved_batch_size = batch_size or DEFAULT_BATCH_SIZE

    train_mixed_precision = GLOBAL_TRAIN_MIXED_PRECISION
    train_mixed_precision_dtype = GLOBAL_TRAIN_MIXED_PRECISION_DTYPE
    if model_type in {"deltanet"} and train_mixed_precision_dtype == "fp32":
        train_mixed_precision = True
        train_mixed_precision_dtype = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
        # ChunkDeltaRuleFunction does not support fp32.
        print(
            f"Enabling mixed precision with {train_mixed_precision_dtype} training for model_type {model_type!r}"
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

    # Build backbone kwargs.
    resolved_config_kwargs = dict(model_settings["config_kwargs"])
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
    if hidden_size is not None:
        resolved_hidden_size = int(hidden_size)
        resolved_config_kwargs["hidden_size"] = resolved_hidden_size
        # Preserve FFN ratio unless explicitly overridden by config_kwargs_override.
        if (
            "intermediate_size" in resolved_config_kwargs
            and not (
                isinstance(config_kwargs_override, dict)
                and "intermediate_size" in config_kwargs_override
            )
        ):
            resolved_config_kwargs["intermediate_size"] = resolved_hidden_size * 2

    # FLA backbone requires ninp == hidden_size.
    resolved_emsize = int(
        resolved_config_kwargs.get("hidden_size", model_settings["emsize"])
    )
    resolved_include_self_term = bool(include_self_term)

    backbone_kwargs = {
        "model_type": model_type,
        "config_kwargs": resolved_config_kwargs,
        "sequence_mode": sequence_mode,
        "bidirectional": bidirectional,
        "bidirectional_share_weights": bidirectional_share_weights,
        "bidirectional_state_fusion": bidirectional_state_fusion,
        "state_passing": bool(state_passing),
        "state_passing_dropout": float(state_passing_dropout),
        "include_self_term": resolved_include_self_term,
        "mimetic_init": mimetic_init,
        "mimetic_init_mode": mimetic_init_mode,
        "mimetic_init_layer_indices": mimetic_init_layer_indices,
    }
    if cache_chunk_size is not None:
        backbone_kwargs["cache_chunk_size"] = cache_chunk_size

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
        backbone=FLABackboneConfig(**backbone_kwargs),
        features_per_group=20,
        attention_between_features=False,
        feature_positional_embedding=feature_positional_embedding,
        interleave_x_y_pairs=sequence_mode.startswith("Int"),
    )

    # Build optimizer and logging config.
    optimizer = OptimizerConfig(
        optimizer="adamw",
        lr=resolved_lr,
        weight_decay=0.01,
    )

    # Build descriptive wandb run name
    effective_hidden_size = resolved_config_kwargs.get("hidden_size")
    effective_num_layers = resolved_config_kwargs.get("num_hidden_layers")
    effective_num_heads = resolved_config_kwargs.get("num_heads")

    extras = [
        f"emb{resolved_emsize}",
        f"hid{effective_hidden_size}" if effective_hidden_size is not None else None,
        f"layers{effective_num_layers}" if effective_num_layers is not None else None,
        f"heads{effective_num_heads}" if effective_num_heads is not None else None,
        f"bs{resolved_batch_size}" if batch_size else None,
        f"bsstages{len(resolved_batch_size_stages)}" if resolved_batch_size_stages else None,
        (
            "dynbs_compagg"
            if resolved_dynamic_batch_size_compensate_grad_accumulation
            else None
        ),
        f"seq{resolved_max_seq_len}" if max_seq_len else None,
        "evalsplit" if eval_pos_split_pct is not None else None,
        f"stages{len(resolved_seq_len_stages)}" if resolved_seq_len_stages else None,
        f"cache{cache_chunk_size}" if cache_chunk_size else None,
        "sp" if state_passing else None,
        f"spd{state_passing_dropout:g}" if state_passing and state_passing_dropout != 0.1 else None,
        f"lr{resolved_lr:g}" if lr else None,
        f"agg{resolved_aggregate_k}" if aggregate_k_gradients else None,
        f"steps{resolved_steps_per_epoch}" if steps_per_epoch else None,
        f"shortconv_{use_short_conv}" if use_short_conv is not None else None,
        "nocat" if not use_categorical_features else None,
        f"mimetic_{mimetic_init_mode}" if mimetic_init else None,
        "bidir" if bidirectional else None,
        (
            f"bidirshare_{int(bidirectional_share_weights)}"
            if bidirectional
            else None
        ),
        f"sfusion_{bidirectional_state_fusion}" if bidirectional else None,
        f"fpe_{feature_positional_embedding}",
    ]
    extras_str = "_".join(e for e in extras if e)
    wandb_name = (
        f"{model_type}_{sequence_mode}_{training_setup}_{extras_str}_config_{config_index}_matched"
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
            f"model_{model_type}",
            f"emb_{resolved_emsize}",
            f"hidden_{effective_hidden_size}" if effective_hidden_size is not None else "hidden_na",
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
        test_steps_per_epoch=500
    )
