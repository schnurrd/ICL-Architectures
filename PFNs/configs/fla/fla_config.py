#!/usr/bin/env python3
"""
Config selector for FLA backbones using CLI-provided arguments.
"""

from __future__ import annotations

import os
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
SUPPORTED_SEQUENCE_MODES = CANONICAL_SEQUENCE_MODES

TRAINING_PROFILES = {
    "debug": {"lr": 6.0e-5, "steps_per_epoch": 10, "epochs": 200},
    "low": {"lr": 6.0e-5, "steps_per_epoch": 1000, "epochs": 200},
    "high": {"lr": 3.0e-5, "steps_per_epoch": 4000, "epochs": 200},
    "ar": {"lr": 3.0e-5, "steps_per_epoch": 500, "epochs": 200},
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
    "mamba2": { 
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


def _default_finetune_save_path(load_path: str) -> str:
    root, ext = os.path.splitext(load_path)
    if ext:
        return f"{root}_finetune{ext}"
    return f"{load_path}_finetune.pt"


def _load_finetune_checkpoint_metadata(
    checkpoint_path: str,
) -> tuple[dict[str, object] | None, str | None]:
    """Extract architecture-relevant metadata from a saved training checkpoint."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    except Exception as exc:
        print(
            f"Warning: Could not read finetune checkpoint metadata from {checkpoint_path}: {exc}"
        )
        return None, None

    if not isinstance(checkpoint, dict):
        return None, None
    config_dict = checkpoint.get("config")
    if not isinstance(config_dict, dict):
        return None, None
    model_dict = config_dict.get("model")
    if not isinstance(model_dict, dict):
        return None, None

    backbone_dict = model_dict.get("backbone")
    backbone_config_kwargs: dict[str, object] | None = None
    if isinstance(backbone_dict, dict):
        raw_kwargs = backbone_dict.get("config_kwargs")
        if isinstance(raw_kwargs, dict):
            backbone_config_kwargs = dict(raw_kwargs)

    checkpoint_fpe = model_dict.get("feature_positional_embedding")
    if checkpoint_fpe is not None and not isinstance(checkpoint_fpe, str):
        checkpoint_fpe = None
    return backbone_config_kwargs, checkpoint_fpe


def get_config(
    config_index: int = 0,
    # Architecture
    model_type: str = "kda",
    hidden_size: int | None = None,
    sequence_mode: str = "Comb_ST",
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
    finetune_from_checkpoint: str | None = None,
    finetune_save_checkpoint: str | None = None,
    finetune_epochs: int | None = None,
    finetune_lr: float | None = None,
    finetune_warmup_epochs: int = 2,
    # Model options
    cache_chunk_size: int | None = None,
    use_short_conv: bool | None = None,
    feature_positional_embedding: str | None = None,
    config_kwargs_override: dict[str, object] | None = None,
) -> MainConfig:
    """Build a MainConfig for FLA backbone training."""
    max_num_classes = int(TABPFN_PRIOR_DEFAULTS["max_num_classes"])
    max_num_features = int(TABPFN_PRIOR_DEFAULTS["max_num_features"])
    
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
    profile = TRAINING_PROFILES[training_setup]
    resolved_lr = float(profile["lr"]) if lr is None else float(lr)
    resolved_steps_per_epoch = (
        int(steps_per_epoch)
        if steps_per_epoch is not None
        else int(profile["steps_per_epoch"])
    )
    resolved_epochs = int(profile.get("epochs", 200))
    resolved_max_seq_len = int(max_seq_len) if max_seq_len is not None else 1000
    resolved_batch_size_stages = resolve_batch_size_stages(batch_size_stages)
    resolved_dynamic_batch_size_compensate_grad_accumulation = bool(
        dynamic_batch_size_compensate_grad_accumulation
    )
    resolved_eval_pos_split_pct_min, resolved_eval_pos_split_pct_max = (
        resolve_eval_pos_split_pct(eval_pos_split_pct)
    )
    resolved_seq_len_stages = seq_len_stages
    if aggregate_k_gradients is not None:
        resolved_aggregate_k = aggregate_k_gradients
    elif is_associative_recall:
        resolved_aggregate_k = 1
    else:
        resolved_aggregate_k = GLOBAL_AGGREGATE_K_GRADIENTS
    resolved_batch_size = batch_size or DEFAULT_BATCH_SIZE
    resolved_warmup_epochs = 10
    resolved_checkpoint_load_mode: str = "resume"
    resolved_train_state_dict_load_path: str | None = None
    resolved_train_state_dict_save_path: str | None = None
    finetune_checkpoint_backbone_kwargs: dict[str, object] | None = None
    finetune_checkpoint_feature_positional_embedding: str | None = None

    if finetune_from_checkpoint is not None:
        if finetune_epochs is not None:
            resolved_epochs = int(finetune_epochs)
            if resolved_epochs <= 0:
                raise ValueError("finetune_epochs must be > 0.")
        if finetune_lr is not None:
            resolved_lr = float(finetune_lr)
        resolved_warmup_epochs = int(finetune_warmup_epochs)
        if resolved_warmup_epochs < 0:
            raise ValueError("finetune_warmup_epochs must be >= 0.")
        resolved_checkpoint_load_mode = "weights_only"
        resolved_train_state_dict_load_path = finetune_from_checkpoint
        resolved_train_state_dict_save_path = (
            finetune_save_checkpoint
            if finetune_save_checkpoint is not None
            else _default_finetune_save_path(finetune_from_checkpoint)
        )
        if os.path.isfile(finetune_from_checkpoint):
            (
                finetune_checkpoint_backbone_kwargs,
                finetune_checkpoint_feature_positional_embedding,
            ) = _load_finetune_checkpoint_metadata(finetune_from_checkpoint)
            if finetune_checkpoint_backbone_kwargs is not None:
                print(
                    "Finetune mode: aligning backbone config kwargs from checkpoint "
                    f"({finetune_from_checkpoint})."
                )
        else:
            print(
                "Warning: finetune_from_checkpoint does not point to an existing file. "
                "Skipping metadata alignment."
            )

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
        max_num_classes=max_num_classes,
        max_num_features=max_num_features,
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
        max_num_features=max_num_features,
        fixed_num_test_instances=None,
    )

    resolved_config_kwargs = dict(MODEL_SETTINGS[model_type]["config_kwargs"])
    if finetune_checkpoint_backbone_kwargs is not None:
        resolved_config_kwargs.update(finetune_checkpoint_backbone_kwargs)
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
        resolved_config_kwargs.get("hidden_size", MODEL_SETTINGS[model_type]["emsize"])
    )

    backbone_kwargs = {
        "model_type": model_type,
        "config_kwargs": resolved_config_kwargs,
        "sequence_mode": sequence_mode,
    }
    if cache_chunk_size is not None:
        backbone_kwargs["cache_chunk_size"] = cache_chunk_size

    resolved_feature_positional_embedding = (
        feature_positional_embedding
        if feature_positional_embedding is not None
        else finetune_checkpoint_feature_positional_embedding
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
        emsize=resolved_emsize,
        backbone=FLABackboneConfig(**backbone_kwargs),
        features_per_group=20,
        attention_between_features=False,
        feature_positional_embedding=resolved_feature_positional_embedding,
        interleave_x_y_pairs=sequence_mode.startswith("Int"),
    )

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
        f"lr{resolved_lr:g}" if lr else None,
        f"agg{resolved_aggregate_k}" if aggregate_k_gradients else None,
        f"steps{resolved_steps_per_epoch}" if steps_per_epoch else None,
        f"ft{resolved_epochs}e" if finetune_from_checkpoint is not None else None,
        f"shortconv_{use_short_conv}" if use_short_conv is not None else None,
        f"fpe_{resolved_feature_positional_embedding}",
    ]
    extras_str = "_".join(e for e in extras if e)
    wandb_name = (
        f"{model_type}_{sequence_mode}_{training_setup}_{extras_str}_config_{config_index}_matched"
    )
    if is_associative_recall:
        wandb_name += "_ar"
    if finetune_from_checkpoint is not None:
        wandb_project = "finetuning"
    elif is_associative_recall:
        wandb_project = ASSOCIATIVE_RECALL_SETTINGS["wandb_project"]
    else:
        wandb_project = "fla_models"

    wandb_config = WandbConfig(
        entity="icl_arch",
        project=wandb_project,
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
        epochs=resolved_epochs,
        warmup_epochs=resolved_warmup_epochs,
        steps_per_epoch=resolved_steps_per_epoch,
        n_targets_per_input=1,
        train_mixed_precision=train_mixed_precision,
        train_mixed_precision_dtype=train_mixed_precision_dtype,
        scheduler="cosine_decay",
        train_state_dict_load_path=resolved_train_state_dict_load_path,
        train_state_dict_save_path=resolved_train_state_dict_save_path,
        checkpoint_load_mode=resolved_checkpoint_load_mode,
        progress_bar=True,
        wandb=wandb_config,
        num_workers=8 if resolved_prior_device == "cpu" else 0,
        aggregate_k_gradients=resolved_aggregate_k,
        validation_period=10,
        test_steps_per_epoch=500
    )
