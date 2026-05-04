#!/usr/bin/env python3
"""
Training config that uses the standalone tabpfn_prior package with the PFNs
training loop with a TabPFN-v1 style transformer backbone.
"""

from __future__ import annotations

from configs.config_utils import normalize_optional_none_string, resolve_prior_device
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
from pfns.prior_defaults import (
    TABPFN_PRIOR_DEFAULTS,
    build_prior_for_task,
    resolve_training_setup_for_task,
)

MAX_NUM_CLASSES = int(TABPFN_PRIOR_DEFAULTS["max_num_classes"])
MAX_NUM_FEATURES = int(TABPFN_PRIOR_DEFAULTS["max_num_features"])

def get_config(
    config_index: int = 0,
    masking: str | None = None,
    max_seq_len: int | None = None,
    interleave_x_y_pairs: bool = False,
    train_mixed_precision: bool = False,
    train_mixed_precision_dtype: str = "fp16",
    task_variant: str = "tabular_prior",
    item_attention_use_rope: bool = False,
    item_attention_rope_base: float = 128_000.0,
    item_attention_rope_pairwise_positions: bool = False,
) -> MainConfig:
    """
    Build a config for training a TabPFN-style classifier on the synthetic
    tabpfn_prior data.
    """

    _, is_associative_recall = resolve_training_setup_for_task(
        training_setup="high",
        task_variant=task_variant,
    )
    assert not is_associative_recall, "Associative recall training setups are not supported in this config"
    
    masking = normalize_optional_none_string(masking)

    # for backward compatibility with older config versions
    if masking == "causal_train_only":
        masking = "Int_ST" if interleave_x_y_pairs else "Comb_ST"
    elif masking == "causal_all":
        masking = "Int_MT" if interleave_x_y_pairs else "Comb_MT"

    assert masking in [
        "test_to_train_only",
        "Comb_ST",
        "Int_ST",
        "Comb_MT",
        "Int_MT",
        None,
    ], f"Invalid masking mode: {masking}"

    resolved_interleave_x_y_pairs = (
        ("Int" in masking) if masking is not None else interleave_x_y_pairs
    )

    print(f"Using masking mode: {masking}")

    resolved_max_seq_len = int(max_seq_len) if max_seq_len is not None else 1000
    
    resolved_prior_device = resolve_prior_device(max_seq_len=resolved_max_seq_len)

    prior = build_prior_for_task(
        task_variant=task_variant,
        prior_device=resolved_prior_device,
        max_num_classes=MAX_NUM_CLASSES,
        max_num_features=MAX_NUM_FEATURES,
    )
    
    batch_shape = BatchShapeSamplerConfig(
        batch_size=8,
        min_single_eval_pos=64,
        max_seq_len=resolved_max_seq_len,
        batch_size_stages=None,
        dynamic_batch_size_compensate_grad_accumulation=False,
        eval_pos_split_pct_min=None,
        eval_pos_split_pct_max=None,
        seq_len_stages=None,
        min_num_features=2,
        max_num_features=MAX_NUM_FEATURES,
        fixed_num_test_instances=None,
    )

    layer_kwargs = {
        "item_attention_mask_mode": masking,
    }
    if item_attention_use_rope:
        layer_kwargs["item_attention_use_rope"] = True
        layer_kwargs["item_attention_rope_base"] = float(item_attention_rope_base)
    if item_attention_rope_pairwise_positions:
        layer_kwargs["item_attention_rope_pairwise_positions"] = True

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
        emsize=320,
        backbone=TransformerBackboneConfig(
            nhid=320 * 2,
            nlayers=15,
            nhead=4,
            layer_kwargs=layer_kwargs,
        ),
        features_per_group=20,
        attention_between_features=False,
        feature_positional_embedding=None,
        interleave_x_y_pairs=resolved_interleave_x_y_pairs,
    )

    optimizer = OptimizerConfig(
        optimizer="adamw",
        lr=3.0e-5,
        weight_decay=0.01,
    )
    
    wandb_name = f"transformer_modified_masking_{masking}"
    if max_seq_len is not None:
        wandb_name += f"_seq{resolved_max_seq_len}"
    if item_attention_use_rope:
        wandb_name += "_item_rope"
        if item_attention_rope_pairwise_positions:
            wandb_name += "_pairwise"

    wandb_config = WandbConfig(
        entity="icl_arch",
        project="tabpfn_transformer_masking_experiments",
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
        steps_per_epoch=4000,
        n_targets_per_input=1,
        train_mixed_precision=train_mixed_precision,
        train_mixed_precision_dtype=train_mixed_precision_dtype.lower(),
        scheduler="cosine_decay",
        progress_bar=True,
        wandb=wandb_config,
        num_workers=8 if resolved_prior_device == "cpu" else 0,
        aggregate_k_gradients=2,
    )
