#!/usr/bin/env python3
"""
Training config that uses the standalone tabpfn_prior package with the PFNs
training loop with a KDA backbone.
"""

from __future__ import annotations

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


def get_config(config_index: int = 0) -> MainConfig:
    """
    Build a config for training a TabPFN-style classifier on the synthetic
    tabpfn_prior data using KDA backbone.
    """

    max_num_classes = 10
    max_num_features = 20

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
        batch_size=2,
        min_single_eval_pos=24,
        max_seq_len=1000,
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
        emsize=320,
        backbone=FLABackboneConfig(
            model_type="kda",
            config_kwargs={
                "hidden_size": 320,
                "num_hidden_layers": 10,
                "num_heads": 4,
                "intermediate_size": 320 * 2,
                "hidden_act": "swish",
                "norm_eps": 1e-4,  # increase in size if nans occur
                "use_cache": True,
            },
            sequence_mode="cached",
        ),
        features_per_group=20,
        attention_between_features=False,
        feature_positional_embedding="subspace",
    )

    optimizer = OptimizerConfig(
        optimizer="adamw",
        lr=3.0e-5,
        weight_decay=0.01,
    )

    wandb_config = WandbConfig(
        entity="icl_arch",
        project="fla_models",
        name=f"kda_cached_performance_long_{config_index}",
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
        train_mixed_precision=True,
        train_mixed_precision_dtype="bf16",  # fp16 will lead to nans
        scheduler="cosine_decay",
        progress_bar=True,
        wandb=wandb_config,
        num_workers=8,
        aggregate_k_gradients=8,
    )
