#!/usr/bin/env python3
"""
Training config that uses the standalone tabpfn_prior package with the PFNs
training loop with a Mamba2 backbone.
"""

from __future__ import annotations

from pfns.model.backbone_config import FLABackboneConfig
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
    tabpfn_prior data using a Mamba2 backbone.
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
        batch_size=1,
        min_single_eval_pos=24,
        max_seq_len=521,
        min_num_features=2,
        max_num_features=max_num_features,
        fixed_num_test_instances=None,
    )
    
    model = ModelConfig(
        criterion=CrossEntropyConfig(num_classes=max_num_classes),
        encoder=EncoderConfig(
            variable_num_features_normalization=True,
            nan_handling=True,
            use_categorical_encoder=True
        ),
        y_encoder=EncoderConfig(
            nan_handling=True,
            constant_normalization_mean=0.0,
            constant_normalization_std=1.0,
        ),
        emsize=192,
        backbone=FLABackboneConfig(
            model_type="mamba2",
            nlayers=4,
            nhead=4,
            intermediate_size=192 * 2,
            dropout=0.1,
            activation="swish",
            norm_eps=1e-4, # increase in size if nans occur
            config_kwargs={
                "hidden_size": 192,
                "d_model": 192,
                "n_embd": 192,
                "n_layer": 6,
                "d_state": 16,
                "d_conv": 4,
                "expand": 2,
                "use_cache": True,
            },
        ),
        features_per_group=20,
        attention_between_features=False,
        feature_positional_embedding="subspace",
    )

    optimizer = OptimizerConfig(
        optimizer="adamw",
        lr=7.5e-5,
        weight_decay=0.01,
    )
    
    wandb_config = WandbConfig(
        entity="icl_arch",
        project="fla_models",
        name=f"mamba2_test_{config_index}",
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
        steps_per_epoch=250,
        n_targets_per_input=1,
        train_mixed_precision=True,
        train_mixed_precision_dtype="bf16",
        scheduler="cosine_decay",
        progress_bar=True,
        wandb=wandb_config,
        num_workers=8,
        aggregate_k_gradients=1
    )
