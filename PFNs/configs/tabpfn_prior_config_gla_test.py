#!/usr/bin/env python3
"""
Training config that uses the standalone tabpfn_prior package with the PFNs
training loop with a GLA (Gated Linear Attention) backbone.
"""

from __future__ import annotations

from pfns.model.backbone_config import FLABackboneConfig
from pfns.model.criterions import CrossEntropyConfig
from pfns.model.encoders import EncoderConfig
from pfns.priors.tabpfn_prior_adapter import TabPFNPriorConfig
from pfns.train import (
    BatchShapeSamplerConfig,
    MainConfig,
    OptimizerConfig,
    ModelConfig,
)


def get_config(config_index: int = 0) -> MainConfig:
    """
    Build a config for training a TabPFN-style classifier on the synthetic
    tabpfn_prior data using GLA (Gated Linear Attention) backbone.
    """

    max_num_classes = 10
    max_num_features = 20

    prior = TabPFNPriorConfig(
        prior_type="mlp",
        max_num_classes=max_num_classes,
        max_num_features=max_num_features,
        flexible=True,
        differentiable=True,
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
            constant_normalization_mean=0.0,
            constant_normalization_std=1.0,
        ),
        y_encoder=EncoderConfig(
            nan_handling=True,
            constant_normalization_mean=0.0,
            constant_normalization_std=1.0,
        ),
        emsize=320,
        backbone=FLABackboneConfig(
            model_type="gla",
            nlayers=8,
            nhead=4,
            intermediate_size=320 * 4,
            dropout=0.1,
            activation="swish",
            norm_eps=1e-6,
            use_cache=False,
        ),
        features_per_group=20,
        attention_between_features=False,
        feature_positional_embedding=None,
    )

    optimizer = OptimizerConfig(
        optimizer="adamw",
        lr=1e-4,
        weight_decay=0.01,
    )

    return MainConfig(
        priors=[prior],
        optimizer=optimizer,
        model=model,
        batch_shape_sampler=batch_shape,
        epochs=200,
        warmup_epochs=10,
        steps_per_epoch=2000,
        n_targets_per_input=1,
        train_mixed_precision=True,
        scheduler="cosine_decay",
        progress_bar=True,
        num_workers=4,
    )
