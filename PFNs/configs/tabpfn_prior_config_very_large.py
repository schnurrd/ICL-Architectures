#!/usr/bin/env python3
"""
Training config that uses the standalone tabpfn_prior package with the PFNs
training loop with a TabPFN-v1 style transformer backbone.
"""

from __future__ import annotations

from pfns.model.backbone_config import TransformerBackboneConfig
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
    tabpfn_prior data.
    """

    max_num_classes = 10

    prior = TabPFNPriorConfig(
        prior_type="mlp",
        max_num_classes=max_num_classes,
        flexible=True,
        differentiable=False,
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
        emsize=512,
        backbone=TransformerBackboneConfig(
            nhid=256 * 4,
            nlayers=12,
            nhead=8,
        ),
        features_per_group=2,
        attention_between_features=True,
    )

    batch_shape = BatchShapeSamplerConfig(
        batch_size=2,
        min_single_eval_pos=24,
        max_seq_len=1024,
        min_num_features=2,
        max_num_features=20,
        fixed_num_test_instances=None,
    )

    optimizer = OptimizerConfig(
        optimizer="adamw",
        lr=1.5e-4,
        weight_decay=0.01,
    )

    return MainConfig(
        priors=[prior],
        optimizer=optimizer,
        model=model,
        batch_shape_sampler=batch_shape,
        epochs=100,
        warmup_epochs=5,
        steps_per_epoch=2000,
        n_targets_per_input=1,
        train_mixed_precision=True,
        scheduler="cosine_decay",
        progress_bar=True,
        num_workers=4,
        aggregate_k_gradients=4,
    )
