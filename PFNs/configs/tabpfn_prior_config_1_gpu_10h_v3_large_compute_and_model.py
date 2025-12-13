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
    max_num_features = 20

    prior = TabPFNPriorConfig(
        prior_type="mlp",       
        max_num_classes=max_num_classes,
        max_num_features=max_num_features,          
        flexible=True,                 
        differentiable=True,
        return_categorical_mask=True,
        nan_handling=True,
    )
    
    batch_shape = BatchShapeSamplerConfig(
        batch_size=1,
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
            nan_handling=True, # currently only nan to mean imputation works
            use_categorical_encoder=True
        ),
        y_encoder=EncoderConfig(
            nan_handling=True,
            constant_normalization_mean=0.0,
            constant_normalization_std=1.0,
        ),
        emsize=512,
        backbone=TransformerBackboneConfig(
            nhid=512 * 4,
            nlayers=12,
            nhead=8,
        ),
        features_per_group=1,
        attention_between_features=True,
        feature_positional_embedding="subspace",
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
        epochs=200,
        warmup_epochs=10,
        steps_per_epoch=4000,
        n_targets_per_input=1,
        train_mixed_precision=True,
        scheduler="cosine_decay",
        progress_bar=True,
        num_workers=4,
        aggregate_k_gradients=8,
    )
