#!/usr/bin/env python3
"""
Combined transformer config with selectable variants.
"""

from __future__ import annotations

from pfns.model.backbones import TransformerBackboneConfig
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

BASE_PROFILE = {
    "nhead": 8,
    "nlayers": 12,
    "layer_kwargs": None,
}

TRAINING_PROFILES = {
    "low": {
        **BASE_PROFILE,
        "emsize": 256,
        "nhid": 256 * 4,
        "lr": 1.5e-4,
        "steps_per_epoch": 250,
        "epochs": 200,
        "warmup_epochs": 10,
        "num_workers": 4,
        "aggregate_k_gradients": 1,
        "wandb_suffix": "",
    },
    "high": {
        **BASE_PROFILE,
        "emsize": 384,
        "nhid": 384 * 4,
        "lr": 7.5e-5,
        "steps_per_epoch": 2000,
        "epochs": 200,
        "warmup_epochs": 10,
        "num_workers": 8,
        "aggregate_k_gradients": 2,
        "wandb_suffix": "_high",
    },
    "very_high": {
        **BASE_PROFILE,
        "emsize": 384,
        "nhid": 384 * 4,
        "lr": 3.0e-5,
        "steps_per_epoch": 2000,
        "epochs": 400,
        "warmup_epochs": 20,
        "num_workers": 8,
        "aggregate_k_gradients": 2,
        "wandb_suffix": "_very_high",
    },
}

def get_config(config_index: int = 0, training_setup: str = "low") -> MainConfig:
    """
    Build a config for training a TabPFN-style classifier on the synthetic
    tabpfn_prior data.
    """

    max_num_classes = 10
    max_num_features = 20

    training_setup = training_setup.strip().lower()
    if training_setup not in TRAINING_PROFILES:
        raise ValueError(
            f"Unknown training_setup {training_setup!r}. "
            f"Available: {sorted(TRAINING_PROFILES)}"
        )
    profile = TRAINING_PROFILES[training_setup]

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
        batch_size=8,
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
            nan_handling=True,  # currently only nan to mean imputation works
            use_categorical_encoder=True,
        ),
        y_encoder=EncoderConfig(
            nan_handling=True,
            constant_normalization_mean=0.0,
            constant_normalization_std=1.0,
        ),
        emsize=profile["emsize"],
        backbone=TransformerBackboneConfig(
            nhid=profile["nhid"],
            nlayers=profile["nlayers"],
            nhead=profile["nhead"],
            layer_kwargs=profile["layer_kwargs"],
        ),
        features_per_group=20,
        attention_between_features=False, # was True before
        feature_positional_embedding="subspace",
    )

    optimizer = OptimizerConfig(
        optimizer="adamw",
        lr=profile["lr"],
        weight_decay=0.01,
    )

    wandb_config = WandbConfig(
        entity="icl_arch",
        project="tabpfn_transformer",
        name=f"transformer_1_gpu_v4{profile['wandb_suffix']}_{config_index}",
        mode="online",
        log_every_n_steps=10,
    )

    return MainConfig(
        priors=[prior],
        optimizer=optimizer,
        model=model,
        batch_shape_sampler=batch_shape,
        epochs=profile["epochs"],
        warmup_epochs=profile["warmup_epochs"],
        steps_per_epoch=profile["steps_per_epoch"],
        n_targets_per_input=1,
        train_mixed_precision=True,
        train_mixed_precision_dtype="fp16",
        scheduler="cosine_decay",
        progress_bar=True,
        wandb=wandb_config,
        num_workers=profile["num_workers"],
        aggregate_k_gradients=profile["aggregate_k_gradients"],
    )
