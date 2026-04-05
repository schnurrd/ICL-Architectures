#!/usr/bin/env python3
"""
Matched config selector for a LinearAttention -> BatchDelta hybrid backbone.
"""

from __future__ import annotations

from configs.transformer._hybrid_batch_delta_config_utils import (
    build_hybrid_batch_delta_main_config,
)
from pfns.model.backbones import HybridLinearBatchDeltaBackboneConfig
from pfns.train import MainConfig

MATCHED_ARCHITECTURE = {
    "emsize": 320,
    "mlp_hidden_dim": 736,
    "nhead": 4,
    "features_per_group": 20,
    "lower_nlayers": 11,
    "upper_nlayers": 4,
    "batch_delta_state_dim": 64,
}


def get_config(
    config_index: int = 0,
    training_setup: str = "high",
    task_variant: str = "tabular_prior",
    batch_size: int | None = None,
    max_seq_len: int | None = None,
    batch_size_stages: list[tuple[int, int]] | tuple[tuple[int, int], ...] | None = None,
    dynamic_batch_size_compensate_grad_accumulation: bool = False,
    eval_pos_split_pct: float | tuple[float, float] | list[float] | None = None,
    seq_len_stages: list[tuple[int | float | str, ...]] | tuple[tuple[int | float | str, ...], ...] | None = None,
    lr: float | None = None,
    aggregate_k_gradients: int | None = None,
    feature_positional_embedding: str | None = None,
    emsize: int | None = None,
    mlp_hidden_dim: int | None = None,
    nhead: int | None = None,
    features_per_group: int | None = None,
    lower_nlayers: int | None = None,
    upper_nlayers: int | None = None,
    batch_delta_state_dim: int | None = None,
    num_solver_steps: int = 1,
    support_target_mode: str = "hidden_plus_label",
    target_bilinear_rank: int = 0,
    fast_weight_rank: int = 0,
    base_fast_weight_context_rank: int = 0,
    incontext_opt_rank: int = 0,
    incontext_opt_steps: int = 0,
    incontext_opt_lr: float = 5e-2,
    incontext_opt_weight_decay: float = 0.0,
    ridge_lambda_init: float = 1e-1,
    learnable_ridge_lambda: bool = False,
    qk_l2_normalize: bool = True,
    residual_scale_init: float = 3e-2,
    interleave_x_y_pairs: bool = False,
) -> MainConfig:
    if interleave_x_y_pairs:
        raise ValueError(
            "Linear-BatchDelta config does not support interleaved x/y pairs."
        )

    resolved_emsize = int(MATCHED_ARCHITECTURE["emsize"] if emsize is None else emsize)
    resolved_mlp_hidden_dim = int(
        MATCHED_ARCHITECTURE["mlp_hidden_dim"] if mlp_hidden_dim is None else mlp_hidden_dim
    )
    resolved_nhead = int(MATCHED_ARCHITECTURE["nhead"] if nhead is None else nhead)
    resolved_features_per_group = int(
        MATCHED_ARCHITECTURE["features_per_group"]
        if features_per_group is None
        else features_per_group
    )
    resolved_lower_nlayers = int(
        MATCHED_ARCHITECTURE["lower_nlayers"] if lower_nlayers is None else lower_nlayers
    )
    resolved_upper_nlayers = int(
        MATCHED_ARCHITECTURE["upper_nlayers"] if upper_nlayers is None else upper_nlayers
    )
    resolved_batch_delta_state_dim = int(
        MATCHED_ARCHITECTURE["batch_delta_state_dim"]
        if batch_delta_state_dim is None
        else batch_delta_state_dim
    )
    batch_delta_layer_kwargs = {
        "num_solver_steps": int(num_solver_steps),
        "support_target_mode": support_target_mode,
        "target_bilinear_rank": int(target_bilinear_rank),
        "fast_weight_rank": int(fast_weight_rank),
        "base_fast_weight_context_rank": int(base_fast_weight_context_rank),
        "incontext_opt_rank": int(incontext_opt_rank),
        "incontext_opt_steps": int(incontext_opt_steps),
        "incontext_opt_lr": float(incontext_opt_lr),
        "incontext_opt_weight_decay": float(incontext_opt_weight_decay),
        "ridge_lambda_init": float(ridge_lambda_init),
        "learnable_ridge_lambda": bool(learnable_ridge_lambda),
        "qk_l2_normalize": bool(qk_l2_normalize),
        "residual_scale_init": float(residual_scale_init),
    }

    backbone = HybridLinearBatchDeltaBackboneConfig(
        lower_nlayers=resolved_lower_nlayers,
        upper_nlayers=resolved_upper_nlayers,
        nhead=resolved_nhead,
        mlp_hidden_dim=resolved_mlp_hidden_dim,
        batch_delta_state_dim=resolved_batch_delta_state_dim,
        dropout=0.0,
        activation="silu",
        linear_layer_kwargs={
            "feature_attention_softmax": False,
            "causal": False,
            "causal_train_only": False,
        },
        batch_delta_layer_kwargs=batch_delta_layer_kwargs,
    )
    return build_hybrid_batch_delta_main_config(
        config_index=config_index,
        training_setup=training_setup,
        task_variant=task_variant,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        batch_size_stages=batch_size_stages,
        dynamic_batch_size_compensate_grad_accumulation=dynamic_batch_size_compensate_grad_accumulation,
        eval_pos_split_pct=eval_pos_split_pct,
        seq_len_stages=seq_len_stages,
        lr=lr,
        aggregate_k_gradients=aggregate_k_gradients,
        feature_positional_embedding=feature_positional_embedding,
        emsize=resolved_emsize,
        features_per_group=resolved_features_per_group,
        backbone=backbone,
        wandb_name_prefix="linear_batch_delta",
        wandb_tags=[
            "matched_high_config",
            "model_linear_batch_delta",
            f"target_{support_target_mode}",
            f"emb_{resolved_emsize}",
            f"lower_{resolved_lower_nlayers}",
            f"upper_{resolved_upper_nlayers}",
        ],
        extra_name_parts=[
            f"emb{resolved_emsize}",
            f"mlp{resolved_mlp_hidden_dim}",
            f"heads{resolved_nhead}",
            f"lower{resolved_lower_nlayers}",
            f"upper{resolved_upper_nlayers}",
            f"state{resolved_batch_delta_state_dim}",
            f"steps{int(num_solver_steps)}",
            f"target{support_target_mode}",
            f"bilin{int(target_bilinear_rank)}",
            f"fwr{int(fast_weight_rank)}",
            f"ctxb{int(base_fast_weight_context_rank)}",
            f"ictx{int(incontext_opt_rank)}x{int(incontext_opt_steps)}",
            f"ridge{float(ridge_lambda_init):g}",
            "lridge" if bool(learnable_ridge_lambda) else "fixedridge",
        ],
    )
