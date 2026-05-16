from __future__ import annotations

import os
from typing import Any

from pfns.run_logger import download_model_from_wandb
from pfns.model.linear_attention import LinearAttention
from pfns.scripts.tabpfn_interface import load_model_workflow
from pfns.utils import get_default_device

from .oracle_hidden_state_baseline import build_oracle_hidden_state_baseline


def _apply_linear_attention_state_update_override(
    model: Any,
    model_config: dict[str, Any],
) -> None:
    state_update_rule = model_config.get("linear_attention_state_update_rule")
    if state_update_rule is None:
        return

    normalized_rule = LinearAttention._normalize_state_update_rule(
        str(state_update_rule)
    )
    layers = [m for m in model.modules() if isinstance(m, LinearAttention)]
    if not layers:
        raise ValueError(
            "linear_attention_state_update_rule override requires a model with "
            "pfns.model.linear_attention.LinearAttention layers."
        )

    for module in layers:
        module.state_update_rule = normalized_rule
        if normalized_rule != "linear":
            module.use_k_sum_normalization = False
            module.state_renormalization = None
            module.state_renormalization_target_norm = None


def load_models_for_benchmark(
    models_to_compare: dict[str, dict[str, Any]],
    *,
    device: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved_device = device or get_default_device()
    models: dict[str, Any] = {}
    configs: dict[str, Any] = {}

    for model_name, model_config in models_to_compare.items():
        base_path = model_config.get("base_path", ".")
        checkpoint_name = model_config.get("checkpoint_name", "checkpoint.pt")

        if model_config.get("wandb_run_id"):
            target_path = download_model_from_wandb(
                model_config["wandb_run_id"],
                destination_path=model_config.get("destination_path"),
            )
            base_path = os.path.dirname(target_path)
            checkpoint_name = os.path.basename(target_path)

        model, config, _ = load_model_workflow(
            name=checkpoint_name,
            base_path=base_path,
            device=resolved_device,
            high_cardinality_categorical_threshold=model_config.get(
                "high_cardinality_categorical_threshold"
            ),
            make_causal=bool(model_config.get("make_causal", False)),
            make_non_causal=bool(model_config.get("make_non_causal", False)),
        )
        _apply_linear_attention_state_update_override(model, model_config)
        if model_config.get("oracle_hidden_state_baseline"):
            model = build_oracle_hidden_state_baseline(
                base_model=model,
                base_config=config,
                model_config=model_config,
            )
        model.eval()
        models[model_name] = model
        configs[model_name] = config

    return models, configs
