from __future__ import annotations

import os

import torch


def _filter_model_types(model_types: tuple[str, ...]) -> tuple[str, ...]:
    raw_excluded = os.getenv("FLA_EXCLUDE_MODEL_TYPES", "")
    if not raw_excluded:
        return model_types
    excluded = {item.strip() for item in raw_excluded.split(",") if item.strip()}
    if not excluded:
        return model_types
    return tuple(model_type for model_type in model_types if model_type not in excluded)


FLA_MODEL_TYPES = _filter_model_types(
    ("gla", "kda", "deltanet", "gated_deltanet", "mamba2", "linear_attn")
)


def fla_model_config_kwargs(
    model_type: str,
    *,
    size: str = "small",
) -> dict[str, object]:
    """Get config kwargs for a given FLA model type and preset size."""
    if size == "small":
        hidden_size = 8 if model_type != "mamba2" else 64
        num_heads = 2
        num_layers = 2
        intermediate_size = 32 if model_type != "mamba2" else 128
        state_size = 64 if model_type == "mamba2" else None
    elif size == "medium":
        hidden_size = 32 if model_type != "mamba2" else 128
        num_heads = 4
        num_layers = 3
        intermediate_size = 128 if model_type != "mamba2" else 256
        state_size = 64 if model_type == "mamba2" else None
    elif size == "equivalence":
        hidden_size = 32 if model_type != "mamba2" else 64
        num_heads = 4 if model_type != "mamba2" else 2
        num_layers = 2
        intermediate_size = 64 if model_type != "mamba2" else 128
        state_size = 64 if model_type == "mamba2" else None
    else:
        raise ValueError(f"Unsupported size: {size}")

    if model_type == "gla":
        return {
            "hidden_size": hidden_size,
            "num_hidden_layers": num_layers,
            "num_heads": num_heads,
            "intermediate_size": intermediate_size,
            "hidden_act": "swish",
            "norm_eps": 1e-5,
            "use_cache": True,
        }
    if model_type == "mamba2":
        return {
            "hidden_size": hidden_size,
            "num_hidden_layers": num_layers,
            "state_size": state_size,
            "conv_kernel": 4,
            "intermediate_size": intermediate_size,
            "num_heads": num_heads,
            "use_cache": True,
        }
    if model_type == "kda":
        return {
            "hidden_size": hidden_size,
            "num_hidden_layers": num_layers,
            "num_heads": num_heads,
            "intermediate_size": intermediate_size,
            "hidden_act": "swish",
            "norm_eps": 1e-5,
            "use_cache": True,
            "use_short_conv": True,
        }
    if model_type == "deltanet":
        return {
            "hidden_size": hidden_size,
            "num_hidden_layers": num_layers,
            "num_heads": num_heads,
            "intermediate_size": intermediate_size,
            "hidden_act": "swish",
            "norm_eps": 1e-5,
            "use_cache": True,
            "use_short_conv": True,
        }
    if model_type == "gated_deltanet":
        return {
            "hidden_size": hidden_size,
            "num_hidden_layers": num_layers,
            "num_heads": num_heads,
            "head_dim": hidden_size // num_heads,
            "intermediate_size": intermediate_size,
            "hidden_act": "swish",
            "norm_eps": 1e-5,
            "use_cache": True,
            "use_short_conv": True,
        }
    if model_type == "linear_attn":
        return {
            "attn_mode": "fused_recurrent",
            "hidden_size": hidden_size,
            "num_hidden_layers": num_layers,
            "num_heads": num_heads,
            "intermediate_size": intermediate_size,
            "feature_map": "identity",
            "norm_q": False,
            "norm_k": False,
            "norm_feature_map": False,
            "hidden_act": "swish",
            "norm_eps": 1e-5,
            "use_cache": True,
        }
    raise ValueError(f"Unsupported model_type: {model_type}")


def fla_tolerances(
    model_type: str,
    *,
    default: tuple[float, float] = (1e-6, 1e-6),
) -> tuple[float, float]:
    if model_type in {"deltanet", "mamba2"}:
        return 1e-3, 1e-3
    if model_type in {"kda", "gated_deltanet"}:
        return 1e-4, 1e-4
    return default


def fla_cache_equivalence_tolerances(model_type: str) -> tuple[float, float]:
    if model_type in {"kda", "deltanet", "gated_deltanet", "mamba2"}:
        return 1e-4, 1e-4
    return 1e-6, 1e-6


def build_fla_backbone(
    model_type: str,
    *,
    size: str = "small",
    sequence_mode: str = "Comb_ST",
    cache_chunk_size: int | None = None,
    mimetic_init: bool = False,
    mimetic_init_layer_indices: tuple[int, ...] | list[int] | None = None,
    mimetic_init_mode: str = "gates",
    train: bool = False,
) -> torch.nn.Module:
    from pfns.model.backbones import FLABackboneConfig

    kwargs = fla_model_config_kwargs(model_type, size=size)
    config = FLABackboneConfig(
        model_type=model_type,
        config_kwargs=kwargs,
        sequence_mode=sequence_mode,
        cache_chunk_size=cache_chunk_size,
        mimetic_init=mimetic_init,
        mimetic_init_layer_indices=mimetic_init_layer_indices,
        mimetic_init_mode=mimetic_init_mode,
    )
    ninp = int(kwargs["hidden_size"])
    backbone = config.create_backbone(ninp=ninp, attention_between_features=False)
    if train:
        backbone.train()
    else:
        backbone.eval()
    return backbone


def fla_hidden_size(model_type: str, *, size: str = "small") -> int:
    return int(fla_model_config_kwargs(model_type, size=size)["hidden_size"])
