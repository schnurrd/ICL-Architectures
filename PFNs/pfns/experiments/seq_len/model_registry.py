from __future__ import annotations

from typing import Any, Iterable
from pfns.training_utils import resolve_autocast_dtype
from pfns.utils import get_default_device

TRANSFORMER_MODELS: dict[str, dict[str, Any]] = {
    "Softmax_Transformer": {
        "wandb_run_id": "tabpfn_transformer/runs/90rqcrr2",  # no feature attention like fla
    },
    "Softmax_Transformer_with_feature_attention": {
        "wandb_run_id": "tabpfn_transformer/runs/go1re6pr",  # with feature attention (tabpfnv2 default)
    },
    # "Non-Causal_TabPFN": {
    #     "wandb_run_id": "tabpfn_transformer_masking_experiments/runs/pmcn4brd",
    #     "eval_mode": "forward",
    # },
    # "Causal_TabPFN": {
    #     "wandb_run_id": "tabpfn_transformer_masking_experiments/runs/b56ohkmz",
    #     "eval_mode": "forward",
    # },
    # "Test_To_Train_Only_TabPFN": {
    #     "wandb_run_id": "tabpfn_transformer_masking_experiments/runs/1agq90eo",
    #     "eval_mode": "forward",
    # },
}

KDA_MODELS: dict[str, dict[str, Any]] = {
    "KDA_causal": {
        "wandb_run_id": "fla_models/runs/ksmv5v4z",
    },
    "KDA_cached": {
        "wandb_run_id": "fla_models/runs/qkruutrt",
    },
    "KDA_cached_short_conv": {
        "wandb_run_id": "fla_models/runs/z7xfal1g",
    },
    "KDA_cached_interleaved": {
        "wandb_run_id": "fla_models/runs/63y7kc9k",
    },
    "KDA_teacher_forcing": {
        "wandb_run_id": "fla_models/runs/a925p05n",
    },
    "KDA_teacher_forcing_short_conv": {
        "wandb_run_id": "fla_models/runs/ab6fuy9c",
    },
    "KDA_causal_interleaved": {
        "wandb_run_id": "fla_models/runs/cneseyi0",
    },
}

GLA_MODELS: dict[str, dict[str, Any]] = {
    "GLA_Causal": {
        "wandb_run_id": "fla_models/runs/yzw9d63f",
    },
    "GLA_Cached": {
        "wandb_run_id": "fla_models/runs/g1ul5lyc",
    },
    "GLA_Cached_interleaved": {
        "wandb_run_id": "fla_models/runs/9k1i2f9z",
    },
    "GLA_Causal_interleaved": {
        "wandb_run_id": "fla_models/runs/ztdpate1",
    },
    "GLA_Teacher_Forcing": {
        "wandb_run_id": "fla_models/runs/4f224z23",
    },
}

DELTANET_MODELS: dict[str, dict[str, Any]] = {
    "DeltaNet_Cached": {
        "wandb_run_id": "fla_models/runs/q67a0x92", 
        "eval_autocast_dtype": "bf16",
    },
    "DeltaNet_Cached_short_conv": {
        "wandb_run_id": "fla_models/runs/nluohjzz",
        "eval_autocast_dtype": "bf16",
    },
    "DeltaNet_Cached_short_conv": {
        "wandb_run_id": "fla_models/runs/4bvpfdho",
        "eval_autocast_dtype": "bf16",
    },
    "DeltaNet_Teacher_Forcing": {
        "wandb_run_id": "fla_models/runs/alqp1bd2",
        "eval_autocast_dtype": "bf16",
    },
    "DeltaNet_Teacher_Forcing_short_conv": {
        "wandb_run_id": "fla_models/runs/fm8kzerj",
        "eval_autocast_dtype": "bf16",
    },
    # "DeltaNet_Causal": {
    #     "wandb_run_id": "fla_models/runs/0bkajhpw",  # redo and remove
    # },
}

GATED_DELTANET_MODELS: dict[str, dict[str, Any]] = {
    "Gated_DeltaNet_Cached_seq_len_10K": {
        "wandb_run_id": "fla_models/runs/9elhe2fw",
    },
    "Gated_DeltaNet_Cached_seq_len_2K": {
        "wandb_run_id": "fla_models/runs/uah7zywj",
    },
    "Gated_DeltaNet_Cached": {
        "wandb_run_id": "fla_models/runs/abi7ojxu",
    },
    # "Gated_DeltaNet_Teacher_Forcing": {
    #     "wandb_run_id": "fla_models/runs/16n9ti07",  # redo and remove
    # },
}

MAMBA2_MODELS: dict[str, dict[str, Any]] = {
    "Mamba2_Teacher_Forcing": {
        "wandb_run_id": "fla_models/runs/gn5r8yj6",
    },
    "Mamba2_Cached": {
        "wandb_run_id": "fla_models/runs/sac363pc",
    },
}

LINEAR_ATTENTION_MODELS: dict[str, dict[str, Any]] = {
    "Linear_Attention": {
        "wandb_run_id": "linear_attention/runs/zybvsyiv",
    },
}

REBASED_MODELS: dict[str, dict[str, Any]] = {
    "Rebased": {
        "wandb_run_id": "fla_models/runs/72wtj14x"
    },
}

OTHER_MODELS: dict[str, dict[str, Any]] = {}

MODEL_FAMILIES: dict[str, dict[str, dict[str, Any]]] = {
    "transformer": TRANSFORMER_MODELS,
    "kda": KDA_MODELS,
    "gla": GLA_MODELS,
    "deltanet": DELTANET_MODELS,
    "gated_deltanet": GATED_DELTANET_MODELS,
    "mamba2": MAMBA2_MODELS,
    "linear_attention": LINEAR_ATTENTION_MODELS,
    "rebased": REBASED_MODELS,
    "other": OTHER_MODELS,
}

__all__ = [
    "TRANSFORMER_MODELS",
    "KDA_MODELS",
    "GLA_MODELS",
    "DELTANET_MODELS",
    "GATED_DELTANET_MODELS",
    "MAMBA2_MODELS",
    "LINEAR_ATTENTION_MODELS",
    "REBASED_MODELS",
    "OTHER_MODELS",
    "MODEL_FAMILIES",
    "get_models_from_names",
    "get_models_from_families",
    "get_all_models",
    "get_autocast_models_from_registry",
    "get_forward_models_from_registry",
]


def _copy_models(models: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {name: config.copy() for name, config in models.items()}


def get_models_from_names(model_names: Iterable[str]) -> dict[str, dict[str, Any]]:
    all_models = get_all_models()
    missing = [name for name in model_names if name not in all_models]
    if missing:
        available = ", ".join(sorted(all_models))
        missing_str = ", ".join(missing)
        raise KeyError(f"Unknown model name(s): {missing_str}. Available models: {available}")
    return {name: all_models[name].copy() for name in model_names}


def get_models_from_families(family_names: Iterable[str]) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    unknown = [name for name in family_names if name not in MODEL_FAMILIES]
    if unknown:
        available = ", ".join(sorted(MODEL_FAMILIES))
        unknown_str = ", ".join(unknown)
        raise KeyError(f"Unknown family name(s): {unknown_str}. Available families: {available}")
    for family_name in family_names:
        selected.update(_copy_models(MODEL_FAMILIES[family_name]))
    return selected


def get_all_models() -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for models in MODEL_FAMILIES.values():
        selected.update(_copy_models(models))
    return selected


def get_autocast_models_from_registry(
    model_configs: dict[str, dict[str, Any]],
    *,
    device: str | None = None,
) -> dict[str, Any]:

    resolved_device = device or get_default_device()
    autocast_models: dict[str, Any] = {}
    for model_name, model_config in model_configs.items():
        dtype_spec = model_config.get("eval_autocast_dtype")
        if dtype_spec is None:
            continue
        autocast_models[model_name] = resolve_autocast_dtype(resolved_device, dtype_spec)
    return autocast_models


def get_forward_models_from_registry(
    model_configs: dict[str, dict[str, Any]],
) -> list[str]:
    return [
        model_name
        for model_name, model_config in model_configs.items()
        if model_config.get("eval_mode") == "forward"
    ]
