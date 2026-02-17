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
}

KDA_MODELS: dict[str, dict[str, Any]] = {
    "KDA_causal": {
        "wandb_run_id": "fla_models/runs/ksmv5v4z",
    },
    "KDA_causal_interleaved": {
        "wandb_run_id": "fla_models/runs/cneseyi0",
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
    "KDA_cached_interleaved_short_conv": {
        "wandb_run_id": "fla_models/runs/q8l1av2n",
    },
    "KDA_teacher_forcing": {
        "wandb_run_id": "fla_models/runs/a925p05n",
    },
    "KDA_teacher_forcing_short_conv": {
        "wandb_run_id": "fla_models/runs/ab6fuy9c",
    },
}

GLA_MODELS: dict[str, dict[str, Any]] = {
    "GLA_Causal": {
        "wandb_run_id": "fla_models/runs/yzw9d63f",
    },
    "GLA_Causal_interleaved": {
        "wandb_run_id": "fla_models/runs/ztdpate1",
    },
    "GLA_Cached": {
        "wandb_run_id": "fla_models/runs/g1ul5lyc",
    },
    "GLA_Cached_short_conv": {
        "wandb_run_id": "fla_models/runs/47u2og3a",
    },
    "GLA_Cached_interleaved": {
        "wandb_run_id": "fla_models/runs/9k1i2f9z",
    },
    "GLA_Cached_interleaved_short_conv": {
        "wandb_run_id": "fla_models/runs/do2tv5da",
    },
    "GLA_Teacher_Forcing": {
        "wandb_run_id": "fla_models/runs/4f224z23",
    },
    
}

DELTANET_MODELS: dict[str, dict[str, Any]] = {
    "DeltaNet_Causal": {
        "wandb_run_id": "fla_models/runs/iwaesmvk",
        "eval_autocast_dtype": "bf16",
    },
    "DeltaNet_Causal_short_conv": {
        "wandb_run_id": "fla_models/runs/j735qiit",
        "eval_autocast_dtype": "bf16",
    },
    "DeltaNet_Cached": {
        "wandb_run_id": "fla_models/runs/q67a0x92", 
        "eval_autocast_dtype": "bf16",
    },
    # "DeltaNet_Cached_Layers_24": {
    #     "wandb_run_id": "fla_models/runs/zbcsdb9h", # Twice the number of layers, currently running
    #     "eval_autocast_dtype": "bf16",
    # },
    "DeltaNet_Cached_Hidden_Size_480": {
        "wandb_run_id": "fla_models/runs/tr0jxu69", # 1.5x hidden size, currently running
        "eval_autocast_dtype": "bf16",
    },
    # "DeltaNet_Cached_Hidden_Size_480_Heads_6": {
    #     "wandb_run_id": "fla_models/runs/gzag08i9", # 1.5x hidden size, 1.5x heads, currently running
    #     "eval_autocast_dtype": "bf16",
    # },
    "DeltaNet_Cached_Hidden_Size_640_Heads_8": {
        "wandb_run_id": "fla_models/runs/j8k7t7nb", # 2x hidden size, 2x heads, currently running
        "eval_autocast_dtype": "bf16",
    },
    "DeltaNet_Cached_Hidden_Size_640": {
        "wandb_run_id": "fla_models/runs/niytteb0", # 2x hidden size,
        "eval_autocast_dtype": "bf16",
    },
    "DeltaNet_Cached_short_conv": {
        "wandb_run_id": "fla_models/runs/nluohjzz", # second model nluohjzz
        "eval_autocast_dtype": "bf16",
    },
    "DeltaNet_Cached_Interleaved": {
        "wandb_run_id": "fla_models/runs/0r7dz00x",
        "eval_autocast_dtype": "bf16",
    },
    "DeltaNet_Cached_Interleaved_short_conv": {
        "wandb_run_id": "fla_models/runs/9v4hbvug",
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
}

GATED_DELTANET_MODELS: dict[str, dict[str, Any]] = {
    "Gated_DeltaNet_Causal": {
        "wandb_run_id": "fla_models/runs/h5xhs15j",
    },
    "Gated_DeltaNet_Cached": {
        "wandb_run_id": "fla_models/runs/abi7ojxu",
    },
    "Gated_DeltaNet_Cached_seq_len_2K": {
        "wandb_run_id": "fla_models/runs/uah7zywj",
    },
    "Gated_DeltaNet_Cached_seq_len_10K": {
        "wandb_run_id": "fla_models/runs/9elhe2fw",
    },
    "Gated_DeltaNet_Cached_Interleaved": {
        "wandb_run_id": "fla_models/runs/6temwkyx",
    },
    "Gated_DeltaNet_Teacher_Forcing": {
        "wandb_run_id": "fla_models/runs/sjkv0db4",
    },
}

MAMBA2_MODELS: dict[str, dict[str, Any]] = {
    "Mamba2_Causal": {
        "wandb_run_id": "fla_models/runs/wccjh2ye",
    },
    "Mamba2_Cached": {
        "wandb_run_id": "fla_models/runs/sac363pc",
    },
    "Mamba2_Cached_Interleaved": {
        "wandb_run_id": "fla_models/runs/kfgmmqu5",
    },
    "Mamba2_Teacher_Forcing": {
        "wandb_run_id": "fla_models/runs/gn5r8yj6",
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

EQUAL_PARAMS_MODELS: dict[str, dict[str, Any]] = {
    "GLA_Combined_Embbedding_Single_Target": {
        "wandb_run_id": "fla_models/runs/4vsqz1ee",
    },
    "Mamba2_Combined_Embbedding_Single_Target": {
        "wandb_run_id": "fla_models/runs/o9e00w17",
    },
    "DeltaNet_Combined_Embbedding_Single_Target": {
        "wandb_run_id": "fla_models/runs/ob2m9rth",
        "eval_autocast_dtype": "bf16",
    },
    "DeltaNet_Interleaved_Embbedding_Multi_Target": {
        "wandb_run_id": "fla_models/runs/v18qqmbk",  # second run 2m9zukic on obsession 0  to check variance
        "eval_autocast_dtype": "bf16",
    },
    "Gated_DeltaNet_Combined_Embbedding_Single_Target": {
        "wandb_run_id": "fla_models/runs/g7rh5nv9",  
    },
    "Gated_DeltaNet_Interleaved_Embbedding_Multi_Target": {
        "wandb_run_id": "fla_models/runs/cpcq82tx", # second run 2cm1gdi5 on obsession 0 to check variance
    },
    "KDA_Combined_Embbedding_Single_Target": {
        "wandb_run_id": "fla_models/runs/qaskm2mq",
    },
    "Rebased_Combined_Embbedding_Single_Target": {
        "wandb_run_id": "fla_models/runs/ntkpkzf3", 
    },
    "Transformer_Combined_Embbedding_Single_Target": {
        "wandb_run_id": "tabpfn_transformer/runs/nb5hz44b",
    },
    "Linear_Attention_Combined_Embbedding_Single_Target": {
        "wandb_run_id": "linear_attention/runs/ygawhsm9",
    },
}

TRANSFORMER_MASKED_MODELS: dict[str, dict[str, Any]] = {
    "Non-Causal_TabPFN_fp16": {
        "wandb_run_id": "tabpfn_transformer_masking_experiments/runs/d4mttnjl", # fp16 version d4mttnjl, fp 32 version pmcn4brd
        "eval_mode": "forward",
    },
    "Non-Causal_TabPFN_fp32": {
        "wandb_run_id": "tabpfn_transformer_masking_experiments/runs/pmcn4brd", # fp16 version d4mttnjl, fp 32 version pmcn4brd
        "eval_mode": "forward",
    },
    # "Non-Causal_TabPFN_with_RoPE": {
    #     "wandb_run_id": "tabpfn_transformer_masking_experiments/runs/ifvo080r", # second run with fp32 as comparison: 0xi6dcvc
    #     "eval_mode": "forward",
    # },
    "Non-Causal_TabPFN_interleaved_with_RoPE": {
        "wandb_run_id": "tabpfn_transformer_masking_experiments/runs/jzs97xfg",
        "eval_mode": "forward",
    },
    "Causal_Train_Only_TabPFN": {
        "wandb_run_id": "tabpfn_transformer_masking_experiments/runs/b56ohkmz", # fp 16 version 2wrxsh60
        "eval_mode": "forward",
    },
    # "Causal_Train_Only_TabPFN_interleaved_with_RoPE": {
    #     "wandb_run_id": "tabpfn_transformer_masking_experiments/runs/7yzlf15p", 
    #     "eval_mode": "forward",
    # },
    "Test_To_Train_Only_TabPFN": {
        "wandb_run_id": "tabpfn_transformer_masking_experiments/runs/1agq90eo",
        "eval_mode": "forward",
    },
    # "Causal_All_TabPFN": {
    #     "wandb_run_id": "tabpfn_transformer_masking_experiments/runs/81g04qla",
    #     "eval_mode": "forward",
    # },
    # "Causal_All_TabPFN_interleaved_with_RoPE": {
    #     "wandb_run_id": "tabpfn_transformer_masking_experiments/runs/m74u7psh",
    #     "eval_mode": "forward",
    # },
}

OTHER_MODELS: dict[str, dict[str, Any]] = {}

BASELINE_MODEL_NAMES: tuple[str, ...] = (
    "RandomForest",
    "XGBoost",
    "CatBoost",
    "TabICL",
    "TabPFNv2.5",
    "TabFlex",
)

MODEL_FAMILIES: dict[str, dict[str, dict[str, Any]]] = {
    "transformer": TRANSFORMER_MODELS,
    "kda": KDA_MODELS,
    "gla": GLA_MODELS,
    "deltanet": DELTANET_MODELS,
    "gated_deltanet": GATED_DELTANET_MODELS,
    "mamba2": MAMBA2_MODELS,
    "linear_attention": LINEAR_ATTENTION_MODELS,
    "rebased": REBASED_MODELS,
    "equal_params": EQUAL_PARAMS_MODELS,
    "transformer_masked": TRANSFORMER_MASKED_MODELS,
    "other": OTHER_MODELS,
}

def _copy_models(models: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {name: config.copy() for name, config in models.items()}


def get_baseline_models() -> dict[str, dict[str, Any]]:
    return {name: {} for name in BASELINE_MODEL_NAMES}


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
