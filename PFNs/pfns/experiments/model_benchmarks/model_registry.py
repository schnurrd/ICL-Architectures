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
    "KDA_Comb_MT": {
        "wandb_run_id": "fla_models/runs/ksmv5v4z",
    },
    "KDA_Comb_ST": {
        "wandb_run_id": "fla_models/runs/qkruutrt",
    },
    "KDA_Comb_ST_short_conv": {
        "wandb_run_id": "fla_models/runs/z7xfal1g",
    },
    "KDA_Int_ST": {
        "wandb_run_id": "fla_models/runs/63y7kc9k",
    },
    "KDA_Int_ST_short_conv": {
        "wandb_run_id": "fla_models/runs/q8l1av2n",
    },
    "KDA_Int_MT": {
        "wandb_run_id": "fla_models/runs/a925p05n",
    },
    "KDA_Int_MT_short_conv": {
        "wandb_run_id": "fla_models/runs/ab6fuy9c",
    },
}

GLA_MODELS: dict[str, dict[str, Any]] = {
    "GLA_Comb_MT": {
        "wandb_run_id": "fla_models/runs/yzw9d63f",
    },
    "gla:GLA_Comb_ST": {
        "display_name": "GLA_Comb_ST",
        "wandb_run_id": "fla_models/runs/g1ul5lyc",
    },
    "GLA_Comb_ST_short_conv": {
        "wandb_run_id": "fla_models/runs/47u2og3a",
    },
    "GLA_Int_ST": {
        "wandb_run_id": "fla_models/runs/9k1i2f9z",
    },
    "GLA_Int_ST_short_conv": {
        "wandb_run_id": "fla_models/runs/do2tv5da",
    },
    "GLA_Int_MT": {
        "wandb_run_id": "fla_models/runs/4f224z23",
    },
    
}

DELTANET_MODELS_SIZE_CHANGES: dict[str, dict[str, Any]] = {
    "DeltaNet_Comb_ST": {
        "wandb_run_id": "fla_models/runs/q67a0x92", 
        "eval_autocast_dtype": "bf16",
    },
    "DeltaNet_Comb_ST_Layers_24": {
        "wandb_run_id": "fla_models/runs/zbcsdb9h", # Twice the number of layers, currently running
        "eval_autocast_dtype": "bf16",
    },
    "DeltaNet_Comb_ST_Hidden_Size_480": {
        "wandb_run_id": "fla_models/runs/tr0jxu69", # 1.5x hidden size, currently running
        "eval_autocast_dtype": "bf16",
    },
    "DeltaNet_Comb_ST_Hidden_Size_480_Heads_6": {
        "wandb_run_id": "fla_models/runs/gzag08i9", # 1.5x hidden size, 1.5x heads, currently running
        "eval_autocast_dtype": "bf16",
    },
    "DeltaNet_Comb_ST_Hidden_Size_640_Heads_8": {
        "wandb_run_id": "fla_models/runs/j8k7t7nb", # 2x hidden size, 2x heads, currently running
        "eval_autocast_dtype": "bf16",
    },
    "DeltaNet_Comb_ST_Hidden_Size_640": {
        "wandb_run_id": "fla_models/runs/niytteb0", # 2x hidden size,
        "eval_autocast_dtype": "bf16",
    },
}

DELTANET_MODELS: dict[str, dict[str, Any]] = {
    "DeltaNet_Comb_MT": {
        "wandb_run_id": "fla_models/runs/iwaesmvk",
        "eval_autocast_dtype": "bf16",
    },
    "DeltaNet_Comb_MT_short_conv": {
        "wandb_run_id": "fla_models/runs/j735qiit",
        "eval_autocast_dtype": "bf16",
    },
    "DeltaNet_Comb_ST": {
        "wandb_run_id": "fla_models/runs/q67a0x92", 
        "eval_autocast_dtype": "bf16",
    },
    "DeltaNet_Comb_ST_short_conv": {
        "wandb_run_id": "fla_models/runs/nluohjzz", # second model nluohjzz
        "eval_autocast_dtype": "bf16",
    },
    "DeltaNet_Int_ST": {
        "wandb_run_id": "fla_models/runs/0r7dz00x",
        "eval_autocast_dtype": "bf16",
    },
    "DeltaNet_Int_ST_short_conv": {
        "wandb_run_id": "fla_models/runs/9v4hbvug",
        "eval_autocast_dtype": "bf16",
    },
    "DeltaNet_Int_MT": {
        "wandb_run_id": "fla_models/runs/alqp1bd2",
        "eval_autocast_dtype": "bf16",
    },
    "DeltaNet_Int_MT_short_conv": {
        "wandb_run_id": "fla_models/runs/fm8kzerj",
        "eval_autocast_dtype": "bf16",
    },
}

GATED_DELTANET_MODELS: dict[str, dict[str, Any]] = {
    "Gated_DeltaNet_Comb_MT": {
        "wandb_run_id": "fla_models/runs/h5xhs15j",
    },
    "Gated_DeltaNet_Comb_ST": {
        "wandb_run_id": "fla_models/runs/abi7ojxu",
    },
    "Gated_DeltaNet_Comb_ST_seq_len_2K": {
        "wandb_run_id": "fla_models/runs/uah7zywj",
    },
    "Gated_DeltaNet_Comb_ST_seq_len_10K": {
        "wandb_run_id": "fla_models/runs/9elhe2fw",
    },
    "Gated_DeltaNet_Int_ST": {
        "wandb_run_id": "fla_models/runs/6temwkyx",
    },
    "Gated_DeltaNet_Int_MT": {
        "wandb_run_id": "fla_models/runs/sjkv0db4",
    },
}

MAMBA2_MODELS: dict[str, dict[str, Any]] = {
    "Mamba2_Comb_MT": {
        "wandb_run_id": "fla_models/runs/wccjh2ye",
    },
    "Mamba2_Comb_ST": {
        "wandb_run_id": "fla_models/runs/sac363pc",
    },
    "Mamba2_Int_ST": {
        "wandb_run_id": "fla_models/runs/kfgmmqu5",
    },
    "Mamba2_Int_MT": {
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
    "equal_params:GLA_Comb_ST": {
        "display_name": "GLA_Comb_ST",
        "wandb_run_id": "fla_models/runs/4vsqz1ee",
    },
    "equal_params:Mamba2_Comb_ST": {
        "display_name": "Mamba2_Comb_ST",
        "wandb_run_id": "fla_models/runs/o9e00w17",
    },
    "equal_params:DeltaNet_Comb_ST": {
        "display_name": "DeltaNet_Comb_ST",
        "wandb_run_id": "fla_models/runs/ob2m9rth",
        "eval_autocast_dtype": "bf16",
    },
    # "DeltaNet_Int_MT": {
    #     "wandb_run_id": "fla_models/runs/v18qqmbk",  # second run 2m9zukic on obsession 0  to check variance
    #     "eval_autocast_dtype": "bf16",
    # },
    "equal_params:Gated_DeltaNet_Comb_ST": {
        "display_name": "Gated_DeltaNet_Comb_ST",
        "wandb_run_id": "fla_models/runs/g7rh5nv9",  
    },
    # "Gated_DeltaNet_Int_MT": {
    #     "wandb_run_id": "fla_models/runs/cpcq82tx", # second run 2cm1gdi5 on obsession 0 to check variance
    # },
    "equal_params:KDA_Comb_ST": {
        "display_name": "KDA_Comb_ST",
        "wandb_run_id": "fla_models/runs/qaskm2mq",
    },
    "equal_params:Rebased_Comb_ST": {
        "display_name": "Rebased_Comb_ST",
        "wandb_run_id": "fla_models/runs/ntkpkzf3", 
    },
    "equal_params:Transformer_Comb_ST": {
        "display_name": "Transformer_Comb_ST",
        "wandb_run_id": "tabpfn_transformer/runs/nb5hz44b",
    },
    "equal_params:Linear_Attention_Comb_ST": {
        "display_name": "Linear_Attention_Comb_ST",
        "wandb_run_id": "linear_attention/runs/ygawhsm9",
    },
}

TRANSFORMER_MASKED_MODELS: dict[str, dict[str, Any]] = {
    "Transformer_Non_Causal": { # fp16 version
        "wandb_run_id": "tabpfn_transformer_masking_experiments/runs/d4mttnjl", # fp16 version d4mttnjl, fp 32 version pmcn4brd
        "eval_mode": "forward",
    },
    # "Transformer_Non_Causal_fp32": {
    #     "wandb_run_id": "tabpfn_transformer_masking_experiments/runs/pmcn4brd", # fp16 version d4mttnjl, fp 32 version pmcn4brd
    #     "eval_mode": "forward",
    # },
    "Transformer_Non_Causal_with_RoPE": {
        "wandb_run_id": "tabpfn_transformer_masking_experiments/runs/xsbe5y6d", # old runs: xsbe5y6d, second run with fp32 as comparison: 0xi6dcvc
        "eval_mode": "forward",
    },
    "Transformer_Non_Causal_interleaved_with_RoPE_paired": {
        "wandb_run_id": "tabpfn_transformer_masking_experiments/runs/6kid4bgi",   # new one uses pairwise rope while old one does not jzs97xfg
        "eval_mode": "forward",
    },
    "Transformer_Comb_ST": {
        "wandb_run_id": "tabpfn_transformer_masking_experiments/runs/b56ohkmz", # fp 16 version 2wrxsh60
        "eval_mode": "forward",
    },
    "Transformer_Int_ST_with_RoPE_paired": {
        "wandb_run_id": "tabpfn_transformer_masking_experiments/runs/z36s69e0",  # new one uses pairwise rope while old one does not 7yzlf15p
        "eval_mode": "forward",
    },
    # "Transformer_Test_To_Train_Only": {
    #     "wandb_run_id": "tabpfn_transformer_masking_experiments/runs/1agq90eo",
    #     "eval_mode": "forward",
    # },
    "Transformer_Comb_MT": {
        "wandb_run_id": "tabpfn_transformer_masking_experiments/runs/81g04qla",
        "eval_mode": "forward",
    },
    "Transformer_Int_MT_with_RoPE": { 
        "wandb_run_id": "tabpfn_transformer_masking_experiments/runs/xiv7f2z3", # old model without pairwise rope m74u7psh
        "eval_mode": "forward",
    },
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
    "deltanet_size_changes": DELTANET_MODELS_SIZE_CHANGES,
    "gated_deltanet": GATED_DELTANET_MODELS,
    "mamba2": MAMBA2_MODELS,
    "linear_attention": LINEAR_ATTENTION_MODELS,
    "rebased": REBASED_MODELS,
    "equal_params": EQUAL_PARAMS_MODELS,
    "transformer_masked": TRANSFORMER_MASKED_MODELS,
    "fla_models": {
        **KDA_MODELS,
        **GLA_MODELS,
        **DELTANET_MODELS,
        **GATED_DELTANET_MODELS,
        **MAMBA2_MODELS,
        **DELTANET_MODELS_SIZE_CHANGES,
    },
    "other": OTHER_MODELS,
}

NON_FUNCTIONAL_CONFIG_KEYS = frozenset({"display_name"})


def _default_display_name(model_name: str) -> str:
    if ":" in model_name:
        return model_name.split(":", maxsplit=1)[1]
    return model_name


def _copy_model_config_with_display_name(
    model_name: str,
    model_config: dict[str, Any],
) -> dict[str, Any]:
    copied = model_config.copy()
    copied.setdefault("display_name", _default_display_name(model_name))
    return copied


def functional_model_config(model_config: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in model_config.items()
        if key not in NON_FUNCTIONAL_CONFIG_KEYS
    }

def _merge_models_with_conflict_check(
    *,
    selected: dict[str, dict[str, Any]],
    selected_sources: dict[str, str],
    family_name: str,
    models: dict[str, dict[str, Any]],
    allowed_names: set[str] | None = None,
) -> None:
    for model_name, model_config in models.items():
        if allowed_names is not None and model_name not in allowed_names:
            continue
        existing = selected.get(model_name)
        existing_functional = functional_model_config(existing) if existing is not None else None
        new_functional = functional_model_config(model_config)
        if existing is not None and existing_functional != new_functional:
            previous_family = selected_sources[model_name]
            raise ValueError(
                f"Model {model_name!r} has conflicting configs across selections: "
                f"{previous_family!r} vs {family_name!r}. "
                f"Existing={existing!r}, new={model_config!r}"
            )
        if existing is None:
            selected[model_name] = _copy_model_config_with_display_name(model_name, model_config)
            selected_sources[model_name] = family_name


def get_baseline_models() -> dict[str, dict[str, Any]]:
    return {
        name: {
            "runner": "baseline",
            "baseline_name": name,
            "display_name": name,
        }
        for name in BASELINE_MODEL_NAMES
    }


def get_models_from_names(model_names: Iterable[str]) -> dict[str, dict[str, Any]]:
    model_names = list(model_names)
    model_names_set = set(model_names)
    selected: dict[str, dict[str, Any]] = {}
    selected_sources: dict[str, str] = {}
    for family_name, family_models in MODEL_FAMILIES.items():
        _merge_models_with_conflict_check(
            selected=selected,
            selected_sources=selected_sources,
            family_name=family_name,
            models=family_models,
            allowed_names=model_names_set,
        )
    missing = [name for name in model_names if name not in selected]
    if missing:
        available = ", ".join(
            sorted({name for models in MODEL_FAMILIES.values() for name in models})
        )
        missing_str = ", ".join(missing)
        raise KeyError(f"Unknown model name(s): {missing_str}. Available models: {available}")
    return {name: selected[name].copy() for name in model_names}


def get_models_from_families(family_names: Iterable[str]) -> dict[str, dict[str, Any]]:
    family_names = list(family_names)
    selected: dict[str, dict[str, Any]] = {}
    selected_sources: dict[str, str] = {}
    unknown = [name for name in family_names if name not in MODEL_FAMILIES]
    if unknown:
        available = ", ".join(sorted(MODEL_FAMILIES))
        unknown_str = ", ".join(unknown)
        raise KeyError(f"Unknown family name(s): {unknown_str}. Available families: {available}")

    for family_name in family_names:
        _merge_models_with_conflict_check(
            selected=selected,
            selected_sources=selected_sources,
            family_name=family_name,
            models=MODEL_FAMILIES[family_name],
        )
    return selected


def get_all_models() -> dict[str, dict[str, Any]]:
    return get_models_from_families(MODEL_FAMILIES)


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
