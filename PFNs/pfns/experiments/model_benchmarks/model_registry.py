from __future__ import annotations

from typing import Any, Iterable
from pfns.training_utils import is_autocast_dtype_enabled, resolve_autocast_dtype
from pfns.utils import get_default_device

TRANSFORMER_MODELS: dict[str, dict[str, Any]] = {
    "Softmax_Transformer": {
        "wandb_run_id": "tabpfn_transformer/runs/lqft3oxa",
        "eval_autocast_dtype": "fp16", # bf16 does not work on rtx 2080 ti due to the GPU being too old -> OOM error in scaled dot product attention
    },
}


GLA_MODELS: dict[str, dict[str, Any]] = {
    "GLA_Comb_MT": {
        "display_name": "GLA Combined Multi Target",
        "wandb_run_id": "fla_models/runs/yzw9d63f",
    },
    "GLA_Comb_ST": {
        "display_name": "GLA Combined Single Target",
        "wandb_run_id": "fla_models/runs/g1ul5lyc",
    },
    "GLA_Int_ST": {
        "display_name": "GLA Interleaved Single Target",
        "wandb_run_id": "fla_models/runs/9k1i2f9z",
    },
    "GLA_Int_MT": {
        "display_name": "GLA Interleaved Multi Target",
        "wandb_run_id": "fla_models/runs/4f224z23",
    },
}


DELTANET_MODELS: dict[str, dict[str, Any]] = {
    "DeltaNet_Comb_MT": {
        "display_name": "DeltaNet Combined Multi Target",
        "wandb_run_id": "fla_models/runs/iwaesmvk",
    },
    "DeltaNet_Comb_ST": {
        "display_name": "DeltaNet Combined Single Target",
        "wandb_run_id": "fla_models/runs/q67a0x92", 
    },
    "DeltaNet_Int_ST": {
        "display_name": "DeltaNet Interleaved Single Target",
        "wandb_run_id": "fla_models/runs/0r7dz00x",
    },
    "DeltaNet_Int_MT": {
        "display_name": "DeltaNet Interleaved Multi Target",
        "wandb_run_id": "fla_models/runs/alqp1bd2",
    },
}


GATED_DELTANET_MODELS: dict[str, dict[str, Any]] = {
    "Gated_DeltaNet_Comb_MT": {
        "display_name": "Gated DeltaNet Combined Multi Target",
        "wandb_run_id": "fla_models/runs/h5xhs15j",
    },
    "Gated_DeltaNet_Comb_ST": {
        "display_name": "Gated DeltaNet Combined Single Target",
        "wandb_run_id": "fla_models/runs/abi7ojxu",
    },
    "Gated_DeltaNet_Int_ST": {
        "display_name": "Gated DeltaNet Interleaved Single Target",
        "wandb_run_id": "fla_models/runs/6temwkyx",
    },
    "Gated_DeltaNet_Int_MT": {
        "display_name": "Gated DeltaNet Interleaved Multi Target",
        "wandb_run_id": "fla_models/runs/sjkv0db4",
    },
}

MAMBA2_MODELS: dict[str, dict[str, Any]] = {
    "Mamba2_Comb_MT": {
        "display_name": "Mamba-2 Combined Multi Target",
        "wandb_run_id": "fla_models/runs/ku412muw",
    },
    "Mamba2_Comb_ST": {
        "display_name": "Mamba-2 Combined Single Target",
        "wandb_run_id": "fla_models/runs/arzdn9rh",
    },
    "Mamba2_Int_ST": {
        "display_name": "Mamba-2 Interleaved Single Target",
        "wandb_run_id": "fla_models/runs/cdyctzjo",
    },
    "Mamba2_Int_MT": {
        "display_name": "Mamba-2 Interleaved Multi Target",
        "wandb_run_id": "fla_models/runs/hvmrqqbi",
    },
}


LINEAR_ATTENTION_MODELS: dict[str, dict[str, Any]] = {
    "Linear_Attention_Non_Causal": {
      "wandb_run_id": "linear_attention/runs/83hs69fa",
      "display_name": "Linear Attention\nNon-Causal",
    },
    "Linear_Attention_Non_Causal_fro_norm": {
      "wandb_run_id": "linear_attention/runs/i960z4r7",
      "display_name": "Linear Attention\n(Non-Causal) w. Fro Norm",
    },
    "Linear_Attention_Comb_ST": {
      "wandb_run_id": "linear_attention/runs/3jq88aqt",
      "display_name": "Linear Attention\nCausal",
    },
    "Linear_Attention_Comb_ST_fro_norm": {
      "wandb_run_id": "linear_attention/runs/rrakg728",
      "display_name": "Linear Attention\n(Comb ST) w. Fro Norm",
    },
}


DELTANET_HIGH_SEQ_LEN_MODELS: dict[str, dict[str, Any]] = {
    "DeltaNet_Comb_ST_Reference": {
        "wandb_run_id": "fla_models/runs/ob2m9rth",
        "display_name": "DeltaNet Reference",
    },
    "DeltaNet_Comb_ST_Seq_Len_200-4K_loguniform": {
        "wandb_run_id": "fla_models/runs/caeqvp6x",
        "display_name": "DeltaNet Seq Len\n200-4K loguniform",
    },
    "DeltaNet_Comb_ST_Seq_Len_200-8K_loguniform": {
        "wandb_run_id": "fla_models/runs/r59sudxf",
        "display_name": "DeltaNet Seq Len\n200-8K loguniform",
    },
    "DeltaNet_Comb_ST_Seq_Len_200-16K_loguniform": {
        "wandb_run_id": "fla_models/runs/didft89r",
        "display_name": "DeltaNet Seq Len\n200-16K loguniform",
    },
    "DeltaNet_Comb_ST_Seq_Len_200-32K_loguniform": {
        "wandb_run_id": "fla_models/runs/fmudiy3w",
        "display_name": "DeltaNet Seq Len\n200-32K loguniform",
    },
    "DeltaNet_Comb_ST_Seq_Len_200-64K_loguniform": {
        "wandb_run_id": "fla_models/runs/9llxebf9",
        "display_name": "DeltaNet Seq Len\n200-64K loguniform",
    },
    "DeltaNet_Comb_ST_Seq_Len_200-100K_loguniform": {
        "wandb_run_id": "fla_models/runs/oh6n51z3",
        "display_name": "DeltaNet Seq Len\n200-100K loguniform",
    },
    "DeltaNet_Comb_ST_Seq_Len_200-64K_lognormal_dataset_matched": {
        "wandb_run_id": "fla_models/runs/a34treix",
        "display_name": "DeltaNet High Seq Len\ncompute matched",
    },
}


EQUAL_PARAMS_MODELS_RETRAINED: dict[str, dict[str, Any]] = {
    "equal_params:Transformer_Comb_ST": { # non-causal version
        "display_name": "Non-Causal Transformer",
        "eval_autocast_dtype": "fp16",
        "wandb_run_id": "tabpfn_transformer/runs/nb5hz44b",
    },
    "Linear_Attention_Non_Causal": {
      "wandb_run_id": "linear_attention/runs/83hs69fa",
      "display_name": "Linear Attention\n(Non-Causal)",
    },
    "equal_params_new:Linear_Attention_Comb_ST": {
      "wandb_run_id": "fla_models/runs/743tmxot",
      "display_name": "Linear Attention\n(Comb_ST)",
    },
    "equal_params_new:DeltaNet_Comb_ST": {
        "display_name": "Delta",
        "wandb_run_id": "fla_models/runs/e9fvlq9x",
    },
    "equal_params_new:GLA_Comb_ST": {
        "display_name": "Gated Linear\nAttention",
        "wandb_run_id": "fla_models/runs/ue3648xk",
    },
    "equal_params_new:Gated_DeltaNet_Comb_ST": {
        "display_name": "Delta Gated",
        "wandb_run_id": "fla_models/runs/bevop0pw",
    },
    "equal_params_new:Mamba2_Comb_ST": {
        "display_name": "Mamba-2",
        "wandb_run_id": "fla_models/runs/mrqlajz1",
    },
    "equal_params_new:GLA_Comb_ST_matched_v2": {
        "display_name": "Gated Linear\nAttention (matched v2)",
        "wandb_run_id": "fla_models/runs/guhcl3zg",
    },
    "equal_params_new:Transformer_Comb_ST_matched_v2": { # non-causal version
        "display_name": "Non-Causal Transformer (matched v2)",
        "eval_autocast_dtype": "fp16",
        "wandb_run_id": "icl_arch/tabpfn_transformer/2sx0des5",
    },
    "equal_params_new:Transformer_Comb_ST_Causal_matched_v2": { # causal version
        "display_name": "Causal Transformer (matched v2)",
        "eval_autocast_dtype": "fp16",
        "wandb_run_id": "icl_arch/tabpfn_transformer_masking_experiments/w5ywh97a", # updated
    },
    "equal_params_new:Gated_DeltaNet_Comb_ST_matched_v2": {
        "display_name": "Gated DeltaNet (matched v2)",
        "wandb_run_id": "fla_models/runs/z51sz4kj",
    },
    # Two-axis (row + feature attention) retrainings of the causal backbones.
    # The `_two_axis` suffix keeps them out of `get_canonical_setting_models`,
    # which selects only names ending exactly in the setting.
    "equal_params_new:Linear_Attention_Causal_Comb_ST_two_axis": {
        "display_name": "Causal Linear Attention (Two Axis)",
        "wandb_run_id": "fla_models/runs/dkyzp2gs",
    },
    "equal_params_new:DeltaNet_Comb_ST_two_axis": {
        "display_name": "DeltaNet (Two Axis)",
        "wandb_run_id": "fla_models/runs/9inbilyc",
    },
}

EQUAL_PARAMS_MODELS: dict[str, dict[str, Any]] = {
    "equal_params:Transformer_Comb_ST": { # non-causal version
        "display_name": "Non-Causal Transformer",
        "eval_autocast_dtype": "fp16",
        "wandb_run_id": "tabpfn_transformer/runs/nb5hz44b",
    },
    "equal_params:Linear_Attention_Non_Causal": {
      "wandb_run_id": "linear_attention/runs/83hs69fa", # new default implementation
      "display_name": "Linear Attention\n(Non-Causal)",
    },
    "equal_params:Linear_Attention_Comb_ST": {
      "wandb_run_id": "linear_attention/runs/3jq88aqt", # new default implementation
      "display_name": "Linear Attention\n(Comb_ST)",
    },
    "equal_params:DeltaNet_Comb_ST": {
        "display_name": "Delta",
        "wandb_run_id": "fla_models/runs/ob2m9rth",
    },
    "equal_params:GLA_Comb_ST": {
        "display_name": "Gated Linear\nAttention",
        "wandb_run_id": "fla_models/runs/4vsqz1ee",
    },
    "equal_params:Gated_DeltaNet_Comb_ST": {
        "display_name": "Delta Gated",
        "wandb_run_id": "fla_models/runs/g7rh5nv9",  
    },
    "equal_params:Mamba2_Comb_ST": {
        "display_name": "Mamba-2",
        "wandb_run_id": "fla_models/runs/o9e00w17",
    },
}

TRANSFORMER_MASKED_MODELS: dict[str, dict[str, Any]] = {
    "Transformer_Non_Causal": {
        "display_name": "Softmax Attention\nNon-Causal", #"Non-Causal (Default)",
        "wandb_run_id": "tabpfn_transformer_masking_experiments/runs/f1lg4ch9",
        "eval_mode": "forward",
        "eval_autocast_dtype": "fp16",
    },
    "Transformer_Non_Causal_new": {
        "display_name": "Softmax Attention\nNon-Causal new", #"Non-Causal (Default)",
        "wandb_run_id": "tabpfn_transformer/runs/2sx0des5", 
        "eval_mode": "forward",
        "eval_autocast_dtype": "fp16",
    },
    "masked:Transformer_Comb_ST": {
        "display_name": "Softmax Attention\nCausal",
        "wandb_run_id": "tabpfn_transformer_masking_experiments/runs/gex7h68b", 
        "eval_mode": "forward",
        "eval_autocast_dtype": "fp16",
    },
    "Transformer_Comb_MT": {
        "display_name": "Softmax Attention (Causal MT)",
        "wandb_run_id": "tabpfn_transformer_masking_experiments/runs/81g04qla",
        "eval_mode": "forward",
        "eval_autocast_dtype": "fp16",
    },
}

STATE_PASSING_MODELS: dict[str, dict[str, Any]] = {
    "State_Passing_GLA_Comb_ST": {
        "display_name": "GLA with\nState Passing",
        "wandb_run_id": "fla_models/runs/66hynh1d"
    },
    "GLA_Comb_ST": {
        "display_name": "GLA Combined\nSingle Target",
        "wandb_run_id": "fla_models/runs/g1ul5lyc",
    },
    "State_Passing_DeltaNet_Comb_ST": {
        "display_name": "DeltaNet with\nState Passing",
        "wandb_run_id": "fla_models/runs/lngpf21c",
    },
    "State_Weaving_DeltaNet_Comb_ST": {
        "display_name": "DeltaNet with\nState Weaving",
        "wandb_run_id": "fla_models/runs/6cu4o5e5",
    },
    "State_Weaving_GLA_Comb_ST": {
        "display_name": "GLA with\nState Weaving",
        "wandb_run_id": "fla_models/runs/ta8ft332",
    },
}

ORACLE_PERFORMANCE_MODELS: dict[str, dict[str, Any]] = {
    "equal_params:Transformer_Comb_ST": { # non-causal version
        "display_name": "Softmax Attention\nNon-Causal",
        "eval_autocast_dtype": "fp16",
        "wandb_run_id": "tabpfn_transformer/runs/nb5hz44b",
    },
    "oracles:DeltaNet_Comb_ST": {
        "display_name": "DeltaNet",
        "wandb_run_id": "fla_models/runs/ob2m9rth",
    },
    "oracles:subsampled:DeltaNet_Comb_ST_3K": {
        "display_name": "DeltaNet\n(Subsampled 3K)",
        "wandb_run_id": "fla_models/runs/ob2m9rth",
        "subsample_dataset_size": 3_000
    },
    "oracles:Oracle_Hidden_State_DeltaNet_Comb_ST": {
        **DELTANET_MODELS["DeltaNet_Comb_ST"],
        "display_name": "Oracle Hidden State", # from deltanet
        "oracle_hidden_state_baseline": True,
        "oracle_num_epochs": 400,
        "oracle_lr": 3e-3,
        "oracle_weight_decay": 1e-5,
        "oracle_patience": 20,
        "oracle_query_batch_size": 4000, # increasing or decreasing batch size hurt at seq len 128k
        "oracle_selection_fraction": 0.1,
        "oracle_evaluate_only_max_seqlen": True,
        "oracle_verbose": False,
        "eval_autocast_dtype": "bf16",
    },
    # Same oracle procedure and hyperparameters, but on the parameter-matched
    # DeltaNet that `oracles:DeltaNet_Comb_ST` uses, so the oracle and its
    # baseline share one checkpoint. `oracles:Oracle_Hidden_State_DeltaNet_Comb_ST`
    # sits on the 10-layer `DeltaNet_Comb_ST` instead, which is a different model
    # from the DeltaNet curve it is plotted against.
    "oracles:Oracle_Hidden_State_DeltaNet_Comb_ST_Matched": {
        "display_name": "Oracle Hidden State\n(Matched DeltaNet)",
        "wandb_run_id": "fla_models/runs/ob2m9rth",
        "oracle_hidden_state_baseline": True,
        "oracle_num_epochs": 400,
        "oracle_lr": 3e-3,
        "oracle_weight_decay": 1e-5,
        "oracle_patience": 20,
        "oracle_query_batch_size": 4000,
        "oracle_selection_fraction": 0.1,
        "oracle_evaluate_only_max_seqlen": True,
        "oracle_verbose": False,
        "eval_autocast_dtype": "bf16",
    },
}

LAYER_ORACLE_PERFORMANCE_MODELS: dict[str, dict[str, Any]] = {
    "oracles:Linear_Attention_Non_Causal_Ridge_scaled_0.1": {
        "display_name": "Linear Attention Non-Causal\nRidge Oracle (Scaled 0.1)",
        "wandb_run_id": "linear_attention/runs/4i36ib45",
        "eval_autocast_dtype": "fp32",
    },
}

SUBSAMPLED_MODELS: dict[str, dict[str, Any]] = {
    "subsampled:DeltaNet_Comb_ST_3K": {
        "display_name": "DeltaNet Comb ST\n(Subsampled 3K)",
        "wandb_run_id": "fla_models/runs/ob2m9rth",
        "subsample_dataset_size": 3_000
    },
    "subsampled:GLA_Comb_ST_3K": {
        "display_name": "GLA Comb ST\n(Subsampled 3K)",
        "wandb_run_id": "fla_models/runs/2v2xw7d2",
        "subsample_dataset_size": 3_000
    },
}

MIMETIC_INITIALIZATION_MODELS: dict[str, dict[str, Any]] = {
    "mimetic:GLA_Comb_ST_Ref": {
        "display_name": "GLA",
        "wandb_run_id": "fla_models/runs/2v2xw7d2",
    },
    "mimetic:GLA_Comb_ST_mimetic_full": {
        "display_name": "GLA Comb ST (Full Mimetic)",
        "wandb_run_id": "fla_models/runs/dthrura3",
    },
    "mimetic:GLA_Comb_ST_mimetic_gate_only": {
        "display_name": "GLA with Mimetic",
        "wandb_run_id": "fla_models/runs/l9lcdj1f",
    },
    
}

BIDIRECTIONAL_MODELS: dict[str, dict[str, Any]] = {
    "Bidirectional_DeltaNet_Comb_ST_mean_output_mean_cache": {
        "display_name": "Bidirectional DeltaNet",
        "wandb_run_id": "icl_arch/fla_models/5rv92df5",
    },
    "Bidirectional_GLA_Comb_ST_mean_output_mean_cache": {
        "display_name": "Bidirectional GLA ",
        "wandb_run_id": "icl_arch/fla_models/iw22mtux",
    },
    "DeltaNet_Comb_ST_Reference_New": {
        "display_name": "DeltaNet",
        "wandb_run_id": "fla_models/runs/tuj1kct1",
    },
}


NON_CAUSAL_FLA_MODELS: dict[str, dict[str, Any]] = {
    "DeltaNet_Comb_ST_online_inverse_eval_only": {
        "display_name": "DeltaNet + Online Inverse LR Decay (eval only)",
        "wandb_run_id": "fla_models/runs/dfzrvzcz",
        "deltanet_beta_decay": "online_inverse",
        "deltanet_beta_decay_t0": 256,
    },
    "Non_Causal_DeltaNet": {
        "display_name": "Non-Causal DeltaNet (Comb ST)",
        "wandb_run_id": "icl_arch/fla_models/dj7xmlsb", # fp32 8cpcrc2e
    },
    "Non_Causal_DeltaNet_loguniform_64K": {
        "display_name": "Non-Causal DeltaNet loguniform 64K(Comb ST)",
        "wandb_run_id": "icl_arch/fla_models/p70y8140", # fp32 8cpcrc2e
    },
    "Non_Causal_DeltaNet_online_inverse_eval_only": {
        "display_name": "Non-Causal DeltaNet + Online Inverse LR Decay (Comb ST)",
        "wandb_run_id": "icl_arch/fla_models/dj7xmlsb", # fp32 8cpcrc2e
        "deltanet_beta_decay": "online_inverse",
        "deltanet_beta_decay_t0": 256,
    },
    "Non_Causal_DeltaNet_with_lr_decay_online_inverse_t0_256": {
        "display_name": "Non-Causal DeltaNet with LR Decay: Online Inverse T0=256 (Comb ST)",
        "wandb_run_id": "fla_models/runs/feoe37tc", # online_inverse, beta_decay_t0=256
    },
    "DeltaNet_with_lr_decay_online_inverse_t0_256": {
        "display_name": "DeltaNet with LR Decay: Online Inverse T0=256 (Comb ST)",
        "wandb_run_id": "fla_models/runs/nw73xsr1", # online_inverse, beta_decay_t0=256
    },
    "DeltaNet_Int_MT_12L_no_decay": {
        "display_name": "DeltaNet Int MT 12L (no decay)",
        "wandb_run_id": "icl_arch/fla_models/06k7fek6",  # Delta (Int-MT)
    },
    "DeltaNet_Int_MT_with_lr_decay_online_inverse_t0_256": {
        "display_name": "DeltaNet Int MT with LR Decay: Online Inverse T0=256",
        "wandb_run_id": "icl_arch/fla_models/e6008u2i",  # online_inverse, t0=256, tokens_per_step=2
    },
    "Non_Causal_GLA": {
        "display_name": "Non-Causal GLA",
        "wandb_run_id": "icl_arch/fla_models/j5vkgn2l",
    },
}


BASELINE_MODEL_NAMES: tuple[str, ...] = (
    "RandomForest",
    "XGBoost",
    "CatBoost",
    "TabICLv2",
    "TabPFNv2.5",
    "TabFlex",
)


MODEL_FAMILIES: dict[str, dict[str, dict[str, Any]]] = {
    "transformer": TRANSFORMER_MODELS,
    "gla": GLA_MODELS,
    "deltanet": DELTANET_MODELS,
    "oracles": ORACLE_PERFORMANCE_MODELS,
    "layer_oracles": LAYER_ORACLE_PERFORMANCE_MODELS,
    "gated_deltanet": GATED_DELTANET_MODELS,
    "mamba2": MAMBA2_MODELS,
    "linear_attention": LINEAR_ATTENTION_MODELS,
    "equal_params": EQUAL_PARAMS_MODELS,
    "equal_params_new": EQUAL_PARAMS_MODELS_RETRAINED, 
    "transformer_masked": TRANSFORMER_MASKED_MODELS,
    "deltanet_high_seq_len": DELTANET_HIGH_SEQ_LEN_MODELS,
    "mimetic_initialization": MIMETIC_INITIALIZATION_MODELS,
    "subsampled": SUBSAMPLED_MODELS,
    "bidirectional": BIDIRECTIONAL_MODELS,
    "state_passing": STATE_PASSING_MODELS,
    "fla_models": {
        **GLA_MODELS,
        **DELTANET_MODELS,
        **GATED_DELTANET_MODELS,
        **MAMBA2_MODELS,
    },
    "non_causal_fla": NON_CAUSAL_FLA_MODELS,
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


def is_oracle_model(model_name: str, model_config: dict[str, Any]) -> bool:
    return bool(model_config.get("oracle_hidden_state_baseline"))


def exclude_oracle_models(
    model_configs: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        model_name: model_config
        for model_name, model_config in model_configs.items()
        if not is_oracle_model(model_name, model_config)
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
        dtype_spec = model_config.get("eval_autocast_dtype", "auto")
        resolved_dtype = resolve_autocast_dtype(resolved_device, dtype_spec)
        if not is_autocast_dtype_enabled(resolved_dtype):
            continue
        autocast_models[model_name] = resolved_dtype
    return autocast_models


def get_forward_models_from_registry(
    model_configs: dict[str, dict[str, Any]],
) -> list[str]:
    return [
        model_name
        for model_name, model_config in model_configs.items()
        if model_config.get("eval_mode") == "forward"
    ]
