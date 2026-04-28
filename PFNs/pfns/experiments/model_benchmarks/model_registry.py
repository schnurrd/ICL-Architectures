from __future__ import annotations

from typing import Any, Iterable
from pfns.training_utils import is_autocast_dtype_enabled, resolve_autocast_dtype
from pfns.utils import get_default_device

TRANSFORMER_MODELS: dict[str, dict[str, Any]] = {
    "Softmax_Transformer": {
        "wandb_run_id": "tabpfn_transformer/runs/lqft3oxa",
        "eval_autocast_dtype": "fp16", # bf16 is broken on rtx 2080 ti due to the GPU being to old -> OOM error in scaled dot product attention
    },
    "Softmax_Transformer_Cat_10_Training": {
        "wandb_run_id": "tabpfn_transformer/runs/m5zgo8r3", 
        "eval_autocast_dtype": "fp16",
    },
    "Softmax_Transformer_No_Cat_Norm": {
        "wandb_run_id": "tabpfn_transformer/runs/ajttwh65",  
        "eval_autocast_dtype": "fp16",
    },
    "Softmax_Transformer_Full_Cat_Norm": {
        "wandb_run_id": "tabpfn_transformer/runs/ajttwh65",  
        "high_cardinality_categorical_threshold": 0,
        "eval_autocast_dtype": "fp16",
    },
    # "Softmax_Transformer_fp32": {
    #     "wandb_run_id": "tabpfn_transformer/runs/lqft3oxa",
    #     "eval_autocast_dtype": "fp32",
    # },
    # "Softmax_Transformer_with_feature_attention_group_2": {
    #     "wandb_run_id": "tabpfn_transformer/runs/ec8120cw", # strong but oom at 128k
    #     "eval_autocast_dtype": "fp16",
    # },
    # "Softmax_Transformer_with_feature_attention_group_4": {
    #     "wandb_run_id": "tabpfn_transformer/runs/8l966af8", # features per group 4 (fp32)
    #     "eval_autocast_dtype": "fp16",
    # },
}

KDA_MODELS: dict[str, dict[str, Any]] = {
    "KDA_Comb_MT": {
        "display_name": "KDA Combined Multi Target",
        "wandb_run_id": "fla_models/runs/ksmv5v4z",
    },
    "KDA_Comb_ST": {
        "display_name": "KDA Combined Single Target",
        "wandb_run_id": "fla_models/runs/qkruutrt",
    },
    "KDA_Comb_ST_short_conv": {
        "display_name": "KDA Combined Single Target (Short Conv)",
        "wandb_run_id": "fla_models/runs/z7xfal1g",
    },
    "KDA_Int_ST": {
        "display_name": "KDA Interleaved Single Target",
        "wandb_run_id": "fla_models/runs/63y7kc9k",
    },
    "KDA_Int_ST_short_conv": {
        "display_name": "KDA Interleaved Single Target (Short Conv)",
        "wandb_run_id": "fla_models/runs/q8l1av2n",
    },
    "KDA_Int_MT": {
        "display_name": "KDA Interleaved Multi Target",
        "wandb_run_id": "fla_models/runs/a925p05n",
    },
    "KDA_Int_MT_short_conv": {
        "display_name": "KDA Interleaved Multi Target (Short Conv)",
        "wandb_run_id": "fla_models/runs/ab6fuy9c",
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
    "GLA_Comb_ST_short_conv": {
        "display_name": "GLA Combined Single Target (Short Conv)",
        "wandb_run_id": "fla_models/runs/47u2og3a",
    },
    "GLA_Int_ST": {
        "display_name": "GLA Interleaved Single Target",
        "wandb_run_id": "fla_models/runs/9k1i2f9z",
    },
    "GLA_Int_ST_short_conv": {
        "display_name": "GLA Interleaved Single Target (Short Conv)",
        "wandb_run_id": "fla_models/runs/do2tv5da",
    },
    "GLA_Int_MT": {
        "display_name": "GLA Interleaved Multi Target",
        "wandb_run_id": "fla_models/runs/4f224z23",
    },
}

DELTANET_MODELS_SIZE_CHANGES: dict[str, dict[str, Any]] = {
    "size_changes:DeltaNet_Comb_ST": {
        "display_name": "12 Layers, Hid. S. 320, Heads 6", # reference model for size changes
        "wandb_run_id": "fla_models/runs/q67a0x92", 
    },
    "DeltaNet_Comb_ST_Layers_24": {
        "display_name": "24 Layers",
        "wandb_run_id": "fla_models/runs/zbcsdb9h", # Twice the number of layers, currently running
    },
    "DeltaNet_Comb_ST_Hidden_Size_480": {
        "display_name": "Hidden Size 480", 
        "wandb_run_id": "fla_models/runs/tr0jxu69", # 1.5x hidden size, currently running
    },
    "DeltaNet_Comb_ST_Hidden_Size_480_Heads_6": {
        "display_name": "Hidden Size 480, Heads 6",
        "wandb_run_id": "fla_models/runs/gzag08i9", # 1.5x hidden size, 1.5x heads, currently running
    },
    "DeltaNet_Comb_ST_Hidden_Size_640_Heads_8": {
        "display_name": "Hidden Size 640, Heads 8",
        "wandb_run_id": "fla_models/runs/j8k7t7nb", # 2x hidden size, 2x heads, currently running
    },
    "DeltaNet_Comb_ST_Hidden_Size_640": {
        "display_name": "Hidden Size 640",
        "wandb_run_id": "fla_models/runs/niytteb0", # 2x hidden size,
    },
}

DELTANET_MODELS: dict[str, dict[str, Any]] = {
    "DeltaNet_Comb_MT": {
        "display_name": "DeltaNet Combined Multi Target",
        "wandb_run_id": "fla_models/runs/iwaesmvk",
    },
    "DeltaNet_Comb_MT_short_conv": {
        "display_name": "DeltaNet Combined Multi Target (Short Conv)",
        "wandb_run_id": "fla_models/runs/j735qiit",
    },
    "DeltaNet_Comb_ST": {
        "display_name": "DeltaNet Combined Single Target",
        "wandb_run_id": "fla_models/runs/q67a0x92", 
    },
    "DeltaNet_Comb_ST_no_self_attn": {
        "display_name": "DeltaNet Combined Single Target (No Self-Attention)",
        "wandb_run_id": "fla_models/runs/4vxeqnat", 
    },
    "DeltaNet_Comb_ST_short_conv": {
        "display_name": "DeltaNet Combined Single Target (Short Conv)",
        "wandb_run_id": "fla_models/runs/nluohjzz", # second model nluohjzz
    },
    "DeltaNet_Int_ST": {
        "display_name": "DeltaNet Interleaved Single Target",
        "wandb_run_id": "fla_models/runs/0r7dz00x",
    },
    "DeltaNet_Int_ST_short_conv": {
        "display_name": "DeltaNet Interleaved Single Target (Short Conv)",
        "wandb_run_id": "fla_models/runs/9v4hbvug",
    },
    "DeltaNet_Int_MT": {
        "display_name": "DeltaNet Interleaved Multi Target",
        "wandb_run_id": "fla_models/runs/alqp1bd2",
    },
    "DeltaNet_Int_MT_short_conv": {
        "display_name": "DeltaNet Interleaved Multi Target (Short Conv)",
        "wandb_run_id": "fla_models/runs/fm8kzerj",
    },
}

GATED_DELTANET_MODELS_SEQ_LEN_CHANGES: dict[str, dict[str, Any]] = {
    "seq_len_changes:Gated_DeltaNet_Comb_ST": {
        "display_name": "Gated DeltaNet (Default)",
        "wandb_run_id": "fla_models/runs/abi7ojxu",
    },
    "Gated_DeltaNet_Comb_ST_seq_len_2K": {
        "display_name": "Gated DeltaNet (Seq Len 2K)",
        "wandb_run_id": "fla_models/runs/uah7zywj",
    },
    "Gated_DeltaNet_Comb_ST_seq_len_10K": {
        "display_name": "Gated DeltaNet (Seq Len 10K)",
        "wandb_run_id": "fla_models/runs/9elhe2fw",
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
        "display_name": "Mamba2 Combined Multi Target",
        "wandb_run_id": "fla_models/runs/ku412muw",
    },
    "Mamba2_Comb_ST": {
        "display_name": "Mamba2 Combined Single Target",
        "wandb_run_id": "fla_models/runs/arzdn9rh",
    },
    "Mamba2_Int_ST": {
        "display_name": "Mamba2 Interleaved Single Target",
        "wandb_run_id": "fla_models/runs/cdyctzjo",
    },
    "Mamba2_Int_MT": {
        "display_name": "Mamba2 Interleaved Multi Target",
        "wandb_run_id": "fla_models/runs/hvmrqqbi",
    },
}

MESANET_MODELS: dict[str, dict[str, Any]] = {
    "MesaNet_Int_MT": {
        "display_name": "MesaNet Interleaved Multi Target",
        "wandb_run_id": "fla_models/runs/bd9rtoku",
    }
}

LINEAR_ATTENTION_MODELS: dict[str, dict[str, Any]] = {
    "Linear_Attention_Non_Causal": {
      "wandb_run_id": "linear_attention/runs/83hs69fa", # new default implementation
      "display_name": "Linear Attention\n(Non-Causal)",
    },
    "Linear_Attention_Non_Causal_updated": {  # Beats Non-Causal version from before
      "wandb_run_id": "linear_attention/runs/tn6h2qyb", # new default implementation
      "display_name": "Linear Attention\n(Non-Causal updated)",
    },
    # "Linear_Attention_Non_Causal_with_k_sum_norm": { # worse/equal than without k_sum_norm
    #   "wandb_run_id": "linear_attention/runs/nnon9bb8", # new default implementation
    #   "display_name": "Linear Attention\n(Non-Causal w. k-sum Norm)",
    # },
    "Linear_Attention_Non_Causal_fro_norm": {
      "wandb_run_id": "linear_attention/runs/i960z4r7", # new default implementation
      "display_name": "Linear Attention\n(Non-Causal) w. Fro Norm",
    },
    # ------linear and element_wise product feature map are worse than the elu one -----
    # "Linear_Attention_Non_Causal_feat_map_elem_product": {
    #   "wandb_run_id": "linear_attention/runs/02rush9s", # new default implementation
    #   "display_name": "Linear Attention\n(Non-Causal) w. Feature Map Element-wise Product",
    # },
    # "Linear_Attention_Non_Causal_feat_map_linear": {
    #   "wandb_run_id": "linear_attention/runs/a4rrhp8v", # new default implementation
    #   "display_name": "Linear Attention\n(Non-Causal) w. Feature Map Linear",
    # },
    # ----------- Causal Models -----------
    "Linear_Attention_Causal_fro_norm_from_non_causal": {
      "wandb_run_id": "linear_attention/runs/i960z4r7", # new default implementation
      "display_name": "Linear Attention\n(Causal) w. Fro Norm From Non-Causal",
      "make_causal": True,
    },
    "Linear_Attention_Non_Causal_fro_norm_from_causal": {
      "wandb_run_id": "linear_attention/runs/rrakg728", # new default implementation
      "display_name": "Linear Attention\n(Non-Causal) w. Fro Norm From Causal",
      "make_non_causal": True,
    },
    "Linear_Attention_Comb_ST": {
      "wandb_run_id": "linear_attention/runs/3jq88aqt", # new default implementation
      "display_name": "Linear Attention\n(Comb_ST)",
    },
    "Linear_Attention_Comb_ST_updated": {
      "wandb_run_id": "linear_attention/runs/mwelekke", # new default implementation
      "display_name": "Linear Attention\n(Comb_ST updated)",
    },
    "Linear_Attention_Comb_ST_fro_norm": {
      "wandb_run_id": "linear_attention/runs/rrakg728", # new default implementation
      "display_name": "Linear Attention\n(Comb_ST) w. Fro Norm",
    },
    "Linear_Attention_FLA_Comb_ST": { 
        "wandb_run_id": "icl_arch/fla_models/hqzpuaso",
        "display_name": "Linear Attention (FLA; Comb ST)",
    },
    "Linear_Attention_FLA_Comb_ST_wo_self_term": { # slightly better ce and roc, and less degradation at long seq lens
        "wandb_run_id": "icl_arch/fla_models/o404kcbf",
        "display_name": "Linear Attention (FLA; Comb ST) w/o Self-Term",
    },
    "Linear_Attention_Comb_ST_old_setup": { # slightly worse acc and ce but slightly better roc auc then new default
        "wandb_run_id": "linear_attention/runs/7kig0n7a", # with layernorm, no output norm an k_sum_normalization
        "display_name": "Linear Attention (Comb ST old setup)",
    },
}

BASED_MODELS: dict[str, dict[str, Any]] = {
    "Rebased_feat_dim_80": {
        "display_name": "Rebased $\\phi$ with 80-dim features",
        "wandb_run_id": "fla_models/runs/2s9ngyny"
    },
    "Rebased_feat_dim_32": {
        "display_name": "Rebased $\\phi$ with 32-dim features", 
        "wandb_run_id": "fla_models/runs/on6k2nl3"
    },
    "Based_feat_dim_32": {
        "display_name": "Based $\\phi$ with 32-dim features",
        "wandb_run_id": "fla_models/runs/wnyl21ly"
    },
}

DELTANET_HIGH_SEQ_LEN_MODELS: dict[str, dict[str, Any]] = {
    "DeltaNet_Int_MT_Increasing_seq_1K->32K": {
        "wandb_run_id": "fla_models/runs/vo5mkuwt",
    },
    "DeltaNet_Comb_ST_Increasing_seq_1K->32K": {
        "wandb_run_id": "fla_models/runs/58w3kifz",
    },
    "DeltaNet_Int_MT_Seq_Len_500-64K_uniform": {
        "wandb_run_id": "fla_models/runs/tou1nzi5",
    },
    "DeltaNet_Int_MT_Seq_Len_500-64K_loguniform": {
        "wandb_run_id": "fla_models/runs/pyfldrsm",
    },
    "DeltaNet_Comb_ST_Seq_Len_200-64K_loguniform": {
        "wandb_run_id": "fla_models/runs/9llxebf9",
    },
    "DeltaNet_Comb_ST_Seq_Len_200-100K_loguniform": {
        "wandb_run_id": "fla_models/runs/oh6n51z3",
    },
    "DeltaNet_Int_MT_Seq_Len_1K": {
        "wandb_run_id": "fla_models/runs/ji6lw9hu",
    },
}

DELTANET_ADDED_REGULARIZATION: dict[str, dict[str, Any]] = {
    "DeltaNet_Comb_ST_reg_1e-6": {
        "wandb_run_id": "fla_models/runs/f6ynrp4l",
    },
    "DeltaNet_Comb_ST_reg_1e-5": {
        "wandb_run_id": "fla_models/runs/bfr8qhfh",
    },
    "DeltaNet_Comb_ST_reg_1e-4": {
        "wandb_run_id": "fla_models/runs/lzodfrv5",
    },
    "DeltaNet_Comb_ST_Reference": {
        "wandb_run_id": "fla_models/runs/ob2m9rth",
    }
}

DELTANET_FINETUNED_MODELS: dict[str, dict[str, Any]] = {
    "DeltaNet_Comb_ST_Finetuned_64K_1_e-5_new": {
        "wandb_run_id": "icl_arch/fla_models/leaywm94",
    },
    "DeltaNet_Comb_ST_Finetuned_64K_5_e-6_new": {
        "wandb_run_id": "icl_arch/fla_models/zmvzjsep",
    },
    "DeltaNet_Comb_ST_Reference": {
        "display_name": "DeltaNet Reference",
        "wandb_run_id": "fla_models/runs/ob2m9rth",
    },
}

EQUAL_PARAMS_MODELS: dict[str, dict[str, Any]] = {
    "equal_params:Transformer_Comb_ST": { # non-causal version
        "display_name": "Non-Causal Transformer",
        "eval_autocast_dtype": "fp16",
        "wandb_run_id": "tabpfn_transformer/runs/nb5hz44b",
    },
    "equal_params:Rebased_Comb_ST": {
        "display_name": "Non-Causal Linear\nAttention (Rebased $\\phi$)",
        "wandb_run_id": "fla_models/runs/7z1vh7vl", 
    },
    "equal_params:Linear_Attention_Comb_ST": {
        "display_name": "Non-Causal Linear\nAttention",
        "wandb_run_id": "linear_attention/runs/0j5sy87c",
    },
    # "Linear_Attention_FLA_Comb_ST": { 
    #     "wandb_run_id": "icl_arch/fla_models/f4rsksje",
    #     "display_name": "Causal Linear\nAttention (Comb ST)",
    # },
    "equal_params:DeltaNet_Comb_ST": {
        "display_name": "DeltaNet",
        "wandb_run_id": "fla_models/runs/ob2m9rth",
    },
    "equal_params:GLA_Comb_ST": {
        "display_name": "Gated Linear\nAttention",
        "wandb_run_id": "fla_models/runs/4vsqz1ee",
    },
    "equal_params:Gated_DeltaNet_Comb_ST": {
        "display_name": "Gated DeltaNet",
        "wandb_run_id": "fla_models/runs/g7rh5nv9",  
    },
    # "equal_params:DeltaNet_Int_MT": {
    #     "display_name": "DeltaNet (Int MT)",
    #     "wandb_run_id": "fla_models/runs/v18qqmbk",  # second run 2m9zukic
    # },
    # "equal_params:Gated_DeltaNet_Int_MT": {
    #     "display_name": "Gated DeltaNet (Int MT)",
    #     "wandb_run_id": "fla_models/runs/cpcq82tx", # second run 2cm1gdi5
    # },
    "equal_params:KDA_Comb_ST": {
        "display_name": "Kimi Delta\nAttention",
        "wandb_run_id": "fla_models/runs/5jfgan9d", # old run qaskm2mq
    },
    # "equal_params:Mamba2_Comb_ST": {
    #     "display_name": "Mamba2",
    #     "wandb_run_id": "fla_models/runs/o9e00w17",
    # },
}

TRANSFORMER_MASKED_MODELS: dict[str, dict[str, Any]] = {
    "Transformer_Non_Causal": {
        "display_name": "Non-Causal", #"Non-Causal (Default)",
        "wandb_run_id": "tabpfn_transformer_masking_experiments/runs/f1lg4ch9", # fp16 version d4mttnjl, fp 32 version pmcn4brd, old 15.2M params version
        "eval_mode": "forward",
        "eval_autocast_dtype": "fp16",
    },
    # "Transformer_Non_Causal_with_RoPE_pairwise": {
    #     "wandb_run_id": "tabpfn_transformer_masking_experiments/runs/xsbe5y6d", # old runs: xsbe5y6d, second run with fp32 as comparison: 0xi6dcvc
    #     "eval_mode": "forward",
    #     "eval_autocast_dtype": "fp16",
    # },
    # "Transformer_Non_Causal_interleaved_with_RoPE_pairwise": {
    #     "display_name": "Non-Causal Interleaved",
    #     "wandb_run_id": "tabpfn_transformer_masking_experiments/runs/6kid4bgi",   # new one uses pairwise rope while old one does not jzs97xfg
    #     "eval_mode": "forward",
    #     "eval_autocast_dtype": "fp16",
    # },
    "masked:Transformer_Comb_ST": {
        "display_name": "Causal Single\nTarget",
        "wandb_run_id": "tabpfn_transformer_masking_experiments/runs/gex7h68b", # fp 16 version 2wrxsh60, old 15.2M params version b56ohkmz
        "eval_mode": "forward",
        "eval_autocast_dtype": "fp16",
    },
    # "Transformer_Test_To_Train_Only": {
    #     "wandb_run_id": "tabpfn_transformer_masking_experiments/runs/1agq90eo",
    #     "eval_mode": "forward",
    #     "eval_autocast_dtype": "fp16",
    # },
    "Transformer_Comb_MT": {
        "display_name": "Causal Multi\nTarget",
        "wandb_run_id": "tabpfn_transformer_masking_experiments/runs/81g04qla",
        "eval_mode": "forward",
        "eval_autocast_dtype": "fp16",
    },
    # "Transformer_Int_ST_with_RoPE_pairwise": { 
    #     "display_name": "Causal Interleaved\nSingle Target",
    #     "wandb_run_id": "tabpfn_transformer_masking_experiments/runs/z36s69e0",  # new one uses pairwise rope while old one does not 7yzlf15p
    #     "eval_mode": "forward",
    #     "eval_autocast_dtype": "fp16",
    # },
    # "Transformer_Int_MT_with_RoPE_pairwise": { 
    #     "display_name": "Causal Interleaved\nMulti Target",
    #     "wandb_run_id": "tabpfn_transformer_masking_experiments/runs/xiv7f2z3", # old model without pairwise rope m74u7psh
    #     "eval_mode": "forward",
    #     "eval_autocast_dtype": "fp16",
    # },
}

STATE_PASSING_MODELS: dict[str, dict[str, Any]] = {
    "State_Passing_GLA_Comb_ST": {
        "display_name": "GLA Combined Single\nTarget with State Passing",
        "wandb_run_id": "fla_models/runs/66hynh1d"
    },
    "GLA_Comb_ST": {
        "display_name": "GLA Combined\nSingle Target",
        "wandb_run_id": "fla_models/runs/g1ul5lyc",
    },
}

SUBSAMPLED_MODELS: dict[str, dict[str, Any]] = {
    "subsampled:DeltaNet_Comb_ST": {
        "display_name": "DeltaNet Comb ST",
        "wandb_run_id": "fla_models/runs/ob2m9rth",
    },
    "subsampled:DeltaNet_Comb_ST_1K": {
        "display_name": "DeltaNet Comb ST (Subsampled 1K)",
        "wandb_run_id": "fla_models/runs/ob2m9rth",
        "subsample_dataset_size": 1_000,
    },
    "subsampled:DeltaNet_Comb_ST_3K": {
        "display_name": "DeltaNet Comb ST\n(Subsampled 3K)",
        "wandb_run_id": "fla_models/runs/ob2m9rth",
        "subsample_dataset_size": 3_000
    },
    "subsampled:DeltaNet_Comb_ST_10K": {
        "display_name": "DeltaNet Comb ST (Subsampled 10K)",
        "wandb_run_id": "fla_models/runs/ob2m9rth",
        "subsample_dataset_size": 10_000
    },
    "subsampled:Transformer_Comb_ST_1K": {
        "display_name": "Transformer Comb ST (Subsampled 1K)",
        "wandb_run_id": "tabpfn_transformer/runs/nb5hz44b",
        "subsample_dataset_size": 1_000,
        "eval_autocast_dtype": "fp16",
    },
    "subsampled:Transformer_Comb_ST_3K": {
        "display_name": "Transformer Comb ST\n(Subsampled 3K)",
        "wandb_run_id": "tabpfn_transformer/runs/nb5hz44b",
        "subsample_dataset_size": 3_000,
        "eval_autocast_dtype": "fp16",
    },
    "subsampled:Transformer_Comb_ST_10K": {
        "display_name": "Transformer Comb ST (Subsampled 10K)",
        "wandb_run_id": "tabpfn_transformer/runs/nb5hz44b",
        "subsample_dataset_size": 10_000,
        "eval_autocast_dtype": "fp16",
    },
}

MIMETIC_INITIALIZATION_MODELS: dict[str, dict[str, Any]] = {
    # "mimetic:Linear_Attention_FLA_Comb_ST": { 
    #     "display_name": "Linear Attention\nComb ST (Reference)",
    #     "wandb_run_id": "icl_arch/fla_models/f4rsksje",
    # },
    "mimetic:GLA_Comb_ST_Ref": {
        "display_name": "GLA Comb ST (Reference)",
        "wandb_run_id": "fla_models/runs/2v2xw7d2",
    },
    "mimetic:GLA_Comb_ST_mimetic_full_wo_output_gate_init": {
        "display_name": "GLA Comb ST (Full Mimetic wo. output gate init)",
        "wandb_run_id": "fla_models/runs/brps0byt",
    },
    "mimetic:GLA_Comb_ST_mimetic_full": {
        "display_name": "GLA Comb ST (Full Mimetic)",
        "wandb_run_id": "fla_models/runs/dthrura3",
    },
    "mimetic:GLA_Comb_ST_mimetic_gate_only": {
        "display_name": "GLA Comb ST (Mimetic gate only)",
        "wandb_run_id": "fla_models/runs/l9lcdj1f",
    },
    
    "mimetic:DeltaNet_Comb_ST": {
        "display_name": "DeltaNet\nComb ST (Reference)",
        "wandb_run_id": "fla_models/runs/ob2m9rth",
    },
    "mimetic:Gated_DeltaNet_Comb_ST": {
        "display_name": "Gated DeltaNet\nComb ST (Reference)",
        "wandb_run_id": "fla_models/runs/g7rh5nv9",  
    },
    "mimetic:Gated_DeltaNet_Comb_ST_mimetic_full_wo_output_gate_init": {
        "display_name": "Gated DeltaNet\nComb ST (Full Mimetic wo. output gate init)",
        "wandb_run_id": "fla_models/runs/g7rh5nv9",  
    },
    "mimetic:Gated_DeltaNet_Comb_ST_mimetic_full": {
        "display_name": "Gated DeltaNet\nComb ST (Full Mimetic)",
        "wandb_run_id": "fla_models/runs/p8uli0zu",  
    },
     "mimetic:Gated_DeltaNet_Comb_ST_mimetic_gate_only": {
        "display_name": "Gated DeltaNet\nComb ST (Mimetic gate only)",
        "wandb_run_id": "fla_models/runs/xvjvp179",  
    },
}

BIDIRECTIONAL_MODELS: dict[str, dict[str, Any]] = {
    "Bidirectional_DeltaNet_Comb_ST_linear_output_two_cache": {
        "display_name": "Bidirectional Linear Output Two Cache (DeltaNet)",
        "wandb_run_id": "icl_arch/fla_models/5htl7smo",
    },
    "Bidirectional_DeltaNet_Comb_ST_mean_output_mean_cache": {
        "display_name": "Bidirectional Mean Output Mean Cache (DeltaNet)",
        "wandb_run_id": "icl_arch/fla_models/vn8w3gjo",
    },
    "Bidirectional_DeltaNet_Comb_ST_mean_output_two_cache": {
        "display_name": "Bidirectional Mean Output Two Cache (DeltaNet)",
        "wandb_run_id": "icl_arch/fla_models/hv2uv8nq",
    },
    "Bidirectional_DeltaNet_Comb_ST_mean_output_two_cache_separate_weights": {
        "display_name": "Bidirectional Mean Output Two Cache Separate Weights (DeltaNet)",
        "wandb_run_id": "icl_arch/fla_models/6lby3bdm",
    },
    "Bidirectional_Linear_Attention_Comb_ST_mean_output_mean_cache": { # todo retrain as linear attention conf is outdated
        "display_name": "Bidirectional Mean Output Mean Cache (Linear Attention)",
        "wandb_run_id": "icl_arch/fla_models/3j9jgdvx",
    },
    "Bidirectional_GLA_Comb_ST_mean_output_mean_cache": {
        "display_name": "Bidirectional Mean Output Mean Cache (GLA)",
        "wandb_run_id": "icl_arch/fla_models/iw22mtux",
    },
    "DeltaNet_Comb_ST_Reference_New": {
        "display_name": "DeltaNet Comb ST (Reference)",
        "wandb_run_id": "fla_models/runs/tuj1kct1",
    },
}

ORACLE_HIDDEN_STATE_MODELS: dict[str, dict[str, Any]] = {
    "Oracle_Hidden_State_GLA_Comb_ST": {
        **GLA_MODELS["GLA_Comb_ST"],
        "display_name": "Oracle Hidden State (GLA) New Base",
        "oracle_hidden_state_baseline": True,
        "oracle_num_epochs": 400,
        "oracle_lr": 4e-1, # set lr by checking with oracle_verbose the relative state update sizes to be around 1e-2-1e-1
        "oracle_weight_decay": 1e-5,
        "oracle_patience": 20,
        "oracle_query_batch_size": 4000,
        "oracle_selection_fraction": 0.1,
        "oracle_evaluate_only_max_seqlen": True,
        "oracle_verbose": False,
        "eval_autocast_dtype": "bf16",
    },
    # "Oracle_Hidden_State_DeltaNet_Comb_ST_state_init": {
    #     **DELTANET_MODELS["DeltaNet_Comb_ST"],
    #     "display_name": "Oracle Hidden State (DeltaNet) New Base with State Init",
    #     "oracle_hidden_state_baseline": True,
    #     "oracle_num_epochs": 400,
    #     "oracle_lr": 3e-3,
    #     "oracle_weight_decay": 1e-5,
    #     "oracle_patience": 20,
    #     "oracle_query_batch_size": 4000,
    #     "oracle_selection_fraction": 0.1,
    #     "oracle_evaluate_only_max_seqlen": True,
    #     "oracle_random_init_hidden_state": True,
    #     "oracle_verbose": False,
    #     "eval_autocast_dtype": "bf16",
    # },
    # "Oracle_Hidden_State_DeltaNet_Comb_ST_state_init_v2": {
    #     **DELTANET_MODELS["DeltaNet_Comb_ST"],
    #     "display_name": "Oracle Hidden State (DeltaNet) New Base with State Init",
    #     "oracle_hidden_state_baseline": True,
    #     "oracle_num_epochs": 400,
    #     "oracle_lr": 5e-3,
    #     "oracle_weight_decay": 1e-5,
    #     "oracle_patience": 40,
    #     "oracle_query_batch_fraction": 0.04,
    #     "oracle_selection_fraction": 0.1,
    #     "oracle_evaluate_only_max_seqlen": True,
    #     "oracle_random_init_hidden_state": True,
    #     "oracle_verbose": False,
    #     "eval_autocast_dtype": "bf16",
    # },
    "Oracle_Hidden_State_DeltaNet_Comb_ST": {
        **DELTANET_MODELS["DeltaNet_Comb_ST"],
        "display_name": "Oracle Hidden State\n(DeltaNet)",
        "oracle_hidden_state_baseline": True,
        "oracle_num_epochs": 400,
        "oracle_lr": 3e-3,
        "oracle_weight_decay": 1e-5,
        "oracle_patience": 20,
        "oracle_query_batch_size": 4000, # increasing or decreasing batch size hurt at seq len 128k
        #"oracle_query_batch_fraction": 0.04,
        "oracle_selection_fraction": 0.1,
        "oracle_evaluate_only_max_seqlen": True,
        "oracle_verbose": False,
        "eval_autocast_dtype": "bf16",
    },
    # "Oracle_Hidden_State_Linear_Attention_Non_Causal": {
    #     **LINEAR_ATTENTION_MODELS["Linear_Attention_Non_Causal"],
    #     "display_name": "Oracle Hidden State (Linear Attention)",
    #     "oracle_hidden_state_baseline": True,
    #     "oracle_num_epochs": 400,
    #     "oracle_lr": 3e1, # set lr by checking with oracle_verbose the relative state update sizes to be around 1e-2-1e-1
    #     "oracle_auto_scale_lr": False,
    #     "oracle_weight_decay": 1e-5,
    #     "oracle_patience": 20,
    #     "oracle_query_batch_size": 4000,
    #     "oracle_selection_fraction": 0.1,
    #     "oracle_evaluate_only_max_seqlen": False,
    #     "oracle_verbose": False,
    #     "oracle_verbose_every_n_epochs": 10,
    #     "display_name": "Oracle Hidden State Optimization (Non-CausalLinear Attention)",
    #     "note": "with new linear attention backbone"
    # },
    # "Oracle_Hidden_State_Rebased_feat_dim_32_base": {
    #     **BASED_MODELS["Rebased_feat_dim_32"],
    #     "display_name": "Oracle Hidden State (Rebased)",
    #     "oracle_hidden_state_baseline": True,
    #     "oracle_num_epochs": 400,
    #     "oracle_lr": 5e1, 
    #     "oracle_weight_decay": 1e-5,
    #     "oracle_patience": 20,
    #     "oracle_query_batch_size": 4000,
    #     "oracle_selection_fraction": 0.1,
    #     "oracle_evaluate_only_max_seqlen": True,
    #     "oracle_verbose": True,
    # },
}

OTHER_MODELS: dict[str, dict[str, Any]] = {}

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
    "kda": KDA_MODELS,
    "gla": GLA_MODELS,
    "deltanet": DELTANET_MODELS,
    "oracle_hidden_state": ORACLE_HIDDEN_STATE_MODELS,
    "deltanet_size_changes": DELTANET_MODELS_SIZE_CHANGES,
    "gated_deltanet": GATED_DELTANET_MODELS,
    "gated_deltanet_seq_len_changes": GATED_DELTANET_MODELS_SEQ_LEN_CHANGES,
    "mamba2": MAMBA2_MODELS,
    "mesanet": MESANET_MODELS,
    "linear_attention": LINEAR_ATTENTION_MODELS,
    "based": BASED_MODELS,
    "equal_params": EQUAL_PARAMS_MODELS,
    "transformer_masked": TRANSFORMER_MASKED_MODELS,
    "deltanet_high_seq_len": DELTANET_HIGH_SEQ_LEN_MODELS,
    "deltanet_added_regularization": DELTANET_ADDED_REGULARIZATION,
    "deltanet_finetuned": DELTANET_FINETUNED_MODELS,
    "mimetic_initialization": MIMETIC_INITIALIZATION_MODELS,
    "subsampled": SUBSAMPLED_MODELS,
    "bidirectional": BIDIRECTIONAL_MODELS,
    "state_passing": STATE_PASSING_MODELS,
    "fla_models": {
        **KDA_MODELS,
        **GLA_MODELS,
        **DELTANET_MODELS,
        **GATED_DELTANET_MODELS,
        **MAMBA2_MODELS,
        **MESANET_MODELS,
        #**GATED_DELTANET_MODELS_SEQ_LEN_CHANGES,
        #**DELTANET_MODELS_SIZE_CHANGES,
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


def is_oracle_model(model_name: str, model_config: dict[str, Any]) -> bool:
    return (
        bool(model_config.get("oracle_hidden_state_baseline"))
        or model_name in ORACLE_HIDDEN_STATE_MODELS
    )


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
