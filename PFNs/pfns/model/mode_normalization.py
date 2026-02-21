from __future__ import annotations

CANONICAL_SEQUENCE_MODES = ("Comb_ST", "Int_ST", "Comb_MT", "Int_MT")

SEQUENCE_MODE_ALIASES = {
    **{mode.lower(): mode for mode in CANONICAL_SEQUENCE_MODES},
    "cached": "Comb_ST",
    "cached_interleaved": "Int_ST",
    "causal": "Comb_MT",
    "causal_interleaved": "Int_MT",
    "teacher_forcing": "Int_MT",
}

ITEM_ATTENTION_MASK_MODE_ALIASES = {
    "causal_train_only": "Comb_ST",
    "causal_all": "Comb_MT",
    "comb_st": "Comb_ST",
    "int_st": "Int_ST",
    "comb_mt": "Comb_MT",
    "int_mt": "Int_MT",
}


def normalize_mode_name(mode: str) -> str:
    return mode.strip().lower().replace("-", "_").replace(" ", "_")


def resolve_sequence_mode(sequence_mode: str) -> str:
    normalized = normalize_mode_name(sequence_mode)
    canonical = SEQUENCE_MODE_ALIASES.get(normalized, normalized)
    if canonical not in CANONICAL_SEQUENCE_MODES:
        available = sorted({*CANONICAL_SEQUENCE_MODES, *SEQUENCE_MODE_ALIASES.keys()})
        raise ValueError(
            f"Unknown sequence_mode {sequence_mode!r}. Available: {available}"
        )
    return canonical


def resolve_item_attention_mask_mode(mask_mode: str | None) -> str | None:
    if not isinstance(mask_mode, str):
        return mask_mode
    normalized = normalize_mode_name(mask_mode)
    return ITEM_ATTENTION_MASK_MODE_ALIASES.get(normalized, mask_mode)
