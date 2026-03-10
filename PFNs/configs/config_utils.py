from __future__ import annotations

from collections.abc import Sequence

import torch


def resolve_eval_pos_split_pct(
    eval_pos_split_pct: float | tuple[float, float] | list[float] | None,
) -> tuple[float | None, float | None]:
    if eval_pos_split_pct is None:
        return None, None
    if isinstance(eval_pos_split_pct, (int, float)):
        value = float(eval_pos_split_pct)
        return value, value
    if (
        isinstance(eval_pos_split_pct, (list, tuple))
        and len(eval_pos_split_pct) == 2
    ):
        return float(eval_pos_split_pct[0]), float(eval_pos_split_pct[1])
    raise ValueError(
        "eval_pos_split_pct must be a number (fixed percent) or a pair "
        "(min_percent, max_percent)."
    )


def normalize_optional_none_string(value: str | None) -> str | None:
    if value == "None":
        return None
    return value


def resolve_batch_size_stages(
    batch_size_stages: Sequence[tuple[int, int]] | None,
) -> list[tuple[int, int]] | None:
    if batch_size_stages is None:
        return None
    return [
        (int(seq_len_threshold), int(stage_batch_size))
        for seq_len_threshold, stage_batch_size in batch_size_stages
    ]


def resolve_prior_device(
    *,
    max_seq_len: int,
    cuda_seq_len_threshold: int = 2000,
) -> str:
    # if torch.cuda.is_available() and max_seq_len > int(cuda_seq_len_threshold):
    #     return "cuda"
    # return "cpu"
    return "cpu" # current tests with the updated prior suggest cpu is even better for long sequences (tested up to 32K)

