from __future__ import annotations

import torch


_ROPE_UNFROZEN_MASK_NAMES = frozenset({"Comb_MT", "Int_MT", "causal_all"})


def build_rope_inv_freq(
    d_k: int,
    *,
    rope_base: float,
    device: torch.device | None = None,
) -> torch.Tensor:
    if d_k % 2 != 0:
        raise ValueError(f"RoPE requires even hidden dimension, got {d_k}.")
    exponents = torch.arange(0, d_k, 2, device=device, dtype=torch.float32) / d_k
    return 1.0 / (rope_base ** exponents)


def build_rope_positions(
    seq_len: int,
    *,
    device: torch.device,
    position_offset: int = 0,
    rope_pairwise_positions: bool = False,
    mask_name: str | None = None,
    eval_pos: int | None = None,
    use_cached_kv: bool = False,
    is_training: bool = True,
) -> torch.Tensor:
    positions = torch.arange(position_offset, position_offset + seq_len, device=device)
    allow_unfrozen_test_positions = (
        is_training
        and mask_name in _ROPE_UNFROZEN_MASK_NAMES
        and not use_cached_kv
    )
    if not allow_unfrozen_test_positions:
        freeze_after = None
        if eval_pos is not None and eval_pos > 0:
            freeze_after = int(eval_pos)
        if freeze_after is None and use_cached_kv:
            freeze_after = int(position_offset) if position_offset > 0 else None
        if freeze_after is not None:
            positions = positions.clamp_max(freeze_after)
    if rope_pairwise_positions:
        positions = torch.div(positions, 2, rounding_mode="floor")
    return positions


def apply_rope(
    x: torch.Tensor,
    *,
    inv_freq: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    if x.shape[-1] % 2 != 0:
        raise ValueError(
            f"RoPE requires even hidden dimension, got {x.shape[-1]}."
        )
    seq_len = x.shape[-3]
    batch_size = x.shape[0]

    if positions.dim() == 1:
        positions = positions[None, :]
    if positions.dim() != 2:
        raise ValueError(
            f"Expected positions to have shape [seq] or [batch, seq], got {tuple(positions.shape)}."
        )
    if positions.shape[1] != seq_len:
        raise ValueError(
            f"Expected positions length {seq_len}, got {positions.shape[1]}."
        )
    if positions.shape[0] == 1:
        positions = positions.expand(batch_size, -1)
    elif positions.shape[0] != batch_size:
        raise ValueError(
            f"Expected positions batch {batch_size}, got {positions.shape[0]}."
        )

    freqs = positions.to(dtype=inv_freq.dtype).unsqueeze(-1) * inv_freq
    cos = freqs.cos().to(dtype=x.dtype).unsqueeze(2)
    sin = freqs.sin().to(dtype=x.dtype).unsqueeze(2)

    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    return torch.stack(
        (
            x_even * cos - x_odd * sin,
            x_even * sin + x_odd * cos,
        ),
        dim=-1,
    ).flatten(-2)
