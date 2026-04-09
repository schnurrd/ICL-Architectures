from __future__ import annotations

import math
from typing import Literal

import torch

InterleavedPairPositionalEmbedding = Literal[
    "none",
    "sinusoidal",
    "shared_random_pair",
]


_UNFROZEN_TEST_POSITION_MASK_NAMES = frozenset({"Comb_MT", "Int_MT", "causal_all"})


def build_position_indices(
    seq_len: int,
    *,
    device: torch.device,
    position_offset: int = 0,
    pairwise_positions: bool = False,
    mask_name: str | None = None,
    eval_pos: int | None = None,
    use_cached_kv: bool = False,
    is_training: bool = True,
) -> torch.Tensor:
    positions = torch.arange(position_offset, position_offset + seq_len, device=device)
    allow_unfrozen_test_positions = (
        is_training
        and mask_name in _UNFROZEN_TEST_POSITION_MASK_NAMES
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
    if pairwise_positions:
        positions = torch.div(positions, 2, rounding_mode="floor")
    return positions


def build_sinusoidal_position_embeddings(
    positions: torch.Tensor,
    *,
    dim: int,
    dtype: torch.dtype,
    base: float = 128_000.0,
) -> torch.Tensor:
    if positions.ndim != 1:
        raise ValueError(
            f"Expected 1D positions tensor, got shape {tuple(positions.shape)}."
        )
    if positions.numel() == 0:
        return torch.empty(0, dim, device=positions.device, dtype=dtype)
    if base <= 0:
        raise ValueError(f"Expected positive sinusoidal base, got {base}.")

    half_dim = (dim + 1) // 2
    freq_indices = torch.arange(half_dim, device=positions.device, dtype=torch.float32)
    div_term = torch.exp(
        -torch.log(torch.tensor(base, device=positions.device))
        * freq_indices
        / max(half_dim - 1, 1)
    )
    angles = positions.to(torch.float32).unsqueeze(1) * div_term.unsqueeze(0)
    embeddings = torch.zeros(
        positions.numel(),
        dim,
        device=positions.device,
        dtype=torch.float32,
    )
    embeddings[:, 0::2] = torch.sin(angles)
    embeddings[:, 1::2] = torch.cos(angles[:, : embeddings[:, 1::2].shape[1]])
    return embeddings.to(dtype=dtype)


def build_shared_random_pair_embeddings(
    positions: torch.Tensor,
    *,
    dim: int,
    dtype: torch.dtype,
    seed: int,
) -> torch.Tensor:
    if positions.ndim != 1:
        raise ValueError(
            f"Expected 1D positions tensor, got shape {tuple(positions.shape)}."
        )
    if positions.numel() == 0:
        return torch.empty(0, dim, device=positions.device, dtype=dtype)

    max_position = int(positions.max().item()) + 1
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    pair_embeddings = torch.randn(
        max_position,
        dim,
        generator=generator,
        device="cpu",
        dtype=torch.float32,
    )
    pair_embeddings = pair_embeddings / pair_embeddings.norm(
        dim=-1,
        keepdim=True,
    ).clamp_min(1e-12)
    return pair_embeddings.index_select(
        0,
        positions.to(device="cpu", dtype=torch.long),
    ).to(device=positions.device, dtype=dtype)


def standardize_embedding_signal(
    embeddings: torch.Tensor,
    *,
    feature_dim: int,
    scale: float | torch.Tensor = 1.0,
) -> torch.Tensor:
    embeddings = embeddings - embeddings.mean(dim=-1, keepdim=True)
    embeddings = embeddings / math.sqrt(feature_dim)
    return embeddings * scale


def apply_interleaved_pair_and_role_embeddings(
    embedded_x: torch.Tensor,
    embedded_y: torch.Tensor,
    *,
    embedding_type: InterleavedPairPositionalEmbedding,
    role_embeddings: torch.Tensor,
    position_offset: int,
    eval_pos: int | None,
    use_cached_positions: bool,
    mask_name: str,
    is_training: bool,
    seed: int = 0,
    position_base: float = 128_000.0,
    pair_position_scale: float | torch.Tensor = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if role_embeddings.shape != (2, embedded_x.shape[-1]):
        raise ValueError(
            "Expected role_embeddings to have shape "
            f"(2, {embedded_x.shape[-1]}), got {tuple(role_embeddings.shape)}."
        )

    pair_positions = build_position_indices(
        embedded_x.shape[1],
        device=embedded_x.device,
        position_offset=position_offset,
        pairwise_positions=False,
        mask_name=mask_name,
        eval_pos=eval_pos,
        use_cached_kv=use_cached_positions,
        is_training=is_training,
    )
    if embedding_type == "sinusoidal":
        pair_position_embeddings = build_sinusoidal_position_embeddings(
            pair_positions,
            dim=embedded_x.shape[-1],
            dtype=embedded_x.dtype,
            base=position_base,
        )
    elif embedding_type == "shared_random_pair":
        pair_position_embeddings = build_shared_random_pair_embeddings(
            pair_positions,
            dim=embedded_x.shape[-1],
            dtype=embedded_x.dtype,
            seed=seed,
        )
    else:
        raise ValueError(
            "embedding_type must be one of "
            f"{{'sinusoidal', 'shared_random_pair'}}, got {embedding_type!r}."
        )
    pair_position_embeddings = standardize_embedding_signal(
        pair_position_embeddings,
        feature_dim=embedded_x.shape[-1],
        scale=pair_position_scale,
    )
    pair_position_embeddings_x = pair_position_embeddings.view(1, -1, 1, embedded_x.shape[-1])
    pair_position_embeddings_y = pair_position_embeddings.view(1, -1, embedded_y.shape[-1])
    role_embeddings = standardize_embedding_signal(
        role_embeddings,
        feature_dim=embedded_x.shape[-1],
    )
    x_role_embedding = role_embeddings[0].view(1, 1, 1, -1)
    y_role_embedding = role_embeddings[1].view(1, 1, -1)
    return (
        embedded_x + pair_position_embeddings_x + x_role_embedding,
        embedded_y + pair_position_embeddings_y + y_role_embedding,
    )


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
    return build_position_indices(
        seq_len,
        device=device,
        position_offset=position_offset,
        pairwise_positions=rope_pairwise_positions,
        mask_name=mask_name,
        eval_pos=eval_pos,
        use_cached_kv=use_cached_kv,
        is_training=is_training,
    )


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
