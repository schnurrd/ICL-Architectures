from __future__ import annotations

import torch
from torch import nn


def build_activation(activation: str) -> nn.Module:
    """Return the requested pointwise activation."""
    if activation == "gelu":
        return nn.GELU()
    if activation == "relu":
        return nn.ReLU()
    if activation in {"swish", "silu"}:
        return nn.SiLU()
    raise ValueError(f"Unsupported activation: {activation}")


def build_mlp(
    d_model: int,
    dim_feedforward: int | None,
    dropout: float,
    activation: str,
) -> nn.Sequential:
    """Build MLP(x) = Dropout(W2 Dropout(act(W1 x)))."""
    if dim_feedforward is None:
        dim_feedforward = 4 * d_model
    act = build_activation(activation)
    return nn.Sequential(
        nn.Linear(d_model, dim_feedforward),
        act,
        nn.Dropout(dropout),
        nn.Linear(dim_feedforward, d_model),
        nn.Dropout(dropout),
    )


def clip_linear_attention_state_frobenius_norm(
    kv_state: torch.Tensor,
    k_sum: torch.Tensor,
    max_frobenius_norm: float | None,
    *,
    target: str = "joint",
    length_normalization: str = "none",
    state_length: int | float | torch.Tensor | None = None,
    min_length: int | float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Clip batched linear-attention states using one shared scale per state."""
    if max_frobenius_norm is None or kv_state.numel() == 0:
        return kv_state, k_sum

    if max_frobenius_norm <= 0.0:
        raise ValueError("max_frobenius_norm must be > 0.")
    if target not in {"joint", "kv_state", "k_sum", "kv_over_ksum_ratio"}:
        raise ValueError(
            "target must be one of {'joint', 'kv_state', 'k_sum', 'kv_over_ksum_ratio'}."
        )
    if length_normalization not in {"none", "sqrt_length", "length"}:
        raise ValueError(
            "length_normalization must be one of {'none', 'sqrt_length', 'length'}."
        )

    kv_norm_input = kv_state.float()
    k_norm_input = k_sum.float()
    kv_norm = kv_norm_input.square().sum(dim=(-2, -1)).sqrt()
    k_norm = k_norm_input.square().sum(dim=-1).sqrt()
    tiny = torch.finfo(kv_norm_input.dtype).tiny

    state_length_tensor = None
    if state_length is None or length_normalization == "none":
        length_scale = 1.0
    else:
        state_length_tensor = torch.as_tensor(
            state_length,
            device=kv_norm_input.device,
            dtype=kv_norm_input.dtype,
        ).clamp_min(1.0)
        length_scale = state_length_tensor
        if length_normalization == "sqrt_length":
            length_scale = length_scale.sqrt()

    clip_limit = float(max_frobenius_norm) * length_scale
    if min_length is not None and state_length is not None:
        if state_length_tensor is None:
            state_length_tensor = torch.as_tensor(
                state_length,
                device=kv_norm_input.device,
                dtype=kv_norm_input.dtype,
            )
        clip_limit = torch.where(
            state_length_tensor >= float(min_length),
            torch.as_tensor(clip_limit, device=kv_norm_input.device, dtype=kv_norm_input.dtype),
            torch.full_like(state_length_tensor, float("inf")),
        )
    if target == "joint":
        state_norm = (kv_norm.square() + k_norm.square()).sqrt()
        scale = torch.clamp(clip_limit / state_norm.clamp_min(tiny), max=1.0)
        return (
            kv_state * scale.to(kv_state.dtype).unsqueeze(-1).unsqueeze(-1),
            k_sum * scale.to(k_sum.dtype).unsqueeze(-1),
        )
    if target == "kv_state":
        scale = torch.clamp(clip_limit / kv_norm.clamp_min(tiny), max=1.0)
        return kv_state * scale.to(kv_state.dtype).unsqueeze(-1).unsqueeze(-1), k_sum
    if target == "k_sum":
        scale = torch.clamp(clip_limit / k_norm.clamp_min(tiny), max=1.0)
        return kv_state, k_sum * scale.to(k_sum.dtype).unsqueeze(-1)

    ratio = kv_norm / k_norm.clamp_min(tiny)
    scale = torch.clamp(clip_limit / ratio.clamp_min(tiny), max=1.0)
    return kv_state * scale.to(kv_state.dtype).unsqueeze(-1).unsqueeze(-1), k_sum


def clip_linear_attention_output_norm(
    attn: torch.Tensor,
    max_output_norm: float | None,
    *,
    length_normalization: str = "none",
    state_length: int | float | torch.Tensor | None = None,
    min_length: int | float | None = None,
) -> torch.Tensor:
    """Clip raw linear-attention outputs before output projection."""
    if max_output_norm is None or attn.numel() == 0:
        return attn

    if max_output_norm <= 0.0:
        raise ValueError("max_output_norm must be > 0.")
    if length_normalization not in {"none", "sqrt_length", "length"}:
        raise ValueError(
            "length_normalization must be one of {'none', 'sqrt_length', 'length'}."
        )

    attn_norm_input = attn.float()
    attn_norm = attn_norm_input.square().sum(dim=-1).sqrt()
    tiny = torch.finfo(attn_norm_input.dtype).tiny
    state_length_tensor = None
    if state_length is None or length_normalization == "none":
        length_scale = 1.0
    else:
        state_length_tensor = torch.as_tensor(
            state_length,
            device=attn_norm_input.device,
            dtype=attn_norm_input.dtype,
        ).clamp_min(1.0)
        length_scale = state_length_tensor
        if length_normalization == "sqrt_length":
            length_scale = length_scale.sqrt()

    clip_limit = float(max_output_norm) * length_scale
    if min_length is not None and state_length is not None:
        if state_length_tensor is None:
            state_length_tensor = torch.as_tensor(
                state_length,
                device=attn_norm_input.device,
                dtype=attn_norm_input.dtype,
            )
        clip_limit = torch.where(
            state_length_tensor >= float(min_length),
            torch.as_tensor(clip_limit, device=attn_norm_input.device, dtype=attn_norm_input.dtype),
            torch.full_like(state_length_tensor, float("inf")),
        )
    scale = torch.clamp(clip_limit / attn_norm.clamp_min(tiny), max=1.0)
    return attn * scale.to(attn.dtype).unsqueeze(-1)


def compute_kv_state_4d(
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute K_sum = sum_t k_t and KV = sum_t k_t v_t^T."""
    # k: (batch, seq, heads, qk_dim)
    # v: (batch, seq, heads, v_dim)
    k_sum = k.sum(dim=1)
    kv_state = torch.einsum("bshf,bshd->bhfd", k, v)
    return kv_state, k_sum


def apply_state_to_query_4d(
    q: torch.Tensor,
    kv_state: torch.Tensor,
    k_sum: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor:
    """Apply cached state.

    Base form:
        out = (q^T KV) / (q^T K_sum + eps)
    """
    # q: (batch, seq, heads, qk_dim)
    # kv_state: (batch, heads, qk_dim, v_dim)
    # k_sum: (batch, heads, qk_dim)
    num = torch.einsum("bshf,bhfd->bshd", q, kv_state)
    denom = torch.einsum("bshf,bhf->bsh", q, k_sum)
    return num / (denom.unsqueeze(-1) + eps)


def compute_kv_state_5d(
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute K_sum = sum_t k_t and KV = sum_t k_t v_t^T for per-feature states."""
    # k: (batch, seq, features, heads, qk_dim)
    # v: (batch, seq, features, heads, v_dim)
    k_sum = torch.einsum("bsnhf->bnhf", k)
    kv_state = torch.einsum("bsnhf,bsnhd->bnhfd", k, v)
    return kv_state, k_sum


def apply_state_to_query_5d(
    q: torch.Tensor,
    kv_state: torch.Tensor,
    k_sum: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor:
    """Apply cached per-feature state.

    Base form:
        out = (q^T KV) / (q^T K_sum + eps)
    """
    # q: (batch, seq, features, heads, qk_dim)
    # kv_state: (batch, features, heads, qk_dim, v_dim)
    # k_sum: (batch, features, heads, qk_dim)
    num = torch.einsum("bsnhf,bnhfd->bsnhd", q, kv_state)
    denom = torch.einsum("bsnhf,bnhf->bsnh", q, k_sum)
    return num / (denom.unsqueeze(-1) + eps)
