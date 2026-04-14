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
