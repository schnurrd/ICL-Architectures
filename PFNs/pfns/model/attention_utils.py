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


def renormalize_state_frobenius(
    state: torch.Tensor,
    *,
    mode: str | None,
    target_norm: float | None = None,
    head_scale: torch.Tensor | None = None,
    eps: float,
) -> torch.Tensor:
    """Renormalize matrix-valued recurrent states over their last two dims."""
    if mode in {None, "none"}:
        return state
    if mode != "sqrt_d_fro":
        raise ValueError(f"Unsupported state renormalization mode: {mode}")

    if target_norm is None:
        target_norm = float(state.shape[-1]) ** 0.5
    current_norm = torch.linalg.matrix_norm(
        state,
        ord="fro",
        dim=(-2, -1),
        keepdim=True,
    )
    target = state.new_tensor(target_norm)
    if head_scale is not None:
        # Expand per-head scales to `(heads, 1, 1)` so they broadcast over `state`.
        target = target * head_scale[..., None, None]
    scale = target / current_norm.clamp_min(eps)
    return state * scale
