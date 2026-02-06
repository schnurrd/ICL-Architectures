from __future__ import annotations

import torch
from torch import nn
import torch.utils.checkpoint
from contextlib import contextmanager

@contextmanager
def _checkpoint_no_reentrant():
    checkpoint_module = torch.utils.checkpoint
    original_checkpoint = checkpoint_module.checkpoint
    def checkpoint_wrapper(function, *args, **kwargs):
        kwargs.setdefault("use_reentrant", False)
        return original_checkpoint(function, *args, **kwargs)
    checkpoint_module.checkpoint = checkpoint_wrapper
    try:
        yield
    finally:
        checkpoint_module.checkpoint = original_checkpoint

from fla.modules.feature_map import RebasedFeatureMap
from pfns.model.attention_utils import (
    apply_state_to_query_4d,
    build_mlp,
    compute_kv_state_4d,
)


class RebasedLinearAttention(nn.Module):
    """
    PyTorch implementation of Rebased Linear Attention.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_mlp_hidden: int,
        dropout: float = 0.1,
        activation: str = "silu",
        feature_dim: int | None = None,
        use_gamma: bool = True,
        use_beta: bool = True,
        normalize: bool = True,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads."

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.feature_dim = feature_dim if feature_dim is not None else self.head_dim
        
        self.eps = eps
        self.dropout = nn.Dropout(dropout)

        # Q, K: d_model -> to feature_dim * heads
        self.q_proj = nn.Linear(d_model, num_heads * self.feature_dim)
        self.k_proj = nn.Linear(d_model, num_heads * self.feature_dim)
        # V : d_model -> d_model
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        
        self.feature_map = RebasedFeatureMap(self.feature_dim, use_gamma, use_beta, normalize)

        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(2)])
        self.mlp = build_mlp(d_model, dim_mlp_hidden, dropout, activation)

    def _prepare_input(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, bool, int, int, int, int]:
        is_three_dim = x.dim() == 3
        if is_three_dim:
            x = x.unsqueeze(2)
        b, s, n, d = x.shape
        return x, is_three_dim, b, s, n, d

    def _project_qkv_with_feature_map(
        self,
        x: torch.Tensor,
        *,
        b: int,
        s: int,
        n: int,
        d: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_norm = self.norms[0](x)
        x_flat = x_norm.transpose(1, 2).reshape(b * n, s, d)

        q = self.q_proj(x_flat).view(b * n, s, self.num_heads, self.feature_dim)
        k = self.k_proj(x_flat).view(b * n, s, self.num_heads, self.feature_dim)
        v = self.v_proj(x_flat).view(b * n, s, self.num_heads, self.head_dim)

        with _checkpoint_no_reentrant():
            q, k = self.feature_map(q), self.feature_map(k)

        return q, k, v

    def _apply_output_residual_and_mlp(
        self,
        x: torch.Tensor,
        attn_out: torch.Tensor,
        *,
        b: int,
        s: int,
        n: int,
        d: int,
        is_three_dim: bool,
    ) -> torch.Tensor:
        attn_out = attn_out.reshape(b * n, s, self.num_heads * self.head_dim)
        attn_out = self.dropout(self.o_proj(attn_out))
        attn_out = attn_out.reshape(b, n, s, d).transpose(1, 2)

        x = x + attn_out
        x = x + self.mlp(self.norms[1](x))

        if is_three_dim:
            x = x.squeeze(2)
        return x


    def forward(
        self,
        x: torch.Tensor,
        *,
        single_eval_pos: int = None,
        **kwargs,
    ) -> torch.Tensor:
        
        assert single_eval_pos is not None, (
            "single_eval_pos must be provided for RebasedLinearAttention."
        )
        x, is_three_dim, b, s, n, d = self._prepare_input(x)
        assert 0 < single_eval_pos < s, (
            f"single_eval_pos must be in the range [1, {s} - 1], got {single_eval_pos}."
        )
        q, k, v = self._project_qkv_with_feature_map(x, b=b, s=s, n=n, d=d)

        q_train = q[:, :single_eval_pos]
        k_train = k[:, :single_eval_pos]
        v_train = v[:, :single_eval_pos]
        
        # A. Compute Train output (non-causal full attention over train prefix)
        kv_state_train, k_sum_train = compute_kv_state_4d(k_train, v_train)
        attn_out_train = apply_state_to_query_4d(
            q_train, kv_state_train, k_sum_train, eps=self.eps
        )
        
        # B. Test Part
        q_test = q[:, single_eval_pos:]
        k_test = k[:, single_eval_pos:]
        v_test = v[:, single_eval_pos:]
        
        # Test tokens attend to Train State and themselves
        attn_out_test = apply_state_to_query_4d(
            q_test, 
            kv_state_train, 
            k_sum_train,
            eps=self.eps,
            k_self=k_test,
            v_self=v_test
        )
        
        attn_out = torch.cat([attn_out_train, attn_out_test], dim=1)        

        return self._apply_output_residual_and_mlp(
            x,
            attn_out,
            b=b,
            s=s,
            n=n,
            d=d,
            is_three_dim=is_three_dim,
        )

    def incontext_fit(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Process the training context and return the cached KV state."""
        x, is_three_dim, b, s, n, d = self._prepare_input(x)
        q, k, v = self._project_qkv_with_feature_map(x, b=b, s=s, n=n, d=d)

        kv_state, k_sum = compute_kv_state_4d(k, v)
        attn_out = apply_state_to_query_4d(q, kv_state, k_sum, eps=self.eps)
        x = self._apply_output_residual_and_mlp(
            x,
            attn_out,
            b=b,
            s=s,
            n=n,
            d=d,
            is_three_dim=is_three_dim,
        )
        return x, {"kv_state": kv_state, "k_sum": k_sum}

    def incontext_predict(
        self,
        x: torch.Tensor,
        state: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Process test tokens using cached KV state from the training context."""
        x, is_three_dim, b, s, n, d = self._prepare_input(x)
        q, k, v = self._project_qkv_with_feature_map(x, b=b, s=s, n=n, d=d)

        kv_state = state["kv_state"]
        k_sum = state["k_sum"]
        attn_out = apply_state_to_query_4d(
            q,
            kv_state,
            k_sum,
            eps=self.eps,
            k_self=k,
            v_self=v,
        )

        return self._apply_output_residual_and_mlp(
            x,
            attn_out,
            b=b,
            s=s,
            n=n,
            d=d,
            is_three_dim=is_three_dim,
        )
