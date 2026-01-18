from __future__ import annotations

import torch
from torch import nn
import torch.utils.checkpoint

# This intercepts calls (like from fla) that don't pass use_reentrant and sets it to False.
if not hasattr(torch.utils.checkpoint, "_original_checkpoint"):
    torch.utils.checkpoint._original_checkpoint = torch.utils.checkpoint.checkpoint
    def checkpoint_wrapper(function, *args, **kwargs):
        if "use_reentrant" not in kwargs:
            kwargs["use_reentrant"] = False
        return torch.utils.checkpoint._original_checkpoint(function, *args, **kwargs)
    torch.utils.checkpoint.checkpoint = checkpoint_wrapper

from fla.modules.feature_map import RebasedFeatureMap


class RebasedLinearAttention(nn.Module):
    """
    PyTorch implementation of Rebased Linear Attention.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int | None = None,
        feature_dim: int = 16,
        dropout: float = 0.1,
        activation: str = "silu",
        use_gamma: bool = True,
        use_beta: bool = True,
        normalize: bool = True,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_key_value_heads = num_heads
        self.feature_dim = feature_dim
        self.head_dim = d_model // num_heads
        self.eps = eps

        # Q, K project to feature_dim * heads
        self.q_proj = nn.Linear(d_model, num_heads * feature_dim, bias=False)
        self.k_proj = nn.Linear(d_model, num_heads * feature_dim, bias=False)
        # V projects to head_dim * heads
        self.v_proj = nn.Linear(d_model, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, d_model, bias=False)
        
        self.feature_map = RebasedFeatureMap(feature_dim, use_gamma, use_beta, normalize)

        if activation == "gelu":
            act = nn.GELU()
        elif activation == "relu":
            act = nn.ReLU()
        elif activation in {"silu", "swish"}:
            act = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(2)])
        if dim_feedforward is None:
            dim_feedforward = 4 * d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            act,
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def _compute_kv_state(
        self, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # k: (batch*n, seq, n_heads, feat_dim)
        # v: (batch*n, seq, n_heads, head_dim)
        k_sum = k.sum(dim=1)
        kv_state = torch.einsum("bshf,bshd->bhfd", k, v)
        return kv_state, k_sum

    def _apply_state_to_query(
        self,
        q: torch.Tensor,
        kv_state: torch.Tensor,
        k_sum: torch.Tensor,
        k_self: torch.Tensor | None = None,
        v_self: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # q: (batch*n, seq_test, n_heads, feat_dim)
        # kv_state: (batch*n, n_heads, feat_dim, head_dim)
        # k_sum: (batch*n, n_heads, feat_dim)
        
        # Numerator: Q * State
        # b s h f, b h f d -> b s h d
        num = torch.einsum("bshf,bhfd->bshd", q, kv_state)
        
        # Denominator: Q * Z
        # b s h f, b h f -> b s h
        denom = torch.einsum("bshf,bhf->bsh", q, k_sum)

        # Add self-attention contribution for test tokens if provided
        if k_self is not None and v_self is not None:
            # (b, s, h, f) * (b, s, h, f) -> (b, s, h)
            attn_self = (q * k_self).sum(dim=-1)
            
            # num += (q.k) * v
            num = num + attn_self.unsqueeze(-1) * v_self
            # denom += (q.k)
            denom = denom + attn_self

        denom = denom.unsqueeze(-1) # (b, s, h, 1)
        
        out = num / (denom + self.eps)
        return out

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
        
        is_three_dim = x.dim() == 3
        if is_three_dim:
            x = x.unsqueeze(2)
        
        b, s, n, d = x.shape
        if single_eval_pos <= 0 or single_eval_pos > s:
            raise ValueError(
                f"single_eval_pos must be in the range [1, {s}], got {single_eval_pos}."
            )

        x_norm = self.norms[0](x)
        x_flat = x_norm.transpose(1, 2).reshape(b * n, s, d)
        
        # Projects: (B*N, S, D) -> Proj Dim
        q = self.q_proj(x_flat).view(b*n, s, self.num_heads, self.feature_dim)
        k = self.k_proj(x_flat).view(b*n, s, self.num_heads, self.feature_dim)
        v = self.v_proj(x_flat).view(b*n, s, self.num_key_value_heads, self.head_dim)
        
        # Apply Feature Map
        q, k = self.feature_map(q), self.feature_map(k)

        q_train = q[:, :single_eval_pos]
        k_train = k[:, :single_eval_pos]
        v_train = v[:, :single_eval_pos]
        
        # A. Compute Train output (non-causal full attention over train prefix)
        kv_state_train, k_sum_train = self._compute_kv_state(k_train, v_train)
        attn_out_train = self._apply_state_to_query(q_train, kv_state_train, k_sum_train)
        
        # B. Test Part
        q_test = q[:, single_eval_pos:]
        k_test = k[:, single_eval_pos:]
        v_test = v[:, single_eval_pos:]
        
        # Test tokens attend to Train State and themselves
        attn_out_test = self._apply_state_to_query(
            q_test, 
            kv_state_train, 
            k_sum_train,
            k_self=k_test,
            v_self=v_test
        )
        
        attn_out = torch.cat([attn_out_train, attn_out_test], dim=1)        

        # Output projection
        attn_out = attn_out.reshape(b*n, s, self.num_heads * self.head_dim)
        attn_out = self.o_proj(attn_out)
        
        # Rearrange
        attn_out = attn_out.reshape(b, n, s, d).transpose(1, 2)
        
        x = x + attn_out
        
        x = x + self.mlp(self.norms[1](x))

        if is_three_dim:
            x = x.squeeze(2)
        return x
