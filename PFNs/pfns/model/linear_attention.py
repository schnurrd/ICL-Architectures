import torch
from torch import nn
import torch.nn.functional as F

from pfns.model.attention_utils import (
    apply_state_to_query_5d,
    build_mlp,
    compute_kv_state_5d,
)


class LinearAttention(nn.Module):
    """
    Linear attention layer with optional attention between feature blocks,
    following the same ordering as PerFeatureLayer (features -> items -> MLP).
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_mlp_hidden: int,
        dropout: float = 0.1,
        activation: str = "silu",
        attention_between_features: bool = False,
        feature_attention_softmax: bool = False,
        feature_dim: int | None = None,
        eps: float = 1e-6,
    ):
        super().__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads."
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.feature_dim = feature_dim if feature_dim is not None else self.head_dim
        
        self.attention_between_features = attention_between_features
        self.feature_attention_softmax = feature_attention_softmax
        
        self.dropout = nn.Dropout(dropout)
        self.eps = eps
        self.save_peak_mem_factor = None

        if attention_between_features:
            self.q_proj_feat = nn.Linear(d_model, d_model)
            self.k_proj_feat = nn.Linear(d_model, d_model)
            self.v_proj_feat = nn.Linear(d_model, d_model)
            self.out_proj_feat = nn.Linear(d_model, d_model)

        self.q_proj_item = nn.Linear(d_model, self.num_heads * self.feature_dim)
        self.k_proj_item = nn.Linear(d_model, self.num_heads * self.feature_dim)
        
        self.v_proj_item = nn.Linear(d_model, d_model)
        self.out_proj_item = nn.Linear(d_model, d_model)

        num_norms = 3 if attention_between_features else 2
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_norms)])
        self.mlp = build_mlp(d_model, dim_mlp_hidden, dropout, activation)

    def _feature_map(self, x: torch.Tensor) -> torch.Tensor:
        return F.elu(x) + 1.0

    def _linear_attention_features(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        # q, k, v: (batch, seq_len, num_feature_blocks, nhead, head_dim)
        q = self._feature_map(q)
        k = self._feature_map(k)
        k_sum = torch.einsum("bsnhd->bshd", k)
        denom = torch.einsum("bsnhd,bshd->bsnh", q, k_sum).unsqueeze(-1)
        kv = torch.einsum("bsnhd,bsnhe->bshde", k, v)
        return torch.einsum("bsnhd,bshde->bsnhe", q, kv) / (denom + self.eps)

    def _linear_attention_items(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        k_self: torch.Tensor | None = None,
        v_self: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # q, k, v: (batch, seq_len, num_feature_blocks, nhead, head_dim)
        q = self._feature_map(q)
        k = self._feature_map(k)
        kv_state, k_sum = compute_kv_state_5d(k, v)
        if k_self is not None and v_self is not None:
            k_self = self._feature_map(k_self)
        return apply_state_to_query_5d(
            q,
            kv_state,
            k_sum,
            eps=self.eps,
            k_self=k_self,
            v_self=v_self,
        )

    def _softmax_attention_features(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        # q, k, v: (batch, seq_len, num_feature_blocks, nhead, head_dim)
        q = q.permute(0, 1, 3, 2, 4)  # (b, s, h, n, d)
        k = k.permute(0, 1, 3, 2, 4)
        v = v.permute(0, 1, 3, 2, 4)
        scores = torch.einsum("bshnd,bshmd->bshnm", q, k)
        attn = torch.softmax(scores, dim=-1)
        out = torch.einsum("bshnm,bshmd->bshnd", attn, v)
        return out.permute(0, 1, 3, 2, 4)

    def _apply_feature_attention_block(
        self,
        x: torch.Tensor,
        norm_idx: int,
    ) -> tuple[torch.Tensor, int]:
        if not self.attention_between_features:
            return x, norm_idx

        x_norm = self.norms[norm_idx](x)
        b, s, n, e = x_norm.shape
        q = self.q_proj_feat(x_norm).view(b, s, n, self.num_heads, self.head_dim)
        k = self.k_proj_feat(x_norm).view(b, s, n, self.num_heads, self.head_dim)
        v = self.v_proj_feat(x_norm).view(b, s, n, self.num_heads, self.head_dim)
        if self.feature_attention_softmax:
            attn_feat = self._softmax_attention_features(q, k, v)
        else:
            attn_feat = self._linear_attention_features(q, k, v)
        attn_feat = attn_feat.reshape(b, s, n, e)
        attn_feat = self.dropout(self.out_proj_feat(attn_feat))
        return x + attn_feat, norm_idx + 1

    def _project_item_qkv(
        self,
        x: torch.Tensor,
        norm_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_norm = self.norms[norm_idx](x)
        b, s, n, _ = x_norm.shape
        q = self.q_proj_item(x_norm).view(b, s, n, self.num_heads, self.feature_dim)
        k = self.k_proj_item(x_norm).view(b, s, n, self.num_heads, self.feature_dim)
        v = self.v_proj_item(x_norm).view(b, s, n, self.num_heads, self.head_dim)
        return q, k, v

    def _apply_item_output_and_mlp(
        self,
        x: torch.Tensor,
        attn: torch.Tensor,
        norm_idx: int,
    ) -> torch.Tensor:
        b, s, n, _ = x.shape
        attn = attn.reshape(b, s, n, self.d_model)
        attn = self.dropout(self.out_proj_item(attn))
        x = x + attn
        norm_idx += 1
        return x + self.mlp(self.norms[norm_idx](x))

    def forward(self, x, *, single_eval_pos: int | None = None, **kwargs):
        # x: (batch, num_items, num_feature_blocks, embed_dim)
        norm_idx = 0
        assert x.dim() == 4, f"Expected x to have 4 dims, got shape {tuple(x.shape)}."
        b, s, n, _ = x.shape
        assert single_eval_pos is not None, (
            "single_eval_pos must be provided for LinearAttention."
        )
        assert 0 < single_eval_pos < s, (
            f"single_eval_pos must be in the range [1, {s} - 1], got {single_eval_pos}."
        )

        x, norm_idx = self._apply_feature_attention_block(x, norm_idx)
        q_all, k_all, v_all = self._project_item_qkv(x, norm_idx)

        # Train Part
        q_train = q_all[:, :single_eval_pos]
        k_train = k_all[:, :single_eval_pos]
        v_train = v_all[:, :single_eval_pos]
        
        # Test Part
        q_test = q_all[:, single_eval_pos:]
        k_test = k_all[:, single_eval_pos:]
        v_test = v_all[:, single_eval_pos:]

        kv_state_train, k_sum_train = compute_kv_state_5d(
            self._feature_map(k_train), v_train
        )

        attn_train = apply_state_to_query_5d(
            self._feature_map(q_train),
            kv_state_train,
            k_sum_train,
            eps=self.eps,
        )
        
        attn_test = apply_state_to_query_5d(
            self._feature_map(q_test),
            kv_state_train,
            k_sum_train,
            eps=self.eps,
            k_self=self._feature_map(k_test),
            v_self=v_test,
        )

        attn_all = torch.cat([attn_train, attn_test], dim=1)
        return self._apply_item_output_and_mlp(x, attn_all, norm_idx)

    def incontext_fit(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Process the training context and return the cached KV state.

        Args:
            x: (batch, seq_len_train, num_feature_blocks, embed_dim)

        Returns:
            Tuple of (output, state) where state contains kv_state and k_sum.
        """
        norm_idx = 0
        assert x.dim() == 4, f"Expected x to have 4 dims, got shape {tuple(x.shape)}."
        x, norm_idx = self._apply_feature_attention_block(x, norm_idx)
        q, k, v = self._project_item_qkv(x, norm_idx)

        q = self._feature_map(q)
        k = self._feature_map(k)
        kv_state, k_sum = compute_kv_state_5d(k, v)

        attn = apply_state_to_query_5d(q, kv_state, k_sum, eps=self.eps)
        x = self._apply_item_output_and_mlp(x, attn, norm_idx)
        return x, {"kv_state": kv_state, "k_sum": k_sum}

    def incontext_predict(
        self,
        x: torch.Tensor,
        state: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Process test tokens using cached KV state from the training context."""
        norm_idx = 0
        assert x.dim() == 4, f"Expected x to have 4 dims, got shape {tuple(x.shape)}."
        x, norm_idx = self._apply_feature_attention_block(x, norm_idx)
        q, k, v = self._project_item_qkv(x, norm_idx)

        q = self._feature_map(q)
        k = self._feature_map(k)
        kv_state = state["kv_state"]
        k_sum = state["k_sum"]

        attn = apply_state_to_query_5d(
            q,
            kv_state,
            k_sum,
            eps=self.eps,
            k_self=k,
            v_self=v,
        )
        return self._apply_item_output_and_mlp(x, attn, norm_idx)

    def empty_trainset_representation_cache(self) -> None:
        return None
