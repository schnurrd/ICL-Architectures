import torch
from torch import nn
import torch.nn.functional as F


class LinearAttention(nn.Module):
    """
    Linear attention layer with optional attention between feature blocks,
    following the same ordering as PerFeatureLayer (features -> items -> MLP).
    """
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward,
        dropout=0.1,
        activation="relu",
        attention_between_features=False,
        feature_attention_softmax: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        assert d_model % nhead == 0, "d_model must be divisible by nhead."
        self.head_dim = d_model // nhead
        self.attention_between_features = attention_between_features
        self.feature_attention_softmax = feature_attention_softmax
        self.dropout = nn.Dropout(dropout)
        self.eps = 1e-6
        self.save_peak_mem_factor = None

        if attention_between_features:
            self.q_proj_feat = nn.Linear(d_model, d_model)
            self.k_proj_feat = nn.Linear(d_model, d_model)
            self.v_proj_feat = nn.Linear(d_model, d_model)
            self.out_proj_feat = nn.Linear(d_model, d_model)

        self.q_proj_item = nn.Linear(d_model, d_model)
        self.k_proj_item = nn.Linear(d_model, d_model)
        self.v_proj_item = nn.Linear(d_model, d_model)
        self.out_proj_item = nn.Linear(d_model, d_model)

        num_norms = 3 if attention_between_features else 2
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_norms)])
        if dim_feedforward is None:
            dim_feedforward = 4 * d_model

        if activation == "gelu":
            activation_fn = nn.GELU()
        elif activation == "relu":
            activation_fn = nn.ReLU()
        elif activation in {"swish", "silu"}:
            activation_fn = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            activation_fn,
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

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
        kv = torch.einsum("bsnhd,bsnhe->bnhde", k, v)
        k_sum = torch.einsum("bsnhd->bnhd", k)
        
        # Denominator: Q * Z (Train)
        denom = torch.einsum("bsnhd,bnhd->bsnh", q, k_sum)
        # Numerator: Q * State (Train)
        num = torch.einsum("bsnhd,bnhde->bsnhe", q, kv)

        # Add Self-Attention (Test -> Self)
        if k_self is not None and v_self is not None:
             k_self = self._feature_map(k_self)
             # (b, s, n, h, d) * (b, s, n, h, d) -> (b, s, n, h)
             attn_self = (q * k_self).sum(dim=-1)
             
             num = num + attn_self.unsqueeze(-1) * v_self
             denom = denom + attn_self

        return num / (denom.unsqueeze(-1) + self.eps)

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

    def _apply_attention(
        self,
        x: torch.Tensor,
        *,
        q_proj: nn.Linear,
        k_proj: nn.Linear,
        v_proj: nn.Linear,
        out_proj: nn.Linear,
        kv_x: torch.Tensor | None = None,
        attention_across_features: bool = False,
        use_self_attention_for_items: bool = False,
    ) -> torch.Tensor:
        q_x = x
        kv_x = x if kv_x is None else kv_x
        b, s_q, n, e = q_x.shape
        b_kv, s_kv, n_kv, e_kv = kv_x.shape
        assert b == b_kv and n == n_kv and e == e_kv, "q_x and kv_x must share batch, token, and embedding dims."

        if attention_across_features:
            assert s_q == s_kv, "We do not support mismatched sequence lengths."
            q = q_proj(q_x).view(b, s_q, n, self.nhead, self.head_dim)
            k = k_proj(kv_x).view(b, s_kv, n, self.nhead, self.head_dim)
            v = v_proj(kv_x).view(b, s_kv, n, self.nhead, self.head_dim)
            if self.feature_attention_softmax:
                out = self._softmax_attention_features(q, k, v)
            else:
                out = self._linear_attention_features(q, k, v)
        else:
            q = q_proj(q_x).view(b, s_q, n, self.nhead, self.head_dim)
            k = k_proj(kv_x).view(b, s_kv, n, self.nhead, self.head_dim)
            v = v_proj(kv_x).view(b, s_kv, n, self.nhead, self.head_dim)
            
            k_self = None
            v_self = None
            if use_self_attention_for_items:
                k_self = k_proj(q_x).view(b, s_q, n, self.nhead, self.head_dim)
                v_self = v_proj(q_x).view(b, s_q, n, self.nhead, self.head_dim)
            
            out = self._linear_attention_items(q, k, v, k_self=k_self, v_self=v_self)

        out = out.reshape(b, s_q, n, e)
        return self.dropout(out_proj(out))

    def forward(self, x, *, single_eval_pos: int | None = None, **kwargs):
        # x: (batch, num_items, num_feature_blocks, embed_dim)
        norm_idx = 0

        if self.attention_between_features:
            x_norm = self.norms[norm_idx](x)
            attn_feat = self._apply_attention(
                x_norm,
                q_proj=self.q_proj_feat,
                k_proj=self.k_proj_feat,
                v_proj=self.v_proj_feat,
                out_proj=self.out_proj_feat,
                attention_across_features=True,
            )
            x = x + attn_feat
            norm_idx += 1

        x_norm = self.norms[norm_idx](x)
        train_x = x_norm[:, :single_eval_pos]
        test_x = x_norm[:, single_eval_pos:]
        
        attn_train = self._apply_attention(
            train_x,
            q_proj=self.q_proj_item,
            k_proj=self.k_proj_item,
            v_proj=self.v_proj_item,
            out_proj=self.out_proj_item,
            kv_x=train_x,
        )
        attn_test = self._apply_attention(
            test_x,
            q_proj=self.q_proj_item,
            k_proj=self.k_proj_item,
            v_proj=self.v_proj_item,
            out_proj=self.out_proj_item,
            kv_x=train_x,
            use_self_attention_for_items=True,
        )
        attn_item = torch.cat([attn_train, attn_test], dim=1)
        
        x = x + attn_item
        norm_idx += 1

        x = x + self.ff(self.norms[norm_idx](x))
        return x

    def empty_trainset_representation_cache(self) -> None:
        return None
