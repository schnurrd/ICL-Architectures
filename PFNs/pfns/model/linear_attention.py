import torch
from torch import nn
import torch.nn.functional as F

from pfns.model.attention_utils import (
    apply_state_to_query_5d,
    build_mlp,
    clip_linear_attention_output_norm,
    clip_linear_attention_state_frobenius_norm,
    compute_kv_state_5d,
)


class LinearAttention(nn.Module):
    AUTO_CAUSAL_EVAL_CHUNK_SIZE = 2000
    HIDDEN_STATE_FROBENIUS_NORM_APPLY_MODES = {
        "pre_attention",
        "pre_prediction",
    }
    HIDDEN_STATE_FROBENIUS_NORM_TARGETS = {
        "joint",
        "kv_state",
        "k_sum",
        "kv_over_ksum_ratio",
    }
    LENGTH_NORMALIZATION_MODES = {"none", "sqrt_length", "length"}

    """
    Linear attention layer with optional attention between feature blocks,
    following the same ordering as PerFeatureLayer (features -> items -> MLP).

    Item attention supports three masking modes:
    - default: train tokens attend bidirectionally within train; test tokens attend
      to train only
    - causal_train_only: train tokens attend causally; test tokens attend only to train
    - causal: full autoregressive attention during training; switches to
      causal_train_only during inference
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_mlp_hidden: int,
        dropout: float = 0.1,
        activation: str = "silu",
        attention_between_features: bool = False,
        causal: bool = False,
        causal_train_only: bool = False,
        causal_chunk_size: int | None = None,
        feature_attention_softmax: bool = False,
        feature_dim: int | None = None,
        hidden_state_frobenius_norm_max: float | None = None,
        hidden_state_frobenius_norm_apply: str = "pre_prediction",
        hidden_state_frobenius_norm_target: str = "joint",
        hidden_state_frobenius_norm_length_normalization: str = "none",
        attention_output_norm_max: float | None = None,
        attention_output_norm_length_normalization: str = "none",
        eps: float = 1e-6,
    ):
        """Initialize projections, norms, and masking flags."""
        super().__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads."
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.feature_dim = feature_dim if feature_dim is not None else self.head_dim
        
        self.attention_between_features = attention_between_features
        self.causal = bool(causal)
        self.causal_train_only = bool(causal_train_only)
        if self.causal and self.causal_train_only:
            raise ValueError(
                "causal and causal_train_only are mutually exclusive."
            )
        if causal_chunk_size is not None and causal_chunk_size <= 0:
            raise ValueError("causal_chunk_size must be >= 1.")
        self.causal_chunk_size = causal_chunk_size
        self.feature_attention_softmax = feature_attention_softmax
        
        self.dropout = nn.Dropout(dropout)
        self.eps = eps
        if (
            hidden_state_frobenius_norm_max is not None
            and hidden_state_frobenius_norm_max <= 0.0
        ):
            raise ValueError("hidden_state_frobenius_norm_max must be > 0.")
        self.hidden_state_frobenius_norm_max = hidden_state_frobenius_norm_max
        hidden_state_frobenius_norm_apply = (
            hidden_state_frobenius_norm_apply.strip().lower().replace("-", "_")
        )
        if (
            hidden_state_frobenius_norm_apply
            not in self.HIDDEN_STATE_FROBENIUS_NORM_APPLY_MODES
        ):
            raise ValueError(
                "hidden_state_frobenius_norm_apply must be one of "
                f"{sorted(self.HIDDEN_STATE_FROBENIUS_NORM_APPLY_MODES)}, got "
                f"{hidden_state_frobenius_norm_apply!r}."
            )
        self.hidden_state_frobenius_norm_apply = hidden_state_frobenius_norm_apply
        hidden_state_frobenius_norm_target = (
            hidden_state_frobenius_norm_target.strip().lower().replace("-", "_")
        )
        if hidden_state_frobenius_norm_target not in self.HIDDEN_STATE_FROBENIUS_NORM_TARGETS:
            raise ValueError(
                "hidden_state_frobenius_norm_target must be one of "
                f"{sorted(self.HIDDEN_STATE_FROBENIUS_NORM_TARGETS)}, got "
                f"{hidden_state_frobenius_norm_target!r}."
            )
        self.hidden_state_frobenius_norm_target = hidden_state_frobenius_norm_target
        hidden_state_frobenius_norm_length_normalization = (
            hidden_state_frobenius_norm_length_normalization.strip().lower().replace("-", "_")
        )
        if (
            hidden_state_frobenius_norm_length_normalization
            not in self.LENGTH_NORMALIZATION_MODES
        ):
            raise ValueError(
                "hidden_state_frobenius_norm_length_normalization must be one of "
                f"{sorted(self.LENGTH_NORMALIZATION_MODES)}, got "
                f"{hidden_state_frobenius_norm_length_normalization!r}."
            )
        self.hidden_state_frobenius_norm_length_normalization = (
            hidden_state_frobenius_norm_length_normalization
        )
        if attention_output_norm_max is not None and attention_output_norm_max <= 0.0:
            raise ValueError("attention_output_norm_max must be > 0.")
        attention_output_norm_length_normalization = (
            attention_output_norm_length_normalization.strip().lower().replace("-", "_")
        )
        if attention_output_norm_length_normalization not in self.LENGTH_NORMALIZATION_MODES:
            raise ValueError(
                "attention_output_norm_length_normalization must be one of "
                f"{sorted(self.LENGTH_NORMALIZATION_MODES)}, got "
                f"{attention_output_norm_length_normalization!r}."
            )
        self.attention_output_norm_max = attention_output_norm_max
        self.attention_output_norm_length_normalization = (
            attention_output_norm_length_normalization
        )

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

    def _clip_hidden_state_for_attention(
        self,
        kv_state: torch.Tensor,
        k_sum: torch.Tensor,
        *,
        state_length: int | float | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.hidden_state_frobenius_norm_apply != "pre_attention":
            return kv_state, k_sum
        return clip_linear_attention_state_frobenius_norm(
            kv_state,
            k_sum,
            self.hidden_state_frobenius_norm_max,
            target=self.hidden_state_frobenius_norm_target,
            length_normalization=self.hidden_state_frobenius_norm_length_normalization,
            state_length=state_length,
        )

    def _clip_hidden_state_for_prediction(
        self,
        kv_state: torch.Tensor,
        k_sum: torch.Tensor,
        *,
        state_length: int | float | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.hidden_state_frobenius_norm_apply not in {
            "pre_attention",
            "pre_prediction",
        }:
            return kv_state, k_sum
        return clip_linear_attention_state_frobenius_norm(
            kv_state,
            k_sum,
            self.hidden_state_frobenius_norm_max,
            target=self.hidden_state_frobenius_norm_target,
            length_normalization=self.hidden_state_frobenius_norm_length_normalization,
            state_length=state_length,
        )

    def _clip_attention_output(
        self,
        attn: torch.Tensor,
        *,
        state_length: int | float | torch.Tensor | None = None,
    ) -> torch.Tensor:
        return clip_linear_attention_output_norm(
            attn,
            self.attention_output_norm_max,
            length_normalization=self.attention_output_norm_length_normalization,
            state_length=state_length,
        )
        
    def _feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """phi(x) = ELU(x) + 1."""
        return F.elu(x) + 1.0

    def _linear_attention_features(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Feature linear attention: out_n = (phi(q_n)^T sum_m phi(k_m)v_m^T) / (phi(q_n)^T sum_m phi(k_m) + eps)."""
        # q, k, v: (batch, seq_len, num_feature_blocks, nhead, head_dim)
        q = self._feature_map(q)
        k = self._feature_map(k)
        k_sum = torch.einsum("bsnhd->bshd", k)
        denom = torch.einsum("bsnhd,bshd->bsnh", q, k_sum).unsqueeze(-1)
        kv = torch.einsum("bsnhd,bsnhe->bshde", k, v)
        return torch.einsum("bsnhd,bshde->bsnhe", q, kv) / (denom + self.eps)

    def _softmax_attention_features(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Feature softmax attention: out_n = sum_m softmax_m(q_n^T k_m) v_m."""
        # q, k, v: (batch, seq_len, num_feature_blocks, nhead, head_dim)
        q = q.permute(0, 1, 3, 2, 4)  # (b, s, h, n, d)
        k = k.permute(0, 1, 3, 2, 4)
        v = v.permute(0, 1, 3, 2, 4)
        scores = torch.einsum("bshnd,bshmd->bshnm", q, k)
        attn = torch.softmax(scores, dim=-1)
        out = torch.einsum("bshnm,bshmd->bshnd", attn, v)
        return out.permute(0, 1, 3, 2, 4)

    def _causal_linear_attention_chunk(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        kv_state_prefix: torch.Tensor | None = None,
        k_sum_prefix: torch.Tensor | None = None,
        state_length_prefix: int | float | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process one causal chunk exactly, optionally seeded from a cached prefix state."""
        if q.shape[1] == 0:
            assert kv_state_prefix is not None and k_sum_prefix is not None
            return v, kv_state_prefix, k_sum_prefix

        kv_prefix = torch.cumsum(torch.einsum("bsnhf,bsnhd->bsnhfd", k, v), dim=1)
        k_prefix = torch.cumsum(k, dim=1)
        if kv_state_prefix is not None:
            kv_prefix = kv_prefix + kv_state_prefix.unsqueeze(1)
        if k_sum_prefix is not None:
            k_prefix = k_prefix + k_sum_prefix.unsqueeze(1)
        prefix_length = float(torch.as_tensor(state_length_prefix).item()) if state_length_prefix is not None else 0.0
        state_lengths = (
            torch.arange(1, q.shape[1] + 1, device=q.device, dtype=q.dtype).view(1, -1, 1, 1)
            + prefix_length
        )
        kv_prefix_for_attn, k_prefix_for_attn = self._clip_hidden_state_for_attention(
            kv_prefix,
            k_prefix,
            state_length=state_lengths,
        )

        num = torch.einsum("bsnhf,bsnhfd->bsnhd", q, kv_prefix_for_attn)
        denom = torch.einsum("bsnhf,bsnhf->bsnh", q, k_prefix_for_attn)
        attn = num / (denom.unsqueeze(-1) + self.eps)
        attn = self._clip_attention_output(attn, state_length=state_lengths)
        return attn, kv_prefix[:, -1], k_prefix[:, -1]

    def _resolved_causal_chunk_size(self, seq_len: int) -> int | None:
        """Resolve the chunk size for causal recurrence."""
        if (
            self.causal_train_only
            and self.hidden_state_frobenius_norm_max is not None
            and self.hidden_state_frobenius_norm_apply == "pre_attention"
        ):
            return None
        if self.causal_chunk_size is not None:
            return self.causal_chunk_size
        if (
            (self.causal or self.causal_train_only)
            and not self.training
            and seq_len > self.AUTO_CAUSAL_EVAL_CHUNK_SIZE
        ):
            return self.AUTO_CAUSAL_EVAL_CHUNK_SIZE
        return None

    def _causal_linear_attention_items(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        kv_state_prefix: torch.Tensor | None = None,
        k_sum_prefix: torch.Tensor | None = None,
        state_length_prefix: int | float | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Causal item attention with optional chunking over the sequence dimension."""
        if q.shape[1] == 0:
            return self._causal_linear_attention_chunk(
                q,
                k,
                v,
                kv_state_prefix=kv_state_prefix,
                k_sum_prefix=k_sum_prefix,
                state_length_prefix=state_length_prefix,
            )

        chunk_size = self._resolved_causal_chunk_size(q.shape[1])
        if chunk_size is None or q.shape[1] <= chunk_size:
            return self._causal_linear_attention_chunk(
                q,
                k,
                v,
                kv_state_prefix=kv_state_prefix,
                k_sum_prefix=k_sum_prefix,
                state_length_prefix=state_length_prefix,
            )

        outputs = []
        kv_state = kv_state_prefix
        k_sum = k_sum_prefix
        state_length = int(torch.as_tensor(state_length_prefix).item()) if state_length_prefix is not None else 0
        for chunk_start in range(0, q.shape[1], chunk_size):
            chunk_end = min(chunk_start + chunk_size, q.shape[1])
            attn_chunk, kv_state, k_sum = self._causal_linear_attention_chunk(
                q[:, chunk_start:chunk_end],
                k[:, chunk_start:chunk_end],
                v[:, chunk_start:chunk_end],
                kv_state_prefix=kv_state,
                k_sum_prefix=k_sum,
                state_length_prefix=state_length,
            )
            outputs.append(attn_chunk)
            state_length += chunk_end - chunk_start

        return torch.cat(outputs, dim=1), kv_state, k_sum

    def _apply_feature_attention_block(
        self,
        x: torch.Tensor,
        norm_idx: int,
    ) -> tuple[torch.Tensor, int]:
        """Feature block: x' = x + Dropout(W_o Attn_feat(W_q LN(x), W_k LN(x), W_v LN(x)))."""
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
        """Item projections: q = W_q LN(x), k = W_k LN(x), v = W_v LN(x)."""
        x_norm = self.norms[norm_idx](x)
        b, s, n, _ = x_norm.shape
        q = self.q_proj_item(x_norm).view(b, s, n, self.num_heads, self.feature_dim)
        k = self.k_proj_item(x_norm).view(b, s, n, self.num_heads, self.feature_dim)
        v = self.v_proj_item(x_norm).view(b, s, n, self.num_heads, self.head_dim)
        return q, k, v

    def _compute_train_attention_and_state(
        self,
        q_train: torch.Tensor,
        k_train: torch.Tensor,
        v_train: torch.Tensor,
        *,
        causal_train: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute train outputs and the train state consumed by test tokens."""
        q_train = self._feature_map(q_train)
        k_train = self._feature_map(k_train)

        if causal_train:
            return self._causal_linear_attention_items(q_train, k_train, v_train)

        kv_state_train, k_sum_train = compute_kv_state_5d(k_train, v_train)
        state_length = int(q_train.shape[1])
        kv_state_train_for_attn, k_sum_train_for_attn = self._clip_hidden_state_for_attention(
            kv_state_train,
            k_sum_train,
            state_length=state_length,
        )
        attn_train = apply_state_to_query_5d(
            q_train,
            kv_state_train_for_attn,
            k_sum_train_for_attn,
            eps=self.eps,
        )
        attn_train = self._clip_attention_output(attn_train, state_length=state_length)
        return attn_train, kv_state_train, k_sum_train

    def _apply_split_item_attention(
        self,
        q_train: torch.Tensor,
        k_train: torch.Tensor,
        v_train: torch.Tensor,
        q_test: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the split train/test attention path.

        Test tokens attend only to the aggregated train state. The difference
        between modes is whether the train state itself was built causally or
        non-causally.
        """
        use_causal_train_only = self.causal_train_only or (self.causal and not self.training)
        attn_train, kv_state_train, k_sum_train = self._compute_train_attention_and_state(
            q_train,
            k_train,
            v_train,
            causal_train=use_causal_train_only,
        )
        q_test = self._feature_map(q_test)
        state_length = int(q_train.shape[1])
        kv_state_train, k_sum_train = self._clip_hidden_state_for_prediction(
            kv_state_train,
            k_sum_train,
            state_length=state_length,
        )
        attn_test = apply_state_to_query_5d(
            q_test,
            kv_state_train,
            k_sum_train,
            eps=self.eps,
        )
        attn_test = self._clip_attention_output(attn_test, state_length=state_length)
        return torch.cat([attn_train, attn_test], dim=1)

    def _apply_item_output_and_mlp(
        self,
        x: torch.Tensor,
        attn: torch.Tensor,
        norm_idx: int,
    ) -> torch.Tensor:
        """Output block: x1 = x + Dropout(W_o attn), out = x1 + MLP(LN(x1))."""
        b, s, n, _ = x.shape
        attn = attn.reshape(b, s, n, self.d_model)
        attn = self.dropout(self.out_proj_item(attn))
        x = x + attn
        norm_idx += 1
        return x + self.mlp(self.norms[norm_idx](x))

    def forward(self, x, *, single_eval_pos: int | None = None, **kwargs):
        """Run feature attention, item attention, and the residual MLP.

        `causal=True` uses full autoregressive attention only in training mode.
        All other cases use the split train/test path.
        """
        # x: (batch, num_items, num_feature_blocks, embed_dim)
        norm_idx = 0
        assert x.dim() == 4, f"Expected x to have 4 dims, got shape {tuple(x.shape)}."
        _, s, _, _ = x.shape
        assert single_eval_pos is not None, (
            "single_eval_pos must be provided for LinearAttention."
        )
        assert 0 < single_eval_pos < s, (
            f"single_eval_pos must be in the range [1, {s} - 1], got {single_eval_pos}."
        )

        x, norm_idx = self._apply_feature_attention_block(x, norm_idx)
        q_all, k_all, v_all = self._project_item_qkv(x, norm_idx)

        if self.causal and self.training:
            q_all = self._feature_map(q_all)
            k_all = self._feature_map(k_all)
            attn_all, _, _ = self._causal_linear_attention_items(q_all, k_all, v_all)
            return self._apply_item_output_and_mlp(x, attn_all, norm_idx)

        q_train = q_all[:, :single_eval_pos]
        k_train = k_all[:, :single_eval_pos]
        v_train = v_all[:, :single_eval_pos]
        q_test = q_all[:, single_eval_pos:]
        attn_all = self._apply_split_item_attention(
            q_train,
            k_train,
            v_train,
            q_test,
        )
        return self._apply_item_output_and_mlp(x, attn_all, norm_idx)

    def incontext_fit(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Encode the train context and return cached `(kv_state, k_sum)`."""
        norm_idx = 0
        assert x.dim() == 4, f"Expected x to have 4 dims, got shape {tuple(x.shape)}."
        x, norm_idx = self._apply_feature_attention_block(x, norm_idx)
        q, k, v = self._project_item_qkv(x, norm_idx)

        if self.causal or self.causal_train_only:
            attn, kv_state, k_sum = self._compute_train_attention_and_state(
                q,
                k,
                v,
                causal_train=True,
            )
        else:
            q = self._feature_map(q)
            k = self._feature_map(k)
            kv_state, k_sum = compute_kv_state_5d(k, v)
            kv_state_for_attn, k_sum_for_attn = self._clip_hidden_state_for_attention(
                kv_state,
                k_sum,
                state_length=int(q.shape[1]),
            )
            attn = apply_state_to_query_5d(
                q,
                kv_state_for_attn,
                k_sum_for_attn,
                eps=self.eps,
            )
            attn = self._clip_attention_output(attn, state_length=int(q.shape[1]))
        x = self._apply_item_output_and_mlp(x, attn, norm_idx)
        return x, {
            "kv_state": kv_state,
            "k_sum": k_sum,
            "state_length": torch.tensor(int(q.shape[1]), device=q.device),
        }

    def incontext_predict(
        self,
        x: torch.Tensor,
        state: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Apply cached train state to test tokens.

        Depending on the mode, this is either:
        - causal continuation from the cached prefix, or
        - attention to cached train state only
        """
        norm_idx = 0
        assert x.dim() == 4, f"Expected x to have 4 dims, got shape {tuple(x.shape)}."
        x, norm_idx = self._apply_feature_attention_block(x, norm_idx)
        q, k, v = self._project_item_qkv(x, norm_idx)
        kv_state = state["kv_state"]
        k_sum = state["k_sum"]
        state_length = state.get("state_length")
        q = self._feature_map(q)
        if self.causal and self.training:
            k = self._feature_map(k)
            attn, _, _ = self._causal_linear_attention_items(
                q,
                k,
                v,
                kv_state_prefix=kv_state,
                k_sum_prefix=k_sum,
                state_length_prefix=state_length,
            )
        else:
            kv_state, k_sum = self._clip_hidden_state_for_prediction(
                kv_state,
                k_sum,
                state_length=state_length,
            )
            attn = apply_state_to_query_5d(
                q,
                kv_state,
                k_sum,
                eps=self.eps,
            )
            attn = self._clip_attention_output(attn, state_length=state_length)
        return self._apply_item_output_and_mlp(x, attn, norm_idx)

    def empty_trainset_representation_cache(self) -> None:
        """No internal cache object."""
        return None
