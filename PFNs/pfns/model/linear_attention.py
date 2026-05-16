from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch import nn
from fla.modules import GatedMLP as FLAGatedMLP
from fla.modules.feature_map import HadamardFeatureMap

from pfns.model.attention_utils import (
    build_norm,
    renormalize_state_frobenius,
)


class GatedMLP(FLAGatedMLP):
    """Upstream FLA GatedMLP with CPU-safe forward fallback."""

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        if x.is_cuda:
            return super().forward(x, **kwargs)
        gate, y = self.gate_proj(x), self.up_proj(x)
        return self.down_proj(F.silu(gate) * y)


def init_linear_attention_weights_like_fla(
    module: nn.Module,
    *,
    initializer_range: float = 0.02,
) -> None:
    """Initialize linear-attention modules using FLA's linear-weight scheme."""
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=initializer_range)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class LinearAttention(nn.Module):
    """
    Linear attention layer following the same high-level ordering as
    PerFeatureLayer, but without the feature-attention block
    (items -> MLP).

    Item attention supports three masking modes:
    - default: train tokens attend bidirectionally within train; test tokens attend
      to train only
    - causal_train_only: train tokens attend causally; test tokens attend only to train
    - causal: full autoregressive attention during training; switches to
      causal_train_only during inference
    """
    DEFAULT_CAUSAL_CHUNK_SIZE = 64

    @staticmethod
    def _normalize_state_update_rule(state_update_rule: str) -> str:
        normalized = state_update_rule.strip().lower().replace("-", "_")
        if normalized in {"rls", "least_squares_oracle", "oracle"}:
            normalized = "least_squares"
        if normalized == "ridge_oracle":
            normalized = "ridge"
        if normalized not in {"linear", "least_squares", "ridge"}:
            raise ValueError(
                "state_update_rule must be one of {'linear', 'least_squares', 'ridge', 'rls'}, "
                f"got {state_update_rule!r}."
            )
        return normalized

    @staticmethod
    def _build_feature_maps(
        feature_map: str,
        head_k_dim: int,
    ) -> tuple[nn.Module, nn.Module]:
        if feature_map == "elementwise_product":
            return HadamardFeatureMap(head_k_dim), HadamardFeatureMap(head_k_dim)
        if feature_map == "elu":
            class _EluPlusOne(nn.Module):
                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    return F.elu(x) + 1.0

            return _EluPlusOne(), _EluPlusOne()
        if feature_map == "identity":
            return nn.Identity(), nn.Identity()
        raise ValueError(f"Unsupported feature_map: {feature_map}")

    def __init__(
        self,
        # Model dimensions.
        d_model: int,
        num_heads: int,
        expand_k: float = 1.0,
        expand_v: float = 1.0,
        # MLP block.
        mlp_hidden_dim: int | None = None,
        use_mlp_norm: bool = True,
        # Attention feature map.
        feature_map: str = "elu",
        # Sequence mixing mode.
        causal: bool = False,
        causal_train_only: bool = False,
        causal_chunk_size: int | None = None,
        # Attention feature map and readout.
        normalize_q_sum: bool = False,
        normalize_k_sum: bool = False,
        use_k_sum_normalization: bool = False,
        use_query_scale: bool = True,
        # Attention/output blocks.
        use_attention_norm: bool = True,
        use_output_norm: bool = True,
        norm_type: str = "rmsnorm",
        fuse_swiglu: bool = False,
        # Recurrent state handling.
        state_update_rule: str = "linear",
        ridge_lambda: float = 1.0,
        state_renormalization: str | None = None,
        learnable_state_renorm_scale: bool = True,
        state_renormalization_target_norm: float | None = None,
        # Numerical stability.
        eps: float = 1e-6,
    ):
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads."
        assert expand_k > 0 and expand_v > 0, "expand_k and expand_v must be > 0."
        assert not (causal and causal_train_only), (
            "causal and causal_train_only are mutually exclusive."
        )
        assert causal_chunk_size is None or causal_chunk_size > 0, (
            "causal_chunk_size must be >= 1."
        )
        assert not (
            use_k_sum_normalization and state_renormalization not in {None, "none"}
        ), (
            "use_k_sum_normalization and state_renormalization are mutually "
            "exclusive."
        )
        assert (
            state_renormalization_target_norm is None
            or state_renormalization_target_norm > 0
        ), "state_renormalization_target_norm must be > 0."
        normalized_state_update_rule = self._normalize_state_update_rule(
            state_update_rule
        )
        if ridge_lambda <= 0:
            raise ValueError("ridge_lambda must be > 0.")
        if normalized_state_update_rule != "linear" and use_k_sum_normalization:
            raise ValueError(
                "use_k_sum_normalization is only defined for state_update_rule='linear'."
            )
        if (
            normalized_state_update_rule != "linear"
            and state_renormalization not in {None, "none"}
        ):
            raise ValueError(
                "state_renormalization is only defined for state_update_rule='linear'."
            )

        self.d_model = d_model
        self.num_heads = num_heads

        self.key_dim = int(d_model * expand_k)
        self.value_dim = int(d_model * expand_v)
        assert self.key_dim % self.num_heads == 0, (
            "int(d_model * expand_k) must be divisible by num_heads."
        )
        assert self.value_dim % self.num_heads == 0, (
            "int(d_model * expand_v) must be divisible by num_heads."
        )

        self.head_k_dim = self.key_dim // self.num_heads
        self.head_v_dim = self.value_dim // self.num_heads
        self.query_scale = self.head_k_dim ** -0.5
        self.use_query_scale = bool(use_query_scale)

        self.causal = causal
        self.causal_train_only = causal_train_only
        self.causal_chunk_size = causal_chunk_size

        self.normalize_q_sum = normalize_q_sum
        self.normalize_k_sum = normalize_k_sum
        self.use_k_sum_normalization = use_k_sum_normalization

        self.state_update_rule = normalized_state_update_rule
        self.ridge_lambda = float(ridge_lambda)
        self.state_renormalization = state_renormalization
        self.state_renormalization_target_norm = state_renormalization_target_norm
        self.eps = eps
        if state_renormalization in {None, "none"}:
            self.state_renorm_log_scale = None
        elif learnable_state_renorm_scale:
            self.state_renorm_log_scale = nn.Parameter(torch.zeros(self.num_heads))
        else:
            self.register_buffer(
                "state_renorm_log_scale",
                torch.zeros(self.num_heads),
            )

        self.feature_map_q, self.feature_map_k = self._build_feature_maps(
            feature_map,
            self.head_k_dim,
        )

        self.q_proj_item = nn.Linear(
            d_model,
            self.key_dim,
            bias=False,
        )
        self.k_proj_item = nn.Linear(
            d_model,
            self.key_dim,
            bias=False,
        )
        self.v_proj_item = nn.Linear(d_model, self.value_dim, bias=False)
        self.out_proj_item = nn.Linear(self.value_dim, d_model, bias=False)

        self.attention_norm = build_norm(
            d_model,
            enabled=use_attention_norm,
            norm_type=norm_type,
        )
        self.output_norm = build_norm(
            self.head_v_dim,
            enabled=use_output_norm,
            norm_type=norm_type,
        )

        self.mlp_norm = build_norm(
            d_model,
            enabled=use_mlp_norm,
            norm_type=norm_type,
        )
        self.mlp = GatedMLP(
            hidden_size=d_model,
            intermediate_size=mlp_hidden_dim,
            hidden_act="swish",
            fuse_swiglu=fuse_swiglu,
        )

    def _apply_feature_map(
        self,
        x: torch.Tensor,
        feature_map: nn.Module,
        *,
        normalize_sum: bool,
    ) -> torch.Tensor:
        x = feature_map(x)
        if normalize_sum:
            x = x / (x.sum(dim=-1, keepdim=True) + self.eps)
        return x

    def _scale_queries(self, q: torch.Tensor) -> torch.Tensor:
        if not self.use_query_scale:
            return q
        return q * self.query_scale

    def _project_qkv(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.attention_norm(x) # normalize last dim (d_model)
        batch_size, seq_len, num_features, _ = x.shape
        assert num_features == 1, f"LinearAttention only supports num_features=1, got {num_features}."
        x_flat = x.transpose(1, 2).reshape(batch_size * num_features, seq_len, self.d_model)
        q = self.q_proj_item(x_flat).view(batch_size * num_features, seq_len, self.num_heads, self.head_k_dim)
        k = self.k_proj_item(x_flat).view(
            batch_size * num_features,
            seq_len,
            self.num_heads,
            self.head_k_dim,
        )
        v = self.v_proj_item(x_flat).view(
            batch_size * num_features,
            seq_len,
            self.num_heads,
            self.head_v_dim,
        )
        return q, k, v

    def _renormalize_state(self, kv_state: torch.Tensor) -> torch.Tensor:
        return renormalize_state_frobenius(
            kv_state,
            mode=self.state_renormalization,
            target_norm=self.state_renormalization_target_norm,
            head_scale=(
                None
                if self.state_renorm_log_scale is None
                else self.state_renorm_log_scale.exp()
            ),
            eps=self.eps,
        )

    def _apply_query_key_feature_maps(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = self._apply_feature_map(
            q,
            self.feature_map_q,
            normalize_sum=self.normalize_q_sum,
        )
        q = self._scale_queries(q)
        k = self._apply_feature_map(
            k,
            self.feature_map_k,
            normalize_sum=self.normalize_k_sum,
        )
        return q, k

    def _read_from_kv_state(
        self,
        q: torch.Tensor,
        kv_state: torch.Tensor,
        k_sum: torch.Tensor | None,
    ) -> torch.Tensor:
        """Apply cached per-feature state.

        If k_sum is not None computes:
            out = (q^T KV) / (q^T K_sum + eps)
        else:
            out = q^T KV
        """
        # q: (batch_times_features, seq, heads, qk_dim)
        # kv_state: (batch_times_features, heads, qk_dim, v_dim)
        # k_sum: (batch_times_features, heads, qk_dim)
        kv_state = self._renormalize_state(kv_state)
        num = torch.einsum("bshf,bhfd->bshd", q, kv_state)
        if k_sum is None:
            return num
        denom = torch.einsum("bshf,bhf->bsh", q, k_sum)
        return num / (denom.unsqueeze(-1) + self.eps)

    def _least_squares_state(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Return the ridgeless minimum-norm linear least-squares memory."""
        if k.shape[1] == 0:
            return k.new_zeros(k.shape[0], k.shape[2], k.shape[3], v.shape[-1])

        compute_dtype = (
            torch.float32
            if k.dtype in {torch.float16, torch.bfloat16}
            else k.dtype
        )
        autocast_context = (
            torch.autocast(device_type=k.device.type, enabled=False)
            if k.device.type in {"cpu", "cuda"}
            else nullcontext()
        )
        with autocast_context:
            k_bh = k.transpose(1, 2).to(compute_dtype)
            v_bh = v.transpose(1, 2).to(compute_dtype)
            state = torch.matmul(torch.linalg.pinv(k_bh), v_bh)
        return state.to(v.dtype)

    def _ridge_state(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        if k.shape[1] == 0:
            return k.new_zeros(k.shape[0], k.shape[2], k.shape[3], v.shape[-1])

        compute_dtype = (
            torch.float32
            if k.dtype in {torch.float16, torch.bfloat16}
            else k.dtype
        )
        autocast_context = (
            torch.autocast(device_type=k.device.type, enabled=False)
            if k.device.type in {"cpu", "cuda"}
            else nullcontext()
        )
        with autocast_context:
            k_bh = k.transpose(1, 2).to(compute_dtype)
            v_bh = v.transpose(1, 2).to(compute_dtype)
            gram = torch.matmul(k_bh.transpose(-1, -2), k_bh)
            cross = torch.matmul(k_bh.transpose(-1, -2), v_bh)
            eye = torch.eye(k.shape[-1], dtype=compute_dtype, device=k.device)
            state = torch.linalg.solve(
                gram + self.ridge_lambda * eye,
                cross,
            )
        return state.to(v.dtype)

    def _oracle_state(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        if self.state_update_rule == "ridge":
            return self._ridge_state(k, v)
        return self._least_squares_state(k, v)

    def _least_squares_causal_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = []
        for t in range(k.shape[1]):
            state_t = self._oracle_state(k[:, : t + 1], v[:, : t + 1])
            outputs.append(
                torch.einsum("bhf,bhfd->bhd", q[:, t], state_t).unsqueeze(1)
            )
        kv_state = self._oracle_state(k, v)
        if not outputs:
            return v, kv_state
        return torch.cat(outputs, dim=1), kv_state

    def _ridge_causal_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Causal ridge reads via a streaming Sherman-Morrison update.

        This computes the same prefix solution as repeatedly solving
        ``(K_t^T K_t + lambda I)^-1 K_t^T V_t``, but avoids materializing every
        growing prefix during causal training.
        """
        if k.shape[1] == 0:
            return v, self._ridge_state(k, v)

        compute_dtype = (
            torch.float32
            if k.dtype in {torch.float16, torch.bfloat16}
            else k.dtype
        )
        autocast_context = (
            torch.autocast(device_type=k.device.type, enabled=False)
            if k.device.type in {"cpu", "cuda"}
            else nullcontext()
        )
        with autocast_context:
            q_compute = q.to(compute_dtype)
            k_compute = k.to(compute_dtype)
            v_compute = v.to(compute_dtype)

            batch_size, _, num_heads, qk_dim = k.shape
            value_dim = v.shape[-1]
            eye = torch.eye(qk_dim, dtype=compute_dtype, device=k.device)
            precision = eye.expand(batch_size, num_heads, qk_dim, qk_dim).clone()
            precision = precision / self.ridge_lambda
            cross = k.new_zeros(
                batch_size,
                num_heads,
                qk_dim,
                value_dim,
                dtype=compute_dtype,
            )

            outputs = []
            state = cross
            for t in range(k.shape[1]):
                k_t = k_compute[:, t]
                v_t = v_compute[:, t]
                precision_k = torch.einsum("bhfg,bhg->bhf", precision, k_t)
                denom = 1.0 + torch.einsum("bhf,bhf->bh", k_t, precision_k)
                precision = precision - torch.einsum(
                    "bhf,bhg->bhfg", precision_k, precision_k
                ) / denom[..., None, None]
                cross = cross + torch.einsum("bhf,bhd->bhfd", k_t, v_t)
                state = torch.einsum("bhfg,bhgd->bhfd", precision, cross)
                outputs.append(
                    torch.einsum("bhf,bhfd->bhd", q_compute[:, t], state).unsqueeze(1)
                )

        return torch.cat(outputs, dim=1).to(v.dtype), state.to(v.dtype)

    def _noncausal_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        q, k = self._apply_query_key_feature_maps(q, k)
        if self.state_update_rule != "linear":
            kv_state = self._oracle_state(k, v)
            return self._read_from_kv_state(q, kv_state, None), kv_state, None
        # k: (batch_times_features, seq, heads, qk_dim)
        # v: (batch_times_features, seq, heads, v_dim)
        kv_state = torch.einsum("bshf,bshd->bhfd", k, v)
        k_sum = None
        if self.use_k_sum_normalization:
            k_sum = k.sum(dim=1)
        return self._read_from_kv_state(q, kv_state, k_sum), kv_state, k_sum

    def _causal_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        kv_state_prefix: torch.Tensor | None = None,
        k_sum_prefix: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        q, k = self._apply_query_key_feature_maps(q, k)

        if self.state_update_rule != "linear":
            if kv_state_prefix is not None:
                raise ValueError(
                    "Least-squares oracle cannot be continued from a fitted "
                    "memory state alone."
                )
            if self.state_update_rule == "ridge":
                attn, kv_state = self._ridge_causal_attention(q, k, v)
            else:
                attn, kv_state = self._least_squares_causal_attention(q, k, v)
            return attn, kv_state, None

        if q.shape[1] == 0:
            return v, kv_state_prefix, k_sum_prefix

        chunk_size = self.causal_chunk_size
        if chunk_size is None:
            chunk_size = min(q.shape[1], self.DEFAULT_CAUSAL_CHUNK_SIZE)

        outputs = []
        kv_state = kv_state_prefix
        k_sum = k_sum_prefix
        for chunk_start in range(0, q.shape[1], chunk_size):
            chunk_end = min(chunk_start + chunk_size, q.shape[1])
            q_chunk = q[:, chunk_start:chunk_end]
            k_chunk = k[:, chunk_start:chunk_end]
            v_chunk = v[:, chunk_start:chunk_end]

            kv_chunk_raw = torch.cumsum(
                torch.einsum("bshf,bshd->bshfd", k_chunk, v_chunk),
                dim=1,
            ) # outer kv product then cumulative sum over sequence
            if kv_state is not None:
                kv_chunk_raw = kv_chunk_raw + kv_state.unsqueeze(1)
            kv_chunk = self._renormalize_state(kv_chunk_raw)

            if self.use_k_sum_normalization:
                k_chunk_sum = torch.cumsum(k_chunk, dim=1)
                if k_sum is not None:
                    k_chunk_sum = k_chunk_sum + k_sum.unsqueeze(1)
            else:
                k_chunk_sum = None

            num = torch.einsum("bshf,bshfd->bshd", q_chunk, kv_chunk) # cumulative state readout
            if k_chunk_sum is None:
                outputs.append(num)
            else:
                denom = torch.einsum("bshf,bshf->bsh", q_chunk, k_chunk_sum)
                outputs.append(num / (denom.unsqueeze(-1) + self.eps))
            kv_state = kv_chunk_raw[:, -1]
            k_sum = None if k_chunk_sum is None else k_chunk_sum[:, -1]

        return torch.cat(outputs, dim=1), kv_state, k_sum

    def _train_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        causal: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if causal:
            attn, kv_state, k_sum = self._causal_attention(q, k, v)
            assert kv_state is not None
            return attn, kv_state, k_sum
        return self._noncausal_attention(q, k, v)

    def _split_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        single_eval_pos: int,
    ) -> torch.Tensor:
        assert not (self.causal and self.training), (
            "split_attention should not be used in full causal training"
        )
        attn_train, kv_state, k_sum = self._train_attention(
            q[:, :single_eval_pos],
            k[:, :single_eval_pos],
            v[:, :single_eval_pos],
            causal=self.causal_train_only or self.causal,
        )
        q_test = self._apply_feature_map(
            q[:, single_eval_pos:],
            self.feature_map_q,
            normalize_sum=self.normalize_q_sum,
        )
        q_test = self._scale_queries(q_test)
        attn_test = self._read_from_kv_state(
            q_test,
            kv_state,
            k_sum,
        )
        return torch.cat([attn_train, attn_test], dim=1)

    def _apply_output(self, x: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, num_features, _ = x.shape
        attn = attn.reshape(
            batch_size,
            num_features,
            seq_len,
            self.num_heads,
            self.head_v_dim,
        )
        attn = self.output_norm(attn)
        attn = attn.reshape(
            batch_size,
            num_features,
            seq_len,
            self.value_dim,
        ).transpose(1, 2)
        x = x + self.out_proj_item(attn)
        return x + self.mlp(self.mlp_norm(x))

    def forward(self, x, *, single_eval_pos: int | None = None, **kwargs):
        assert x.dim() == 4, f"Expected x to have 4 dims, got shape {tuple(x.shape)}."
        _, seq_len, _, _ = x.shape
        assert single_eval_pos is not None, "single_eval_pos must be provided for LinearAttention."
        assert 0 < single_eval_pos < seq_len, (
            f"single_eval_pos must be in the range [1, {seq_len} - 1], got {single_eval_pos}."
        )

        q, k, v = self._project_qkv(x)
        if self.causal and self.training:
            attn, _, _ = self._causal_attention(q, k, v)
        else:
            attn = self._split_attention(q, k, v, single_eval_pos)
        return self._apply_output(x, attn)

    def incontext_fit(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | None]]:
        assert x.dim() == 4, f"Expected x to have 4 dims, got shape {tuple(x.shape)}."

        q, k, v = self._project_qkv(x)
        attn, kv_state, k_sum = self._train_attention(
            q,
            k,
            v,
            causal=self.causal or self.causal_train_only,
        )
        return self._apply_output(x, attn), {"kv_state": kv_state, "k_sum": k_sum}

    def incontext_predict(
        self,
        x: torch.Tensor,
        state: dict[str, torch.Tensor | None],
    ) -> torch.Tensor:
        assert x.dim() == 4, f"Expected x to have 4 dims, got shape {tuple(x.shape)}."

        q, k, v = self._project_qkv(x)
        if self.causal and self.training:
            attn, _, _ = self._causal_attention(
                q,
                k,
                v,
                kv_state_prefix=state["kv_state"],
                k_sum_prefix=state.get("k_sum"),
            )
        else:
            q = self._apply_feature_map(
                q,
                self.feature_map_q,
                normalize_sum=self.normalize_q_sum,
            )
            q = self._scale_queries(q)
            attn = self._read_from_kv_state(
                q,
                state["kv_state"],
                state.get("k_sum"),
            )
        return self._apply_output(x, attn)

    def empty_trainset_representation_cache(self) -> None:
        return None
