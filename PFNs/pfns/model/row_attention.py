"""Row (item) attention primitives for the two-axis tabular architecture."""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


class RowAttentionBase(nn.Module, ABC):
    """Shared projection/folding/split machinery for row attention variants."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        *,
        zero_init: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if d_model <= 0 or num_heads <= 0:
            raise ValueError("d_model and num_heads must be > 0.")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        if eps <= 0:
            raise ValueError("eps must be > 0.")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.eps = eps

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        if zero_init:
            nn.init.zeros_(self.out_proj.weight)

    def _project_qkv(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int]]:
        """Fold feature blocks into the batch dim and project to q/k/v.

        Returns q, k, v, the flattened hidden states (for any extra
        per-variant projections), and the original (batch, num_blocks).
        """
        assert x.dim() == 4, f"Expected x to have 4 dims, got shape {tuple(x.shape)}."
        batch_size, seq_len, num_blocks, _ = x.shape
        x_flat = x.transpose(1, 2).reshape(batch_size * num_blocks, seq_len, self.d_model)

        def heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(batch_size * num_blocks, seq_len, self.num_heads, self.head_dim)

        q = heads(self.q_proj(x_flat))
        k = heads(self.k_proj(x_flat))
        v = heads(self.v_proj(x_flat))
        return q, k, v, x_flat, (batch_size, num_blocks)

    def _project_out(
        self, out: torch.Tensor, batch_size: int, num_blocks: int
    ) -> torch.Tensor:
        """Apply the output projection and unfold feature blocks back out.

        The projection runs while the tensor is still contiguous
        `(batch*blocks, seq, d_model)`; unfolding first would hand `nn.Linear`
        a strided view and force a full contiguous copy.
        """
        bb, seq_len = out.shape[0], out.shape[1]
        assert bb == batch_size * num_blocks
        out = self.out_proj(out.reshape(bb, seq_len, self.d_model))
        return out.reshape(batch_size, num_blocks, seq_len, self.d_model).transpose(1, 2)

    @staticmethod
    def _compute_dtype(dtype: torch.dtype) -> torch.dtype:
        return torch.float32 if dtype in {torch.float16, torch.bfloat16} else dtype

    def _empty_prediction(
        self, cached: torch.Tensor, v: torch.Tensor, batch_size: int, num_blocks: int
    ) -> torch.Tensor | None:
        """Validate the cached state; return an empty output for empty input."""
        bb = cached.shape[0]
        if bb != batch_size * num_blocks:
            raise ValueError(
                f"Cached state batch dim {bb} does not match input "
                f"batch*feature_blocks {batch_size * num_blocks}."
            )
        if v.shape[1] == 0:
            empty = v.new_zeros(bb, 0, self.num_heads, self.head_dim)
            return self._project_out(empty, batch_size, num_blocks)
        return None

    def forward(self, x: torch.Tensor, *, single_eval_pos: int | None = None) -> torch.Tensor:
        assert x.dim() == 4, f"Expected x to have 4 dims, got shape {tuple(x.shape)}."
        seq_len = x.shape[1]
        assert single_eval_pos is not None, (
            f"single_eval_pos must be provided for {type(self).__name__}."
        )
        assert 0 <= single_eval_pos <= seq_len, (
            f"single_eval_pos must satisfy 0 <= single_eval_pos <= {seq_len}, "
            f"got {single_eval_pos}."
        )

        out_train, state = self.incontext_fit(x[:, :single_eval_pos])
        out_test = self.incontext_predict(x[:, single_eval_pos:], state)
        return torch.cat([out_train, out_test], dim=1)

    @abstractmethod
    def incontext_fit(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Run the recurrence over the context and return the output + state."""

    @abstractmethod
    def incontext_predict(
        self, x: torch.Tensor, state: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Read test tokens against a fixed cached state."""

    def empty_trainset_representation_cache(self) -> None:
        return None


class DeltaNetRowAttention(RowAttentionBase):
    """Row attention using the DeltaNet delta-rule recurrence.  """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        *,
        qk_norm: bool | str | None = "l2",
        zero_init: bool = True,
        checkpoint_chunk_size: int | None = 64,
        eps: float = 1e-6,
    ) -> None:
        super().__init__(d_model, num_heads, zero_init=zero_init, eps=eps)
        if checkpoint_chunk_size is not None and checkpoint_chunk_size <= 0:
            raise ValueError("checkpoint_chunk_size must be >= 1 or None.")
        self.qk_norm = self._normalize_qk_norm(qk_norm)
        self.checkpoint_chunk_size = checkpoint_chunk_size

        self.beta_proj = nn.Linear(d_model, num_heads)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.constant_(self.beta_proj.bias, 2.0)

    @staticmethod
    def _normalize_qk_norm(qk_norm: bool | str | None) -> str | None:
        if qk_norm is None or qk_norm is False:
            return None
        if qk_norm is True:
            return "l2"
        normalized = str(qk_norm).strip().lower().replace("-", "_")
        if normalized in {"", "none", "false"}:
            return None
        if normalized in {"l2", "l2norm", "l2_norm"}:
            return "l2"
        raise ValueError(
            f"qk_norm must be one of {{None, False, True, none, l2}}, got {qk_norm!r}."
        )

    def _project(self, x: torch.Tensor):
        q, k, v, x_flat, shape = self._project_qkv(x)
        beta = torch.sigmoid(self.beta_proj(x_flat))
        if self.qk_norm == "l2":
            q = F.normalize(q, p=2, dim=-1, eps=self.eps)
            k = F.normalize(k, p=2, dim=-1, eps=self.eps)
        return q, k, v, beta, shape

    def _delta_rule(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        *,
        initial_state: torch.Tensor | None,
        output_final_state: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if q.is_cuda and q.shape[1] > 0:
            from fla.layers.delta_net import fused_recurrent_delta_rule

            return fused_recurrent_delta_rule(
                q=q,
                k=k,
                v=v,
                beta=beta,
                initial_state=initial_state,
                output_final_state=output_final_state,
                use_qk_l2norm_in_kernel=False,
            )
        return self._delta_rule_pytorch(
            q, k, v, beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
        )

    @staticmethod
    def _run_delta_rule_chunk(
        state: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = []
        for t in range(k.shape[1]):
            k_t, v_t, q_t, beta_t = k[:, t], v[:, t], q[:, t], beta[:, t]
            prediction = torch.einsum("bhk,bhkv->bhv", k_t, state)
            error = (v_t - prediction) * beta_t.unsqueeze(-1)
            state = state + torch.einsum("bhk,bhv->bhkv", k_t, error)
            outputs.append(torch.einsum("bhk,bhkv->bhv", q_t, state).unsqueeze(1))

        return torch.cat(outputs, dim=1), state

    def _delta_rule_pytorch(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        *,
        initial_state: torch.Tensor | None,
        output_final_state: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, seq_len, num_heads, head_dim = k.shape
        value_dim = v.shape[-1]
        orig_dtype = v.dtype
        compute_dtype = self._compute_dtype(orig_dtype)

        q = q.to(compute_dtype) * (head_dim**-0.5)
        k = k.to(compute_dtype)
        v = v.to(compute_dtype)
        beta = beta.to(compute_dtype)

        if initial_state is not None:
            state = initial_state.to(compute_dtype)
        else:
            state = v.new_zeros(batch_size, num_heads, head_dim, value_dim)

        if seq_len == 0:
            final_state = state.to(orig_dtype) if output_final_state else None
            return v.new_zeros(batch_size, 0, num_heads, value_dim, dtype=orig_dtype), final_state

        chunk_size = self.checkpoint_chunk_size or seq_len
        use_checkpoint = (
            torch.is_grad_enabled()
            and chunk_size < seq_len
            and (q.requires_grad or k.requires_grad or v.requires_grad or beta.requires_grad)
        )

        outputs = []
        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)
            q_c, k_c, v_c, beta_c = q[:, start:end], k[:, start:end], v[:, start:end], beta[:, start:end]
            if use_checkpoint:
                out_c, state = checkpoint(
                    self._run_delta_rule_chunk, state, q_c, k_c, v_c, beta_c,
                    use_reentrant=False,
                )
            else:
                out_c, state = self._run_delta_rule_chunk(state, q_c, k_c, v_c, beta_c)
            outputs.append(out_c.to(orig_dtype))

        out = torch.cat(outputs, dim=1)
        final_state = state.to(orig_dtype) if output_final_state else None
        return out, final_state

    def incontext_fit(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        q, k, v, beta, (batch_size, num_blocks) = self._project(x)
        out, state = self._delta_rule(q, k, v, beta, initial_state=None, output_final_state=True)
        assert state is not None
        return self._project_out(out, batch_size, num_blocks), {"recurrent_state": state}

    def incontext_predict(self, x: torch.Tensor, state: dict[str, torch.Tensor]) -> torch.Tensor:
        q, k, v, beta, (batch_size, num_blocks) = self._project(x)
        recurrent_state = state["recurrent_state"]
        empty = self._empty_prediction(recurrent_state, v, batch_size, num_blocks)
        if empty is not None:
            return empty
        
        recurrent_state = recurrent_state.to(q.dtype)
        q = q * (self.head_dim**-0.5)
        q_state = torch.einsum("bshk,bhkv->bshv", q, recurrent_state)
        k_state = torch.einsum("bshk,bhkv->bshv", k, recurrent_state)
        error = beta.unsqueeze(-1) * (v - k_state)
        q_dot_k = torch.einsum("bshk,bshk->bsh", q, k)
        out = q_state + q_dot_k.unsqueeze(-1) * error

        return self._project_out(out, batch_size, num_blocks)


class LinearRowAttention(RowAttentionBase):
    """Row attention using causal linear attention."""

    SUPPORTED_FEATURE_MAPS = ("elu", "identity", "relu")

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        *,
        feature_map: str = "elu",
        use_k_sum_normalization: bool = True,
        use_query_scale: bool = True,
        include_self_term: bool = True,
        zero_init: bool = True,
        chunk_size: int | None = 64,
        eps: float = 1e-6,
    ) -> None:
        super().__init__(d_model, num_heads, zero_init=zero_init, eps=eps)
        normalized = feature_map.strip().lower()
        if normalized not in self.SUPPORTED_FEATURE_MAPS:
            raise ValueError(
                f"feature_map must be one of {self.SUPPORTED_FEATURE_MAPS}, got {feature_map!r}."
            )
        if chunk_size is not None and chunk_size <= 0:
            raise ValueError("chunk_size must be >= 1 or None.")
        self.feature_map = normalized
        self.use_k_sum_normalization = bool(use_k_sum_normalization)
        self.use_query_scale = bool(use_query_scale)
        self.include_self_term = bool(include_self_term)
        self.chunk_size = chunk_size

    def _apply_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        if self.feature_map == "elu":
            return F.elu(x) + 1.0
        if self.feature_map == "relu":
            return F.relu(x)
        return x

    def _project(self, x: torch.Tensor):
        q, k, v, _, shape = self._project_qkv(x)
        q = self._apply_feature_map(q)
        k = self._apply_feature_map(k)
        if self.use_query_scale:
            q = q * (self.head_dim**-0.5)
        return q, k, v, shape

    @staticmethod
    def _run_linear_chunk(
        kv_state: torch.Tensor,
        k_sum: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Cumulative outer products within the chunk, offset by the state
        # carried in from previous chunks.
        kv = torch.cumsum(torch.einsum("bshk,bshv->bshkv", k, v), dim=1)
        kv = kv + kv_state.unsqueeze(1)
        num = torch.einsum("bshk,bshkv->bshv", q, kv)
        ks = torch.cumsum(k, dim=1) + k_sum.unsqueeze(1)
        return num, ks, kv[:, -1]

    def _linear_causal(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bb, seq_len, num_heads, head_dim = k.shape
        value_dim = v.shape[-1]
        orig_dtype = v.dtype
        if seq_len == 0:
            return (
                v.new_zeros(bb, 0, num_heads, value_dim),
                v.new_zeros(bb, num_heads, head_dim, value_dim),
                k.new_zeros(bb, num_heads, head_dim),
            )

        if q.is_cuda and seq_len >= num_heads:
            from fla.ops.linear_attn import chunk_linear_attn

            out, kv_state = chunk_linear_attn(
                q=q,
                k=k,
                v=v,
                scale=1.0,
                initial_state=None,
                output_final_state=True,
                normalize=self.use_k_sum_normalization,
            )
            k_sum = (
                k.sum(dim=1)
                if self.use_k_sum_normalization
                else k.new_zeros(bb, num_heads, head_dim)
            )
            return out.to(orig_dtype), kv_state.to(orig_dtype), k_sum.to(orig_dtype)

        # PyTorch fallback (CPU): accumulate in fp32 for stability.
        compute_dtype = self._compute_dtype(orig_dtype)
        q, k, v = q.to(compute_dtype), k.to(compute_dtype), v.to(compute_dtype)
        kv_state = v.new_zeros(bb, num_heads, head_dim, value_dim)
        k_sum = k.new_zeros(bb, num_heads, head_dim)
        chunk = self.chunk_size or seq_len
        use_checkpoint = (
            torch.is_grad_enabled()
            and chunk < seq_len
            and (q.requires_grad or k.requires_grad or v.requires_grad)
        )

        outputs = []
        for start in range(0, seq_len, chunk):
            end = min(start + chunk, seq_len)
            args = (kv_state, k_sum, q[:, start:end], k[:, start:end], v[:, start:end])
            if use_checkpoint:
                num, ks, kv_state = checkpoint(self._run_linear_chunk, *args, use_reentrant=False)
            else:
                num, ks, kv_state = self._run_linear_chunk(*args)
            if self.use_k_sum_normalization:
                denom = torch.einsum("bshk,bshk->bsh", q[:, start:end], ks)
                num = num / (denom.unsqueeze(-1) + self.eps)
            outputs.append(num.to(orig_dtype))
            k_sum = ks[:, -1]

        return torch.cat(outputs, dim=1), kv_state.to(orig_dtype), k_sum.to(orig_dtype)

    def incontext_fit(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        q, k, v, (batch_size, num_blocks) = self._project(x)
        out, kv_state, k_sum = self._linear_causal(q, k, v)
        return (
            self._project_out(out, batch_size, num_blocks),
            {"recurrent_state": kv_state, "k_sum": k_sum},
        )

    def incontext_predict(self, x: torch.Tensor, state: dict[str, torch.Tensor]) -> torch.Tensor:
        q, k, v, (batch_size, num_blocks) = self._project(x)
        kv_state, k_sum = state["recurrent_state"], state["k_sum"]
        empty = self._empty_prediction(kv_state, v, batch_size, num_blocks)
        if empty is not None:
            return empty

        kv_state = kv_state.to(q.dtype)
        num = torch.einsum("bshk,bhkv->bshv", q, kv_state)
        q_dot_k = (
            torch.einsum("bshk,bshk->bsh", q, k) if self.include_self_term else None
        )
        if q_dot_k is not None:
            num = num + q_dot_k.unsqueeze(-1) * v
        if self.use_k_sum_normalization:
            denom = torch.einsum("bshk,bhk->bsh", q, k_sum.to(q.dtype))
            if q_dot_k is not None:
                denom = denom + q_dot_k
            num = num / (denom.unsqueeze(-1) + self.eps)

        return self._project_out(num, batch_size, num_blocks)


ROW_ATTENTION_REGISTRY = {
    "deltanet": DeltaNetRowAttention,
    "linear": LinearRowAttention,
}


def build_row_attention(row_attention: str, **kwargs) -> RowAttentionBase:
    """Instantiate a row-attention primitive by name."""
    try:
        cls = ROW_ATTENTION_REGISTRY[row_attention.strip().lower()]
    except KeyError:
        raise ValueError(
            f"row_attention must be one of {sorted(ROW_ATTENTION_REGISTRY)}, "
            f"got {row_attention!r}."
        ) from None
    return cls(**kwargs)
