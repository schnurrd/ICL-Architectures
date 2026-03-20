from __future__ import annotations

import math
from functools import partial

import torch

from pfns.model.save_peak_memory import support_save_peak_mem_factor
from torch.utils.checkpoint import checkpoint
from typing_extensions import override

TORCH_VERSION = torch.__version__.split(".")

TORCH_2_ATTENTION_POSSIBLE = int(TORCH_VERSION[0]) >= 2


def _gqa_is_supported() -> bool: # function copied from  https://github.com/PriorLabs/TabPFN/blob/ac6d961b052cdac5288a6b3be071f7e61029ad09/src/tabpfn/architectures/base/attention/full_attention.py#L25
    """Check if PyTorch's scaled_dot_product_attention supports enable_gqa parameter.

    This checks whether torch.nn.functional.scaled_dot_product_attention has a
    kwarg enable_gqa and if we have sufficient NVIDIA compute capability.
    PyTorch 2.5+ includes enable_gqa support.
    """
    if not TORCH_2_ATTENTION_POSSIBLE or not torch.cuda.is_available():
        return False

    # Check if PyTorch version is 2.5 or higher for enable_gqa support
    torch_major, torch_minor = int(TORCH_VERSION[0]), int(TORCH_VERSION[1])
    has_enable_gqa = torch_major > 2 or (torch_major == 2 and torch_minor >= 5)

    if not has_enable_gqa:
        return False

    # We need compute capability >= 8.0 for efficient GQA
    device = torch.cuda.current_device()
    nvidia_compute_capability = torch.cuda.get_device_capability(device)
    return nvidia_compute_capability[0] >= 8


USE_TORCH_2_GQA = _gqa_is_supported()


def _should_run_bf16_sdpa_in_fp16(x: torch.Tensor) -> bool:
    return (
        x.is_cuda
        and x.dtype == torch.bfloat16
        and torch.cuda.get_device_capability(x.device)[0] < 8
    )


class MultiHeadAttention(torch.nn.Module):
    """
    An implementation of multi-head attention, heavily relying on the pytorch
    implementation of multi-head attention, but with some modeifications, namely:
        - simple multiquery attention for the test set
        - support caching
        - flash attention implementation from the flash_attn library, if installed
        - memory saving, by sequentially working off the sequences (seq. in batch dim.)
    """
    _input_size: int
    _output_size: int
    _nhead: int
    _nhead_kv: int
    _d_k: int
    _d_v: int
    _share_kv_across_n_heads: int
    dropout_p: float | None
    softmax_scale: float | None
    _k_cache: torch.Tensor | None
    _v_cache: torch.Tensor | None
    _kv_cache: torch.Tensor | None
    _w_q: torch.nn.Parameter | None
    _w_k: torch.nn.Parameter | None
    _w_v: torch.nn.Parameter | None
    _w_kv: torch.nn.Parameter | None
    _w_qkv: torch.nn.Parameter | None
    _w_out: torch.nn.Parameter
    _use_rope: bool
    _rope_base: float
    inv_freq: torch.Tensor

    @property
    def w_q(self) -> torch.nn.Parameter | None:
        return self._w_q

    @property
    def w_k(self) -> torch.nn.Parameter | None:
        return self._w_k

    @property
    def w_v(self) -> torch.nn.Parameter | None:
        return self._w_v

    @property
    def w_qkv(self) -> torch.nn.Parameter | None:
        return self._w_qkv

    @property
    def w_kv(self) -> torch.nn.Parameter | None:
        return self._w_kv

    @property
    def w_out(self) -> torch.nn.Parameter:
        return self._w_out

    @property
    def has_cached_kv(self) -> bool:
        assert (self._k_cache is None) == (self._v_cache is None)
        assert self._kv_cache is None or (
            self._k_cache is None and self._v_cache is None
        )
        return (
            self._k_cache is not None and self._v_cache is not None
        ) or self._kv_cache is not None

    def empty_kv_cache(self) -> None:
        self._k_cache = None
        self._v_cache = None
        self._kv_cache = None

    def set_parameters(
        self,
        w_out: torch.nn.Parameter,
        w_q: torch.nn.Parameter | None = None,
        w_k: torch.nn.Parameter | None = None,
        w_v: torch.nn.Parameter | None = None,
        w_kv: torch.nn.Parameter | None = None,
        w_qkv: torch.nn.Parameter | None = None,
        precomputed_k: torch.Tensor | None = None,
        precomputed_v: torch.Tensor | None = None,
        precomputed_kv: torch.Tensor | None = None,
    ) -> None:
        assert (precomputed_k is None) == (precomputed_v is None)
        assert (precomputed_kv is None) or (precomputed_k is None)
        assert (precomputed_kv is None and precomputed_k is None) != (
            w_qkv is None and w_kv is None and w_k is None and w_v is None
        )
        assert (w_qkv is None) != (w_q is None)
        assert (w_qkv is None) or (w_kv is None and w_k is None and w_v is None)
        assert w_kv is None or (w_k is None and w_v is None)
        assert (w_k is None) == (w_v is None)

        def assert_tensor_shape(
            tensor: torch.Tensor | None,
            expected_shape: list[int | None],
        ) -> None:
            if tensor is None:
                return
            actual_shape = tensor.size()
            err = f"Tensor {actual_shape=} does not match {expected_shape=}."
            assert len(actual_shape) == len(expected_shape), err
            for actual_dim, expected_dim in zip(actual_shape, expected_shape):
                if expected_dim is not None:
                    assert actual_dim == expected_dim, err

        assert_tensor_shape(precomputed_k, [None, None, self._nhead_kv, self._d_k])
        assert_tensor_shape(precomputed_v, [None, None, self._nhead_kv, self._d_v])
        assert_tensor_shape(precomputed_kv, [None, None, 2, self._nhead_kv, self._d_k])
        assert_tensor_shape(
            w_q,
            [
                self._nhead,
                self._d_k,
                self._input_size,
            ],
        )
        assert_tensor_shape(w_k, [self._nhead_kv, self._d_k, self._input_size])
        assert_tensor_shape(w_v, [self._nhead_kv, self._d_v, self._input_size])
        assert_tensor_shape(w_kv, [2, self._nhead_kv, self._d_k, self._input_size])
        assert_tensor_shape(w_qkv, [3, self._nhead, self._d_k, self._input_size])
        assert_tensor_shape(w_out, [self._nhead, self._d_v, self._output_size])

        self.register_parameter("_w_out", w_out)
        self.register_parameter("_w_q", w_q)
        self.register_parameter("_w_k", w_k)
        self.register_parameter("_w_v", w_v)
        self.register_parameter("_w_kv", w_kv)
        self.register_parameter("_w_qkv", w_qkv)

        self.register_buffer("_k_cache", precomputed_k)
        self.register_buffer("_v_cache", precomputed_v)
        self.register_buffer("_kv_cache", precomputed_kv)

    def newly_initialized_input_weight(
        self,
        dims: list[int],
        nhead: int,
        device: torch.device | None,
        dtype: torch.dtype | None,
    ) -> torch.nn.Parameter:
        assert 3 <= len(dims) <= 4  # ([stack,] nhead_, d, input_size)
        w = torch.nn.Parameter(torch.empty(*dims, device=device, dtype=dtype))
        d, input_size = dims[-2:]
        std = math.sqrt(2.0 / float(nhead * d + input_size)) * self.init_gain
        a = math.sqrt(3.0) * std
        torch.nn.init.uniform_(w, -a, a)
        return w

    def __init__(  # noqa: PLR0913
        self,
        *,
        input_size: int,
        output_size: int,
        d_k: int,
        d_v: int,
        nhead: int,
        device: torch.device | None,
        dtype: torch.dtype | None,
        share_kv_across_n_heads: int = 1,
        dropout_p: float | None = None,
        softmax_scale: float | None = None,
        initialize_output_to_zero: bool = True, # prior labs set this to False
        precomputed_k: torch.Tensor | None = None,
        precomputed_v: torch.Tensor | None = None,
        precomputed_kv: torch.Tensor | None = None,
        recompute: bool = False,
        init_gain: float = 1.0,
        use_rope: bool = False,
        rope_base: float = 128_000.0,
    ):
        super().__init__()
        assert nhead % share_kv_across_n_heads == 0
        self._input_size = input_size
        self._output_size = output_size
        self._d_k = d_k
        self._d_v = d_v
        self._nhead = nhead
        self._nhead_kv = nhead // share_kv_across_n_heads
        self._device = device
        self._dtype = dtype
        self.dropout_p = dropout_p
        self.softmax_scale = softmax_scale
        self.recompute = recompute
        self.init_gain = init_gain
        self._use_rope = use_rope
        self._rope_base = rope_base
        if self._use_rope:
            if self._d_k % 2 != 0:
                raise ValueError(
                    f"RoPE requires even d_k, got d_k={self._d_k}."
                )
            inv_freq = 1.0 / (
                self._rope_base
                ** (
                    torch.arange(
                        0,
                        self._d_k,
                        2,
                        device=device,
                        dtype=torch.float32,
                    )
                    / self._d_k
                )
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        w_out = torch.nn.Parameter(
            torch.empty(nhead, d_v, output_size, device=device, dtype=dtype),
        )
        if initialize_output_to_zero:
            torch.nn.init.zeros_(w_out)
        else:
            torch.nn.init.xavier_uniform_(w_out)

        assert (precomputed_k is None) == (precomputed_v is None)
        has_precomputed_kv = precomputed_kv is not None or precomputed_k is not None
        w_q = None
        w_k = None
        w_v = None
        w_kv = None
        w_qkv = None
        if d_k == d_v and self._nhead == self._nhead_kv and not has_precomputed_kv:
            w_qkv = self.newly_initialized_input_weight(
                [3, self._nhead, self._d_k, self._input_size],
                nhead=self._nhead,
                device=device,
                dtype=dtype,
            )
        else:
            w_q = self.newly_initialized_input_weight(
                [
                    self._nhead,
                    self._d_k,
                    self._input_size,
                ],
                nhead=self._nhead,
                device=device,
                dtype=dtype,
            )
            if not has_precomputed_kv:
                if d_k == d_v:
                    w_kv = self.newly_initialized_input_weight(
                        [2, self._nhead_kv, self._d_k, self._input_size],
                        nhead=self._nhead,
                        device=device,
                        dtype=dtype,
                    )
                else:
                    w_k = self.newly_initialized_input_weight(
                        [self._nhead_kv, self._d_k, self._input_size],
                        nhead=self._nhead,
                        device=device,
                        dtype=dtype,
                    )
                    w_v = self.newly_initialized_input_weight(
                        [self._nhead_kv, self._d_v, self._input_size],
                        nhead=self._nhead,
                        device=device,
                        dtype=dtype,
                    )
        self.set_parameters(
            w_out,
            w_q,
            w_k,
            w_v,
            w_kv,
            w_qkv,
            precomputed_k,
            precomputed_v,
            precomputed_kv,
        )
        if recompute:
            self.forward = partial(checkpoint, self.forward, use_reentrant=False)  # type: ignore

    @override
    def forward(
        self,
        x: torch.Tensor,
        x_kv: torch.Tensor | None = None,
        *,
        cache_kv: bool = False,
        add_input: bool = False,
        # Indicates that 'x' is not used after the call and its buffer can be reused
        # for the output. The operation is not guaranteed to be inplace.
        allow_inplace: bool = False,
        # This requires 'add_input' and 'allow_inplace'. See the documentation of
        # the decorator 'support_save_peak_mem_factor' for details.
        save_peak_mem_factor: int | None = None,
        reuse_first_head_kv: bool = False,
        only_cache_first_head_kv: bool = False,
        use_cached_kv: bool = False,
        attn_mask: torch.Tensor | None = None,
        is_causal: bool = False,
        q_position_offset: int | None = None,
        k_position_offset: int = 0,
        rope_pairwise_positions: bool = False,
        mask_name: str | None = None,
        eval_pos: int | None = None,
    ):
        """X is the current hidden and has a shape of [batch, ..., seq_len, input_size].
        If keys and values are present in the cache and 'freeze_kv' is not set, they
        are obtained from there and 'x_kv' has to be None.
        Else, if 'x_kv' is not None, keys and values are obtained by applying the
        respective linear transformations to 'x_kv'.
        Else, keys and values are attained by applying the respective linear
        transformations to 'x' (self attention).
        """
        assert not (
            cache_kv and use_cached_kv
        ), "Cannot cache and use cached keys and values at the same time."
        assert (
            not x.requires_grad or not cache_kv
        ), "Saving keys and values will stop gradients to flow into trainset."

        # Flatten the batch dimensions
        x, x_kv, x_shape_after_transpose = self._rearrange_inputs_to_flat_batch(x, x_kv)

        nhead_kv = 1 if reuse_first_head_kv else self._nhead_kv

        if cache_kv:
            # Reset cache first so memory is freed before new cache is allocated.
            self._k_cache = self._v_cache = self._kv_cache = None

            if x_kv is not None:
                batch_size, seqlen_kv = x_kv.shape[:2]
            else:
                batch_size, seqlen_kv = x.shape[:2]

            # TODO: handling of device and dtype.
            if self._w_kv is not None or self._w_qkv is not None:
                self._kv_cache = torch.empty(
                    batch_size,
                    seqlen_kv,
                    2,
                    1 if only_cache_first_head_kv else nhead_kv,
                    self._d_k,
                    device=x.device,
                    dtype=x.dtype,
                )
            else:
                self._k_cache = torch.empty(
                    batch_size,
                    seqlen_kv,
                    nhead_kv,
                    self._d_k,
                    device=x.device,
                    dtype=x.dtype,
                )
                self._v_cache = torch.empty(
                    batch_size,
                    seqlen_kv,
                    nhead_kv,
                    self._d_v,
                    device=x.device,
                    dtype=x.dtype,
                )

        output: torch.Tensor = self._compute(
            x,
            x_kv,
            self._k_cache,
            self._v_cache,
            self._kv_cache,
            cache_kv=cache_kv,
            use_cached_kv=use_cached_kv,
            add_input=add_input,
            allow_inplace=allow_inplace,
            save_peak_mem_factor=save_peak_mem_factor,
            reuse_first_head_kv=reuse_first_head_kv,
            attn_mask=attn_mask,
            is_causal=is_causal,
            q_position_offset=q_position_offset,
            k_position_offset=k_position_offset,
            rope_pairwise_positions=rope_pairwise_positions,
            mask_name=mask_name,
            eval_pos=eval_pos,
        )
        return output.reshape(x_shape_after_transpose[:-1] + output.shape[-1:])

    def compute_qkv(  # noqa: PLR0912, C901
        self,
        x: torch.Tensor,
        x_kv: torch.Tensor | None,
        k_cache: torch.Tensor | None,
        v_cache: torch.Tensor | None,
        kv_cache: torch.Tensor | None,
        *,
        cache_kv: bool,
        use_cached_kv: bool,
        reuse_first_head_kv: bool,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        assert not (
            cache_kv and use_cached_kv
        ), "You cannot both cache new KV and use the cached KV at once."
        if reuse_first_head_kv:
            assert x is not x_kv, (
                "x and x_kv must be different tensors. That means reuse_first_head_kv"
                "is not compatible with self attention only cross attention."
            )
        if x_kv is None:
            x_kv = x

        k = v = kv = None
        if use_cached_kv:
            assert (
                self.has_cached_kv
            ), "You try to use cached keys and values but the cache is empty."
            k = k_cache
            v = v_cache
            kv = kv_cache

        assert (k is None) == (v is None)

        if self._w_qkv is None:
            w_q, w_kv = self._w_q, self._w_kv
        else:
            w_q, w_kv = self._w_qkv[0], self._w_qkv[1:]

        if (
            self._w_qkv is not None
            and x is x_kv
            and kv is None
            and k is None
            and v is None
        ):
            qkv = torch.einsum("... s, j h d s -> ... j h d", x, self._w_qkv)
            q = None
        else:
            qkv = None
            q = torch.einsum("... s, h d s -> ... h d", x, w_q)

        if kv is None and k is None and v is None and qkv is None:
            if w_kv is not None:
                if reuse_first_head_kv:
                    orig_num_heads = w_kv.shape[1]
                    w_kv = w_kv[:, :1]
                kv = torch.einsum("... s, j h d s -> ... j h d", x_kv, w_kv)
                if reuse_first_head_kv:
                    expand_shape = [-1 for _ in kv.shape]
                    expand_shape[-2] = orig_num_heads
                    kv = kv.expand(*expand_shape)
            else:
                w_k = self._w_k
                w_v = self._w_v
                if reuse_first_head_kv:
                    orig_num_heads = w_k.shape[0]
                    w_k = w_k[:1]
                    w_v = w_v[:1]
                k = torch.einsum("... s, h d s -> ... h d", x_kv, w_k)
                v = torch.einsum("... s, h d s -> ... h d", x_kv, w_v)
                if reuse_first_head_kv:
                    expand_shape = [-1 for _ in k.shape]
                    expand_shape[-2] = orig_num_heads
                    k = k.expand(*expand_shape)
                    v = v.expand(*expand_shape)

        if cache_kv:
            kv_for_cache = kv
            if kv_cache is not None and kv_for_cache is None:
                # Fast self-attention path computes a stacked qkv tensor directly.
                assert qkv is not None, "Expected qkv when caching stacked KV projections."
                kv_for_cache = qkv[..., 1:, :, :]
            if k_cache is not None:
                k_cache[:] = k
            if v_cache is not None:
                v_cache[:] = v
            if kv_cache is not None:
                if kv_cache.shape[-2] == 1:
                    # we are in the case where only the first head kv is cached
                    # that is the case when we only neeed that for inference
                    kv_cache[:] = kv_for_cache[..., :1, :]
                else:
                    kv_cache[:] = kv_for_cache

        return q, k, v, kv, qkv

    @support_save_peak_mem_factor  # type: ignore
    def _compute(
        self,
        x: torch.Tensor,
        x_kv: torch.Tensor | None,
        k_cache: torch.Tensor | None,
        v_cache: torch.Tensor | None,
        kv_cache: torch.Tensor | None,
        *,
        cache_kv: bool,
        use_cached_kv: bool,
        reuse_first_head_kv: bool,
        attn_mask: torch.Tensor | None,
        is_causal: bool,
        q_position_offset: int | None,
        k_position_offset: int,
        rope_pairwise_positions: bool,
        mask_name: str | None,
        eval_pos: int | None,
    ) -> torch.Tensor:
        """Attention computation.
        Called by 'forward', potentially on shards, once shapes have been normalized.
        """
        q, k, v, kv, qkv = self.compute_qkv(
            x,
            x_kv,
            k_cache,
            v_cache,
            kv_cache,
            cache_kv=cache_kv,
            use_cached_kv=use_cached_kv,
            reuse_first_head_kv=reuse_first_head_kv,
        )
        if self._use_rope:
            if q_position_offset is None:
                q_position_offset = (
                    MultiHeadAttention._cached_kv_len(k_cache, v_cache, kv_cache)
                    if use_cached_kv
                    else 0
                )
            if qkv is not None:
                q, k, v = qkv.unbind(dim=-3)
                q = self._apply_rope(
                    q,
                    position_offset=q_position_offset,
                    rope_pairwise_positions=rope_pairwise_positions,
                    mask_name=mask_name,
                    eval_pos=eval_pos,
                    use_cached_kv=use_cached_kv,
                )
                k = self._apply_rope( # keys are already cut to training part up to eval_pos so we don't need to worry about fixing rope positions
                    k,
                    position_offset=k_position_offset,
                    rope_pairwise_positions=rope_pairwise_positions,
                )
                qkv = None
            else:
                assert q is not None
                q = self._apply_rope(
                    q,
                    position_offset=q_position_offset,
                    rope_pairwise_positions=rope_pairwise_positions,
                    mask_name=mask_name,
                    eval_pos=eval_pos,
                    use_cached_kv=use_cached_kv,
                )
                if kv is not None:
                    k_unbound, v_unbound = kv.unbind(dim=-3)
                    k_unbound = self._apply_rope(
                        k_unbound,
                        position_offset=k_position_offset,
                        rope_pairwise_positions=rope_pairwise_positions,
                    )
                    kv = torch.stack((k_unbound, v_unbound), dim=-3)
                elif k is not None:
                    k = self._apply_rope(
                        k,
                        position_offset=k_position_offset,
                        rope_pairwise_positions=rope_pairwise_positions,
                    )
        attention_head_outputs = MultiHeadAttention.compute_attention_heads(
            q,
            k,
            v,
            kv,
            qkv,
            self.dropout_p,
            self.softmax_scale,
            attn_mask,
            is_causal=is_causal,
        )
        return torch.einsum(
            "... h d, h d s -> ... s",
            attention_head_outputs,
            self._w_out,
        )

    def _rearrange_inputs_to_flat_batch(
        self,
        x: torch.Tensor,
        x_kv: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Size]:
        # TODO: This presumably creates potential memory overhead not captured
        # by save_peak_mem_factor.
        x_shape_after_transpose = x.shape
        if x_kv is not None:
            assert x.shape[:-2] == x_kv.shape[:-2]
        x = x.reshape(-1, *x.shape[-2:])
        if x_kv is not None:
            x_kv = x_kv.reshape(-1, *x_kv.shape[-2:])
        return x, x_kv, x_shape_after_transpose

    @staticmethod
    def _cached_kv_len(
        k_cache: torch.Tensor | None,
        v_cache: torch.Tensor | None,
        kv_cache: torch.Tensor | None,
    ) -> int:
        if kv_cache is not None:
            return int(kv_cache.shape[1])
        if k_cache is not None:
            return int(k_cache.shape[1])
        if v_cache is not None:
            return int(v_cache.shape[1])
        return 0

    def _apply_rope(
        self,
        x: torch.Tensor,
        *,
        position_offset: int = 0,
        rope_pairwise_positions: bool = False,
        mask_name: str | None = None,
        eval_pos: int | None = None,
        use_cached_kv: bool = False,
    ) -> torch.Tensor:
        if x.shape[-1] % 2 != 0:
            raise ValueError(
                f"RoPE requires even hidden dimension, got {x.shape[-1]}."
            )
        seq_len = x.shape[-3]
        half_dim = x.shape[-1] // 2
        t = torch.arange(
            position_offset,
            position_offset + seq_len,
            device=x.device,
            dtype=torch.long,
        )
        if mask_name is None or mask_name not in {"Comb_MT", "Int_MT", "causal_all"}:
            freeze_after = int(eval_pos) if eval_pos is not None and eval_pos > 0 else None
            if freeze_after is None and use_cached_kv:
                freeze_after = int(position_offset) if position_offset > 0 else None
            if freeze_after is not None:
                frozen = t >= freeze_after
                if frozen.any():
                    t[frozen] = freeze_after
        if rope_pairwise_positions:
            t = torch.div(t, 2, rounding_mode="floor")
        t = t.to(dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        cos = freqs.cos().to(dtype=x.dtype).view(1, seq_len, 1, half_dim)
        sin = freqs.sin().to(dtype=x.dtype).view(1, seq_len, 1, half_dim)

        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        return torch.stack(
            (
                x_even * cos - x_odd * sin,
                x_even * sin + x_odd * cos,
            ),
            dim=-1,
        ).flatten(-2)

    @staticmethod
    def broadcast_kv_across_heads(
        kv: torch.Tensor,
        share_kv_across_n_heads: int,
    ) -> torch.Tensor:
        if share_kv_across_n_heads == 1:
            return kv

        nhead, d = kv.shape[-2:]
        kv = kv[..., None, :].expand(
            *([-1] * (kv.dim() - 1)),
            share_kv_across_n_heads,
            -1,
        )
        return kv.reshape(*kv.shape[:-3], nhead * share_kv_across_n_heads, d)

    @staticmethod
    def compute_attention_heads(  # noqa: C901, PLR0912
        q: torch.Tensor | None,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        kv: torch.Tensor | None,
        qkv: torch.Tensor | None,
        dropout_p: float | None = None,
        softmax_scale: float | None = None,
        attn_mask: torch.Tensor | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        assert (k is None) == (v is None)
        assert sum([qkv is None, kv is None, k is None and v is None]) == 2
        assert (qkv is None) != (q is None)
        if attn_mask is not None and is_causal:
            raise ValueError("Use either attn_mask or is_causal, not both.")

        if qkv is not None:
            q, k, v = qkv.unbind(dim=-3)
        elif kv is not None:
            k, v = kv.unbind(dim=-3)

        assert q is not None
        assert k is not None
        assert v is not None

        batch_size, seqlen_q, nhead, d_k = q.shape
        _, _seqlen_kv, nhead_kv, d_v = v.shape
        share_kv_across_n_heads = nhead // nhead_kv
        if dropout_p is None:
            dropout_p = 0.0  # TODO: necessary?

        if TORCH_2_ATTENTION_POSSIBLE:
            extra_inputs = {}
            sdpa_dtype = (
                torch.float16 if _should_run_bf16_sdpa_in_fp16(q) else q.dtype
            )
            if softmax_scale is not None:
                extra_inputs["scale"] = (
                    softmax_scale  # defaults to 1/sqrt(d_k) if None or not provided
                )
            if attn_mask is not None:
                if attn_mask.dim() == 2:
                    attn_mask = attn_mask.unsqueeze(0)
                if attn_mask.dim() == 3:
                    if attn_mask.shape[0] == 1 and batch_size > 1:
                        attn_mask = attn_mask.expand(batch_size, -1, -1)
                    attn_mask = attn_mask.unsqueeze(1)
                if attn_mask.dim() != 4:
                    raise ValueError(
                        f"Expected attn_mask to have 2, 3, or 4 dims, got {attn_mask.shape}."
                    )
                attn_mask = attn_mask.to(device=q.device, dtype=sdpa_dtype)
                extra_inputs["attn_mask"] = attn_mask
            if is_causal:
                extra_inputs["is_causal"] = True
            if USE_TORCH_2_GQA:
                extra_inputs["enable_gqa"] = True
            else:
                k = MultiHeadAttention.broadcast_kv_across_heads(
                    k,
                    share_kv_across_n_heads,
                )
                v = MultiHeadAttention.broadcast_kv_across_heads(
                    v,
                    share_kv_across_n_heads,
                )
            attention_head_outputs = torch.nn.functional.scaled_dot_product_attention(
                # transposing just before and keeping the state in the transposed state
                # makes things faster as this function internally transposes the state
                # back and forth
                    q.transpose(1, 2).to(sdpa_dtype),
                    k.transpose(1, 2).to(sdpa_dtype),
                    v.transpose(1, 2).to(sdpa_dtype),
                    dropout_p=dropout_p,
                    **extra_inputs,
            )
            attention_head_outputs = attention_head_outputs.transpose(1, 2).to(q.dtype)
        else:
            k = MultiHeadAttention.broadcast_kv_across_heads(k, share_kv_across_n_heads)
            v = MultiHeadAttention.broadcast_kv_across_heads(v, share_kv_across_n_heads)
            logits = torch.einsum("b q h d, b k h d -> b q k h", q, k)
            logits *= (
                torch.sqrt(torch.tensor(1.0 / d_k)).to(k.device)
                if softmax_scale is None
                else softmax_scale
            )
            if attn_mask is not None:
                if attn_mask.dim() == 2:
                    attn_mask = attn_mask.unsqueeze(0)
                if attn_mask.dim() != 3:
                    raise ValueError(
                        f"Expected attn_mask to have 2 or 3 dims, got {attn_mask.shape}."
                    )
                if attn_mask.shape[0] == 1 and batch_size > 1:
                    attn_mask = attn_mask.expand(batch_size, -1, -1)
                attn_mask = attn_mask.to(device=logits.device, dtype=logits.dtype)
                logits = logits + attn_mask.unsqueeze(-1)
            if is_causal:
                q_idx = torch.arange(seqlen_q, device=logits.device).unsqueeze(1)
                k_idx = torch.arange(_seqlen_kv, device=logits.device).unsqueeze(0)
                causal_mask = k_idx > q_idx
                logits = logits.masked_fill(
                    causal_mask.unsqueeze(0).unsqueeze(-1),
                    float("-inf"),
                )
            ps = torch.softmax(logits, dim=2)
            ps = torch.dropout(ps, dropout_p, train=True)
            attention_head_outputs = torch.einsum("b q k h, b k h d -> b q h d", ps, v)

        return attention_head_outputs.reshape(
            batch_size,
            seqlen_q,
            nhead,
            d_v,
        )

    @staticmethod
    def convert_torch_nn_multihead_attention_state_dict(
        state_dict: dict,
        nhead: int,
        *,
        disable_stacked_w_qkv: bool = False,
    ) -> dict:
        in_proj_weight = state_dict["in_proj_weight"]
        out_proj_weight = state_dict["out_proj.weight"]

        embed_dim = in_proj_weight.shape[1]
        assert embed_dim % nhead == 0
        assert in_proj_weight.shape[0] == 3 * embed_dim
        assert out_proj_weight.shape == (embed_dim, embed_dim)
        in_proj_weight = in_proj_weight.reshape(3, nhead, -1, embed_dim)

        state_dict = {}
        if disable_stacked_w_qkv:
            state_dict["_w_q"], state_dict["_w_kv"] = torch.split(
                in_proj_weight,
                [1, 2],
            )
            state_dict["_w_q"] = state_dict["_w_q"].squeeze(0)
        else:
            state_dict["_w_qkv"] = in_proj_weight
        state_dict["_w_out"] = out_proj_weight.T.reshape(nhead, -1, embed_dim)
        return state_dict
