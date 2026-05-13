"""Context managers for patching FLA/Gla ops during evaluation."""
from __future__ import annotations

import typing as tp
from contextlib import contextmanager

import torch
import torch.nn.functional as F

from fla.modules.l2norm import l2norm


def _delta_rule_step_decay_is_enabled(power: float) -> bool:
    return abs(float(power)) > 0.0


def _apply_delta_rule_step_decay(
    beta: torch.Tensor,
    *,
    power: float,
    train_length: int | None,
    clamp_max: bool,
    sequence_offset: int = 0,
    position_dim: int = 1,
) -> torch.Tensor:
    """Scale DeltaNet beta by a normalized online-LR decay schedule."""
    if not _delta_rule_step_decay_is_enabled(power):
        return beta

    position_dim = position_dim % beta.ndim
    seq_len = beta.shape[position_dim]
    if seq_len == 0:
        return beta

    reference_length = seq_len if train_length is None else int(train_length)
    if reference_length <= 0:
        raise ValueError("Delta rule step-decay train_length must be positive.")
    if sequence_offset < 0:
        raise ValueError("Delta rule step-decay sequence_offset must be non-negative.")

    positions = torch.arange(
        sequence_offset + 1,
        sequence_offset + seq_len + 1,
        device=beta.device,
        dtype=torch.float32,
    )
    weights = (float(reference_length) / positions).pow(float(power))
    if clamp_max:
        weights = weights.clamp(max=1.0)

    shape = [1] * beta.ndim
    shape[position_dim] = seq_len
    return beta * weights.to(dtype=beta.dtype).view(shape)


@contextmanager
def _maybe_patch_shortconv_forward_pytorch(enabled: bool):
    if not enabled:
        yield
        return
    try:
        import fla.modules.convolution as conv_module
    except Exception:
        yield
        return

    if not hasattr(conv_module, "ShortConvolution"):
        yield
        return

    original_forward = conv_module.ShortConvolution.forward

    def _forward_pytorch(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        cache: torch.Tensor | None = None,
        output_final_state: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        chunk_indices: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Copied from FLA ShortConvolution forward with modifications for stateless processing in PyTorch. 
        Only supports decode-like 1 step per sequence path.
        Args:
            x (`torch.Tensor`):
                Tensor of shape `[B, T, D]`. `B` must be 1 if `cu_seqlens` is provided.
            residual (`Optional[torch.Tensor]`):
                Residual tensor of shape `[B, T, D]`. Default: `None`.
            mask (`Optional[torch.Tensor]`):
                Attention mask dealing with padded positions.
            cache (`Optional[torch.Tensor]`):
                Previous cache tensor of shape `[N, D, W]`, where `W` is the kernel size.
                If provided, the cache is updated **inplace**.
            output_final_state (Optional[bool]):
                Whether to output the final state of shape `[N, D, W]`. Default: `False`.
            cu_seqlens (Optional[torch.LongTensor]):
                Cumulative sequence lengths for each batch. Used for varlen. Default: `None`.
                Shape: [B+1]
            chunk_indices (Optional[torch.LongTensor]):
                Chunk indices for variable-length sequences. Default: `None`.

        Returns:
            Tensor of shape `[B, T, D]`.
        """
        assert chunk_indices is None, "chunk_indices not supported in pytorch ShortConvolution patch."
        assert output_final_state is False, "output_final_state must be False in pytorch ShortConvolution patch."
        assert cu_seqlens is None, "cu_seqlens must be None in pytorch ShortConvolution patch."
        assert mask is None, "mask must be None in pytorch ShortConvolution patch."

        assert cache is not None, "cache must be provided in pytorch ShortConvolution patch."

        x = x.contiguous()
        cache = cache.contiguous()
        if residual is not None:
            residual = residual.contiguous()

        B, T, D = x.shape

        assert T == 1, "We only support decode-like path in pytorch ShortConvolution patch."

        weight = self.weight.squeeze(1)
        W = weight.shape[1]

        cache_batch = cache.shape[0]
        assert B % cache_batch == 0, "B must be divisible by cache_batch."
        flat_len = B // cache_batch

        x3d = x.view(cache_batch, flat_len, D).float()
        weight_f = weight.float()

        if W > 1:
            cache_f = cache.float()
            cache_window = cache_f[:, :, 1:].unsqueeze(1).expand(cache_batch, flat_len, D, W - 1)
            cache_shifted = torch.cat([cache_window, x3d.unsqueeze(-1)], dim=-1)
        else:
            cache_shifted = x3d.unsqueeze(-1)

        y3d = (cache_shifted * weight_f.view(1, 1, D, W)).sum(dim=-1)

        if self.bias is not None:
            y3d = y3d + self.bias.float().view(1, 1, D)
        if self.activation in ("silu", "swish"):
            y3d = F.silu(y3d)
        if residual is not None:
            y3d = y3d + residual.reshape(cache_batch, flat_len, D).float()

        y = y3d.to(x.dtype).reshape(B, T, D)
        
        # Stateless path: expand cache to match flattened batch and delegate to original forward.
        # cache_for_real = cache.repeat_interleave(flat_len, dim=0)
        # residual_for_real = residual if residual is not None else None
        # y_real, cache_real = original_forward(
        #     self,
        #     x,
        #     residual=residual_for_real,
        #     mask=mask,
        #     cache=cache_for_real,
        #     output_final_state=output_final_state,
        #     cu_seqlens=cu_seqlens,
        #     chunk_indices=chunk_indices,
        #     **kwargs,
        # )
        # return y_real, cache_real
        return y, cache

    conv_module.ShortConvolution.forward = _forward_pytorch
    try:
        yield
    finally:
        conv_module.ShortConvolution.forward = original_forward


@contextmanager
def _maybe_patch_gla_with_stateless_recurrent(
    enabled: bool,
    *,
    include_self_term: bool = True,
):
    if not enabled:
        yield
        return
    import fla.layers.gla as gla_layer

    @torch.compiler.disable
    def _stateless_gla_kernel(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        gk: torch.Tensor | None = None,
        g: torch.Tensor | None = None,
        gv: torch.Tensor | None = None,
        scale: int | None = None,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        reverse: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        **kwargs: tp.Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert not kwargs, f"Unsupported extra args: {sorted(kwargs)}"
        assert gv is None, "naive_recurrent_gla does not support gv."
        assert not reverse, "naive_recurrent_gla does not support reverse processing."
        assert cu_seqlens is None, "naive_recurrent_gla does not support cu_seqlens."
        assert output_final_state is False, (
            "output_final_state must be False in stateless_gla patch."
        )
        assert q.shape[1] == 1, "stateless_gla patch only supports decode-like T=1."
        if gk is None:
            gk = g
        assert gk is not None, "gk is required for naive_recurrent_gla."
        assert initial_state is not None, "stateless mode requires an initial_state."
        if scale is None:
            scale = k.shape[-1] ** -0.5

        dtype = q.dtype
        q, k, v, gk = (t.float() for t in (q, k, v, gk))
        h0 = initial_state.float()
        orig_batch = q.shape[0]
        cache_batch = h0.shape[0]
        if orig_batch != cache_batch:
            assert orig_batch % cache_batch == 0, "orig_batch must be divisible by cache_batch."
            flat_len = orig_batch // cache_batch
            q = q.reshape(cache_batch, flat_len, *q.shape[2:])
            k = k.reshape(cache_batch, flat_len, *k.shape[2:])
            v = v.reshape(cache_batch, flat_len, *v.shape[2:])
            gk = gk.reshape(cache_batch, flat_len, *gk.shape[2:])
        
        q = q * scale
        qg = q * gk.exp()
        term1 = torch.einsum("bthk,bhkv->bthv", qg, h0)
        o = term1
        if include_self_term:
            qk = (q * k).sum(-1, keepdim=True)
            o = o + qk * v

        if orig_batch != cache_batch:
            o = o.reshape(orig_batch, 1, *o.shape[2:])
        return o.to(dtype), None

    original_fused_recurrent = gla_layer.fused_recurrent_gla
    original_chunked = gla_layer.chunk_gla
    original_fused_chunked = gla_layer.fused_chunk_gla
    gla_layer.fused_recurrent_gla = _stateless_gla_kernel
    gla_layer.fused_chunk_gla = _stateless_gla_kernel
    gla_layer.chunk_gla = _stateless_gla_kernel
    try:
        yield
    finally:
        gla_layer.fused_recurrent_gla = original_fused_recurrent
        gla_layer.chunk_gla = original_chunked
        gla_layer.fused_chunk_gla = original_fused_chunked


@contextmanager
def _maybe_patch_kda_with_stateless_recurrent(
    enabled: bool,
    *,
    include_self_term: bool = True,
):
    if not enabled:
        yield
        return
    import fla.layers.kda as kda_layer
    import torch.nn.functional as F

    @torch.compiler.disable
    def _stateless_kda_kernel(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor | None = None,
        beta: torch.Tensor | None = None,
        scale: float | None = None,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        reverse: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        use_qk_l2norm_in_kernel: bool = False,
        use_gate_in_kernel: bool = False,
        A_log: torch.Tensor | None = None,
        dt_bias: torch.Tensor | None = None,
        **kwargs: tp.Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert not kwargs, f"Unsupported extra args: {sorted(kwargs)}"
        assert g is not None, "g is required for stateless_kda."
        assert beta is not None, "beta is required for stateless_kda."
        assert initial_state is not None, "stateless mode requires an initial_state."
        assert not reverse, "stateless_kda does not support reverse processing."
        assert cu_seqlens is None, "stateless_kda does not support cu_seqlens."
        assert output_final_state is False, (
            "output_final_state must be False in stateless_kda patch."
        )
        assert q.shape[1] == 1, "stateless_kda patch only supports decode-like T=1."

        scale = k.shape[-1] ** -0.5 if scale is None else scale
        dtype = q.dtype
        
        q, k, v, g, beta = (t.float() for t in (q, k, v, g, beta))
        s0 = initial_state.float() # (B_cache, H, K, V)

        orig_batch = q.shape[0]
        cache_batch = s0.shape[0]

        if orig_batch != cache_batch:
            assert orig_batch % cache_batch == 0, "orig_batch must be divisible by cache_batch."
            flat_len = orig_batch // cache_batch
            
            T = q.shape[1]
            q = q.reshape(cache_batch, flat_len, T, *q.shape[2:])
            k = k.reshape(cache_batch, flat_len, T, *k.shape[2:])
            v = v.reshape(cache_batch, flat_len, T, *v.shape[2:])
            g = g.reshape(cache_batch, flat_len, T, *g.shape[2:])
            beta = beta.reshape(cache_batch, flat_len, T, *beta.shape[2:])

        # Use FLA's l2norm to match the original kernel's normalization and don't fail tests
        if use_qk_l2norm_in_kernel:
            q = l2norm(q)
            k = l2norm(k)

        q = q * scale

        if use_gate_in_kernel and A_log is not None:
            if dt_bias is not None:
                 H, K = q.shape[-2], q.shape[-1]
                 if dt_bias.ndim == 1 and dt_bias.numel() == H * K:
                     dt_bias = dt_bias.view(H, K)
            
            A = A_log.exp().view(1, 1, -1, 1) # (1, 1, H, 1)
            bias = dt_bias.view(1, 1, *dt_bias.shape) if dt_bias is not None else 0.0
            
            if g.ndim == 5:
                A = A.unsqueeze(0)
                if isinstance(bias, torch.Tensor):
                    bias = bias.unsqueeze(0)

            g = -A * F.softplus(g + bias)

        g_exp = g.exp()
        if g_exp.ndim == q.ndim - 1:
            g_exp = g_exp.unsqueeze(-1)
        q_decayed = q * g_exp
        
        if q_decayed.ndim == 5: # (B, L, T, H, K)
             o_base = torch.einsum("blthk,bhkv->blthv", q_decayed, s0)
        else: # (B, T, H, K)
             o_base = torch.einsum("bthk,bhkv->bthv", q_decayed, s0)

        k_decayed = k * g_exp
        if k_decayed.ndim == 5:
            k_s0 = torch.einsum("blthk,bhkv->blthv", k_decayed, s0)
        else:
            k_s0 = torch.einsum("bthk,bhkv->bthv", k_decayed, s0)
            
        o = o_base
        if include_self_term:
            delta = v - k_s0
            qk_dot = (q * k).sum(dim=-1, keepdim=True) # (..., H, 1)
            scaling = beta.unsqueeze(-1) * qk_dot      # (..., H, 1)
            o = o + delta * scaling

        if orig_batch != cache_batch:
            o = o.reshape(orig_batch, *o.shape[2:])

        return o.to(dtype), None

    original_fused_recurrent_kda = kda_layer.fused_recurrent_kda
    original_chunk_kda = kda_layer.chunk_kda
    kda_layer.fused_recurrent_kda = _stateless_kda_kernel
    kda_layer.chunk_kda = _stateless_kda_kernel
    try:
        yield
    finally:
        kda_layer.fused_recurrent_kda = original_fused_recurrent_kda
        kda_layer.chunk_kda = original_chunk_kda

@contextmanager
def _maybe_patch_deltanet_with_stateless_recurrent(
    enabled: bool,
    *,
    include_self_term: bool = True,
    step_decay_power: float = 0.0,
    step_decay_train_length: int | None = None,
    step_decay_clamp_max: bool = True,
    sequence_offset: int = 0,
):
    step_decay_enabled = _delta_rule_step_decay_is_enabled(step_decay_power)
    if not enabled and not step_decay_enabled:
        yield
        return
    import fla.layers.delta_net as deltanet_layer

    original_fused_recurrent_delta_rule = deltanet_layer.fused_recurrent_delta_rule
    original_chunk_delta_rule = deltanet_layer.chunk_delta_rule

    def _decay_beta(beta: torch.Tensor, *, position_dim: int) -> torch.Tensor:
        return _apply_delta_rule_step_decay(
            beta,
            power=step_decay_power,
            train_length=step_decay_train_length,
            clamp_max=step_decay_clamp_max,
            sequence_offset=sequence_offset,
            position_dim=position_dim,
        )

    def _wrap_delta_rule_kernel(original_kernel: tp.Callable[..., tp.Any]):
        @torch.compiler.disable
        def _kernel_with_step_decay(*args: tp.Any, **kwargs: tp.Any) -> tp.Any:
            if "beta" in kwargs:
                beta = kwargs["beta"]
                if beta is not None:
                    kwargs["beta"] = _decay_beta(beta, position_dim=1)
                return original_kernel(*args, **kwargs)

            if len(args) >= 4 and args[3] is not None:
                args = (*args[:3], _decay_beta(args[3], position_dim=1), *args[4:])
            return original_kernel(*args, **kwargs)

        return _kernel_with_step_decay

    @torch.compiler.disable
    def _stateless_deltanet_kernel(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor | None = None,
        scale: float | None = None,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        use_qk_l2norm_in_kernel: bool = False,
        **kwargs: tp.Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert not kwargs, f"Unsupported extra args: {sorted(kwargs)}"
        assert cu_seqlens is None, "native_recurrent_deltanet does not support cu_seqlens."
        assert beta is not None, "beta is required for native_recurrent_deltanet."
        assert initial_state is not None, "stateless mode requires an initial_state."
        assert output_final_state is False, (
            "output_final_state must be False in stateless_deltanet patch."
        )
        assert q.shape[1] == 1, "stateless_deltanet patch only supports decode-like T=1."
        
        scale = k.shape[-1] ** -0.5 if scale is None else scale

        dtype = q.dtype
        q, k, v, beta = (t.float() for t in (q, k, v, beta))
        s0 = initial_state.float()
        
        orig_batch = q.shape[0]
        cache_batch = s0.shape[0]
        
        if orig_batch != cache_batch:
            assert orig_batch % cache_batch == 0, "orig_batch must be divisible by cache_batch."
            flat_len = orig_batch // cache_batch
            q = q.reshape(cache_batch, flat_len, *q.shape[2:])
            k = k.reshape(cache_batch, flat_len, *k.shape[2:])
            v = v.reshape(cache_batch, flat_len, *v.shape[2:])
            beta = beta.reshape(cache_batch, flat_len, *beta.shape[2:])
            beta = _decay_beta(beta, position_dim=1)
        else:
            beta = _decay_beta(beta, position_dim=1)

        # Use FLA's l2norm to match the original kernel's normalization and don't fail tests
        if use_qk_l2norm_in_kernel:
            q = l2norm(q)
            k = l2norm(k)
        
        q = q * scale
        
        if beta.ndim < v.ndim:
            beta = beta.view(*beta.shape, *([1] * (v.ndim - beta.ndim)))

        if q.ndim == 5:
            s0k = torch.einsum("blthd,bhdm->blthm", k, s0)
            term1 = torch.einsum("blthd,bhdm->blthm", q, s0)
        else:
            s0k = torch.einsum("bthd,bhdm->bthm", k, s0)
            term1 = torch.einsum("bthd,bhdm->bthm", q, s0)

        o = term1
        if include_self_term:
            qk = (q * k).sum(-1, keepdim=True)
            scaled_qk = qk * beta
            o = o + scaled_qk * v - scaled_qk * s0k

        if orig_batch != cache_batch:
            if o.ndim == 4:
                o = o.reshape(orig_batch, 1, *o.shape[2:])
            else:
                o = o.reshape(orig_batch, *o.shape[2:])
            
        return o.to(dtype), None

    if enabled:
        deltanet_layer.fused_recurrent_delta_rule = _stateless_deltanet_kernel
        deltanet_layer.chunk_delta_rule = _stateless_deltanet_kernel
    else:
        deltanet_layer.fused_recurrent_delta_rule = _wrap_delta_rule_kernel(
            original_fused_recurrent_delta_rule
        )
        deltanet_layer.chunk_delta_rule = _wrap_delta_rule_kernel(original_chunk_delta_rule)

    try:
        yield
    finally:
        deltanet_layer.fused_recurrent_delta_rule = original_fused_recurrent_delta_rule
        deltanet_layer.chunk_delta_rule = original_chunk_delta_rule

@contextmanager
def _maybe_patch_gated_deltanet_with_stateless_recurrent(
    enabled: bool,
    *,
    include_self_term: bool = True,
):
    if not enabled:
        yield
        return
    import fla.layers.gated_deltanet as gated_deltanet_layer

    @torch.compiler.disable
    def _stateless_gated_deltanet_kernel(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor | None = None,
        beta: torch.Tensor | None = None,
        scale: float | None = None,
        cu_seqlens: torch.LongTensor | None = None,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        num_householder: int = 1,
        use_qk_l2norm_in_kernel: bool = False,
        **kwargs: tp.Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert not kwargs, f"Unsupported extra args: {sorted(kwargs)}"
        assert beta is not None, "beta is required for stateless_gated_deltanet."
        assert initial_state is not None, "stateless mode requires an initial_state."
        assert cu_seqlens is None, "stateless_gated_deltanet does not support cu_seqlens."
        assert num_householder == 1, "stateless_gated_deltanet only supports num_householder=1."
        assert output_final_state is False, (
            "output_final_state must be False in stateless_gated_deltanet patch."
        )
        assert q.shape[1] == 1, "stateless_gated_deltanet patch only supports decode-like T=1."

        scale = scale if scale is not None else k.shape[-1] ** -0.5
        dtype = q.dtype
        
        q, k, v, beta = (t.float() for t in (q, k, v, beta))
        g = g.float() if g is not None else None
        s0 = initial_state.float()

        orig_batch = q.shape[0]
        cache_batch = s0.shape[0]

        if orig_batch != cache_batch:
            assert orig_batch % cache_batch == 0, "orig_batch must be divisible by cache_batch."
            flat_len = orig_batch // cache_batch
            q = q.reshape(cache_batch, flat_len, *q.shape[2:])
            k = k.reshape(cache_batch, flat_len * num_householder, *k.shape[2:])
            v = v.reshape(cache_batch, flat_len * num_householder, *v.shape[2:])
            beta = beta.reshape(cache_batch, flat_len * num_householder, *beta.shape[2:])
            if g is not None:
                g = g.reshape(cache_batch, flat_len, *g.shape[2:])

        # Use FLA's l2norm to match the original kernel's normalization and don't fail tests
        if use_qk_l2norm_in_kernel:
            q = l2norm(q)
            k = l2norm(k)
        
        q = q * scale

        bsz, seq_len, num_heads, key_dim = q.shape
        value_dim = v.shape[-1]

        k = k.view(bsz, seq_len, num_householder, num_heads, key_dim)
        v = v.view(bsz, seq_len, num_householder, num_heads, value_dim)
        beta = beta.view(bsz, seq_len, num_householder, num_heads)
        
        g_exp = None
        if g is not None:
            if g.ndim != 3:
                raise ValueError(f"Expected g.ndim == 3, got g.shape={tuple(g.shape)}")
            if g.shape[1] == num_heads and g.shape[2] == seq_len and num_heads != seq_len:
                g = g.transpose(1, 2)
            elif g.shape != (bsz, seq_len, num_heads):
                raise ValueError(
                    f"Unexpected g shape {tuple(g.shape)} for q shape {(bsz, seq_len, num_heads, key_dim)}"
                )
            g_exp = g.exp().unsqueeze(-1)

        q_decayed = q * g_exp if g_exp is not None else q
        
        o = torch.einsum("bthk,bhkv->bthv", q_decayed, s0)


        k_decayed = k * g_exp.unsqueeze(2) if g_exp is not None else k
        k_s0 = torch.einsum("btlhk,bhkv->btlhv", k_decayed, s0)

        
        if include_self_term:
            k_0 = k[:, :, 0]
            v_0 = v[:, :, 0]
            beta_0 = beta[:, :, 0]
            k_s0_0 = k_s0[:, :, 0]

            u_0 = v_0 - k_s0_0
            qk_score = (q * k_0).sum(dim=-1, keepdim=True)
            o = o + (u_0 * (beta_0.unsqueeze(-1) * qk_score))

        if orig_batch != cache_batch:
            o = o.reshape(orig_batch, 1, *o.shape[2:])

        return o.to(dtype), None

    original_fused = gated_deltanet_layer.fused_recurrent_gated_delta_rule
    original_chunk = gated_deltanet_layer.chunk_gated_delta_rule
    
    gated_deltanet_layer.fused_recurrent_gated_delta_rule = _stateless_gated_deltanet_kernel
    gated_deltanet_layer.chunk_gated_delta_rule = _stateless_gated_deltanet_kernel
    try:
        yield
    finally:
        gated_deltanet_layer.fused_recurrent_gated_delta_rule = original_fused
        gated_deltanet_layer.chunk_gated_delta_rule = original_chunk


@contextmanager
def _maybe_patch_mesanet_with_stateless_recurrent(
    enabled: bool,
    *,
    include_self_term: bool = True,
):
    if not enabled:
        yield
        return
    import fla.layers.mesa_net as mesa_net_layer

    original_forward = mesa_net_layer.MesaNet.forward

    @torch.compiler.disable
    def _stateless_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs: tp.Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tp.Any]:
        del use_cache
        assert output_attentions in (None, False), (
            "stateless MesaNet patch does not support output_attentions=True."
        )
        assert attention_mask is None, (
            "stateless MesaNet patch does not support attention_mask."
        )
        assert not kwargs, (
            f"Unsupported extra args for stateless MesaNet patch: {sorted(kwargs)}"
        )

        last_state = mesa_net_layer.get_layer_cache(self, past_key_values)
        if last_state is None:
            return original_forward(
                self,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=False,
                output_attentions=output_attentions,
            )

        batch_size, seq_len, _ = hidden_states.shape
        assert seq_len == 1, (
            "stateless MesaNet patch only supports decode-like seq_len=1."
        )

        conv_state_q, conv_state_k = last_state["conv_state"]
        q, _ = self.q_conv1d(
            x=self.q_proj(hidden_states),
            cache=conv_state_q,
            output_final_state=False,
            cu_seqlens=None,
        )
        k, _ = self.k_conv1d(
            x=self.k_proj(hidden_states),
            cache=conv_state_k,
            output_final_state=False,
            cu_seqlens=None,
        )
        v = self.v_proj(hidden_states)

        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_k_dim)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_k_dim)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_v_dim)
        beta = self.b_proj(hidden_states).float().sigmoid()
        g = F.logsigmoid(self.a_proj(hidden_states).float())
        lamb = F.softplus(self.lambda_params.float()) + self.lambda_lower_bound
        lamb = lamb.reshape(self.num_heads, self.head_k_dim)

        prev_h_kk, prev_h_kv = last_state["recurrent_state"]
        assert prev_h_kk is not None and prev_h_kv is not None, (
            "stateless MesaNet patch requires recurrent_state in the cache."
        )

        dtype = q.dtype
        q = mesa_net_layer.l2_norm(q).float().squeeze(1)
        k = mesa_net_layer.l2_norm(k).float().squeeze(1)
        v = v.float().squeeze(1)
        beta = beta.float().squeeze(1)
        g = g.float().squeeze(1)
        prev_h_kk = prev_h_kk.float()
        prev_h_kv = prev_h_kv.float()

        cache_batch = prev_h_kk.shape[0]
        assert batch_size % cache_batch == 0, (
            "MesaNet stateless patch expects batch_size to be divisible by cache batch size."
        )
        flat_len = batch_size // cache_batch

        q = q.reshape(cache_batch, flat_len, self.num_heads, self.head_k_dim)
        k = k.reshape(cache_batch, flat_len, self.num_heads, self.head_k_dim)
        v = v.reshape(cache_batch, flat_len, self.num_heads, self.head_v_dim)
        beta = beta.reshape(cache_batch, flat_len, self.num_heads)
        g = g.reshape(cache_batch, flat_len, self.num_heads)

        decay = g.exp().unsqueeze(-1).unsqueeze(-1)
        k_beta = k * beta.unsqueeze(-1)
        h_kk = prev_h_kk.unsqueeze(1) * decay
        h_kv = prev_h_kv.unsqueeze(1) * decay
        if include_self_term:
            h_kk = h_kk + k_beta.unsqueeze(-1) * k.unsqueeze(-2)
            h_kv = h_kv + k_beta.unsqueeze(-1) * v.unsqueeze(-2)

        lamb = lamb.view(1, 1, self.num_heads, self.head_k_dim)
        diag_h = torch.diagonal(h_kk, dim1=-2, dim2=-1)
        x = q / (diag_h + lamb)
        residual = q - (x.unsqueeze(-1) * h_kk).sum(-2) - lamb * x
        direction = residual.clone()
        delta_old = (residual * residual).sum(-1)

        for _ in range(self.max_cg_step_decoding):
            q_cg = (direction.unsqueeze(-1) * h_kk).sum(-2) + lamb * direction
            alpha = delta_old / ((direction * q_cg).sum(-1) + 1e-5)
            x = x + alpha.unsqueeze(-1) * direction
            residual = residual - alpha.unsqueeze(-1) * q_cg
            delta_new = (residual * residual).sum(-1)
            beta_cg = delta_new / (delta_old + 1e-5)
            direction = residual + beta_cg.unsqueeze(-1) * direction
            delta_old = delta_new

        o = (x.unsqueeze(-1) * h_kv).sum(-2)
        o = o.reshape(batch_size, 1, self.num_heads, self.head_v_dim).to(dtype)

        if self.use_output_gate:
            gate = self.g_proj(hidden_states).reshape(batch_size, seq_len, self.num_heads, self.head_v_dim)
            o = self.o_norm(o, gate)
        else:
            o = self.o_norm(o)
        o = o.reshape(batch_size, seq_len, self.value_dim)
        o = self.o_proj(o)
        return o, None, past_key_values

    mesa_net_layer.MesaNet.forward = _stateless_forward
    try:
        yield
    finally:
        mesa_net_layer.MesaNet.forward = original_forward


@contextmanager
def _maybe_patch_mamba2_with_stateless_recurrent(
    enabled: bool,
    *,
    include_self_term: bool = True,
):
    """
    Patch Mamba2 forward for stateless parallel evaluation with cached state.
    Computes SSM output directly from initial state without materializing intermediates:
        h_new = A * h_0 + B * x;  y = C @ h_new + D * x
    """
    if not enabled:
        yield
        return
    import fla.models.mamba2.modeling_mamba2 as mamba_module
    
    original_forward = mamba_module.Mamba2.forward

    @torch.compiler.disable
    def _stateless_forward(
        self,
        hidden_states: torch.Tensor,  # (orig_batch, seq_len, hidden_size)
        cache_params=None,
        cache_position=None,
        attention_mask=None,
        past_key_values=None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        **kwargs
    ):
        # Accepted for API compatibility with upstream Mamba2.forward, but unused here.
        del cache_position
        if cache_params is None:
            cache_params = past_key_values
        assert cache_params is not None, "Stateless mamba2 requires cached state."
        assert attention_mask is None, "stateless mamba2 patch does not support attention_mask."
        assert output_attentions in (None, False), (
            "stateless mamba2 patch does not support output_attentions=True."
        )
        assert use_cache is not False, "stateless mamba2 patch expects use_cache=True."
        assert not kwargs, f"Unsupported extra args for stateless mamba2 patch: {sorted(kwargs)}"
        
        dtype = hidden_states.dtype
        orig_batch, seq_len, _ = hidden_states.shape
        assert seq_len == 1, "stateless mamba2 patch only supports decode-like seq_len=1."
        if hasattr(cache_params, "conv_states") and hasattr(cache_params, "ssm_states"):
            conv_state_layer = cache_params.conv_states[self.layer_idx]
            recurrent_state_layer = cache_params.ssm_states[self.layer_idx]
        elif hasattr(cache_params, "layers"):
            layer_cache = cache_params.layers[self.layer_idx]
            layer_state = getattr(layer_cache, "state", None)
            assert isinstance(layer_state, dict), "Mamba2 cache layer state must be a dict."
            conv_state_layer = layer_state.get("conv_state", None)
            recurrent_state_layer = layer_state.get("recurrent_state", None)
            assert conv_state_layer is not None, "Missing conv_state in Mamba2 cache layer state."
            assert recurrent_state_layer is not None, "Missing recurrent_state in Mamba2 cache layer state."
        else:
            raise AssertionError("Unsupported Mamba2 cache_params structure.")

        cache_batch = conv_state_layer.shape[0]
        assert orig_batch % cache_batch == 0
        flat_len = orig_batch // cache_batch  # number of test samples per cache entry
        
        # Input projection: split into gate, conv input, and dt
        projected_states = self.in_proj(hidden_states.float())
        d_mlp = (projected_states.shape[-1] - 2 * self.intermediate_size 
                 - 2 * self.n_groups * self.ssm_state_size - self.num_heads) // 2
        _, _, gate, hidden_states_B_C, dt = projected_states.split(
            [d_mlp, d_mlp, self.intermediate_size, self.conv_dim, self.num_heads], dim=-1
        )
        
        # Short convolution: prepend cached history and apply depthwise conv
        hidden_states_B_C = hidden_states_B_C.view(cache_batch, flat_len, seq_len, -1)
        conv_state = conv_state_layer.float()  # (cache_batch, conv_dim, kernel_size)
        x_transposed = hidden_states_B_C.transpose(2, 3)  # (cache_batch, flat_len, conv_dim, seq_len=1)
        hist_len = self.conv_kernel_size - 1
        if hist_len > 0:
            conv_history = conv_state[:, :, -hist_len:].unsqueeze(1)  # (cache_batch, 1, conv_dim, k-1)
            conv_window = torch.cat([conv_history.expand(cache_batch, flat_len, -1, -1), x_transposed], dim=-1)
        else:
            conv_window = x_transposed

        weight = self.conv1d.weight.squeeze(1).float()  # (conv_dim, kernel_size)
        bias = self.conv1d.bias.float() if self.conv1d.bias is not None else None
        conv_out = F.conv1d(
            conv_window.view(cache_batch * flat_len, self.conv_dim, -1),
            weight.unsqueeze(1), bias=bias, groups=self.conv_dim,
        ).view(cache_batch, flat_len, self.conv_dim, seq_len)
        hidden_states_B_C = self.act(conv_out).transpose(2, 3)  # (cache_batch, flat_len, seq_len, conv_dim)
        
        # Split conv output into x, B, C and reshape for SSM
        x, B, C = torch.split(hidden_states_B_C, [
            self.intermediate_size, self.n_groups * self.ssm_state_size, self.n_groups * self.ssm_state_size
        ], dim=-1)
        x = x.view(cache_batch, flat_len, seq_len, self.num_heads, self.head_dim)
        B = B.view(cache_batch, flat_len, seq_len, self.n_groups, self.ssm_state_size)
        C = C.view(cache_batch, flat_len, seq_len, self.n_groups, self.ssm_state_size)
        dt = dt.view(cache_batch, flat_len, seq_len, self.num_heads)
        gate = gate.view(cache_batch, flat_len, seq_len, self.intermediate_size)
        
        # SSM discretization: A_bar = exp(A * dt), B_bar = dt * B
        A = -torch.exp(self.A_log.float())  # (num_heads,)
        dt = F.softplus(dt + self.dt_bias.float())  # (cache_batch, flat_len, seq_len, num_heads)
        dt = torch.clamp(dt, self.time_step_limit[0], self.time_step_limit[1])
        A_bar = torch.exp(A.view(1, 1, 1, -1) * dt).unsqueeze(-1)  # (cache_batch, flat_len, seq_len, num_heads, 1)
        h0 = recurrent_state_layer.float()  # (cache_batch, num_heads, head_dim, ssm_state_size)
        
        # Expand B, C from n_groups to num_heads
        assert self.num_heads % self.n_groups == 0, "num_heads must be divisible by n_groups."
        heads_per_group = self.num_heads // self.n_groups
        B = B.repeat_interleave(heads_per_group, dim=-2).contiguous()  # (..., num_heads, ssm_state_size)
        C = C.repeat_interleave(heads_per_group, dim=-2).contiguous()
        B_bar = dt.unsqueeze(-1) * B  # (cache_batch, flat_len, seq_len, num_heads, ssm_state_size)
        
        # Fused SSM output: y = C @ (A_bar * h0) + sum_s(C * B_bar) * x
        # Avoids materializing (batch, flat_len, seq_len, heads, head_dim, state_size) tensor
        C_scaled = C * A_bar  # broadcasts A_bar over ssm_state_size
        y_from_h0 = torch.einsum('bflhs,bhds->bflhd', C_scaled, h0)  # contract over ssm_state_size
        y = y_from_h0
        if include_self_term:
            CB_sum = (C * B_bar).sum(dim=-1)  # (cache_batch, flat_len, seq_len, num_heads)
            y_from_x = CB_sum.unsqueeze(-1) * x  # (cache_batch, flat_len, seq_len, num_heads, head_dim)
            y = y + y_from_x
        
        # D skip connection
        if self.D is not None:
            y = y + self.D.float().view(1, 1, 1, -1, 1) * x
        
        # Reshape and apply output projection with gating
        y = y.reshape(cache_batch * flat_len, seq_len, self.intermediate_size)
        gate = gate.reshape(cache_batch * flat_len, seq_len, self.intermediate_size)
        hidden_states = self.out_proj(self.norm(y, gate)).to(dtype)
        return hidden_states, None, past_key_values

    mamba_module.Mamba2.forward = _stateless_forward
    try:
        yield
    finally:
        mamba_module.Mamba2.forward = original_forward


@contextmanager
def _maybe_patch_linear_attn_with_stateless_recurrent(
    enabled: bool,
    *,
    include_self_term: bool = True,
):
    """Patch linear-attention kernels for stateless decode (GLA-style)."""
    if not enabled:
        yield
        return
    import fla.layers.linear_attn as linear_attn_layer

    @torch.compiler.disable
    def _stateless_linear_attn_kernel(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        normalize: bool = False,
        scale: float | None = None,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        reverse: bool = False,
        head_first: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        offsets: torch.LongTensor | None = None,
        indices: torch.LongTensor | None = None,
        **kwargs: tp.Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert not kwargs, f"Unsupported extra args for stateless linear_attn patch: {sorted(kwargs)}"
        assert not reverse, "stateless linear_attn patch does not support reverse=True."
        assert not head_first, "stateless linear_attn patch expects head_first=False."
        assert cu_seqlens is None, "stateless linear_attn patch does not support cu_seqlens."
        assert offsets is None, "stateless linear_attn patch does not support offsets."
        assert indices is None, "stateless linear_attn patch does not support indices."
        assert initial_state is not None, "stateless linear_attn patch requires initial_state."
        assert output_final_state is False, (
            "output_final_state must be False in stateless linear_attn patch."
        )
        assert q.shape[1] == 1, "stateless linear_attn patch only supports decode-like T=1."
        assert not normalize, ("stateless linear_attn patch does not support normalize=True")

        dtype = q.dtype
        if scale is None:
            scale = q.shape[-1] ** -0.5

        qf = (q * scale).float()
        kf = k.float()
        vf = v.float()
        h0 = initial_state.float()

        orig_batch = qf.shape[0]
        cache_batch = h0.shape[0]
        if orig_batch != cache_batch:
            assert orig_batch % cache_batch == 0, "orig_batch must be divisible by cache_batch."
            flat_len = orig_batch // cache_batch
            qf = qf.reshape(cache_batch, flat_len, *qf.shape[1:])
            kf = kf.reshape(cache_batch, flat_len, *kf.shape[1:])
            vf = vf.reshape(cache_batch, flat_len, *vf.shape[1:])

        if qf.ndim == 5:
            o = torch.einsum("blthk,bhkv->blthv", qf, h0)
        else:
            o = torch.einsum("bthk,bhkv->bthv", qf, h0)
        if include_self_term:
            o = o + (qf * kf).sum(-1, keepdim=True) * vf

        if orig_batch != cache_batch:
            o = o.reshape(orig_batch, *o.shape[2:])

        return o.to(dtype), None

    original_fused_recurrent = linear_attn_layer.fused_recurrent_linear_attn
    original_chunk = linear_attn_layer.chunk_linear_attn
    original_fused_chunk = linear_attn_layer.fused_chunk_linear_attn
    linear_attn_layer.fused_recurrent_linear_attn = _stateless_linear_attn_kernel
    linear_attn_layer.chunk_linear_attn = _stateless_linear_attn_kernel
    linear_attn_layer.fused_chunk_linear_attn = _stateless_linear_attn_kernel
    try:
        yield
    finally:
        linear_attn_layer.fused_recurrent_linear_attn = original_fused_recurrent
        linear_attn_layer.chunk_linear_attn = original_chunk
        linear_attn_layer.fused_chunk_linear_attn = original_fused_chunk
