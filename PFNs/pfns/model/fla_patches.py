"""Context managers for patching FLA/Gla ops during evaluation."""
from __future__ import annotations

import os
import typing as tp
from contextlib import contextmanager

import torch
import torch.nn.functional as F

from fla.modules.l2norm import l2norm

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

        x = x.contiguous()
        cache = cache.contiguous()
        if residual is not None:
            residual = residual.contiguous()

        B, T, D = x.shape

        assert T == 1, "We only support decode-like path in pytorch ShortConvolution patch."
        assert cache is not None, "cache must be provided in pytorch ShortConvolution patch."

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
def _maybe_patch_gla_with_stateless_recurrent(enabled: bool):
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
            flat_len = orig_batch // cache_batch
            q = q.reshape(cache_batch, flat_len, *q.shape[2:])
            k = k.reshape(cache_batch, flat_len, *k.shape[2:])
            v = v.reshape(cache_batch, flat_len, *v.shape[2:])
            gk = gk.reshape(cache_batch, flat_len, *gk.shape[2:])
        
        q = q * scale
        qg = q * gk.exp()
        term1 = torch.einsum("bthk,bhkv->bthv", qg, h0)
        qk = (q * k).sum(-1, keepdim=True)
        term2 = qk * v
        o = term1 + term2

        if orig_batch != cache_batch:
            o = o.reshape(orig_batch, 1, *o.shape[2:])
        final_state = initial_state if output_final_state else None
        return o.to(dtype), final_state

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
def _maybe_patch_gla_with_stateless_recurrent_vmap(enabled: bool):
    if not enabled:
        yield
        return
    import fla.layers.gla as gla_layer
    # from fla.ops.gla.naive import naive_recurrent_gla
    
    def _stateless_gla_kernel_with_vmap(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        gk: torch.Tensor | None = None,
        g: torch.Tensor | None = None,
        gv: torch.Tensor | None = None,
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
        if gk is None:
            gk = g
        assert gk is not None, "gk is required for naive_recurrent_gla."
        assert initial_state is not None, "stateless mode requires an initial_state."
        cache_batch = initial_state.shape[0]
        orig_batch = q.shape[0]
        assert orig_batch % cache_batch == 0, "orig_batch must be divisible by cache_batch."
        flat_len = orig_batch // cache_batch
        dtype = q.dtype
        q = q.view(cache_batch, flat_len, *q.shape[1:])
        k = k.view(cache_batch, flat_len, *k.shape[1:])
        v = v.view(cache_batch, flat_len, *v.shape[1:])
        gk = gk.view(cache_batch, flat_len, *gk.shape[1:])
        
        def naive_recurrent_gla(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            gk: torch.Tensor,
            initial_state: torch.Tensor
        ):
            dtype = q.dtype
            q, k, v, gk = map(lambda x: x.transpose(1, 2).float(), (q, k, v, gk))
            B, H, T, K, V = *q.shape, v.shape[-1]
            o = torch.zeros_like(v)
            scale = K ** -0.5

            for i in range(T):
                q_i = q[:, :, i] * scale
                k_i = k[:, :, i]
                v_i = v[:, :, i]
                gk_i = gk[:, :, i].exp()
                kv_i = k_i[..., None] * v_i[..., None, :]
                o[:, :, i] = (q_i[..., None] * (initial_state * gk_i[..., None] + kv_i)).sum(-2)

            return o.transpose(1, 2).to(dtype)
        
        # def step(q_t, k_t, v_t, gk_t):
        #     o1, _ = naive_recurrent_gla(
        #         q_t, 
        #         k_t, 
        #         v_t, 
        #         gk_t, 
        #         initial_state, 
        #         False
        #     )
        #     return o1

        # 4. vmap over flat_len (dim 1)
        o = torch.vmap(naive_recurrent_gla, in_dims=(1, 1, 1, 1, None), out_dims=1)(q, k, v, gk, initial_state)

        o = o.reshape(orig_batch, *o.shape[2:])
        h = initial_state if output_final_state else None
        return o.to(dtype), h
    
    original_fused_recurrent = gla_layer.fused_recurrent_gla
    original_chunked = gla_layer.chunk_gla
    original_fused_chunked = gla_layer.fused_chunk_gla
    gla_layer.fused_recurrent_gla = _stateless_gla_kernel_with_vmap
    gla_layer.fused_chunk_gla = _stateless_gla_kernel_with_vmap
    gla_layer.chunk_gla = _stateless_gla_kernel_with_vmap
    try:
        yield
    finally:
        gla_layer.fused_recurrent_gla = original_fused_recurrent
        gla_layer.chunk_gla = original_chunked
        gla_layer.fused_chunk_gla = original_fused_chunked


@contextmanager
def _maybe_patch_gla_with_stateless_recurrent_causal(enabled: bool):
    if not enabled:
        yield
        return
    import fla.layers.gla as gla_layer

    def _stateless_gla_kernel_causal(
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
        if gk is None:
            gk = g
        assert gk is not None, "gk is required for naive_recurrent_gla."
        assert initial_state is not None, "stateless mode requires an initial_state."
        if scale is None:
            scale = k.shape[-1] ** -0.5

        dtype = q.dtype
        q, k, v, gk = (t.float() for t in (q, k, v, gk))
        h0 = initial_state.float()

        q = q * scale
        qg = q * gk.exp()
        term1 = torch.einsum("bthk,bhkv->bthv", qg, h0)
        qk = (q * k).sum(-1, keepdim=True)
        term2 = qk * v
        o = term1 + term2

        final_state = initial_state if output_final_state else None
        return o.to(dtype), final_state

    original_fused_recurrent = gla_layer.fused_recurrent_gla
    original_chunked = gla_layer.chunk_gla
    original_fused_chunked = gla_layer.fused_chunk_gla
    gla_layer.fused_recurrent_gla = _stateless_gla_kernel_causal
    gla_layer.fused_chunk_gla = _stateless_gla_kernel_causal
    gla_layer.chunk_gla = _stateless_gla_kernel_causal
    try:
        yield
    finally:
        gla_layer.fused_recurrent_gla = original_fused_recurrent
        gla_layer.chunk_gla = original_chunked
        gla_layer.fused_chunk_gla = original_fused_chunked


@contextmanager
def _maybe_patch_kda_with_stateless_recurrent(enabled: bool):
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
            
        delta = v - k_s0
        
        qk_dot = (q * k).sum(dim=-1, keepdim=True) # (..., H, 1)
        scaling = beta.unsqueeze(-1) * qk_dot      # (..., H, 1)
        
        o_update = delta * scaling

        o = o_base + o_update

        if orig_batch != cache_batch:
            o = o.reshape(orig_batch, *o.shape[2:])

        final_state = initial_state if output_final_state else None
        
        return o.to(dtype), final_state

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
def _maybe_patch_deltanet_with_stateless_recurrent(enabled: bool):
    if not enabled:
        yield
        return
    import fla.layers.delta_net as deltanet_layer

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

        # Use FLA's l2norm to match the original kernel's normalization and don't fail tests
        if use_qk_l2norm_in_kernel:
            q = l2norm(q)
            k = l2norm(k)
        
        q = q * scale
        
        if beta.ndim < v.ndim:
            beta = beta.view(*beta.shape, *([1] * (v.ndim - beta.ndim)))

        s0k = torch.einsum("bthd,bhdm->bthm", k, s0)

        term1 = torch.einsum("bthd,bhdm->bthm", q, s0)

        qk = (q * k).sum(-1, keepdim=True)
        scaled_qk = qk * beta
        term2 = scaled_qk * v - scaled_qk * s0k
        
        o = term1 + term2

        if orig_batch != cache_batch:
            o = o.reshape(orig_batch, 1, *o.shape[2:])
            
        final_state = initial_state if output_final_state else None
        
        return o.to(dtype), final_state

    original_fused_recurrent_delta_rule = deltanet_layer.fused_recurrent_delta_rule
    original_chunk_delta_rule = deltanet_layer.chunk_delta_rule
    
    deltanet_layer.fused_recurrent_delta_rule = _stateless_deltanet_kernel
    deltanet_layer.chunk_delta_rule = _stateless_deltanet_kernel

    try:
        yield
    finally:
        deltanet_layer.fused_recurrent_delta_rule = original_fused_recurrent_delta_rule
        deltanet_layer.chunk_delta_rule = original_chunk_delta_rule

@contextmanager
def _maybe_patch_gated_deltanet_with_stateless_recurrent(enabled: bool):
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

        scale = scale if scale is not None else k.shape[-1] ** -0.5
        dtype = q.dtype
        
        q, k, v, beta = (t.float() for t in (q, k, v, beta))
        g = g.float() if g is not None else None
        s0 = initial_state.float() 

        orig_batch = q.shape[0]
        cache_batch = s0.shape[0]

        if orig_batch != cache_batch:
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
            if g.shape[1] == num_heads and g.shape[2] == seq_len:
                g = g.transpose(1, 2)
            g_exp = g.exp().unsqueeze(-1)

        q_decayed = q * g_exp if g_exp is not None else q
        
        o = torch.einsum("bthk,bhkv->bthv", q_decayed, s0)


        k_decayed = k * g_exp.unsqueeze(2) if g_exp is not None else k
        k_s0 = torch.einsum("btlhk,bhkv->btlhv", k_decayed, s0)

        
        k_0 = k[:, :, 0]
        v_0 = v[:, :, 0]
        beta_0 = beta[:, :, 0]
        k_s0_0 = k_s0[:, :, 0]

        u_0 = v_0 - k_s0_0
        
        qk_score = (q * k_0).sum(dim=-1, keepdim=True)
        o = o + (u_0 * (beta_0.unsqueeze(-1) * qk_score))

        if orig_batch != cache_batch:
            o = o.reshape(orig_batch, 1, *o.shape[2:])

        final_state = initial_state if output_final_state else None
        
        return o.to(dtype), final_state

    original_fused = gated_deltanet_layer.fused_recurrent_gated_delta_rule
    original_chunk = gated_deltanet_layer.chunk_gated_delta_rule
    
    gated_deltanet_layer.fused_recurrent_gated_delta_rule = _stateless_gated_deltanet_kernel
    gated_deltanet_layer.chunk_gated_delta_rule = _stateless_gated_deltanet_kernel
    try:
        yield
    finally:
        gated_deltanet_layer.fused_recurrent_gated_delta_rule = original_fused
        gated_deltanet_layer.chunk_gated_delta_rule = original_chunk
