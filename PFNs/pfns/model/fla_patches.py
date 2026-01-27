"""Context managers for patching FLA/Gla ops during evaluation."""
from __future__ import annotations

import typing as tp
from contextlib import contextmanager

import torch
import torch.nn.functional as F

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

        if use_qk_l2norm_in_kernel:
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)

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

        if use_qk_l2norm_in_kernel:
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)
        
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

        if use_qk_l2norm_in_kernel:
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)
        
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
