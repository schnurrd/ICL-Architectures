"""Context managers for patching FLA/Gla ops during evaluation."""
from __future__ import annotations

import typing as tp
from contextlib import contextmanager

import torch

@contextmanager
def _maybe_patch_gla_native_recurrent(enabled: bool):
    if not enabled:
        yield
        return
    import fla.layers.gla as gla_layer

    @torch.compiler.disable
    def _native_recurrent_gla(
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
    gla_layer.fused_recurrent_gla = _native_recurrent_gla
    gla_layer.fused_chunk_gla = _native_recurrent_gla
    gla_layer.chunk_gla = _native_recurrent_gla
    try:
        yield
    finally:
        gla_layer.fused_recurrent_gla = original_fused_recurrent
        gla_layer.chunk_gla = original_chunked
        gla_layer.fused_chunk_gla = original_fused_chunked


@contextmanager
def _maybe_patch_gla_native_recurrent_vmap(enabled: bool):
    if not enabled:
        yield
        return
    import fla.layers.gla as gla_layer
    from fla.ops.gla.naive import naive_recurrent_gla
    
    def _native_recurrent_gla(
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
        
        def step(q_t, k_t, v_t, gk_t):
            o1, _ = naive_recurrent_gla(
                q_t, 
                k_t, 
                v_t, 
                gk_t, 
                initial_state, 
                False
            )
            return o1

        # 4. vmap over flat_len (dim 1)
        o = torch.vmap(step, in_dims=(1, 1, 1, 1), out_dims=1)(q, k, v, gk)

        o = o.reshape(orig_batch, *o.shape[2:])
        h = initial_state if output_final_state else None
        return o.to(dtype), h
    
    original_fused_recurrent = gla_layer.fused_recurrent_gla
    original_chunked = gla_layer.chunk_gla
    original_fused_chunked = gla_layer.fused_chunk_gla
    gla_layer.fused_recurrent_gla = _native_recurrent_gla
    gla_layer.fused_chunk_gla = _native_recurrent_gla
    gla_layer.chunk_gla = _native_recurrent_gla
    try:
        yield
    finally:
        gla_layer.fused_recurrent_gla = original_fused_recurrent
        gla_layer.chunk_gla = original_chunked
        gla_layer.fused_chunk_gla = original_fused_chunked


@contextmanager
def _maybe_patch_gla_native_recurrent_causal(enabled: bool):
    if not enabled:
        yield
        return
    import fla.layers.gla as gla_layer

    def _native_recurrent_gla(
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
    gla_layer.fused_recurrent_gla = _native_recurrent_gla
    gla_layer.fused_chunk_gla = _native_recurrent_gla
    gla_layer.chunk_gla = _native_recurrent_gla
    try:
        yield
    finally:
        gla_layer.fused_recurrent_gla = original_fused_recurrent
        gla_layer.chunk_gla = original_chunked
        gla_layer.fused_chunk_gla = original_fused_chunked
