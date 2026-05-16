from contextlib import ExitStack

import pytest
import torch

pytest.importorskip("fla")
from fla.layers.gla import fused_recurrent_gla
from fla.layers.linear_attn import fused_recurrent_linear_attn
from fla.modules.l2norm import l2norm

from tests.model.fla_test_utils import (
    fla_cache_equivalence_tolerances,
    FLA_MODEL_TYPES,
    build_fla_backbone,
    fla_model_config_kwargs,
    fla_hidden_size,
    fla_tolerances,
)

def _run_repeated_cache_reference(
    backbone,
    test_x: torch.Tensor,
    past,
    *,
    use_custom_recurrent: bool,
    use_custom_shortconv: bool,
) -> torch.Tensor:
    batch_size, test_len, embed_dim = test_x.shape
    repeated_cache = backbone._repeat_cache(past, test_len)
    test_x_flat = test_x.contiguous().view(batch_size * test_len, 1, embed_dim)
    out_ref, _ = backbone._run_fla(
        test_x_flat,
        cache_params=repeated_cache,
        return_cache=False,
        use_custom_recurrent=use_custom_recurrent,
        use_custom_shortconv=use_custom_shortconv,
    )
    return out_ref.view(batch_size, test_len, embed_dim)


def test_fla_output_unpacker_accepts_tuple_outputs():
    from pfns.model.backbones import FLABackbone

    hidden_state = torch.randn(2, 3, 4)
    cache_params = object()

    out, cache = FLABackbone._unpack_fla_output(
        (hidden_state, cache_params),
        return_cache=True,
    )
    assert out is hidden_state
    assert cache is cache_params

    out, cache = FLABackbone._unpack_fla_output(
        (hidden_state,),
        return_cache=False,
    )
    assert out is hidden_state
    assert cache is None


def test_fla_linear_attn_matches_gla_without_gating():
    if not torch.cuda.is_available():
        pytest.skip("FLA backend requires CUDA/Triton for this test.")

    torch.manual_seed(0)
    device = torch.device("cuda")

    batch_size = 2
    seq_len = 7
    num_heads = 3
    key_dim = 8
    value_dim = 6

    q = torch.randn(batch_size, seq_len, num_heads, key_dim, device=device)
    k = torch.randn(batch_size, seq_len, num_heads, key_dim, device=device)
    v = torch.randn(batch_size, seq_len, num_heads, value_dim, device=device)
    initial_state = torch.randn(batch_size, num_heads, key_dim, value_dim, device=device)

    out_linear, final_state_linear = fused_recurrent_linear_attn(
        q,
        k,
        v,
        initial_state=initial_state,
        output_final_state=True,
        normalize=False,
    )

    neutral_gate = torch.zeros(batch_size, seq_len, num_heads, key_dim, device=device)
    out_gla, final_state_gla = fused_recurrent_gla(
        q,
        k,
        v,
        gk=neutral_gate,
        initial_state=initial_state,
        output_final_state=True,
    )

    torch.testing.assert_close(out_linear, out_gla, rtol=0.0, atol=0.0)
    torch.testing.assert_close(final_state_linear, final_state_gla, rtol=0.0, atol=0.0)


def _expected_final_state_readout(
    q: torch.Tensor,
    k: torch.Tensor,
    final_state: torch.Tensor,
    *,
    scale: float | None,
    use_qk_l2norm_in_kernel: bool = False,
) -> torch.Tensor:
    q_read = l2norm(q.float()) if use_qk_l2norm_in_kernel else q.float()
    q_read = q_read * (k.shape[-1] ** -0.5 if scale is None else scale)
    return torch.einsum("bthd,bhdm->bthm", q_read, final_state.float()).to(q.dtype)


def test_deltanet_beta_decay_scales_late_positions():
    from pfns.model.fla_patches import _apply_deltanet_beta_decay

    beta = torch.ones(1, 5, 2)

    inverse = _apply_deltanet_beta_decay(beta, mode="inverse", t0=2)
    expected_inverse = torch.tensor([1.0, 1.0, 2 / 3, 0.5, 0.4]).view(1, 5, 1)
    torch.testing.assert_close(inverse, expected_inverse.expand_as(beta))

    offset_inverse = _apply_deltanet_beta_decay(
        beta,
        mode="inverse",
        t0=2,
        start_position=2,
    )
    expected_offset = torch.tensor([2 / 3, 0.5, 0.4, 2 / 6, 2 / 7]).view(1, 5, 1)
    torch.testing.assert_close(offset_inverse, expected_offset.expand_as(beta))

    sqrt_inverse = _apply_deltanet_beta_decay(beta, mode="sqrt_inverse", t0=2)
    torch.testing.assert_close(
        sqrt_inverse,
        expected_inverse.sqrt().expand_as(beta),
    )

    sqrt_length_inverse = _apply_deltanet_beta_decay(
        beta,
        mode="sqrt_length_inverse",
        t0=2,
    )
    expected_length_scale = torch.full_like(beta, (2 / 5) ** 0.5)
    torch.testing.assert_close(sqrt_length_inverse, expected_length_scale)

    offset_sqrt_length_inverse = _apply_deltanet_beta_decay(
        beta,
        mode="sqrt_length_inverse",
        t0=8,
        start_position=7,
    )
    expected_offset_length_scale = torch.full_like(beta, (8 / 12) ** 0.5)
    torch.testing.assert_close(offset_sqrt_length_inverse, expected_offset_length_scale)

    online_inverse = _apply_deltanet_beta_decay(beta, mode="online_inverse", t0=2)
    expected_online = torch.tensor([1.0, 2 / 3, 0.5, 0.4, 2 / 6]).view(1, 5, 1)
    torch.testing.assert_close(online_inverse, expected_online.expand_as(beta))

    offset_online_inverse = _apply_deltanet_beta_decay(
        beta,
        mode="online_inverse",
        t0=2,
        start_position=3,
    )
    expected_offset_online = torch.tensor(
        [2 / 5, 2 / 6, 2 / 7, 0.25, 2 / 9]
    ).view(1, 5, 1)
    torch.testing.assert_close(
        offset_online_inverse,
        expected_offset_online.expand_as(beta),
    )

    online_sqrt_inverse = _apply_deltanet_beta_decay(
        beta, mode="online_sqrt_inverse", t0=2
    )
    torch.testing.assert_close(
        online_sqrt_inverse,
        expected_online.sqrt().expand_as(beta),
    )

    k = torch.ones(1, 5, 2, 3)
    nlms = _apply_deltanet_beta_decay(beta, mode="nlms", t0=1, k=k, eps=0.0)
    torch.testing.assert_close(nlms, torch.full_like(beta, 1 / 3))

    nlms_l2 = _apply_deltanet_beta_decay(
        beta,
        mode="nlms",
        t0=1,
        k=k * 2,
        use_qk_l2norm_in_kernel=True,
        eps=0.0,
    )
    torch.testing.assert_close(nlms_l2, beta)

    nlms_inverse = _apply_deltanet_beta_decay(
        beta, mode="nlms_inverse", t0=1, k=k, eps=0.0
    )
    expected_nlms_inverse = torch.tensor(
        [1.0, 0.5, 1 / 3, 0.25, 0.2]
    ).view(1, 5, 1) / 3
    torch.testing.assert_close(nlms_inverse, expected_nlms_inverse.expand_as(beta))

    nlms_sqrt_inverse = _apply_deltanet_beta_decay(
        beta, mode="nlms_sqrt_inverse", t0=1, k=k, eps=0.0
    )
    torch.testing.assert_close(
        nlms_sqrt_inverse,
        expected_nlms_inverse.mul(3).sqrt().div(3).expand_as(beta),
    )


def test_deltanet_beta_decay_validates_backbone_config():
    from pfns.model.backbones import FLABackboneConfig

    config_kwargs = fla_model_config_kwargs("deltanet", size="small")
    config = FLABackboneConfig(
        model_type="deltanet",
        config_kwargs=config_kwargs,
        deltanet_beta_decay="inverse",
        deltanet_beta_decay_t0=2,
    )
    assert config.deltanet_beta_decay == "inverse"

    with pytest.raises(ValueError, match="model_type='deltanet'"):
        FLABackboneConfig(
            model_type="gla",
            config_kwargs=fla_model_config_kwargs("gla", size="small"),
            deltanet_beta_decay="inverse",
        )

    with pytest.raises(ValueError, match="state_passing"):
        FLABackboneConfig(
            model_type="deltanet",
            config_kwargs=config_kwargs,
            deltanet_beta_decay="inverse",
            state_passing=True,
        )


def test_deltanet_beta_decay_patch_matches_manual_decayed_beta():
    if not torch.cuda.is_available():
        pytest.skip("FLA DeltaNet kernel requires CUDA/Triton.")

    import fla.layers.delta_net as deltanet_layer
    from pfns.model.fla_patches import (
        _apply_deltanet_beta_decay,
        _maybe_patch_deltanet_with_stateless_recurrent,
    )

    torch.manual_seed(0)
    device = torch.device("cuda")
    inputs = {
        "q": torch.randn(2, 5, 3, 4, device=device, dtype=torch.float32),
        "k": torch.randn(2, 5, 3, 4, device=device, dtype=torch.float32),
        "v": torch.randn(2, 5, 3, 6, device=device, dtype=torch.float32),
        "beta": torch.rand(2, 5, 3, device=device, dtype=torch.float32),
        "initial_state": torch.randn(2, 3, 4, 6, device=device, dtype=torch.float32),
        "output_final_state": True,
        "use_qk_l2norm_in_kernel": True,
    }
    manual_inputs = {
        **inputs,
        "beta": _apply_deltanet_beta_decay(
            inputs["beta"],
            mode="inverse",
            t0=2,
        ),
    }
    expected_out, expected_state = deltanet_layer.fused_recurrent_delta_rule(
        **manual_inputs,
    )

    with _maybe_patch_deltanet_with_stateless_recurrent(
        False,
        beta_decay="inverse",
        beta_decay_t0=2,
    ):
        actual_out, actual_state = deltanet_layer.fused_recurrent_delta_rule(**inputs)

    torch.testing.assert_close(actual_out, expected_out, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(actual_state, expected_state, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    "case",
    ["linear_attn", "gla", "kda", "deltanet", "gated_deltanet"],
)
def test_final_state_readout_kernels_match_native_final_state_readout(case: str):
    if not torch.cuda.is_available():
        pytest.skip("FLA final-state readout requires CUDA/Triton.")

    torch.manual_seed(0)

    import fla.layers.delta_net as deltanet_layer
    import fla.layers.gated_deltanet as gated_deltanet_layer
    import fla.layers.gla as gla_layer
    import fla.layers.kda as kda_layer
    import fla.layers.linear_attn as linear_attn_layer
    from pfns.model.fla_patches import (
        _maybe_patch_deltanet_with_stateless_recurrent,
        _maybe_patch_gated_deltanet_with_stateless_recurrent,
        _maybe_patch_gla_with_stateless_recurrent,
        _maybe_patch_kda_with_stateless_recurrent,
        _maybe_patch_linear_attn_with_stateless_recurrent,
    )

    device = torch.device("cuda")
    base_inputs = {
        "q": torch.randn(2, 5, 3, 4, device=device),
        "k": torch.randn(2, 5, 3, 4, device=device),
        "v": torch.randn(2, 5, 3, 6, device=device),
        "scale": 0.5,
        "initial_state": torch.randn(2, 3, 4, 6, device=device),
        "output_final_state": True,
    }
    use_l2 = False

    if case == "linear_attn":
        inputs = {**base_inputs, "normalize": False}
        original_kernel = linear_attn_layer.fused_recurrent_linear_attn
        patched_kernel = lambda: linear_attn_layer.fused_recurrent_linear_attn
        patch_context = _maybe_patch_linear_attn_with_stateless_recurrent(
            False,
            final_state_readout=True,
        )
    elif case == "gla":
        gk = torch.randn(2, 5, 3, 4, device=device).clamp(-2.0, 2.0)
        inputs = {**base_inputs, "gk": gk}
        original_kernel = gla_layer.fused_recurrent_gla
        patched_kernel = lambda: gla_layer.fused_recurrent_gla
        patch_context = _maybe_patch_gla_with_stateless_recurrent(
            False,
            final_state_readout=True,
        )
    elif case == "kda":
        g = torch.randn(2, 5, 3, 4, device=device).clamp(-2.0, 2.0)
        inputs = {
            **base_inputs,
            "g": g,
            "beta": torch.rand(2, 5, 3, device=device),
            "use_qk_l2norm_in_kernel": True,
        }
        use_l2 = True
        original_kernel = kda_layer.fused_recurrent_kda
        patched_kernel = lambda: kda_layer.fused_recurrent_kda
        patch_context = _maybe_patch_kda_with_stateless_recurrent(
            False,
            final_state_readout=True,
        )
    elif case == "deltanet":
        inputs = {
            **base_inputs,
            "beta": torch.rand(2, 5, 3, device=device),
            "use_qk_l2norm_in_kernel": True,
        }
        use_l2 = True
        original_kernel = deltanet_layer.fused_recurrent_delta_rule
        patched_kernel = lambda: deltanet_layer.fused_recurrent_delta_rule
        patch_context = _maybe_patch_deltanet_with_stateless_recurrent(
            False,
            final_state_readout=True,
        )
    elif case == "gated_deltanet":
        g = torch.randn(2, 5, 3, device=device).clamp(-2.0, 2.0)
        inputs = {
            **base_inputs,
            "g": g,
            "beta": torch.rand(2, 5, 3, device=device),
        }
        original_kernel = gated_deltanet_layer.fused_recurrent_gated_delta_rule
        patched_kernel = lambda: gated_deltanet_layer.fused_recurrent_gated_delta_rule
        patch_context = _maybe_patch_gated_deltanet_with_stateless_recurrent(
            False,
            final_state_readout=True,
        )
    else:
        raise AssertionError(f"Unhandled FLA final-state readout case: {case}")

    causal_out, expected_state = original_kernel(**inputs)
    expected_out = _expected_final_state_readout(
        inputs["q"],
        inputs["k"],
        expected_state,
        scale=inputs["scale"],
        use_qk_l2norm_in_kernel=use_l2,
    )
    torch.testing.assert_close(
        causal_out[:, -1],
        expected_out[:, -1],
        rtol=1e-4,
        atol=1e-4,
    )
    assert not torch.allclose(causal_out, expected_out, rtol=1e-5, atol=1e-5)

    with patch_context:
        out, state = patched_kernel()(**inputs)

    torch.testing.assert_close(out, expected_out, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(state, expected_state, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    ("case", "chunk_kernel"),
    [
        ("linear_attn", "chunk"),
        ("linear_attn", "fused_chunk"),
        ("gla", "chunk"),
        ("gla", "fused_chunk"),
        ("kda", "chunk"),
        ("deltanet", "chunk"),
        ("gated_deltanet", "chunk"),
    ],
)
def test_final_state_readout_chunk_kernels_match_selected_final_state(
    case: str,
    chunk_kernel: str,
):
    if not torch.cuda.is_available():
        pytest.skip("FLA final-state readout requires CUDA/Triton.")

    torch.manual_seed(0)

    import fla.layers.delta_net as deltanet_layer
    import fla.layers.gated_deltanet as gated_deltanet_layer
    import fla.layers.gla as gla_layer
    import fla.layers.kda as kda_layer
    import fla.layers.linear_attn as linear_attn_layer
    from pfns.model.fla_patches import (
        _maybe_patch_deltanet_with_stateless_recurrent,
        _maybe_patch_gated_deltanet_with_stateless_recurrent,
        _maybe_patch_gla_with_stateless_recurrent,
        _maybe_patch_kda_with_stateless_recurrent,
        _maybe_patch_linear_attn_with_stateless_recurrent,
    )

    device = torch.device("cuda")
    base_inputs = {
        "q": torch.randn(2, 80, 3, 4, device=device),
        "k": torch.randn(2, 80, 3, 4, device=device),
        "v": torch.randn(2, 80, 3, 6, device=device),
        "scale": 0.5,
        "initial_state": torch.randn(2, 3, 4, 6, device=device),
        "output_final_state": True,
    }
    use_l2 = False

    if case == "linear_attn":
        recurrent_inputs = {**base_inputs, "normalize": False}
        chunk_inputs = dict(recurrent_inputs)
        if chunk_kernel == "chunk":
            chunk_inputs["head_first"] = False
        original_recurrent = linear_attn_layer.fused_recurrent_linear_attn
        patched_kernel = (
            lambda: linear_attn_layer.chunk_linear_attn
            if chunk_kernel == "chunk"
            else linear_attn_layer.fused_chunk_linear_attn
        )
        patch_context = _maybe_patch_linear_attn_with_stateless_recurrent(
            False,
            final_state_readout=True,
        )
    elif case == "gla":
        g = torch.randn(2, 80, 3, 4, device=device).clamp(-2.0, 2.0)
        recurrent_inputs = {**base_inputs, "gk": g}
        chunk_inputs = {**base_inputs, "g": g}
        original_recurrent = gla_layer.fused_recurrent_gla
        patched_kernel = (
            lambda: gla_layer.chunk_gla
            if chunk_kernel == "chunk"
            else gla_layer.fused_chunk_gla
        )
        patch_context = _maybe_patch_gla_with_stateless_recurrent(
            False,
            final_state_readout=True,
        )
    elif case == "kda":
        g = torch.randn(2, 80, 3, 4, device=device).clamp(-2.0, 2.0)
        recurrent_inputs = {
            **base_inputs,
            "g": g,
            "beta": torch.rand(2, 80, 3, device=device),
            "A_log": torch.log(torch.rand(3, device=device) + 0.5),
            "dt_bias": torch.randn(12, device=device),
            "use_qk_l2norm_in_kernel": True,
            "use_gate_in_kernel": True,
        }
        chunk_inputs = dict(recurrent_inputs)
        use_l2 = True
        original_recurrent = kda_layer.fused_recurrent_kda
        patched_kernel = lambda: kda_layer.chunk_kda
        patch_context = _maybe_patch_kda_with_stateless_recurrent(
            False,
            final_state_readout=True,
        )
    elif case == "deltanet":
        recurrent_inputs = {
            **{
                name: tensor.to(torch.bfloat16)
                if isinstance(tensor, torch.Tensor) and tensor.is_floating_point()
                else tensor
                for name, tensor in base_inputs.items()
            },
            "beta": torch.rand(2, 80, 3, device=device, dtype=torch.bfloat16),
            "use_qk_l2norm_in_kernel": True,
            "head_first": False,
        }
        chunk_inputs = dict(recurrent_inputs)
        use_l2 = True
        original_recurrent = deltanet_layer.chunk_delta_rule
        patched_kernel = lambda: deltanet_layer.chunk_delta_rule
        patch_context = _maybe_patch_deltanet_with_stateless_recurrent(
            False,
            final_state_readout=True,
        )
    elif case == "gated_deltanet":
        recurrent_inputs = {
            **base_inputs,
            "g": -torch.nn.functional.softplus(torch.randn(2, 80, 3, device=device)),
            "beta": torch.rand(2, 80, 3, device=device),
            "use_qk_l2norm_in_kernel": True,
        }
        chunk_inputs = dict(recurrent_inputs)
        use_l2 = True
        original_recurrent = gated_deltanet_layer.fused_recurrent_gated_delta_rule
        patched_kernel = lambda: gated_deltanet_layer.chunk_gated_delta_rule
        patch_context = _maybe_patch_gated_deltanet_with_stateless_recurrent(
            False,
            final_state_readout=True,
        )
    else:
        raise AssertionError(f"Unhandled FLA final-state readout case: {case}")

    _, expected_state = original_recurrent(**recurrent_inputs)
    expected_out = _expected_final_state_readout(
        recurrent_inputs["q"],
        recurrent_inputs["k"],
        expected_state,
        scale=recurrent_inputs["scale"],
        use_qk_l2norm_in_kernel=use_l2,
    )

    with patch_context:
        out, state = patched_kernel()(**chunk_inputs)

    rtol, atol = (1e-3, 1e-3) if case == "gated_deltanet" else (1e-4, 1e-4)
    torch.testing.assert_close(out, expected_out, rtol=rtol, atol=atol)
    torch.testing.assert_close(state, expected_state, rtol=rtol, atol=atol)


@pytest.mark.filterwarnings(
    "ignore:ShortConvolution is crucial to the performance.*:UserWarning"
)
def test_final_state_readout_gated_deltanet_backbone_training_supports_backward():
    if not torch.cuda.is_available():
        pytest.skip("FLA final-state readout requires CUDA/Triton.")
    if not torch.cuda.is_bf16_supported():
        pytest.skip("This regression test mirrors the bf16 training path.")

    torch.manual_seed(0)
    from pfns.model.backbones import FLABackboneConfig

    config_kwargs = fla_model_config_kwargs("gated_deltanet")
    config_kwargs["use_short_conv"] = False
    backbone = FLABackboneConfig(
        model_type="gated_deltanet",
        config_kwargs=config_kwargs,
        final_state_readout=True,
    ).create_backbone(
        ninp=int(config_kwargs["hidden_size"]),
        attention_between_features=False,
    )
    backbone.train()
    backbone = backbone.to("cuda")
    x = torch.randn(
        1,
        6,
        1,
        int(config_kwargs["hidden_size"]),
        device="cuda",
        requires_grad=True,
    )

    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = backbone(x, single_eval_pos=3)
    out.float().sum().backward()

    assert x.grad is not None
    assert any(
        parameter.grad is not None
        for parameter in backbone.parameters()
        if parameter.requires_grad
    )


def test_final_state_readout_eval_test_tokens_are_independent():
    if not torch.cuda.is_available():
        pytest.skip("FLA final-state readout requires CUDA/Triton.")

    torch.manual_seed(0)
    backbone = build_fla_backbone(
        "linear_attn",
        sequence_mode="Comb_MT",
        final_state_readout=True,
    ).to("cuda")
    backbone.eval()

    x = torch.randn(2, 6, 1, fla_hidden_size("linear_attn"), device="cuda")
    x_perturbed = x.clone()
    x_perturbed[:, 5] += 10.0

    with torch.no_grad():
        out = backbone(x, single_eval_pos=3)
        out_perturbed = backbone(x_perturbed, single_eval_pos=3)

    torch.testing.assert_close(out[:, 3:5], out_perturbed[:, 3:5])
    assert not torch.allclose(out[:, 5], out_perturbed[:, 5])


def test_final_state_readout_cached_stateless_path_skips_self_term():
    import fla.layers.delta_net as deltanet_layer

    backbone = build_fla_backbone("deltanet", final_state_readout=True)

    torch.manual_seed(0)
    q = torch.randn(2, 1, 3, 4)
    k = torch.randn(2, 1, 3, 4)
    v = torch.randn(2, 1, 3, 6)
    beta = torch.rand(2, 1, 3)
    initial_state = torch.randn(2, 3, 4, 6)
    scale = 0.5

    with ExitStack() as stack:
        for ctx in backbone._patch_contexts(use_custom_recurrent=True):
            stack.enter_context(ctx)
        out, _ = deltanet_layer.fused_recurrent_delta_rule(
            q=q,
            k=k,
            v=v,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
        )

    expected = torch.einsum("bthd,bhdm->bthm", q * scale, initial_state)
    self_term = (q * scale * k).sum(-1, keepdim=True) * beta.unsqueeze(-1)
    self_term = self_term * (v - torch.einsum("bthd,bhdm->bthm", k, initial_state))

    torch.testing.assert_close(out, expected)
    assert not torch.allclose(out, expected + self_term)


@pytest.mark.parametrize("model_type", ["gla", "kda", "gated_deltanet"])
def test_final_state_readout_cached_stateless_path_skips_query_decay(model_type: str):
    torch.manual_seed(0)
    backbone = build_fla_backbone(model_type, final_state_readout=True)

    batch_size = 2
    seq_len = 1
    num_heads = 3
    key_dim = 4
    value_dim = 6
    q = torch.randn(batch_size, seq_len, num_heads, key_dim)
    k = torch.randn(batch_size, seq_len, num_heads, key_dim)
    v = torch.randn(batch_size, seq_len, num_heads, value_dim)
    initial_state = torch.randn(batch_size, num_heads, key_dim, value_dim)
    scale = 0.5

    with ExitStack() as stack:
        for ctx in backbone._patch_contexts(use_custom_recurrent=True):
            stack.enter_context(ctx)
        if model_type == "gla":
            import fla.layers.gla as gla_layer

            g = torch.randn(batch_size, seq_len, num_heads, key_dim).clamp(-2.0, 2.0)
            out, _ = gla_layer.fused_recurrent_gla(
                q=q,
                k=k,
                v=v,
                gk=g,
                scale=scale,
                initial_state=initial_state,
            )
            q_read = q * scale
            q_decayed = q_read * g.exp()
        elif model_type == "kda":
            import fla.layers.kda as kda_layer

            g = torch.randn(batch_size, seq_len, num_heads, key_dim).clamp(-2.0, 2.0)
            out, _ = kda_layer.fused_recurrent_kda(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=torch.rand(batch_size, seq_len, num_heads),
                scale=scale,
                initial_state=initial_state,
                use_qk_l2norm_in_kernel=False,
            )
            q_read = q * scale
            q_decayed = q_read * g.exp()
        elif model_type == "gated_deltanet":
            import fla.layers.gated_deltanet as gated_deltanet_layer

            g = torch.randn(batch_size, seq_len, num_heads).clamp(-2.0, 2.0)
            out, _ = gated_deltanet_layer.fused_recurrent_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=torch.rand(batch_size, seq_len, num_heads),
                scale=scale,
                initial_state=initial_state,
                use_qk_l2norm_in_kernel=False,
            )
            q_read = q * scale
            q_decayed = q_read * g.exp().unsqueeze(-1)
        else:
            raise AssertionError(f"Unhandled model_type={model_type}")

    expected = torch.einsum("bthd,bhdm->bthm", q_read, initial_state)
    decayed = torch.einsum("bthd,bhdm->bthm", q_decayed, initial_state)

    torch.testing.assert_close(out, expected)
    assert not torch.allclose(out, decayed)


def test_final_state_readout_stateless_gla_handles_4d_and_5d_readout_shapes():
    torch.manual_seed(0)

    import fla.layers.gla as gla_layer
    from pfns.model.fla_patches import _maybe_patch_gla_with_stateless_recurrent

    batch_size = 2
    flat_len = 3
    num_heads = 2
    key_dim = 4
    value_dim = 5
    scale = 0.5

    q = torch.randn(batch_size * flat_len, 1, num_heads, key_dim)
    k = torch.randn(batch_size * flat_len, 1, num_heads, key_dim)
    v = torch.randn(batch_size * flat_len, 1, num_heads, value_dim)
    gk = torch.randn(batch_size * flat_len, 1, num_heads, key_dim)
    initial_state = torch.randn(batch_size, num_heads, key_dim, value_dim)

    expected_4d = torch.einsum(
        "blthk,bhkv->blthv",
        q.reshape(batch_size, flat_len, 1, num_heads, key_dim) * scale,
        initial_state,
    ).reshape(batch_size * flat_len, 1, num_heads, value_dim)

    with _maybe_patch_gla_with_stateless_recurrent(
        True,
        final_state_readout=True,
    ):
        out_4d, _ = gla_layer.fused_recurrent_gla(
            q=q,
            k=k,
            v=v,
            gk=gk,
            scale=scale,
            initial_state=initial_state,
        )

        out_5d, _ = gla_layer.fused_recurrent_gla(
            q=q.reshape(batch_size, flat_len, 1, num_heads, key_dim),
            k=k.reshape(batch_size, flat_len, 1, num_heads, key_dim),
            v=v.reshape(batch_size, flat_len, 1, num_heads, value_dim),
            gk=gk.reshape(batch_size, flat_len, 1, num_heads, key_dim),
            scale=scale,
            initial_state=initial_state,
        )

    torch.testing.assert_close(out_4d, expected_4d)
    torch.testing.assert_close(
        out_5d,
        expected_4d.reshape(batch_size, flat_len, 1, num_heads, value_dim),
    )


@pytest.mark.parametrize("model_type", FLA_MODEL_TYPES)
def test_fla_test_cache_matches_naive(model_type: str):
    if not torch.cuda.is_available():
        pytest.skip("FLA backend requires CUDA/Triton for this test.")

    torch.manual_seed(0)
    backbone = build_fla_backbone(model_type, train=True)
    if model_type == "gated_deltanet":
        backbone.eval()
    device = torch.device("cuda")
    backbone = backbone.to(device)

    batch_size = 2
    seq_len = 20
    num_tokens = 1
    embed_dim = fla_hidden_size(model_type)
    train_len = 12
    assert 0 < train_len < seq_len

    x = torch.randn(batch_size, seq_len, num_tokens, embed_dim, device=device)
    x_batched = x.transpose(1, 2).reshape(batch_size * num_tokens, seq_len, embed_dim)
    train_x = x_batched[:, :train_len]
    test_x = x_batched[:, train_len:]
    test_len = test_x.size(1)
    assert test_len > 1
    use_custom_reference = model_type == "mesanet"

    with torch.no_grad():
        out_full, _ = backbone._run_fla(x_batched)
        out_full_test = out_full[:, train_len:]

        _, past_1 = backbone._run_fla(train_x)
        assert past_1 is not None
        out_naive = backbone._run_test_with_cache_naive(
            test_x,
            past_1,
            use_custom_recurrent=use_custom_reference,
            use_custom_shortconv=use_custom_reference,
        )

        _, past_2 = backbone._run_fla(train_x)
        assert past_2 is not None
        out_fast = backbone._run_test_with_cache(test_x, past_2)

        _, past_3 = backbone._run_fla(train_x)
        assert past_3 is not None
        perm = torch.randperm(test_len, device=device)
        test_x_swapped = test_x[:, perm, :]
        out_swapped = backbone._run_test_with_cache(test_x_swapped, past_3)
        inv_perm = torch.argsort(perm)
        out_swapped = out_swapped[:, inv_perm, :]

        # perturb a different test token and ensure another position is unchanged
        _, past_4 = backbone._run_fla(train_x)
        assert past_4 is not None
        test_x_pert = test_x.clone()
        test_x_pert[:, 0:1, :] += 10.0
        out_pert = backbone._run_test_with_cache(test_x_pert, past_4)
        
        _, past_5 = backbone._run_fla(train_x)
        assert past_5 is not None
        out_cached_repeat = _run_repeated_cache_reference(
            backbone,
            test_x,
            past_5,
            use_custom_recurrent=use_custom_reference,
            use_custom_shortconv=use_custom_reference,
        )
    
    rtol, atol = fla_cache_equivalence_tolerances(model_type)
    

    torch.testing.assert_close(out_fast, out_naive, rtol=rtol, atol=atol)
    torch.testing.assert_close(out_cached_repeat, out_fast, rtol=rtol, atol=atol)
    torch.testing.assert_close(out_cached_repeat, out_naive, rtol=rtol, atol=atol)
    torch.testing.assert_close(out_fast, out_swapped, rtol=rtol, atol=atol)
    torch.testing.assert_close(out_fast[:, 1:2, :], out_pert[:, 1:2, :], rtol=rtol, atol=atol)
    assert not torch.allclose(out_full_test, out_fast, rtol=1e-8, atol=1e-8)
        

@pytest.mark.parametrize("model_type", FLA_MODEL_TYPES)
def test_fla_cache_allows_train_gradients(model_type: str):
    if not torch.cuda.is_available():
        pytest.skip("FLA backend requires CUDA/Triton for this test.")

    torch.manual_seed(0)
    backbone_naive = build_fla_backbone(model_type, train=True)
    backbone_fast = build_fla_backbone(model_type, train=True)
    backbone_fast.load_state_dict(backbone_naive.state_dict())
    device = torch.device("cuda")
    backbone_naive = backbone_naive.to(device)
    backbone_fast = backbone_fast.to(device)

    batch_size = 2
    seq_len = 20
    num_tokens = 1
    embed_dim = fla_hidden_size(model_type)
    train_len = 12
    assert 0 < train_len < seq_len

    x = torch.randn(batch_size, seq_len, num_tokens, embed_dim, device=device)
    x_batched = x.transpose(1, 2).reshape(batch_size * num_tokens, seq_len, embed_dim)
    train_x_base = x_batched[:, :train_len].detach()
    test_x = x_batched[:, train_len:].detach()

    train_x_naive = train_x_base.clone().requires_grad_(True)
    _, past_naive = backbone_naive._run_fla(train_x_naive)
    assert past_naive is not None
    # For mamba2, native FLA doesn't support gradients through cache, so use custom recurrent
    use_custom_for_naive = model_type in {"mamba2", "mesanet"}
    out_naive = backbone_naive._run_test_with_cache_naive(
        test_x, past_naive, use_custom_recurrent=use_custom_for_naive, use_custom_shortconv=True # to allow gradients through shortconv cache
    )
    out_naive.sum().backward()

    train_x_fast = train_x_base.clone().requires_grad_(True)
    _, past_fast = backbone_fast._run_fla(train_x_fast)
    assert past_fast is not None
    out_fast = backbone_fast._run_test_with_cache(test_x, past_fast)
    out_fast.sum().backward()

    assert train_x_naive.grad is not None, f"train_x_naive.grad is None for {model_type}"
    assert train_x_fast.grad is not None, f"train_x_fast.grad is None for {model_type}"
    
    rtol, atol = fla_tolerances(model_type)
    torch.testing.assert_close(train_x_fast.grad, train_x_naive.grad, rtol=rtol, atol=atol)

@pytest.mark.parametrize("model_type", FLA_MODEL_TYPES)
def test_fla_cache_chunking_matches_gradients(model_type: str):
    if not torch.cuda.is_available():
        pytest.skip("FLA backend requires CUDA/Triton for this test.")

    torch.manual_seed(0)
    backbone_full = build_fla_backbone(model_type, cache_chunk_size=None, train=True)
    backbone_chunked = build_fla_backbone(model_type, cache_chunk_size=4, train=True)
    backbone_chunked.load_state_dict(backbone_full.state_dict())
    device = torch.device("cuda")
    backbone_full = backbone_full.to(device)
    backbone_chunked = backbone_chunked.to(device)

    batch_size = 2
    seq_len = 20
    num_tokens = 1
    embed_dim = fla_hidden_size(model_type)
    train_len = 10
    assert 0 < train_len < seq_len

    x = torch.randn(batch_size, seq_len, num_tokens, embed_dim, device=device)
    x_batched = x.transpose(1, 2).reshape(batch_size * num_tokens, seq_len, embed_dim)
    train_x_base = x_batched[:, :train_len].detach()
    test_x_base = x_batched[:, train_len:].detach()
    assert test_x_base.size(1) > 4

    train_x_full = train_x_base.clone().requires_grad_(True)
    test_x_full = test_x_base.clone().requires_grad_(True)
    _, past_full = backbone_full._run_fla(train_x_full)
    assert past_full is not None
    use_custom_recurrent = model_type == "mesanet"
    out_full = backbone_full._run_test_with_cache(
        test_x_full,
        past_full,
        use_custom_recurrent=use_custom_recurrent,
        use_custom_shortconv=use_custom_recurrent,
    )
    out_full.sum().backward()

    train_x_chunked = train_x_base.clone().requires_grad_(True)
    test_x_chunked = test_x_base.clone().requires_grad_(True)
    _, past_chunked = backbone_chunked._run_fla(train_x_chunked)
    assert past_chunked is not None
    out_chunked = backbone_chunked._run_test_with_cache(
        test_x_chunked,
        past_chunked,
        use_custom_recurrent=use_custom_recurrent,
        use_custom_shortconv=use_custom_recurrent,
    )
    out_chunked.sum().backward()

    torch.testing.assert_close(train_x_chunked.grad, train_x_full.grad, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(test_x_chunked.grad, test_x_full.grad, rtol=1e-4, atol=1e-4)
    
    for (name_full, param_full), (name_chunked, param_chunked) in zip(
        backbone_full.named_parameters(), backbone_chunked.named_parameters()
    ):
        assert name_full == name_chunked
        if param_full.grad is None or param_chunked.grad is None:
            assert param_full.grad is None and param_chunked.grad is None
            continue
        torch.testing.assert_close(param_chunked.grad, param_full.grad, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("model_type", FLA_MODEL_TYPES)
def test_stateless_matches_repeated_cache_outputs_and_grads(model_type: str):
    if not torch.cuda.is_available():
        pytest.skip("FLA backend requires CUDA/Triton for this test.")

    torch.manual_seed(0)
    backbone_stateless = build_fla_backbone(model_type, train=True)
    backbone_reference = build_fla_backbone(model_type, train=True)
    backbone_reference.load_state_dict(backbone_stateless.state_dict())
    device = torch.device("cuda")
    backbone_stateless = backbone_stateless.to(device)
    backbone_reference = backbone_reference.to(device)

    batch_size = 2
    seq_len = 10
    num_tokens = 1
    embed_dim = fla_hidden_size(model_type)
    train_len = 6
    assert 0 < train_len < seq_len

    x = torch.randn(batch_size, seq_len, num_tokens, embed_dim, device=device)
    x_batched = x.transpose(1, 2).reshape(batch_size * num_tokens, seq_len, embed_dim)
    train_x_base = x_batched[:, :train_len].detach()
    test_x_base = x_batched[:, train_len:].detach()
    test_len = test_x_base.size(1)

    train_x_stateless = train_x_base.clone().requires_grad_(True)
    test_x_stateless = test_x_base.clone().requires_grad_(True)
    _, past_stateless = backbone_stateless._run_fla(train_x_stateless)
    assert past_stateless is not None
    out_stateless = backbone_stateless._run_test_with_cache(test_x_stateless, past_stateless)
    out_stateless.sum().backward()

    train_x_ref = train_x_base.clone().requires_grad_(True)
    test_x_ref = test_x_base.clone().requires_grad_(True)
    _, past_ref = backbone_reference._run_fla(train_x_ref)
    assert past_ref is not None

    use_custom_reference = model_type == "mesanet"
    out_ref = _run_repeated_cache_reference(
        backbone_reference,
        test_x_ref,
        past_ref,
        use_custom_recurrent=use_custom_reference,
        use_custom_shortconv=True,
    )
    
    out_ref.sum().backward()

    rtol, atol = fla_tolerances(model_type)

    torch.testing.assert_close(out_stateless, out_ref, rtol=rtol, atol=atol)
    
    # Mamba2's native mamba_chunk_scan_combined doesn't support gradients through initial_states,
    # so we only compare gradients for non-mamba2 models
    if model_type != "mamba2":
        torch.testing.assert_close(train_x_stateless.grad, train_x_ref.grad, rtol=rtol, atol=atol)
        torch.testing.assert_close(test_x_stateless.grad, test_x_ref.grad, rtol=rtol, atol=atol)


@pytest.mark.parametrize("model_type", FLA_MODEL_TYPES)
@pytest.mark.parametrize("batch_size,test_len,size", [(1, 4, "small"), (2, 1, "small"), (2, 6, "medium")])
def test_edge_cases(model_type: str, batch_size: int, test_len: int, size: str):
    if not torch.cuda.is_available():
        pytest.skip("FLA backend requires CUDA/Triton for this test.")

    torch.manual_seed(42)
    backbone = build_fla_backbone(model_type, train=True, size=size)
    device = torch.device("cuda")
    backbone = backbone.to(device)

    embed_dim = fla_hidden_size(model_type, size=size)
    train_len = 8
    train_x = torch.randn(batch_size, train_len, embed_dim, device=device)
    test_x = torch.randn(batch_size, test_len, embed_dim, device=device)

    with torch.no_grad():
        _, past = backbone._run_fla(train_x)
        out_fast = backbone._run_test_with_cache(test_x, past)
        use_custom_reference = model_type == "mesanet"
        out_naive = backbone._run_test_with_cache_naive(
            test_x,
            backbone._copy_cache(past),
            use_custom_recurrent=use_custom_reference,
            use_custom_shortconv=use_custom_reference,
        )

    rtol, atol = fla_tolerances(model_type)
    torch.testing.assert_close(out_fast, out_naive, rtol=rtol, atol=atol)


@pytest.mark.parametrize("model_type", FLA_MODEL_TYPES)
def test_model_parameter_gradients(model_type: str):
    """Test that model parameter gradients match between naive and fast implementations."""
    if not torch.cuda.is_available():
        pytest.skip("FLA backend requires CUDA/Triton for this test.")

    torch.manual_seed(0)
    backbone_naive = build_fla_backbone(model_type, train=True)
    backbone_fast = build_fla_backbone(model_type, train=True)
    backbone_fast.load_state_dict(backbone_naive.state_dict())
    device = torch.device("cuda")
    backbone_naive = backbone_naive.to(device)
    backbone_fast = backbone_fast.to(device)

    batch_size = 2
    embed_dim = fla_hidden_size(model_type)
    train_len = 8
    test_len = 4

    train_x = torch.randn(batch_size, train_len, embed_dim, device=device, requires_grad=True)
    test_x = torch.randn(batch_size, test_len, embed_dim, device=device)

    train_x_naive = train_x.detach().clone().requires_grad_(True)
    _, past_naive = backbone_naive._run_fla(train_x_naive)
    # mamba2 & gated_deltanet: native kernels don't support correct gradients through cache for model parameters
    use_custom = model_type in {"mamba2", "gated_deltanet", "mesanet"}
    out_naive = backbone_naive._run_test_with_cache_naive(
        test_x, past_naive, use_custom_recurrent=use_custom, use_custom_shortconv=True
    )
    out_naive.sum().backward()

    train_x_fast = train_x.detach().clone().requires_grad_(True)
    _, past_fast = backbone_fast._run_fla(train_x_fast)
    out_fast = backbone_fast._run_test_with_cache(test_x, past_fast)
    out_fast.sum().backward()

    rtol, atol = fla_tolerances(model_type)
    if model_type == "mamba2":
        rtol, atol = max(rtol, 5e-3), max(atol, 5e-3)
    
    for (name_naive, param_naive), (name_fast, param_fast) in zip(
        backbone_naive.named_parameters(), backbone_fast.named_parameters()
    ):
        assert name_naive == name_fast, f"Parameter name mismatch: {name_naive} vs {name_fast}"
        if param_naive.grad is None and param_fast.grad is None:
            continue
        assert param_naive.grad is not None, f"param_naive.grad is None for {name_naive}"
        assert param_fast.grad is not None, f"param_fast.grad is None for {name_fast}"
        torch.testing.assert_close(
            param_fast.grad, param_naive.grad, rtol=rtol, atol=atol,
            msg=f"Gradient mismatch for parameter {name_naive}"
        )


@pytest.mark.parametrize("model_type", FLA_MODEL_TYPES)
def test_outputs_different_from_autoregressive(model_type: str):
    """Stateless parallel outputs should differ from autoregressive full-sequence outputs."""
    if not torch.cuda.is_available():
        pytest.skip("FLA backend requires CUDA/Triton for this test.")

    torch.manual_seed(42)
    backbone = build_fla_backbone(model_type, train=True)
    device = torch.device("cuda")
    backbone = backbone.to(device)

    batch_size = 2
    embed_dim = fla_hidden_size(model_type)
    train_len = 8
    test_len = 4
    full_len = train_len + test_len

    full_x = torch.randn(batch_size, full_len, embed_dim, device=device)
    train_x = full_x[:, :train_len]
    test_x = full_x[:, train_len:]

    with torch.no_grad():
        out_full, _ = backbone._run_fla(full_x)
        out_full_test = out_full[:, train_len:]

        _, past = backbone._run_fla(train_x)
        out_cached = backbone._run_test_with_cache(test_x, past)

    assert not torch.allclose(out_full_test, out_cached, rtol=1e-4, atol=1e-4), (
        "Stateless cached output should differ from full autoregressive output"
    )


@pytest.mark.parametrize("model_type", FLA_MODEL_TYPES)
@pytest.mark.parametrize("train_len", [128, 256])
def test_long_training_context(model_type: str, train_len: int):
    """Test with long training contexts to stress-test cache handling.
    Note: DeltaNet requires bfloat16 for sequences ≥64 tokens (chunk kernel limitation).
    """
    if not torch.cuda.is_available():
        pytest.skip("FLA backend requires CUDA/Triton for this test.")

    torch.manual_seed(42)
    backbone = build_fla_backbone(model_type, train=True)
    device = torch.device("cuda")
    
    # DeltaNet's chunk kernel requires bfloat16
    use_bf16 = model_type == "deltanet"
    if use_bf16:
        backbone = backbone.to(device).to(torch.bfloat16)
    else:
        backbone = backbone.to(device)

    batch_size = 2
    embed_dim = fla_hidden_size(model_type)
    test_len = 8
    
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    train_x = torch.randn(batch_size, train_len, embed_dim, device=device, dtype=dtype)
    test_x = torch.randn(batch_size, test_len, embed_dim, device=device, dtype=dtype)

    with torch.no_grad():
        _, past = backbone._run_fla(train_x)
        assert past is not None
        out_fast = backbone._run_test_with_cache(test_x, past)
        use_custom_reference = model_type == "mesanet"
        out_naive = backbone._run_test_with_cache_naive(
            test_x,
            backbone._copy_cache(past),
            use_custom_recurrent=use_custom_reference,
            use_custom_shortconv=use_custom_reference,
        )

    rtol, atol = fla_tolerances(model_type)
    if use_bf16:
        rtol, atol = max(rtol, 5e-3), max(atol, 5e-3)
    torch.testing.assert_close(out_fast, out_naive, rtol=rtol, atol=atol)


if __name__ == "__main__":
    test_fla_test_cache_matches_naive("deltanet")
    test_fla_cache_allows_train_gradients("deltanet")
    test_fla_cache_chunking_matches_gradients("deltanet")
    test_stateless_matches_repeated_cache_outputs_and_grads("deltanet")
