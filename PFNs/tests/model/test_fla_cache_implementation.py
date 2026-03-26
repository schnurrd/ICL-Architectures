import pytest
import torch

pytest.importorskip("fla")
from fla.layers.gla import fused_recurrent_gla
from fla.layers.linear_attn import fused_recurrent_linear_attn

from pfns.model.fla_patches import (
    _maybe_patch_shortconv_forward_pytorch,
)
from tests.model.fla_test_utils import (
    fla_cache_equivalence_tolerances,
    FLA_MODEL_TYPES,
    build_fla_backbone,
    fla_hidden_size,
    fla_tolerances,
)


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

    with torch.no_grad():
        out_full, _ = backbone._run_fla(x_batched)
        out_full_test = out_full[:, train_len:]

        _, past_1 = backbone._run_fla(train_x)
        assert past_1 is not None
        out_naive = backbone._run_test_with_cache_naive(test_x, past_1, use_custom_recurrent=False)

        _, past_2 = backbone._run_fla(train_x)
        assert past_2 is not None
        out_fast = backbone._run_test_with_cache(test_x, past_2, cache_position_start=train_len)

        _, past_3 = backbone._run_fla(train_x)
        assert past_3 is not None
        perm = torch.randperm(test_len, device=device)
        test_x_swapped = test_x[:, perm, :]
        out_swapped = backbone._run_test_with_cache(test_x_swapped, past_3, cache_position_start=train_len)
        inv_perm = torch.argsort(perm)
        out_swapped = out_swapped[:, inv_perm, :]

        # perturb a different test token and ensure another position is unchanged
        _, past_4 = backbone._run_fla(train_x)
        assert past_4 is not None
        test_x_pert = test_x.clone()
        test_x_pert[:, 0:1, :] += 10.0
        out_pert = backbone._run_test_with_cache(test_x_pert, past_4, cache_position_start=train_len)
        
        _, past_5 = backbone._run_fla(train_x)
        assert past_5 is not None
        out_cached_repeat = backbone._run_test_with_cache(test_x, past_5, cache_position_start=train_len, use_custom_recurrent=False)
    
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
    use_custom_for_naive = model_type == "mamba2"
    out_naive = backbone_naive._run_test_with_cache_naive(
        test_x, past_naive, use_custom_recurrent=use_custom_for_naive, use_custom_shortconv=True # to allow gradients through shortconv cache
    )
    out_naive.sum().backward()

    train_x_fast = train_x_base.clone().requires_grad_(True)
    _, past_fast = backbone_fast._run_fla(train_x_fast)
    assert past_fast is not None
    out_fast = backbone_fast._run_test_with_cache(test_x, past_fast, cache_position_start=train_len)
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
    out_full = backbone_full._run_test_with_cache(test_x_full, past_full, use_custom_recurrent=False, cache_position_start=train_len)
    out_full.sum().backward()

    train_x_chunked = train_x_base.clone().requires_grad_(True)
    test_x_chunked = test_x_base.clone().requires_grad_(True)
    _, past_chunked = backbone_chunked._run_fla(train_x_chunked)
    assert past_chunked is not None
    out_chunked = backbone_chunked._run_test_with_cache(test_x_chunked, past_chunked, use_custom_recurrent=False, cache_position_start=train_len)
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

    repeated_cache = backbone_reference._repeat_cache(past_ref, test_len)
    test_x_flat = test_x_ref.contiguous().view(batch_size * num_tokens * test_len, 1, embed_dim)

    out_ref, _ = backbone_reference._run_fla(
        test_x_flat,
        cache_params=repeated_cache,
        cache_position_start=train_len,
        return_cache=False,
        use_custom_recurrent=False,
        use_custom_shortconv=True,
    )
    out_ref = out_ref.view(batch_size * num_tokens, test_len, embed_dim)
    
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
        out_fast = backbone._run_test_with_cache(test_x, past, cache_position_start=train_len)
        out_naive = backbone._run_test_with_cache_naive(test_x, backbone._copy_cache(past), use_custom_recurrent=False)

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
    use_custom = model_type in {"mamba2", "gated_deltanet"}
    out_naive = backbone_naive._run_test_with_cache_naive(
        test_x, past_naive, use_custom_recurrent=use_custom, use_custom_shortconv=True
    )
    out_naive.sum().backward()

    train_x_fast = train_x.detach().clone().requires_grad_(True)
    _, past_fast = backbone_fast._run_fla(train_x_fast)
    out_fast = backbone_fast._run_test_with_cache(test_x, past_fast, cache_position_start=train_len)
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
        out_cached = backbone._run_test_with_cache(test_x, past, cache_position_start=train_len)

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
        out_fast = backbone._run_test_with_cache(test_x, past, cache_position_start=train_len)
        out_naive = backbone._run_test_with_cache_naive(test_x, backbone._copy_cache(past), use_custom_recurrent=False)

    rtol, atol = fla_tolerances(model_type)
    if use_bf16:
        rtol, atol = max(rtol, 5e-3), max(atol, 5e-3)
    torch.testing.assert_close(out_fast, out_naive, rtol=rtol, atol=atol)


if __name__ == "__main__":
    test_fla_test_cache_matches_naive("deltanet")
    test_fla_cache_allows_train_gradients("deltanet")
    test_fla_cache_chunking_matches_gradients("deltanet")
    test_stateless_matches_repeated_cache_outputs_and_grads("deltanet")
