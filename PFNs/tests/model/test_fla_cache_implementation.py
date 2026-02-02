import pytest
import torch

pytest.importorskip("fla")

from pfns.model.backbones import FLABackboneConfig
from pfns.model.fla_patches import (
    _maybe_patch_shortconv_forward_pytorch,
)
MODEL_TYPES = ("gla", "kda", "deltanet", "gated_deltanet", "mamba2")


def _model_config_kwargs(model_type: str) -> dict[str, object]:
    if model_type == "gla":
        return {
            "hidden_size": 8,
            "num_hidden_layers": 2,
            "num_heads": 2,
            "intermediate_size": 32,
            "hidden_act": "swish",
            "norm_eps": 1e-5,
            "use_cache": True,
        }
    if model_type == "mamba2":
        return {
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "state_size": 64,
            "conv_kernel": 4,
            "intermediate_size": 128,
            "num_heads": 2, 
            "use_cache": True,
        }
    if model_type == "kda":
        return {
            "hidden_size": 8,
            "num_hidden_layers": 2,
            "num_heads": 2,
            "intermediate_size": 32,
            "hidden_act": "swish",
            "norm_eps": 1e-5,
            "use_cache": True,
            "use_short_conv": True,
        }
    if model_type == "deltanet":
        return {
            "hidden_size": 8,
            "num_hidden_layers": 2,
            "num_heads": 2,
            "intermediate_size": 32,
            "hidden_act": "swish",
            "norm_eps": 1e-5,
            "use_cache": True,
            "use_short_conv": True,
        }
    if model_type == "gated_deltanet":
        return {
            "hidden_size": 8,
            "num_hidden_layers": 2,
            "num_heads": 2,
            "head_dim": 4,
            "intermediate_size": 32,
            "hidden_act": "swish",
            "norm_eps": 1e-5,
            "use_cache": True,
            "use_short_conv": True,
        }
    raise ValueError(f"Unsupported model_type: {model_type}")


def _build_backbone(model_type: str, cache_chunk_size: int | None = None, train: bool = False) -> torch.nn.Module:
    config = FLABackboneConfig(
        model_type=model_type,
        config_kwargs=_model_config_kwargs(model_type),
        cache_chunk_size=cache_chunk_size,
    )
    backbone = config.create_backbone(ninp=8, attention_between_features=False)
    if train:
        backbone.train()
    else:
        backbone.eval()
    return backbone


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_fla_test_cache_matches_naive(model_type: str):
    if not torch.cuda.is_available():
        pytest.skip("FLA backend requires CUDA/Triton for this test.")

    torch.manual_seed(0)
    backbone = _build_backbone(model_type, train=True)
    if model_type == "gated_deltanet":
        backbone.eval()
    device = torch.device("cuda")
    backbone = backbone.to(device)

    batch_size = 2
    seq_len = 20
    num_tokens = 1
    embed_dim = 8
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
    
    if model_type in ("kda", "deltanet", "gated_deltanet", "mamba2"):
        rtol, atol = 1e-4, 1e-4
    else:
        rtol, atol = 1e-6, 1e-6
    

    torch.testing.assert_close(out_fast, out_naive, rtol=rtol, atol=atol)
    torch.testing.assert_close(out_cached_repeat, out_fast, rtol=rtol, atol=atol)
    torch.testing.assert_close(out_cached_repeat, out_naive, rtol=rtol, atol=atol)
    torch.testing.assert_close(out_fast, out_swapped, rtol=rtol, atol=atol)
    torch.testing.assert_close(out_fast[:, 1:2, :], out_pert[:, 1:2, :], rtol=rtol, atol=atol)
    assert not torch.allclose(out_full_test, out_fast, rtol=1e-8, atol=1e-8)
        

@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_fla_cache_allows_train_gradients(model_type: str):
    if not torch.cuda.is_available():
        pytest.skip("FLA backend requires CUDA/Triton for this test.")

    torch.manual_seed(0)
    backbone_naive = _build_backbone(model_type, train=True)
    backbone_fast = _build_backbone(model_type, train=True)
    backbone_fast.load_state_dict(backbone_naive.state_dict())
    device = torch.device("cuda")
    backbone_naive = backbone_naive.to(device)
    backbone_fast = backbone_fast.to(device)

    batch_size = 2
    seq_len = 20
    num_tokens = 1
    embed_dim = 8
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
    
    if model_type in {"deltanet", "mamba2"}:
        rtol, atol = 1e-3, 1e-3
    elif model_type in {"kda", "gated_deltanet"}:
        rtol, atol = 1e-4, 1e-4
    else:
        rtol, atol = 1e-6, 1e-6
    torch.testing.assert_close(train_x_fast.grad, train_x_naive.grad, rtol=rtol, atol=atol)

@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_fla_cache_chunking_matches_gradients(model_type: str):
    if not torch.cuda.is_available():
        pytest.skip("FLA backend requires CUDA/Triton for this test.")

    torch.manual_seed(0)
    backbone_full = _build_backbone(model_type, cache_chunk_size=None, train=True)
    backbone_chunked = _build_backbone(model_type, cache_chunk_size=4, train=True)
    backbone_chunked.load_state_dict(backbone_full.state_dict())
    device = torch.device("cuda")
    backbone_full = backbone_full.to(device)
    backbone_chunked = backbone_chunked.to(device)

    batch_size = 2
    seq_len = 20
    num_tokens = 1
    embed_dim = 8
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


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_stateless_matches_repeated_cache_outputs_and_grads(model_type: str):
    if not torch.cuda.is_available():
        pytest.skip("FLA backend requires CUDA/Triton for this test.")

    torch.manual_seed(0)
    backbone_stateless = _build_backbone(model_type, train=True)
    backbone_reference = _build_backbone(model_type, train=True)
    backbone_reference.load_state_dict(backbone_stateless.state_dict())
    device = torch.device("cuda")
    backbone_stateless = backbone_stateless.to(device)
    backbone_reference = backbone_reference.to(device)

    batch_size = 2
    seq_len = 10
    num_tokens = 1
    embed_dim = 8
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
    
    if model_type == "mamba2":
        # Mamba2 uses cache_params (not past_key_values) and requires use_cache=True
        cache_position = torch.arange(train_len, train_len + 1, device=device).unsqueeze(0).expand(test_x_flat.size(0), -1)
        out_ref = backbone_reference.fla(
            inputs_embeds=test_x_flat,
            cache_params=repeated_cache,
            cache_position=cache_position,
            use_cache=True,
            return_dict=True,
        ).last_hidden_state
    else:
        with _maybe_patch_shortconv_forward_pytorch(True):
            out_ref = backbone_reference.fla(
                inputs_embeds=test_x_flat,
                past_key_values=repeated_cache,
                use_cache=False,
                return_dict=True,
            ).last_hidden_state
    out_ref = out_ref.view(batch_size * num_tokens, test_len, embed_dim)
    
    out_ref.sum().backward()

    if model_type in {"deltanet", "mamba2"}:
        rtol, atol = 1e-3, 1e-3
    elif model_type in {"kda", "gated_deltanet"}:
        rtol, atol = 1e-4, 1e-4
    else:
        rtol, atol = 1e-6, 1e-6

    torch.testing.assert_close(out_stateless, out_ref, rtol=rtol, atol=atol)
    
    # Mamba2's native mamba_chunk_scan_combined doesn't support gradients through initial_states,
    # so we only compare gradients for non-mamba2 models
    if model_type != "mamba2":
        torch.testing.assert_close(train_x_stateless.grad, train_x_ref.grad, rtol=rtol, atol=atol)
        torch.testing.assert_close(test_x_stateless.grad, test_x_ref.grad, rtol=rtol, atol=atol)

if __name__ == "__main__":
    test_fla_test_cache_matches_naive("deltanet")
    test_fla_cache_allows_train_gradients("deltanet")
    test_fla_cache_chunking_matches_gradients("deltanet")
    test_stateless_matches_repeated_cache_outputs_and_grads("deltanet")