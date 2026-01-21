import pytest
import torch

pytest.importorskip("fla")

from pfns.model.backbones import FLABackboneConfig


def _build_backbone(cache_chunk_size: int | None = None) -> torch.nn.Module:
    config = FLABackboneConfig(
        model_type="gla",
        config_kwargs={
            "hidden_size": 8,
            "num_hidden_layers": 2,
            "num_heads": 2,
            "intermediate_size": 32,
            "hidden_act": "swish",
            "norm_eps": 1e-5,
            "use_cache": True,
        },
        cache_chunk_size=cache_chunk_size,
    )
    backbone = config.create_backbone(ninp=8, attention_between_features=False)
    backbone.eval()
    return backbone


def test_fla_test_cache_matches_naive():
    if not torch.cuda.is_available():
        pytest.skip("FLA GLA backend requires CUDA/Triton for this test.")

    torch.manual_seed(0)
    backbone = _build_backbone()
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

    torch.testing.assert_close(out_fast, out_naive, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(out_fast, out_swapped, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(out_fast[:, 1:2, :], out_pert[:, 1:2, :], rtol=1e-6, atol=1e-6)
    assert not torch.allclose(out_full_test, out_fast, rtol=1e-6, atol=1e-6)
        

def test_fla_cache_allows_train_gradients():
    if not torch.cuda.is_available():
        pytest.skip("FLA GLA backend requires CUDA/Triton for this test.")

    torch.manual_seed(0)
    backbone_naive = _build_backbone()
    backbone_fast = _build_backbone()
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
    out_naive = backbone_naive._run_test_with_cache_naive(test_x, past_naive, use_custom_recurrent=False)
    out_naive.sum().backward()

    train_x_fast = train_x_base.clone().requires_grad_(True)
    _, past_fast = backbone_fast._run_fla(train_x_fast)
    assert past_fast is not None
    out_fast = backbone_fast._run_test_with_cache(test_x, past_fast)
    out_fast.sum().backward()

    assert train_x_naive.grad is not None
    assert train_x_fast.grad is not None
    torch.testing.assert_close(train_x_fast.grad, train_x_naive.grad, rtol=5e-3, atol=1e-3)


def test_fla_cache_chunking_matches_gradients():
    if not torch.cuda.is_available():
        pytest.skip("FLA GLA backend requires CUDA/Triton for this test.")

    torch.manual_seed(0)
    backbone_full = _build_backbone(cache_chunk_size=None)
    backbone_chunked = _build_backbone(cache_chunk_size=4)
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
    out_full = backbone_full._run_test_with_cache(test_x_full, past_full, use_custom_recurrent=False)
    out_full.sum().backward()

    train_x_chunked = train_x_base.clone().requires_grad_(True)
    test_x_chunked = test_x_base.clone().requires_grad_(True)
    _, past_chunked = backbone_chunked._run_fla(train_x_chunked)
    assert past_chunked is not None
    out_chunked = backbone_chunked._run_test_with_cache(test_x_chunked, past_chunked, use_custom_recurrent=False)
    out_chunked.sum().backward()

    torch.testing.assert_close(train_x_chunked.grad, train_x_full.grad, rtol=5e-3, atol=1e-3)
    torch.testing.assert_close(test_x_chunked.grad, test_x_full.grad, rtol=5e-3, atol=1e-3)
    
    for (name_full, param_full), (name_chunked, param_chunked) in zip(
        backbone_full.named_parameters(), backbone_chunked.named_parameters()
    ):
        assert name_full == name_chunked
        if param_full.grad is None or param_chunked.grad is None:
            assert param_full.grad is None and param_chunked.grad is None
            continue
        torch.testing.assert_close(param_chunked.grad, param_full.grad, rtol=5e-3, atol=1e-3)


if __name__ == "__main__":
    test_fla_test_cache_matches_naive()
    test_fla_cache_allows_train_gradients()
    test_fla_cache_chunking_matches_gradients()
