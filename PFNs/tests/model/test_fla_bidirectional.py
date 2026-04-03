import pytest
import torch

pytest.importorskip("fla")

from pfns.model.backbones import (
    BidirectionalFLACache,
    BidirectionalFLALayer,
    FusedBidirectionalFLACache,
)
from tests.model.fla_test_utils import (
    FLA_MODEL_TYPES,
    build_fla_backbone,
    fla_hidden_size,
    fla_model_config_kwargs,
)

BIDIRECTIONAL_FLA_MODEL_TYPES = tuple(
    model_type for model_type in FLA_MODEL_TYPES if model_type != "mamba2"
)


def _enable_bidirectional_fusion(backbone: torch.nn.Module) -> None:
    with torch.no_grad():
        for layer in backbone.layers:
            if isinstance(layer, BidirectionalFLALayer) and layer.fusion_out is not None:
                layer.fusion_out.weight.normal_(mean=0.0, std=0.2)
                if layer.fusion_out.bias is not None:
                    layer.fusion_out.bias.zero_()


@pytest.mark.parametrize("model_type", FLA_MODEL_TYPES)
@pytest.mark.parametrize("sequence_mode", ["Int_ST", "Comb_MT", "Int_MT"])
def test_bidirectional_rejects_unsupported_sequence_modes(
    model_type: str,
    sequence_mode: str,
) -> None:
    with pytest.raises(ValueError, match="supports only sequence_mode"):
        build_fla_backbone(
            model_type,
            size="small",
            sequence_mode=sequence_mode,
            bidirectional=True,
        )


@pytest.mark.parametrize("model_type", BIDIRECTIONAL_FLA_MODEL_TYPES)
def test_bidirectional_wraps_every_layer(model_type: str) -> None:
    backbone = build_fla_backbone(model_type, size="small", bidirectional=True)
    expected_num_layers = int(
        fla_model_config_kwargs(model_type, size="small")["num_hidden_layers"]
    )

    assert len(backbone.layers) == expected_num_layers
    assert all(isinstance(layer, BidirectionalFLALayer) for layer in backbone.layers)


def test_bidirectional_rejects_mamba2() -> None:
    with pytest.raises(ValueError, match="does not support model_type='mamba2'"):
        build_fla_backbone("mamba2", size="small", bidirectional=True)


def test_non_bidirectional_ignores_bidirectional_state_fusion() -> None:
    backbone = build_fla_backbone(
        "gla",
        size="small",
        bidirectional=False,
        bidirectional_state_fusion="mean_output_two_cache",
    )

    assert backbone is not None


def test_bidirectional_state_fusion_requires_shared_weights() -> None:
    with pytest.raises(ValueError, match="bidirectional_state_fusion requires bidirectional_share_weights=True"):
        build_fla_backbone(
            "gla",
            size="small",
            bidirectional=True,
            bidirectional_share_weights=False,
            bidirectional_state_fusion="mean_output_mean_cache",
        )


def test_bidirectional_layer_mean_state_fusion_averages_hidden_states() -> None:
    layer = BidirectionalFLALayer(
        torch.nn.Identity(),
        hidden_size=4,
        state_fusion="mean_output_two_cache",
    )
    assert layer.fusion_out is None
    forward_hidden = torch.tensor([[[1.0, 3.0, 5.0, 7.0]]])
    backward_hidden = torch.tensor([[[3.0, 5.0, 7.0, 9.0]]])

    fused_hidden = layer._fuse_hidden_states(forward_hidden, backward_hidden)

    torch.testing.assert_close(fused_hidden, (forward_hidden + backward_hidden) / 2)


def test_bidirectional_layer_mean_fused_cache_linear_output_keeps_fusion_out() -> None:
    layer = BidirectionalFLALayer(
        torch.nn.Identity(),
        hidden_size=4,
        state_fusion="linear_output_mean_cache",
    )

    assert layer.fusion_out is not None


@pytest.mark.parametrize("model_type", BIDIRECTIONAL_FLA_MODEL_TYPES)
def test_bidirectional_incontext_fit_builds_forward_and_backward_caches(model_type: str) -> None:
    if not torch.cuda.is_available():
        pytest.skip("FLA backend requires CUDA/Triton for this test.")

    torch.manual_seed(0)
    device = torch.device("cuda")
    backbone = build_fla_backbone(
        model_type,
        size="small",
        bidirectional=True,
        bidirectional_state_fusion="linear_output_two_cache",
    ).to(device)
    embed_dim = fla_hidden_size(model_type, size="small")
    train_x = torch.randn(2, 5, embed_dim, device=device)

    with torch.no_grad():
        _, state = backbone.incontext_fit(train_x)

    cache = state["cache_params"]
    assert isinstance(cache, BidirectionalFLACache)
    assert cache.forward_cache is not None
    assert cache.backward_cache is not None


@pytest.mark.parametrize("model_type", BIDIRECTIONAL_FLA_MODEL_TYPES)
def test_bidirectional_default_state_fusion_returns_mean_fused_prediction_cache(model_type: str) -> None:
    if not torch.cuda.is_available():
        pytest.skip("FLA backend requires CUDA/Triton for this test.")

    torch.manual_seed(0)
    device = torch.device("cuda")
    backbone = build_fla_backbone(model_type, size="small", bidirectional=True).to(device)
    embed_dim = fla_hidden_size(model_type, size="small")
    train_x = torch.randn(2, 5, embed_dim, device=device)

    with torch.no_grad():
        _, state = backbone.incontext_fit(train_x)

    cache = state["cache_params"]
    assert isinstance(cache, FusedBidirectionalFLACache)
    assert cache.state_fusion == "mean_output_mean_cache"


@pytest.mark.parametrize("model_type", BIDIRECTIONAL_FLA_MODEL_TYPES)
def test_bidirectional_mean_state_fusion_keeps_bidirectional_prediction_cache(model_type: str) -> None:
    if not torch.cuda.is_available():
        pytest.skip("FLA backend requires CUDA/Triton for this test.")

    torch.manual_seed(0)
    device = torch.device("cuda")
    backbone = build_fla_backbone(
        model_type,
        size="small",
        bidirectional=True,
        bidirectional_state_fusion="mean_output_two_cache",
    ).to(device)
    embed_dim = fla_hidden_size(model_type, size="small")
    train_x = torch.randn(2, 5, embed_dim, device=device)
    test_x = torch.randn(2, 3, embed_dim, device=device)

    with torch.no_grad():
        _, state = backbone.incontext_fit(train_x)
        out = backbone.incontext_predict(test_x, state)

    cache = state["cache_params"]
    assert isinstance(cache, BidirectionalFLACache)
    assert cache.forward_cache is not None
    assert cache.backward_cache is not None
    assert out.shape == test_x.shape
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("state_fusion", ["mean_output_mean_cache", "linear_output_mean_cache"])
@pytest.mark.parametrize("model_type", BIDIRECTIONAL_FLA_MODEL_TYPES)
def test_bidirectional_state_fusion_returns_fused_prediction_cache(
    model_type: str,
    state_fusion: str,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("FLA backend requires CUDA/Triton for this test.")

    torch.manual_seed(0)
    device = torch.device("cuda")
    backbone = build_fla_backbone(
        model_type,
        size="small",
        bidirectional=True,
        bidirectional_state_fusion=state_fusion,
    ).to(device)
    embed_dim = fla_hidden_size(model_type, size="small")
    train_x = torch.randn(2, 5, embed_dim, device=device)
    test_x = torch.randn(2, 3, embed_dim, device=device)

    with torch.no_grad():
        _, state = backbone.incontext_fit(train_x)
        out = backbone.incontext_predict(test_x, state)

    cache = state["cache_params"]
    assert isinstance(cache, FusedBidirectionalFLACache)
    assert cache.state_fusion == state_fusion
    assert cache.cache is not None
    assert out.shape == test_x.shape
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("model_type", BIDIRECTIONAL_FLA_MODEL_TYPES)
def test_bidirectional_train_tokens_can_use_future_train_context(model_type: str) -> None:
    if not torch.cuda.is_available():
        pytest.skip("FLA backend requires CUDA/Triton for this test.")

    torch.manual_seed(0)
    device = torch.device("cuda")
    bidirectional = build_fla_backbone(model_type, size="small", bidirectional=True).to(device)
    _enable_bidirectional_fusion(bidirectional)
    embed_dim = fla_hidden_size(model_type, size="small")
    train_x = torch.randn(2, 5, embed_dim, device=device)
    perturbed_train_x = train_x.clone()
    perturbed_train_x[:, -1, :] += 10.0

    torch.manual_seed(0)
    causal = build_fla_backbone(model_type, size="small", bidirectional=False).to(device)

    with torch.no_grad():
        bidirectional_out, _ = bidirectional.incontext_fit(train_x)
        bidirectional_out_perturbed, _ = bidirectional.incontext_fit(perturbed_train_x)
        causal_out, _ = causal.incontext_fit(train_x)
        causal_out_perturbed, _ = causal.incontext_fit(perturbed_train_x)

    assert not torch.allclose(
        bidirectional_out[:, :-1, :],
        bidirectional_out_perturbed[:, :-1, :],
        rtol=1e-5,
        atol=1e-5,
    )
    torch.testing.assert_close(
        causal_out[:, :-1, :],
        causal_out_perturbed[:, :-1, :],
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.parametrize("model_type", BIDIRECTIONAL_FLA_MODEL_TYPES)
def test_bidirectional_test_tokens_are_independent(model_type: str) -> None:
    if not torch.cuda.is_available():
        pytest.skip("FLA backend requires CUDA/Triton for this test.")

    torch.manual_seed(0)
    device = torch.device("cuda")
    backbone = build_fla_backbone(model_type, size="small", bidirectional=True).to(device)
    embed_dim = fla_hidden_size(model_type, size="small")
    train_x = torch.randn(2, 4, embed_dim, device=device)
    test_x = torch.randn(2, 3, embed_dim, device=device)
    perturbed_test_x = test_x.clone()
    perturbed_test_x[:, 0, :] += 10.0

    with torch.no_grad():
        _, state = backbone.incontext_fit(train_x)
        out = backbone.incontext_predict(test_x, state)
        out_perturbed = backbone.incontext_predict(perturbed_test_x, state)

    torch.testing.assert_close(out[:, 1:, :], out_perturbed[:, 1:, :], rtol=1e-5, atol=1e-5)
