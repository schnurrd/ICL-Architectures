import pytest
import torch

pytest.importorskip("fla")

from pfns.model.backbones import BidirectionalFLALayer
from tests.model.fla_test_utils import (
    FLA_MODEL_TYPES,
    build_fla_backbone,
    fla_hidden_size,
    fla_model_config_kwargs,
)


def _enable_bidirectional_fusion(backbone: torch.nn.Module) -> None:
    with torch.no_grad():
        for layer in backbone.layers:
            if isinstance(layer, BidirectionalFLALayer):
                layer.fusion_out.weight.normal_(mean=0.0, std=0.2)
                if layer.fusion_out.bias is not None:
                    layer.fusion_out.bias.zero_()


@pytest.mark.parametrize("model_type", FLA_MODEL_TYPES)
@pytest.mark.parametrize("sequence_mode", ["Comb_MT", "Int_MT"])
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


@pytest.mark.parametrize("model_type", FLA_MODEL_TYPES)
def test_bidirectional_wraps_every_layer(model_type: str) -> None:
    backbone = build_fla_backbone(model_type, size="small", bidirectional=True)
    expected_num_layers = int(
        fla_model_config_kwargs(model_type, size="small")["num_hidden_layers"]
    )

    assert len(backbone.layers) == expected_num_layers
    assert all(isinstance(layer, BidirectionalFLALayer) for layer in backbone.layers)


def test_bidirectional_mamba2_incontext_fit_uses_single_model_pass() -> None:
    if not torch.cuda.is_available():
        pytest.skip("FLA backend requires CUDA/Triton for this test.")

    torch.manual_seed(0)
    device = torch.device("cuda")
    backbone = build_fla_backbone("mamba2", size="small", bidirectional=True).to(device)
    embed_dim = fla_hidden_size("mamba2", size="small")
    train_x = torch.randn(2, 5, embed_dim, device=device)

    call_count = 0
    original_forward = backbone.fla.forward

    def counted_forward(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_forward(*args, **kwargs)

    backbone.fla.forward = counted_forward

    with torch.no_grad():
        _, state = backbone.incontext_fit(train_x)

    assert call_count == 1
    assert state["cache_params"] is not None


@pytest.mark.parametrize("model_type", FLA_MODEL_TYPES)
def test_bidirectional_incontext_fit_matches_non_cached_train_pass(model_type: str) -> None:
    if not torch.cuda.is_available():
        pytest.skip("FLA backend requires CUDA/Triton for this test.")

    torch.manual_seed(0)
    device = torch.device("cuda")
    backbone = build_fla_backbone(model_type, size="small", bidirectional=True).to(device)
    _enable_bidirectional_fusion(backbone)
    embed_dim = fla_hidden_size(model_type, size="small")
    train_x = torch.randn(2, 5, embed_dim, device=device)

    with torch.no_grad():
        direct_out, _ = backbone._run_fla(train_x, return_cache=False)
        fit_out, _ = backbone.incontext_fit(train_x)

    torch.testing.assert_close(fit_out, direct_out, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("model_type", FLA_MODEL_TYPES)
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


@pytest.mark.parametrize("model_type", FLA_MODEL_TYPES)
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


@pytest.mark.parametrize("model_type", FLA_MODEL_TYPES)
def test_bidirectional_cached_prediction_matches_helper_paths(model_type: str) -> None:
    if not torch.cuda.is_available():
        pytest.skip("FLA backend requires CUDA/Triton for this test.")

    torch.manual_seed(0)
    device = torch.device("cuda")
    backbone = build_fla_backbone(model_type, size="small", bidirectional=True).to(device)
    embed_dim = fla_hidden_size(model_type, size="small")
    train_x = torch.randn(2, 4, embed_dim, device=device)
    test_x = torch.randn(2, 3, embed_dim, device=device)

    with torch.no_grad():
        _, state = backbone.incontext_fit(train_x)
        parallel_out = backbone._run_test_with_cache(
            test_x,
            backbone._copy_cache(state["cache_params"]),
        )
        naive_out = backbone._run_test_with_cache_naive(
            test_x,
            backbone._copy_cache(state["cache_params"]),
        )
        predict_out = backbone.incontext_predict(test_x, state)

    torch.testing.assert_close(parallel_out, naive_out, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(predict_out, parallel_out, rtol=1e-5, atol=1e-5)
