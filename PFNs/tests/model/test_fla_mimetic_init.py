import pytest
import torch

pytest.importorskip("fla")

from pfns.model.fla_mimetic_init import (
    _MIMETIC_A_LOG,
    _MIMETIC_DT_BIAS,
    _MIMETIC_OPEN_GATE_BIAS,
    apply_mimetic_fla_init,
)
from tests.model.fla_test_utils import build_fla_backbone


def _expected_block_identity_like(weight: torch.Tensor) -> torch.Tensor:
    expected = torch.zeros_like(weight)
    diag = min(weight.shape)
    expected[:diag, :diag] = torch.eye(diag, device=weight.device, dtype=weight.dtype)
    return expected


def _assert_encoder_decoder_identity(encoder: torch.nn.Linear, decoder: torch.nn.Linear) -> None:
    composed = decoder.weight @ encoder.weight
    expected = torch.eye(
        composed.shape[0],
        device=composed.device,
        dtype=composed.dtype,
    )
    torch.testing.assert_close(composed, expected)


def test_gla_mimetic_gate_init() -> None:
    backbone = build_fla_backbone("gla", size="small", mimetic_init=True, train=False)
    attn = backbone.fla.layers[0].attn

    torch.testing.assert_close(attn.q_proj.weight, _expected_block_identity_like(attn.q_proj.weight))
    torch.testing.assert_close(attn.k_proj.weight, _expected_block_identity_like(attn.k_proj.weight))
    _assert_encoder_decoder_identity(attn.v_proj, attn.o_proj)
    torch.testing.assert_close(attn.gk_proj[0].weight, torch.zeros_like(attn.gk_proj[0].weight))
    torch.testing.assert_close(attn.gk_proj[1].weight, torch.zeros_like(attn.gk_proj[1].weight))
    torch.testing.assert_close(
        attn.gk_proj[1].bias,
        torch.full_like(attn.gk_proj[1].bias, _MIMETIC_OPEN_GATE_BIAS),
    )


def test_mimetic_init_applies_to_all_layers_by_default() -> None:
    baseline = build_fla_backbone("gla", size="medium", mimetic_init=False, train=False)
    mimetic = build_fla_backbone("gla", size="medium", mimetic_init=True, train=False)

    for idx in range(len(mimetic.fla.layers)):
        baseline_attn = baseline.fla.layers[idx].attn
        mimetic_attn = mimetic.fla.layers[idx].attn
        assert not torch.allclose(mimetic_attn.q_proj.weight, baseline_attn.q_proj.weight)


def test_mimetic_init_honors_explicit_layer_indices() -> None:
    torch.manual_seed(0)
    baseline = build_fla_backbone("gla", size="medium", mimetic_init=False, train=False)
    torch.manual_seed(0)
    targeted = build_fla_backbone(
        "gla",
        size="medium",
        mimetic_init=True,
        mimetic_init_layer_indices=[1],
        train=False,
    )

    torch.testing.assert_close(targeted.fla.layers[0].attn.q_proj.weight, baseline.fla.layers[0].attn.q_proj.weight)
    assert not torch.allclose(targeted.fla.layers[1].attn.q_proj.weight, baseline.fla.layers[1].attn.q_proj.weight)
    torch.testing.assert_close(targeted.fla.layers[-1].attn.q_proj.weight, baseline.fla.layers[-1].attn.q_proj.weight)


def test_mimetic_init_supports_gated_deltanet() -> None:
    backbone = build_fla_backbone("gated_deltanet", size="small", mimetic_init=True, train=False)
    attn = backbone.fla.layers[0].attn

    torch.testing.assert_close(attn.q_proj.weight, _expected_block_identity_like(attn.q_proj.weight))
    torch.testing.assert_close(attn.k_proj.weight, _expected_block_identity_like(attn.k_proj.weight))
    _assert_encoder_decoder_identity(attn.v_proj, attn.o_proj)
    torch.testing.assert_close(attn.a_proj.weight, torch.zeros_like(attn.a_proj.weight))
    torch.testing.assert_close(attn.A_log, torch.full_like(attn.A_log, _MIMETIC_A_LOG))
    torch.testing.assert_close(attn.dt_bias, torch.full_like(attn.dt_bias, _MIMETIC_DT_BIAS))


def test_mimetic_init_rejects_unsupported_models() -> None:
    unsupported = build_fla_backbone("deltanet", size="small", mimetic_init=False, train=False)

    with pytest.raises(ValueError, match="Expected at least one supported mimetic-init layer"):
        apply_mimetic_fla_init(unsupported.fla)
