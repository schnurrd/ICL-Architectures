import pytest
import torch

pytest.importorskip("fla")

from pfns.model.backbones import (
    _MEMETIC_A_LOG,
    _MEMETIC_DT_BIAS,
    _MEMETIC_OPEN_GATE_BIAS,
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


def test_gla_memetic_gate_init() -> None:
    backbone = build_fla_backbone("gla", size="small", memetic_init=True, train=False)
    attn = backbone.fla.layers[0].attn

    torch.testing.assert_close(attn.q_proj.weight, _expected_block_identity_like(attn.q_proj.weight))
    torch.testing.assert_close(attn.k_proj.weight, _expected_block_identity_like(attn.k_proj.weight))
    _assert_encoder_decoder_identity(attn.v_proj, attn.o_proj)
    torch.testing.assert_close(attn.gk_proj[0].weight, torch.zeros_like(attn.gk_proj[0].weight))
    torch.testing.assert_close(attn.gk_proj[1].weight, torch.zeros_like(attn.gk_proj[1].weight))
    torch.testing.assert_close(
        attn.gk_proj[1].bias,
        torch.full_like(attn.gk_proj[1].bias, _MEMETIC_OPEN_GATE_BIAS),
    )

def test_gated_deltanet_memetic_gate_init() -> None:
    backbone = build_fla_backbone("gated_deltanet", size="small", memetic_init=True, train=False)
    attn = backbone.fla.layers[0].attn

    torch.testing.assert_close(attn.q_proj.weight, _expected_block_identity_like(attn.q_proj.weight))
    torch.testing.assert_close(attn.k_proj.weight, _expected_block_identity_like(attn.k_proj.weight))
    _assert_encoder_decoder_identity(attn.v_proj, attn.o_proj)
    torch.testing.assert_close(attn.a_proj.weight, torch.zeros_like(attn.a_proj.weight))
    torch.testing.assert_close(attn.A_log, torch.full_like(attn.A_log, _MEMETIC_A_LOG))
    torch.testing.assert_close(attn.dt_bias, torch.full_like(attn.dt_bias, _MEMETIC_DT_BIAS))
