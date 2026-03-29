import pytest
import torch

pytest.importorskip("fla")

from pfns.base_config import BaseConfig
from pfns.model.backbones import FLABackboneConfig
from pfns.model.fla_mimetic_init import (
    _MIMETIC_A_LOG,
    _MIMETIC_DT_BIAS,
    _MIMETIC_OPEN_GATE_BIAS,
    _MIMETIC_OUTPUT_GATE_SWISH_NEUTRAL_BIAS,
    apply_mimetic_fla_init,
)
from tests.model.fla_test_utils import build_fla_backbone


def test_mimetic_init_modes_smoke() -> None:
    gla_gates = build_fla_backbone("gla", size="small", mimetic_init=True, mimetic_init_mode="gates", train=False)
    gla_full = build_fla_backbone("gla", size="small", mimetic_init=True, mimetic_init_mode="full", train=False)
    gdn_gates = build_fla_backbone(
        "gated_deltanet", size="small", mimetic_init=True, mimetic_init_mode="gates", train=False
    )
    gdn_full = build_fla_backbone(
        "gated_deltanet", size="small", mimetic_init=True, mimetic_init_mode="full", train=False
    )

    gla_gates_attn = gla_gates.fla.layers[0].attn
    torch.testing.assert_close(
        gla_gates_attn.gk_proj[1].bias,
        torch.full_like(gla_gates_attn.gk_proj[1].bias, _MIMETIC_OPEN_GATE_BIAS),
    )
    assert gla_gates_attn.g_proj.bias is not None
    torch.testing.assert_close(gla_gates_attn.g_proj.weight, torch.zeros_like(gla_gates_attn.g_proj.weight))
    torch.testing.assert_close(
        gla_gates_attn.g_proj.bias,
        torch.full_like(gla_gates_attn.g_proj.bias, _MIMETIC_OUTPUT_GATE_SWISH_NEUTRAL_BIAS),
    )
    assert not torch.allclose(gla_full.fla.layers[0].attn.q_proj.weight, gla_gates_attn.q_proj.weight)

    gdn_gates_attn = gdn_gates.fla.layers[0].attn
    gdn_full_attn = gdn_full.fla.layers[0].attn
    torch.testing.assert_close(gdn_gates_attn.a_proj.weight, torch.zeros_like(gdn_gates_attn.a_proj.weight))
    torch.testing.assert_close(gdn_gates_attn.A_log, torch.full_like(gdn_gates_attn.A_log, _MIMETIC_A_LOG))
    torch.testing.assert_close(gdn_gates_attn.dt_bias, torch.full_like(gdn_gates_attn.dt_bias, _MIMETIC_DT_BIAS))
    assert not torch.allclose(gdn_full_attn.q_proj.weight, gdn_gates_attn.q_proj.weight)


def test_mimetic_init_rejects_unsupported_models() -> None:
    unsupported = build_fla_backbone("deltanet", size="small", mimetic_init=False, train=False)

    with pytest.raises(ValueError, match="Expected at least one supported mimetic-init layer"):
        apply_mimetic_fla_init(unsupported.fla)
