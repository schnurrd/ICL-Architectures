import pytest
import torch

pytest.importorskip("fla")

from pfns.model.fla_mimetic_init import (
    _MIMETIC_A_LOG,
    _MIMETIC_DT_BIAS,
    _MIMETIC_OPEN_GATE_BIAS,
    _MIMETIC_OUTPUT_GATE_SWISH_NEUTRAL_BIAS,
    _normalize_mimetic_init_mode,
    apply_mimetic_fla_init,
)
from tests.model.fla_test_utils import build_fla_backbone


def _assert_block_identity(linear: torch.nn.Linear) -> None:
    expected = torch.zeros_like(linear.weight)
    diag = min(linear.weight.shape)
    expected[:diag, :diag] = torch.eye(diag, device=linear.weight.device, dtype=linear.weight.dtype)
    torch.testing.assert_close(linear.weight, expected)
    if linear.bias is not None:
        torch.testing.assert_close(linear.bias, torch.zeros_like(linear.bias))


def test_mimetic_init_mode_behavior() -> None:
    torch.manual_seed(0)
    gdn_baseline = build_fla_backbone("gated_deltanet", size="small", mimetic_init=False, train=False)
    torch.manual_seed(0)
    gla_gate_only = build_fla_backbone(
        "gla", size="small", mimetic_init=True, mimetic_init_mode="gate_only", train=False
    )
    gla_full_with_output_gate = build_fla_backbone(
        "gla",
        size="small",
        mimetic_init=True,
        mimetic_init_mode="full_with_output_gate",
        train=False,
    )
    gla_full_without_output_gate = build_fla_backbone(
        "gla",
        size="small",
        mimetic_init=True,
        mimetic_init_mode="full_without_output_gate",
        train=False,
    )
    torch.manual_seed(0)
    gdn_gate_only = build_fla_backbone(
        "gated_deltanet",
        size="small",
        mimetic_init=True,
        mimetic_init_mode="gate_only",
        train=False,
    )
    gdn_full_with_output_gate = build_fla_backbone(
        "gated_deltanet",
        size="small",
        mimetic_init=True,
        mimetic_init_mode="full_with_output_gate",
        train=False,
    )
    gla_gate_only_attn = gla_gate_only.fla.layers[0].attn
    torch.testing.assert_close(
        gla_gate_only_attn.gk_proj[1].bias,
        torch.full_like(gla_gate_only_attn.gk_proj[1].bias, _MIMETIC_OPEN_GATE_BIAS),
    )
    torch.testing.assert_close(
        gla_gate_only_attn.g_proj.bias,
        torch.full_like(gla_gate_only_attn.g_proj.bias, _MIMETIC_OUTPUT_GATE_SWISH_NEUTRAL_BIAS),
    )
    gla_full_with_output_gate_attn = gla_full_with_output_gate.fla.layers[0].attn
    gla_full_without_output_gate_attn = gla_full_without_output_gate.fla.layers[0].attn
    _assert_block_identity(gla_full_with_output_gate_attn.q_proj)
    assert gla_full_without_output_gate_attn.g_proj.bias is None

    gdn_gate_only_attn = gdn_gate_only.fla.layers[0].attn
    gdn_full_with_output_gate_attn = gdn_full_with_output_gate.fla.layers[0].attn
    gdn_baseline_attn = gdn_baseline.fla.layers[0].attn
    torch.testing.assert_close(
        gdn_gate_only_attn.a_proj.weight, torch.zeros_like(gdn_gate_only_attn.a_proj.weight)
    )
    torch.testing.assert_close(gdn_gate_only_attn.A_log, gdn_baseline_attn.A_log)
    torch.testing.assert_close(gdn_gate_only_attn.dt_bias, gdn_baseline_attn.dt_bias)
    torch.testing.assert_close(
        gdn_full_with_output_gate_attn.A_log,
        torch.full_like(gdn_full_with_output_gate_attn.A_log, _MIMETIC_A_LOG),
    )
    torch.testing.assert_close(
        gdn_full_with_output_gate_attn.dt_bias,
        torch.full_like(gdn_full_with_output_gate_attn.dt_bias, _MIMETIC_DT_BIAS),
    )
    _assert_block_identity(gdn_full_with_output_gate_attn.q_proj)
    assert gdn_gate_only_attn.g_proj.bias is not None


def test_mimetic_init_mode_normalization_and_validation() -> None:
    assert _normalize_mimetic_init_mode(True) == "gate_only"
    assert _normalize_mimetic_init_mode(False) == "full_with_output_gate"

    with pytest.raises(
        ValueError,
        match="mode must be one of 'gate_only', 'full_with_output_gate', or 'full_without_output_gate'",
    ):
        _normalize_mimetic_init_mode("gates")


def test_mimetic_init_rejects_unsupported_models() -> None:
    unsupported = build_fla_backbone("deltanet", size="small", mimetic_init=False, train=False)

    with pytest.raises(ValueError, match="Expected at least one supported mimetic-init layer"):
        apply_mimetic_fla_init(unsupported.fla)
