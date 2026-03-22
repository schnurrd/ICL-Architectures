import pytest
import torch

pytest.importorskip("fla")

from pfns.model.backbones import FLABackboneConfig
from pfns.model.fla_memetic_init import (
    _MEMETIC_OPEN_GATE_BIAS,
    apply_memetic_fla_init,
)
from tests.model.fla_test_utils import build_fla_backbone


def _assert_encoder_decoder_identity(encoder: torch.nn.Linear, decoder: torch.nn.Linear) -> None:
    composed = decoder.weight @ encoder.weight
    expected = torch.eye(
        composed.shape[0],
        device=composed.device,
        dtype=composed.dtype,
    )
    torch.testing.assert_close(composed, expected)


def _query_key_cosine_mean(attn: torch.nn.Module) -> torch.Tensor:
    overlap = min(attn.q_proj.weight.shape[0], attn.k_proj.weight.shape[0])
    return torch.nn.functional.cosine_similarity(
        attn.q_proj.weight[:overlap],
        attn.k_proj.weight[:overlap],
        dim=-1,
    ).mean()


def _build_grouped_kv_backbone(*, memetic_init: bool) -> torch.nn.Module:
    config = FLABackboneConfig(
        model_type="gla",
        config_kwargs={
            "hidden_size": 8,
            "num_hidden_layers": 1,
            "num_heads": 4,
            "num_kv_heads": 1,
            "intermediate_size": 32,
            "hidden_act": "swish",
            "norm_eps": 1e-5,
            "use_cache": True,
        },
        memetic_init=memetic_init,
    )
    return config.create_backbone(ninp=8, attention_between_features=False)


def test_gla_memetic_gate_init() -> None:
    backbone = build_fla_backbone("gla", size="small", memetic_init=True, train=False)
    attn = backbone.fla.layers[0].attn

    assert attn.use_output_gate is True
    assert _query_key_cosine_mean(attn) > 0.9
    assert not torch.allclose(attn.q_proj.weight, attn.k_proj.weight)
    _assert_encoder_decoder_identity(attn.v_proj, attn.o_proj)
    torch.testing.assert_close(attn.gk_proj[1].weight, torch.zeros_like(attn.gk_proj[1].weight))
    assert hasattr(attn, "g_proj")
    assert attn.g_proj.bias is None
    torch.testing.assert_close(
        attn.gk_proj[1].bias,
        torch.full_like(attn.gk_proj[1].bias, _MEMETIC_OPEN_GATE_BIAS),
    )

def test_memetic_init_changes_query_key_structure() -> None:
    torch.manual_seed(0)
    baseline = build_fla_backbone("gla", size="small", memetic_init=False, train=False)
    torch.manual_seed(0)
    memetic = build_fla_backbone("gla", size="small", memetic_init=True, train=False)

    assert _query_key_cosine_mean(memetic.fla.layers[0].attn) > _query_key_cosine_mean(
        baseline.fla.layers[0].attn
    )


def test_memetic_init_applies_to_all_layers_by_default() -> None:
    torch.manual_seed(0)
    baseline = build_fla_backbone("gla", size="medium", memetic_init=False, train=False)
    torch.manual_seed(0)
    memetic = build_fla_backbone("gla", size="medium", memetic_init=True, train=False)

    first_layer = memetic.fla.layers[0].attn
    last_layer = memetic.fla.layers[-1].attn
    baseline_last = baseline.fla.layers[-1].attn

    assert _query_key_cosine_mean(first_layer) > 0.9
    assert _query_key_cosine_mean(last_layer) > 0.9
    assert not torch.allclose(last_layer.q_proj.weight, baseline_last.q_proj.weight)
    assert not torch.allclose(last_layer.k_proj.weight, baseline_last.k_proj.weight)


def test_memetic_init_can_target_specific_layers() -> None:
    torch.manual_seed(0)
    baseline = build_fla_backbone("gla", size="medium", memetic_init=False, train=False)
    torch.manual_seed(0)
    memetic = build_fla_backbone("gla", size="medium", memetic_init=False, train=False)

    apply_memetic_fla_init(memetic.fla, layer_indices=[1])

    unchanged_first = memetic.fla.layers[0].attn
    changed_middle = memetic.fla.layers[1].attn
    unchanged_last = memetic.fla.layers[-1].attn
    baseline_first = baseline.fla.layers[0].attn
    baseline_middle = baseline.fla.layers[1].attn
    baseline_last = baseline.fla.layers[-1].attn

    torch.testing.assert_close(unchanged_first.q_proj.weight, baseline_first.q_proj.weight)
    assert not torch.allclose(changed_middle.q_proj.weight, baseline_middle.q_proj.weight)
    torch.testing.assert_close(unchanged_last.q_proj.weight, baseline_last.q_proj.weight)


def test_grouped_kv_memetic_init_builds_and_aligns_query_key_heads() -> None:
    backbone = _build_grouped_kv_backbone(memetic_init=True)
    attn = backbone.fla.layers[0].attn

    repeated_k = (
        attn.k_proj.weight.view(attn.num_kv_heads, attn.head_k_dim, -1)
        .repeat_interleave(attn.num_kv_groups, dim=0)
        .reshape_as(attn.q_proj.weight)
    )
    cosine = torch.nn.functional.cosine_similarity(
        attn.q_proj.weight,
        repeated_k,
        dim=-1,
    )
    assert torch.all(cosine > 0.9)


def test_memetic_init_rejects_short_conv_by_default() -> None:
    config = FLABackboneConfig(
        model_type="gla",
        config_kwargs={
            "hidden_size": 8,
            "num_hidden_layers": 1,
            "num_heads": 2,
            "intermediate_size": 32,
            "hidden_act": "swish",
            "norm_eps": 1e-5,
            "use_cache": True,
            "use_short_conv": True,
        },
        memetic_init=False,
    )
    backbone = config.create_backbone(ninp=8, attention_between_features=False)

    with pytest.raises(ValueError, match="use_short_conv=False"):
        apply_memetic_fla_init(backbone.fla)


def test_memetic_init_rejects_unsupported_models() -> None:
    backbone = build_fla_backbone("gated_deltanet", size="small", memetic_init=False, train=False)

    with pytest.raises(ValueError, match="Expected at least one GatedLinearAttention layer"):
        apply_memetic_fla_init(backbone.fla)
