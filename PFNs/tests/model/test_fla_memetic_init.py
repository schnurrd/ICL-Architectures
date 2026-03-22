import pytest
import torch

pytest.importorskip("fla")

from pfns.model.backbones import FLABackboneConfig
from pfns.model.fla_memetic_init import _repeat_head_blocks, apply_memetic_fla_init
from tests.model.fla_test_utils import build_fla_backbone


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


def test_memetic_init_defaults_to_middle_layer_only() -> None:
    torch.manual_seed(0)
    baseline = build_fla_backbone("gla", size="medium", memetic_init=False, train=False)
    torch.manual_seed(0)
    memetic = build_fla_backbone("gla", size="medium", memetic_init=True, train=False)

    first_layer = memetic.fla.layers[0].attn
    middle_layer = memetic.fla.layers[1].attn
    last_layer = memetic.fla.layers[-1].attn

    torch.testing.assert_close(first_layer.q_proj.weight, baseline.fla.layers[0].attn.q_proj.weight)
    assert _query_key_cosine_mean(middle_layer) > _query_key_cosine_mean(
        baseline.fla.layers[1].attn
    )
    torch.testing.assert_close(last_layer.q_proj.weight, baseline.fla.layers[-1].attn.q_proj.weight)


def test_memetic_init_all_layers_override_changes_last_layer() -> None:
    torch.manual_seed(0)
    baseline = build_fla_backbone("gla", size="medium", memetic_init=False, train=False)
    torch.manual_seed(0)
    memetic = build_fla_backbone(
        "gla",
        size="medium",
        memetic_init=True,
        memetic_init_layer_indices=None,
        train=False,
    )

    assert _query_key_cosine_mean(memetic.fla.layers[-1].attn) > _query_key_cosine_mean(
        baseline.fla.layers[-1].attn
    )


def test_memetic_init_supports_grouped_kv() -> None:
    attn = _build_grouped_kv_backbone(memetic_init=True).fla.layers[0].attn
    repeated_k = (
        attn.k_proj.weight.view(attn.num_kv_heads, attn.head_k_dim, -1)
        .repeat_interleave(attn.num_kv_groups, dim=0)
        .reshape_as(attn.q_proj.weight)
    )
    cosine = torch.nn.functional.cosine_similarity(attn.q_proj.weight, repeated_k, dim=-1)
    assert torch.all(cosine > 0.9)


def test_memetic_init_rejects_invalid_usage() -> None:
    short_conv = FLABackboneConfig(
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
    ).create_backbone(ninp=8, attention_between_features=False)
    unsupported = build_fla_backbone("gated_deltanet", size="small", memetic_init=False, train=False)

    with pytest.raises(ValueError, match="use_short_conv=False"):
        apply_memetic_fla_init(short_conv.fla)
    with pytest.raises(ValueError, match="Expected at least one GatedLinearAttention layer"):
        apply_memetic_fla_init(unsupported.fla)
    with pytest.raises(ValueError, match="num_heads \\* head_dim"):
        _repeat_head_blocks(torch.randn(3, 8), num_heads=2, repeat_factor=2, head_dim=2)
