from __future__ import annotations

import pytest
import torch

from pfns.model.backbones import FLABackboneConfig
from tests.model.fla_test_utils import build_fla_backbone, fla_hidden_size


def test_state_weaving_rejects_non_deltanet() -> None:
    with pytest.raises(ValueError, match="only DeltaNet"):
        build_fla_backbone("gla", state_weaving=True)


def test_state_weaving_rejects_non_comb_st() -> None:
    with pytest.raises(ValueError, match="Comb_ST"):
        build_fla_backbone("deltanet", sequence_mode="Comb_MT", state_weaving=True)


def test_state_weaving_registers_one_initial_state_per_layer() -> None:
    backbone = build_fla_backbone("deltanet", size="small", state_weaving=True)

    assert len(backbone.state_weaving_initial_states) == len(backbone.layers)
    for parameter, layer in zip(backbone.state_weaving_initial_states, backbone.layers):
        attn = layer.attn
        assert tuple(parameter.shape) == (
            attn.num_heads,
            attn.head_k_dim,
            attn.head_v_dim,
        )


def test_state_weaving_rejects_state_passing_combination() -> None:
    with pytest.raises(ValueError, match="state_passing"):
        FLABackboneConfig(
            model_type="deltanet",
            config_kwargs={"hidden_size": 8, "num_hidden_layers": 1, "num_heads": 2},
            state_weaving=True,
            state_passing=True,
        )


def test_state_weaving_delta_net_forward_and_gradients() -> None:
    pytest.importorskip("fla")
    if not torch.cuda.is_available():
        pytest.skip("FLA backend requires CUDA/Triton for this test.")

    torch.manual_seed(0)
    device = torch.device("cuda")
    backbone = build_fla_backbone(
        "deltanet",
        size="small",
        state_weaving=True,
        train=True,
    ).to(device)

    batch_size = 2
    seq_len = 72
    train_len = 68
    embed_dim = fla_hidden_size("deltanet")
    backbone = backbone.to(torch.bfloat16)
    x = torch.randn(batch_size, seq_len, 1, embed_dim, device=device, dtype=torch.bfloat16)

    out = backbone(x, single_eval_pos=train_len)
    assert out.shape == x.shape

    out.sum().backward()
    grads = [parameter.grad for parameter in backbone.state_weaving_initial_states]
    assert any(grad is not None and torch.count_nonzero(grad) > 0 for grad in grads)
