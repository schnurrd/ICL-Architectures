import pytest
import torch

from pfns.model.attention_utils import clip_linear_attention_state_frobenius_norm
from pfns.model.linear_attention import LinearAttention


STATE_NORM_MAX = 1e-4


def _build_layer(*, causal: bool = False) -> LinearAttention:
    return LinearAttention(
        d_model=8,
        num_heads=2,
        dim_mlp_hidden=16,
        dropout=0.0,
        activation="swish",
        attention_between_features=False,
        causal=causal,
        hidden_state_frobenius_norm_max=STATE_NORM_MAX,
        hidden_state_frobenius_norm_apply="pre_attention",
    )


def _state_norms(kv_state: torch.Tensor, k_sum: torch.Tensor) -> torch.Tensor:
    return (
        kv_state.float().square().sum(dim=(-2, -1))
        + k_sum.float().square().sum(dim=-1)
    ).sqrt()


def test_clip_linear_attention_state_frobenius_norm_caps_joint_state() -> None:
    torch.manual_seed(0)
    kv_state = torch.randn(2, 5, 3, 4, 6, 7) * 100.0
    k_sum = torch.randn(2, 5, 3, 4, 6) * 100.0

    clipped_kv_state, clipped_k_sum = clip_linear_attention_state_frobenius_norm(
        kv_state,
        k_sum,
        STATE_NORM_MAX,
    )

    assert torch.all(
        _state_norms(clipped_kv_state, clipped_k_sum) <= STATE_NORM_MAX * 1.001
    )


@pytest.mark.parametrize("causal", [False, True])
def test_incontext_fit_keeps_cached_state_unclipped_but_clip_hook_caps_it(
    causal: bool,
) -> None:
    torch.manual_seed(0)
    layer = _build_layer(causal=causal)
    x = torch.randn(2, 11, 1, 8) * 100.0

    _, state = layer.incontext_fit(x)

    assert torch.any(_state_norms(state["kv_state"], state["k_sum"]) > STATE_NORM_MAX * 10.0)

    clipped_kv_state, clipped_k_sum = layer._clip_hidden_state_for_attention(
        state["kv_state"],
        state["k_sum"],
    )
    assert torch.all(
        _state_norms(clipped_kv_state, clipped_k_sum) <= STATE_NORM_MAX * 1.001
    )


@pytest.mark.parametrize("causal", [False, True])
def test_linear_attention_forward_backward_supports_state_clipping(causal: bool) -> None:
    torch.manual_seed(0)
    layer = _build_layer(causal=causal)
    layer.train()
    x = torch.randn(2, 9, 1, 8, requires_grad=True)

    out = layer(x, single_eval_pos=5)
    out.square().mean().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
