import pytest
import torch
from unittest.mock import patch

from pfns.model.attention_utils import clip_hidden_state_matrix_frobenius_norm
from pfns.model.linear_attention import LinearAttention


STATE_NORM_MAX = 1e-4


def _build_layer(
    *,
    causal: bool = False,
    causal_train_only: bool = False,
    hidden_state_frobenius_norm_apply: str = "state_update",
) -> LinearAttention:
    return LinearAttention(
        d_model=8,
        num_heads=2,
        dim_mlp_hidden=16,
        dropout=0.0,
        activation="swish",
        attention_between_features=False,
        causal=causal,
        causal_train_only=causal_train_only,
        hidden_state_frobenius_norm_max=STATE_NORM_MAX,
        hidden_state_frobenius_norm_apply=hidden_state_frobenius_norm_apply,
    )


def _frobenius_norms(kv_state: torch.Tensor) -> torch.Tensor:
    return kv_state.float().square().sum(dim=(-2, -1)).sqrt()


def _raw_noncausal_kv_state(layer: LinearAttention, x: torch.Tensor) -> torch.Tensor:
    x, norm_idx = layer._apply_feature_attention_block(x, 0)
    _, k, v = layer._project_item_qkv(x, norm_idx)
    k = layer._feature_map(k)
    return torch.einsum("bsnhf,bsnhd->bnhfd", k, v)


def _raw_full_sequence_kv_state(
    layer: LinearAttention,
    x: torch.Tensor,
) -> torch.Tensor:
    x, norm_idx = layer._apply_feature_attention_block(x, 0)
    _, k, v = layer._project_item_qkv(x, norm_idx)
    k = layer._feature_map(k)
    return torch.einsum("bsnhf,bsnhd->bnhfd", k, v)


def test_clip_hidden_state_matrix_frobenius_norm_caps_batched_states() -> None:
    torch.manual_seed(0)
    kv_state = torch.randn(2, 5, 3, 4, 6, 7) * 100.0

    clipped = clip_hidden_state_matrix_frobenius_norm(kv_state, STATE_NORM_MAX)

    assert torch.all(_frobenius_norms(clipped) <= STATE_NORM_MAX * 1.001)


@pytest.mark.parametrize("causal", [False, True])
def test_linear_attention_state_update_clips_hidden_state_incontext_fit(
    causal: bool,
) -> None:
    torch.manual_seed(0)
    layer = _build_layer(
        causal=causal,
        hidden_state_frobenius_norm_apply="state_update",
    )
    x = torch.randn(2, 11, 1, 8) * 100.0

    _, state = layer.incontext_fit(x)

    assert torch.all(_frobenius_norms(state["kv_state"]) <= STATE_NORM_MAX * 1.001)


@pytest.mark.parametrize(
    ("causal", "causal_train_only"),
    [
        (False, False),
        (False, True),
    ],
)
def test_linear_attention_incontext_predict_only_keeps_cached_fit_state_unclipped(
    causal: bool,
    causal_train_only: bool,
) -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 11, 1, 8) * 100.0
    layer = _build_layer(
        causal=causal,
        causal_train_only=causal_train_only,
        hidden_state_frobenius_norm_apply="incontext_predict_only",
    )

    raw_kv_state = _raw_noncausal_kv_state(layer, x)
    _, state = layer.incontext_fit(x)

    assert torch.any(_frobenius_norms(raw_kv_state) > STATE_NORM_MAX * 10.0)
    assert torch.any(_frobenius_norms(state["kv_state"]) > STATE_NORM_MAX * 10.0)


@pytest.mark.parametrize("causal_train_only", [False, True])
def test_linear_attention_incontext_predict_only_clips_post_test_prediction_state(
    causal_train_only: bool,
) -> None:
    torch.manual_seed(0)
    layer = _build_layer(
        causal_train_only=causal_train_only,
        hidden_state_frobenius_norm_apply="incontext_predict_only",
    )
    x = torch.randn(2, 11, 1, 8) * 100.0

    raw_kv_state = _raw_full_sequence_kv_state(layer, x)
    clipped_kv_state = layer._clip_hidden_state_matrix_before_prediction(raw_kv_state)

    assert torch.any(_frobenius_norms(raw_kv_state) > STATE_NORM_MAX * 10.0)
    assert torch.all(_frobenius_norms(clipped_kv_state) <= STATE_NORM_MAX * 1.001)


@pytest.mark.parametrize("causal", [False, True])
def test_linear_attention_training_loop_supports_hidden_state_clipping(
    causal: bool,
) -> None:
    torch.manual_seed(0)
    layer = _build_layer(causal=causal)
    layer.train()
    x = torch.randn(2, 9, 1, 8, requires_grad=True)

    out = layer(x, single_eval_pos=5)
    loss = out.square().mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
