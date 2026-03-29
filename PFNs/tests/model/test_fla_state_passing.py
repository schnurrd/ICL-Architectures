from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from pfns.model.fla_state_passing import FLAStatePassing, aligned_indices
from tests.model.fla_test_utils import (
    FLA_MODEL_TYPES,
    build_fla_backbone,
    fla_hidden_size,
)


@dataclass
class _LayerCache:
    state: dict[str, object]


@dataclass
class _RecurrentCache:
    layers: list[_LayerCache]


@dataclass
class _MambaCache:
    conv_states: torch.Tensor
    ssm_states: torch.Tensor


def _assert_same_tensors(left: object, right: object) -> None:
    if torch.is_tensor(left):
        assert torch.is_tensor(right)
        torch.testing.assert_close(left, right)
        return
    if isinstance(left, dict):
        assert isinstance(right, dict)
        left_keys = {key for key in left if not str(key).startswith("_")}
        right_keys = {key for key in right if not str(key).startswith("_")}
        assert left_keys == right_keys
        for key in left_keys:
            _assert_same_tensors(left[key], right[key])
        return
    if isinstance(left, (list, tuple)):
        assert isinstance(right, type(left))
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right):
            _assert_same_tensors(left_item, right_item)
        return
    if hasattr(left, "__dict__"):
        assert hasattr(right, "__dict__")
        _assert_same_tensors(vars(left), vars(right))
        return
    assert left == right


def test_state_passing_reuses_and_freezes_layer_cache() -> None:
    recurrent_state = torch.arange(4 * 2 * 3, dtype=torch.float32).reshape(4, 2, 3)
    conv_state = [torch.arange(4 * 5, dtype=torch.float32).reshape(4, 5)]
    cache = _RecurrentCache(
        layers=[
            _LayerCache(
                state={
                    "recurrent_state": recurrent_state.clone(),
                    "conv_state": conv_state,
                }
            )
        ]
    )
    helper = FLAStatePassing(dropout_prob=0.0)
    helper.remember(cache)

    cache.layers[0].state["recurrent_state"].zero_()

    sampled = helper.sample_initial_cache(4, device=torch.device("cpu"))

    sampled_state = sampled.layers[0].state["recurrent_state"]
    torch.testing.assert_close(sampled_state, recurrent_state)
    torch.testing.assert_close(
        sampled.layers[0].state["conv_state"][0],
        conv_state[0],
    )


def test_state_passing_zeroes_mamba_style_cache() -> None:
    cache = _MambaCache(
        conv_states=torch.ones(2, 3, 4, 5),
        ssm_states=torch.ones(2, 3, 6, 7, 8),
    )
    helper = FLAStatePassing(dropout_prob=1.0)
    helper.remember(cache)

    sampled = helper.sample_initial_cache(3, device=torch.device("cpu"))

    assert sampled is not None
    assert torch.count_nonzero(sampled.conv_states) == 0
    assert torch.count_nonzero(sampled.ssm_states) == 0


def test_state_passing_uses_aligned_cycles_for_unequal_batch_sizes() -> None:
    indices = aligned_indices(3, 8, device=torch.device("cpu"))

    assert tuple(indices.shape) == (8,)
    torch.testing.assert_close(indices, torch.tensor([0, 1, 2, 0, 1, 2, 0, 1]))


@pytest.mark.parametrize("model_type", FLA_MODEL_TYPES)
@pytest.mark.parametrize("sequence_mode", ["Comb_ST", "Comb_MT"])
def test_state_passing_changes_training_outputs(
    model_type: str,
    sequence_mode: str,
) -> None:
    pytest.importorskip("fla")
    if not torch.cuda.is_available():
        pytest.skip("FLA backend requires CUDA/Triton for this test.")

    torch.manual_seed(0)
    device = torch.device("cuda")
    backbone = build_fla_backbone(
        model_type,
        sequence_mode=sequence_mode,
        state_passing=True,
        state_passing_dropout=0.0,
        train=True,
    ).to(device)

    batch_size = 2
    seq_len = 8
    train_len = 5
    embed_dim = fla_hidden_size(model_type)
    x_prev = 4.0 * torch.randn(batch_size, seq_len, 1, embed_dim, device=device)
    x_cur = torch.randn(batch_size, seq_len, 1, embed_dim, device=device)

    with torch.no_grad():
        assert backbone.state_passing is not None
        backbone.state_passing.reset()
        _ = backbone(x_prev, single_eval_pos=train_len)
        out_with_prev = backbone(x_cur, single_eval_pos=train_len)

        backbone.state_passing.reset()
        out_without_prev = backbone(x_cur, single_eval_pos=train_len)

    assert not torch.allclose(out_with_prev, out_without_prev, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("model_type", FLA_MODEL_TYPES)
def test_state_passing_remembers_full_sequence_final_cache(model_type: str) -> None:
    pytest.importorskip("fla")
    if not torch.cuda.is_available():
        pytest.skip("FLA backend requires CUDA/Triton for this test.")

    torch.manual_seed(0)
    device = torch.device("cuda")
    backbone = build_fla_backbone(
        model_type,
        sequence_mode="Comb_ST",
        state_passing=True,
        state_passing_dropout=0.0,
        train=True,
    ).to(device)

    batch_size = 2
    seq_len = 8
    train_len = 5
    embed_dim = fla_hidden_size(model_type)
    x = torch.randn(batch_size, seq_len, 1, embed_dim, device=device)
    x_batched = x.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
    train_x = x_batched[:, :train_len]
    test_x = x_batched[:, train_len:]

    with torch.no_grad():
        assert backbone.state_passing is not None
        backbone.state_passing.reset()
        _ = backbone(x, single_eval_pos=train_len)
        remembered_cache = backbone.state_passing.previous_cache

        _, state = backbone.incontext_fit(train_x)
        _, expected_cache = backbone._run_fla(
            test_x,
            cache_params=backbone._copy_cache(state["cache_params"]),
            return_cache=True,
        )

    _assert_same_tensors(remembered_cache, expected_cache)
