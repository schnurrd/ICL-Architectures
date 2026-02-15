import pytest
import torch

from pfns.model.layer import PerFeatureLayer


def _build_layer_with_mask_mode(mask_mode: str) -> PerFeatureLayer:
    return PerFeatureLayer(
        d_model=4,
        nhead=2,
        dim_feedforward=8,
        attention_between_features=False,
        item_attention_mask_mode=mask_mode,
    )


def _build_rope_layer() -> PerFeatureLayer:
    return PerFeatureLayer(
        d_model=4,
        nhead=2,
        dim_feedforward=8,
        attention_between_features=False,
        item_attention_use_rope=True,
        item_attention_rope_pairwise_positions=True,
    )


def test_item_attention_mask_test_to_train_only():
    layer = _build_layer_with_mask_mode("test_to_train_only")
    seq_len = 5
    train_len = 3

    mask = layer._build_item_attention_mask(
        mode="test_to_train_only",
        seq_len_q=seq_len,
        seq_len_kv=seq_len,
        train_len=train_len,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    expected = torch.full((seq_len, seq_len), float("-inf"))
    expected[torch.arange(train_len), torch.arange(train_len)] = 0.0
    expected[train_len:, :train_len] = 0.0

    torch.testing.assert_close(mask, expected)


def test_item_attention_mask_causal_train_only():
    layer = _build_layer_with_mask_mode("causal_train_only")
    seq_len = 5
    train_len = 3

    mask = layer._build_item_attention_mask(
        mode="causal_train_only",
        seq_len_q=seq_len,
        seq_len_kv=seq_len,
        train_len=train_len,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    expected = torch.full((seq_len, seq_len), float("-inf"))
    tril_mask = torch.tril(torch.ones((train_len, train_len), dtype=torch.bool))
    expected[:train_len, :train_len][tril_mask] = 0.0
    expected[train_len:, :train_len] = 0.0

    torch.testing.assert_close(mask, expected)


def test_item_attention_mask_causal_all():
    layer = _build_layer_with_mask_mode("causal_all")
    seq_len = 5
    train_len = 3

    mask = layer._build_item_attention_mask(
        mode="causal_all",
        seq_len_q=seq_len,
        seq_len_kv=seq_len,
        train_len=train_len,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    expected = torch.full((seq_len, seq_len), float("-inf"))
    expected[torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool))] = 0.0

    torch.testing.assert_close(mask, expected)


def test_item_attention_rope_pairwise_forward_runs():
    layer = _build_rope_layer()
    state = torch.randn(2, 9, 1, 4)
    out = layer(state, single_eval_pos=6, rope_pairwise_positions=True)
    assert out.shape == state.shape


@torch.inference_mode()
def test_causal_all_remaps_to_causal_train_only_in_eval_with_cache():
    layer_causal_all = _build_layer_with_mask_mode("causal_all")
    layer_causal_train_only = _build_layer_with_mask_mode("causal_train_only")
    layer_causal_train_only.load_state_dict(layer_causal_all.state_dict())
    layer_causal_all.eval()
    layer_causal_train_only.eval()

    state = torch.randn(2, 10, 1, 4)
    interleaved_train_len = 6  # e.g. (x1,y1,x2,y2,x3,y3) in the train part

    out_causal_all = layer_causal_all(
        state,
        single_eval_pos=interleaved_train_len,
        cache_trainset_representation=True,
    )
    out_causal_train_only = layer_causal_train_only(
        state,
        single_eval_pos=interleaved_train_len,
        cache_trainset_representation=True,
    )
    torch.testing.assert_close(out_causal_all, out_causal_train_only)

    test_state = state[:, interleaved_train_len:]
    out_causal_all_cached = layer_causal_all(
        test_state,
        single_eval_pos=0,
        cache_trainset_representation=True,
    )
    out_causal_train_only_cached = layer_causal_train_only(
        test_state,
        single_eval_pos=0,
        cache_trainset_representation=True,
    )
    torch.testing.assert_close(out_causal_all_cached, out_causal_train_only_cached)
