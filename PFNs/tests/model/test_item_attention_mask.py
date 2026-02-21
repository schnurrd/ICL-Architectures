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
    layer = _build_layer_with_mask_mode("Comb_ST")
    seq_len = 5
    train_len = 3

    mask = layer._build_item_attention_mask(
        mode="Comb_ST",
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
    layer = _build_layer_with_mask_mode("Comb_MT")
    seq_len = 5
    train_len = 3

    mask = layer._build_item_attention_mask(
        mode="Comb_MT",
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
    layer_comb_mt = _build_layer_with_mask_mode("Comb_MT")
    layer_comb_st = _build_layer_with_mask_mode("Comb_ST")
    layer_comb_st.load_state_dict(layer_comb_mt.state_dict())
    layer_comb_mt.eval()
    layer_comb_st.eval()

    state = torch.randn(2, 10, 1, 4)
    interleaved_train_len = 6  # e.g. (x1,y1,x2,y2,x3,y3) in the train part

    out_comb_mt = layer_comb_mt(
        state,
        single_eval_pos=interleaved_train_len,
        cache_trainset_representation=True,
    )
    out_comb_st = layer_comb_st(
        state,
        single_eval_pos=interleaved_train_len,
        cache_trainset_representation=True,
    )
    torch.testing.assert_close(out_comb_mt, out_comb_st)

    test_state = state[:, interleaved_train_len:]
    out_comb_mt_cached = layer_comb_mt(
        test_state,
        single_eval_pos=0,
        cache_trainset_representation=True,
    )
    out_comb_st_cached = layer_comb_st(
        test_state,
        single_eval_pos=0,
        cache_trainset_representation=True,
    )
    torch.testing.assert_close(out_comb_mt_cached, out_comb_st_cached)
