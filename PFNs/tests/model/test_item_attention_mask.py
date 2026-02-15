import torch

from pfns.model.layer import PerFeatureLayer


def _build_layer() -> PerFeatureLayer:
    return PerFeatureLayer(
        d_model=4,
        nhead=2,
        dim_feedforward=8,
        attention_between_features=False,
        item_attention_mask_mode="test_to_train_only",
    )


def test_item_attention_mask_test_to_train_only():
    layer = _build_layer()
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
    layer = _build_layer()
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
    layer = _build_layer()
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
