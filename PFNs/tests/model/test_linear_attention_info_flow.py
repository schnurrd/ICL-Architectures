import torch

from pfns.model.linear_attention import LinearAttention


def _build_layer() -> LinearAttention:
    return LinearAttention(
        d_model=8,
        num_heads=2,
        dim_mlp_hidden=16,
        dropout=0.0,
        activation="swish",
        attention_between_features=False,
    )


def test_linear_attention_test_tokens_independent():
    torch.manual_seed(0)
    layer = _build_layer()
    layer.eval()

    batch_size = 2
    seq_len = 12
    num_features = 1
    embed_dim = 8
    train_len = 7
    test_len = seq_len - train_len
    assert 0 < train_len < seq_len

    x = torch.randn(batch_size, seq_len, num_features, embed_dim)

    with torch.no_grad():
        out = layer(x, single_eval_pos=train_len)
        out_test = out[:, train_len:]

        perm = torch.randperm(test_len)
        x_swapped = x.clone()
        x_swapped[:, train_len:] = x_swapped[:, train_len:][:, perm]
        out_swapped = layer(x_swapped, single_eval_pos=train_len)
        out_swapped_test = out_swapped[:, train_len:][:, torch.argsort(perm)]

        x_pert = x.clone()
        x_pert[:, train_len : train_len + 1] += 10.0
        out_pert = layer(x_pert, single_eval_pos=train_len)
        out_pert_test = out_pert[:, train_len:]

    torch.testing.assert_close(out_test, out_swapped_test, rtol=5e-4, atol=1e-5)
    torch.testing.assert_close(out_test[:, 1:2], out_pert_test[:, 1:2], rtol=5e-4, atol=1e-5)


if __name__ == "__main__":
    test_linear_attention_test_tokens_independent()
