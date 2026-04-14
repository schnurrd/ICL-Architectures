import torch
import pytest

from pfns.model.backbones import LinearAttentionBackboneConfig
from pfns.model.linear_attention import LinearAttention


def _build_layer(
    *,
    causal: bool = False,
    causal_train_only: bool = False,
) -> LinearAttention:
    return LinearAttention(
        d_model=8,
        num_heads=2,
        dim_mlp_hidden=16,
        dropout=0.0,
        activation="swish",
        causal=causal,
        causal_train_only=causal_train_only,
    )


def _run_test_token_permutation_case(layer: LinearAttention) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    train_len = 7
    seq_len = 12
    test_len = seq_len - train_len
    assert 0 < train_len < seq_len

    x = torch.randn(2, seq_len, 1, 8)

    with torch.no_grad():
        out = layer(x, single_eval_pos=train_len)
        out_test = out[:, train_len:]

        perm = torch.randperm(test_len)
        x_swapped = x.clone()
        x_swapped[:, train_len:] = x_swapped[:, train_len:][:, perm]
        out_swapped = layer(x_swapped, single_eval_pos=train_len)
        out_swapped_test = out_swapped[:, train_len:][:, torch.argsort(perm)]
    return x, out_swapped_test


@pytest.mark.parametrize(
    ("layer_kwargs", "check_perturbation"),
    [
        ({}, True),
        ({"causal": True}, False),
        ({"causal_train_only": True}, True),
    ],
)
def test_linear_attention_eval_test_token_behavior(
    layer_kwargs: dict[str, bool],
    check_perturbation: bool,
) -> None:
    layer = _build_layer(**layer_kwargs)
    layer.eval()

    x, out_swapped_test = _run_test_token_permutation_case(layer)
    train_len = 7
    out_test = layer(x, single_eval_pos=train_len)[:, train_len:]
    torch.testing.assert_close(out_test, out_swapped_test, rtol=5e-4, atol=1e-5)

    if check_perturbation:
        with torch.no_grad():
            x_pert = x.clone()
            x_pert[:, train_len : train_len + 1] += 10.0
            out_pert_test = layer(x_pert, single_eval_pos=train_len)[:, train_len:]
        torch.testing.assert_close(
            out_test[:, 1:2],
            out_pert_test[:, 1:2],
            rtol=5e-4,
            atol=1e-5,
        )


def test_linear_attention_causal_train_mode_test_tokens_dependent():
    layer = _build_layer(causal=True)
    layer.train()

    x, out_swapped_test = _run_test_token_permutation_case(layer)
    out_test = layer(x, single_eval_pos=7)[:, 7:]
    assert (out_test - out_swapped_test).abs().max() > 1e-3


def test_linear_attention_backbone_ignores_legacy_layer_kwargs():
    backbone = LinearAttentionBackboneConfig(
        nlayers=2,
        nhead=2,
        mlp_hidden_dim=16,
        layer_kwargs={
            "feature_dim": 3,
            "attention_between_features": False,
            "feature_attention_softmax": False,
            "causal": True,
        },
    ).create_backbone(ninp=8, attention_between_features=False)

    assert len(backbone.layers) == 2
    assert all(layer.causal for layer in backbone.layers)
    assert all(layer.qk_dim == 3 for layer in backbone.layers)


def test_linear_attention_disallows_k_sum_normalization_with_state_renormalization():
    with pytest.raises(AssertionError, match="mutually exclusive"):
        LinearAttention(
            d_model=8,
            num_heads=2,
            dim_mlp_hidden=16,
            dropout=0.0,
            activation="swish",
            use_k_sum_normalization=True,
            state_renormalization="sqrt_d_fro",
        )


def test_linear_attention_causal_matches_prefix_reads_with_scaled_readout():
    torch.manual_seed(0)
    layer = LinearAttention(
        d_model=8,
        num_heads=2,
        dim_mlp_hidden=16,
        dropout=0.0,
        activation="swish",
        causal=True,
        scale_readout_by_sqrt_dk=True,
        use_k_sum_normalization=False,
    )
    layer.eval()

    x = torch.randn(2, 9, 1, 8)
    q_raw, k_raw, v = layer._project_qkv(x)
    q = layer._feature_map_with_sum_normalization(q_raw, normalize_sum=layer.norm_q)
    k = layer._feature_map_with_sum_normalization(k_raw, normalize_sum=layer.norm_k)

    with torch.no_grad():
        attn, _, _ = layer._causal_attention(q_raw, k_raw, v)

    expected = []
    for t in range(q.shape[1]):
        kv_state = torch.einsum("bshf,bshd->bhfd", k[:, : t + 1], v[:, : t + 1])
        expected.append(layer._read_from_kv_state(q[:, t : t + 1], kv_state, None))

    torch.testing.assert_close(
        attn,
        torch.cat(expected, dim=1),
        rtol=5e-4,
        atol=1e-5,
    )


def test_linear_attention_chunked_causal_matches_unchunked_with_state_renormalization():
    torch.manual_seed(0)
    layer_full = LinearAttention(
        d_model=8,
        num_heads=2,
        dim_mlp_hidden=16,
        dropout=0.0,
        activation="swish",
        causal=True,
        use_k_sum_normalization=False,
        state_renormalization="sqrt_d_fro",
    )
    layer_chunked = LinearAttention(
        d_model=8,
        num_heads=2,
        dim_mlp_hidden=16,
        dropout=0.0,
        activation="swish",
        causal=True,
        causal_chunk_size=3,
        use_k_sum_normalization=False,
        state_renormalization="sqrt_d_fro",
    )
    layer_chunked.load_state_dict(layer_full.state_dict())
    layer_full.eval()
    layer_chunked.eval()

    x = torch.randn(2, 9, 1, 8)

    with torch.no_grad():
        out_full, state_full = layer_full.incontext_fit(x)
        out_chunked, state_chunked = layer_chunked.incontext_fit(x)

    torch.testing.assert_close(out_full, out_chunked, rtol=5e-4, atol=1e-5)
    torch.testing.assert_close(
        state_full["kv_state"],
        state_chunked["kv_state"],
        rtol=5e-4,
        atol=1e-5,
    )
    torch.testing.assert_close(
        state_full["k_sum"],
        state_chunked["k_sum"],
        rtol=5e-4,
        atol=1e-5,
    )


def test_linear_attention_chunked_incontext_predict_matches_forward_with_state_renormalization():
    torch.manual_seed(0)
    layer = LinearAttention(
        d_model=8,
        num_heads=2,
        dim_mlp_hidden=16,
        dropout=0.0,
        activation="swish",
        causal=True,
        causal_chunk_size=3,
        use_k_sum_normalization=False,
        state_renormalization="sqrt_d_fro",
    )
    layer.eval()

    x = torch.randn(2, 9, 1, 8)
    train_len = 5

    with torch.no_grad():
        out_full = layer(x, single_eval_pos=train_len)
        train_out, state = layer.incontext_fit(x[:, :train_len])
        test_out = layer.incontext_predict(x[:, train_len:], state)

    torch.testing.assert_close(
        torch.cat([train_out, test_out], dim=1),
        out_full,
        rtol=5e-4,
        atol=1e-5,
    )
