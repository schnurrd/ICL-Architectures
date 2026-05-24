import torch
import pytest

from pfns.model.linear_attention import LinearAttention


RTOL = 5e-4
ATOL = 1e-5
TRAIN_LEN = 7


def _build_layer(
    **kwargs,
) -> LinearAttention:
    defaults = dict(
        d_model=8,
        num_heads=2,
        mlp_hidden_dim=16,
        norm_type="layernorm",
    )
    defaults.update(kwargs)
    return LinearAttention(**defaults)


def _assert_close(actual, expected) -> None:
    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)


def _least_squares_state(k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return torch.matmul(
        torch.linalg.pinv(k.transpose(1, 2)),
        v.transpose(1, 2),
    )


def _ridge_state(
    k: torch.Tensor,
    v: torch.Tensor,
    ridge_lambda: float,
    regularization_scale: float = 1.0,
) -> torch.Tensor:
    k_bh = k.transpose(1, 2)
    v_bh = v.transpose(1, 2)
    gram = torch.matmul(k_bh.transpose(-1, -2), k_bh)
    cross = torch.matmul(k_bh.transpose(-1, -2), v_bh)
    eye = torch.eye(k.shape[-1], dtype=k.dtype, device=k.device)
    regularized_gram = gram + ridge_lambda * regularization_scale * eye
    return torch.linalg.solve(regularized_gram, cross)


def _run_test_token_permutation_case(
    layer: LinearAttention,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    seq_len = 12
    test_len = seq_len - TRAIN_LEN

    x = torch.randn(2, seq_len, 1, 8)
    with torch.no_grad():
        out_test = layer(x, single_eval_pos=TRAIN_LEN)[:, TRAIN_LEN:]
        perm = torch.randperm(test_len)
        x_swapped = x.clone()
        x_swapped[:, TRAIN_LEN:] = x_swapped[:, TRAIN_LEN:][:, perm]
        out_swapped = layer(x_swapped, single_eval_pos=TRAIN_LEN)
    return x, out_test, out_swapped[:, TRAIN_LEN:][:, torch.argsort(perm)]


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

    x, out_test, out_swapped_test = _run_test_token_permutation_case(layer)
    _assert_close(out_test, out_swapped_test)

    if check_perturbation:
        with torch.no_grad():
            x_pert = x.clone()
            x_pert[:, TRAIN_LEN : TRAIN_LEN + 1] += 10.0
            out_pert_test = layer(x_pert, single_eval_pos=TRAIN_LEN)[:, TRAIN_LEN:]
        _assert_close(out_test[:, 1:2], out_pert_test[:, 1:2])


def test_linear_attention_causal_train_mode_test_tokens_dependent():
    layer = _build_layer(causal=True)
    layer.train()

    _, out_test, out_swapped_test = _run_test_token_permutation_case(layer)
    assert (out_test - out_swapped_test).abs().max() > 1e-3


def test_linear_attention_causal_matches_prefix_reads_with_scaled_query():
    torch.manual_seed(0)
    layer = _build_layer(
        causal=True,
        use_k_sum_normalization=False,
    )
    layer.eval()

    x = torch.randn(2, 9, 1, 8)
    q_raw, k_raw, v = layer._project_qkv(x)
    q, k = layer._apply_query_key_feature_maps(q_raw, k_raw)

    with torch.no_grad():
        attn, _, _ = layer._causal_attention(q_raw, k_raw, v)

    expected = []
    for t in range(q.shape[1]):
        kv_state = torch.einsum("bshf,bshd->bhfd", k[:, : t + 1], v[:, : t + 1])
        expected.append(layer._read_from_kv_state(q[:, t : t + 1], kv_state, None))

    _assert_close(attn, torch.cat(expected, dim=1))


def test_linear_attention_least_squares_matches_prefix_solution():
    torch.manual_seed(0)
    layer = _build_layer(
        causal=True,
        feature_map="identity",
        use_query_scale=False,
        use_k_sum_normalization=False,
        state_update_rule="least_squares",
    )
    layer.eval()

    q = torch.randn(2, 5, 2, 4)
    k = torch.randn(2, 5, 2, 4)
    v = torch.randn(2, 5, 2, 3)

    with torch.no_grad():
        attn, state, _ = layer._causal_attention(q, k, v)

    expected = []
    for t in range(k.shape[1]):
        k_prefix = k[:, : t + 1]
        v_prefix = v[:, : t + 1]
        w_star = _least_squares_state(k_prefix, v_prefix)
        expected.append(torch.einsum("bhf,bhfd->bhd", q[:, t], w_star).unsqueeze(1))

    expected_state = _least_squares_state(k, v)

    torch.testing.assert_close(attn, torch.cat(expected, dim=1), rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(state, expected_state, rtol=2e-5, atol=2e-6)


def test_linear_attention_least_squares_matches_noncausal_solution():
    torch.manual_seed(0)
    layer = _build_layer(
        feature_map="identity",
        use_query_scale=False,
        use_k_sum_normalization=False,
        state_update_rule="least_squares",
    )
    layer.eval()

    q = torch.randn(2, 5, 2, 4)
    k = torch.randn(2, 5, 2, 4)
    v = torch.randn(2, 5, 2, 3)

    with torch.no_grad():
        attn, state, k_sum = layer._noncausal_attention(q, k, v)

    expected_state = _least_squares_state(k, v)
    expected_attn = torch.einsum("bshf,bhfd->bshd", q, expected_state)

    torch.testing.assert_close(attn, expected_attn, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(state, expected_state, rtol=2e-5, atol=2e-6)
    assert k_sum is None


@pytest.mark.parametrize("state_update_rule", ["ridge", "scaled_ridge"])
def test_linear_attention_ridge_matches_noncausal_solution(state_update_rule: str):
    torch.manual_seed(0)
    layer = _build_layer(
        feature_map="identity",
        use_query_scale=False,
        use_k_sum_normalization=False,
        state_update_rule=state_update_rule,
        ridge_lambda=0.7,
    )
    layer.eval()

    q = torch.randn(2, 5, 2, 4)
    k = torch.randn(2, 5, 2, 4)
    v = torch.randn(2, 5, 2, 3)

    with torch.no_grad():
        attn, state, k_sum = layer._noncausal_attention(q, k, v)

    regularization_scale = k.shape[1] if state_update_rule == "scaled_ridge" else 1.0
    expected_state = _ridge_state(
        k,
        v,
        layer.ridge_lambda,
        regularization_scale=regularization_scale,
    )
    expected_attn = torch.einsum("bshf,bhfd->bshd", q, expected_state)

    torch.testing.assert_close(attn, expected_attn, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(state, expected_state, rtol=2e-5, atol=2e-6)
    assert k_sum is None


def test_linear_attention_ridge_state_uses_fp32_linalg_under_autocast():
    torch.manual_seed(0)
    layer = _build_layer(
        feature_map="identity",
        use_query_scale=False,
        use_k_sum_normalization=False,
        state_update_rule="ridge",
        ridge_lambda=100.0,
    )
    layer.eval()

    k = torch.randn(2, 5, 2, 4).bfloat16()
    v = torch.randn(2, 5, 2, 3).bfloat16()

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        state = layer._ridge_state(k, v)

    assert state.dtype == v.dtype


@pytest.mark.parametrize("state_update_rule", ["ridge", "scaled_ridge"])
def test_linear_attention_ridge_matches_prefix_solution(state_update_rule: str):
    torch.manual_seed(0)
    layer = _build_layer(
        causal=True,
        feature_map="identity",
        use_query_scale=False,
        use_k_sum_normalization=False,
        state_update_rule=state_update_rule,
        ridge_lambda=0.7,
    )
    layer.eval()

    q = torch.randn(2, 5, 2, 4)
    k = torch.randn(2, 5, 2, 4)
    v = torch.randn(2, 5, 2, 3)

    with torch.no_grad():
        attn, state, _ = layer._causal_attention(q, k, v)

    expected = []
    for t in range(k.shape[1]):
        k_prefix = k[:, : t + 1]
        v_prefix = v[:, : t + 1]
        regularization_scale_t = (
            k_prefix.shape[1] if state_update_rule == "scaled_ridge" else 1.0
        )
        expected_state_t = _ridge_state(
            k_prefix,
            v_prefix,
            layer.ridge_lambda,
            regularization_scale=regularization_scale_t,
        )
        expected.append(
            torch.einsum("bhf,bhfd->bhd", q[:, t], expected_state_t).unsqueeze(1)
        )

    regularization_scale = k.shape[1] if state_update_rule == "scaled_ridge" else 1.0
    expected_state = _ridge_state(
        k,
        v,
        layer.ridge_lambda,
        regularization_scale=regularization_scale,
    )

    torch.testing.assert_close(attn, torch.cat(expected, dim=1), rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(state, expected_state, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize(
    "state_update_rule",
    ["least_squares", "ridge", "scaled_ridge"],
)
def test_linear_attention_oracle_update_incontext_predict_matches_forward(
    state_update_rule: str,
):
    torch.manual_seed(0)
    layer = _build_layer(
        causal_train_only=True,
        causal_chunk_size=3,
        use_k_sum_normalization=False,
        state_update_rule=state_update_rule,
    )
    layer.eval()

    x = torch.randn(2, 9, 1, 8)
    train_len = 5

    with torch.no_grad():
        out_full = layer(x, single_eval_pos=train_len)
        train_out, state = layer.incontext_fit(x[:, :train_len])
        test_out = layer.incontext_predict(x[:, train_len:], state)

    _assert_close(torch.cat([train_out, test_out], dim=1), out_full)


def test_linear_attention_oracle_update_rejects_k_sum_normalization():
    with pytest.raises(ValueError, match="use_k_sum_normalization"):
        _build_layer(
            state_update_rule="least_squares",
            use_k_sum_normalization=True,
        )


@pytest.mark.parametrize(
    "state_update_rule",
    ["rls", "oracle", "least_squares_oracle", "ridge_oracle"],
)
def test_linear_attention_rejects_unsupported_oracle_aliases(
    state_update_rule: str,
):
    with pytest.raises(ValueError, match="state_update_rule"):
        _build_layer(state_update_rule=state_update_rule)


def test_linear_attention_chunked_causal_matches_unchunked_with_state_renormalization():
    torch.manual_seed(0)
    layer_full = _build_layer(
        causal=True,
        use_k_sum_normalization=False,
        state_renormalization="sqrt_d_fro",
    )
    layer_chunked = _build_layer(
        causal=True,
        causal_chunk_size=3,
        use_k_sum_normalization=False,
        state_renormalization="sqrt_d_fro",
    )
    layer_chunked.load_state_dict(layer_full.state_dict())
    for layer in (layer_full, layer_chunked):
        layer.eval()

    x = torch.randn(2, 9, 1, 8)

    with torch.no_grad():
        out_full, state_full = layer_full.incontext_fit(x)
        out_chunked, state_chunked = layer_chunked.incontext_fit(x)

    _assert_close(out_full, out_chunked)
    for key in ("kv_state", "k_sum"):
        _assert_close(state_full[key], state_chunked[key])


def test_linear_attention_chunked_incontext_predict_matches_forward_with_state_renormalization():
    torch.manual_seed(0)
    layer = _build_layer(
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

    _assert_close(torch.cat([train_out, test_out], dim=1), out_full)
