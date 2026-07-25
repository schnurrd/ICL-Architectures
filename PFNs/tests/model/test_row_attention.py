import pytest
import torch

from pfns.model.row_attention import DeltaNetRowAttention, LinearRowAttention


RTOL = 1e-5
ATOL = 1e-5


def _build_attn(**kwargs) -> DeltaNetRowAttention:
    defaults = dict(d_model=8, num_heads=2)
    defaults.update(kwargs)
    return DeltaNetRowAttention(**defaults)


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)


def test_forward_output_shape() -> None:
    torch.manual_seed(0)
    attn = _build_attn(zero_init=False)
    x = torch.randn(2, 6, 3, 8)
    out = attn(x, single_eval_pos=4)
    assert out.shape == x.shape


def test_zero_init_output_is_zero_at_init() -> None:
    torch.manual_seed(0)
    attn = _build_attn(zero_init=True)
    x = torch.randn(2, 6, 3, 8)
    out = attn(x, single_eval_pos=4)
    _assert_close(out, torch.zeros_like(out))


def test_gradients_flow_through_forward() -> None:
    torch.manual_seed(0)
    attn = _build_attn(zero_init=False)
    x = torch.randn(2, 6, 3, 8, requires_grad=True)
    out = attn(x, single_eval_pos=4)
    out.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    for name, param in attn.named_parameters():
        assert param.grad is not None, f"{name} has no gradient"
        assert torch.isfinite(param.grad).all(), f"{name} has non-finite gradient"


def test_incontext_fit_predict_matches_eval_forward() -> None:
    torch.manual_seed(0)
    attn = _build_attn(zero_init=False)
    attn.eval()

    train_len = 4
    x = torch.randn(2, 6, 3, 8)

    with torch.no_grad():
        baseline = attn(x, single_eval_pos=train_len)
        out_fit, state = attn.incontext_fit(x[:, :train_len])
        out_predict = attn.incontext_predict(x[:, train_len:], state)

    _assert_close(torch.cat([out_fit, out_predict], dim=1), baseline)


def test_train_test_split_holds_during_training() -> None:
    """Comb_ST must keep train/test separated in training too, matching
    `FLABackbone.forward`'s `use_split_path` for sequence_mode='Comb_ST'.
    Otherwise the model trains with test-to-test information flow that is
    removed at inference."""
    torch.manual_seed(0)
    attn = _build_attn(zero_init=False)
    x = torch.randn(2, 6, 3, 8)
    train_len = 4

    with torch.no_grad():
        attn.eval()
        out_eval = attn(x, single_eval_pos=train_len)
        attn.train()
        out_train = attn(x, single_eval_pos=train_len)

    _assert_close(out_train, out_eval)


def test_bf16_input_returns_bf16_output() -> None:
    """The CPU fallback accumulates in fp32 but must return the caller's
    dtype -- the fla_config forces bf16 autocast for deltanet, and returning
    fp32 crashes the subsequent out_proj."""
    torch.manual_seed(0)
    attn = _build_attn(zero_init=False).to(torch.bfloat16)
    attn.eval()
    x = torch.randn(2, 6, 3, 8, dtype=torch.bfloat16)

    with torch.no_grad():
        out = attn(x, single_eval_pos=4)

    assert out.dtype == torch.bfloat16


def test_closed_form_predict_matches_explicit_recurrence() -> None:
    """incontext_predict uses a closed-form single-step update instead of
    replaying the recurrence against a per-token copy of the state. It must
    agree with explicitly running one delta-rule step per test token."""
    torch.manual_seed(0)
    attn = _build_attn(zero_init=False).eval()
    train_len = 4
    x = torch.randn(2, 9, 3, 8)

    with torch.no_grad():
        _, state = attn.incontext_fit(x[:, :train_len])
        actual = attn.incontext_predict(x[:, train_len:], state)

        # Explicit reference: each test token independently applies one
        # delta-rule write to its own copy of the cached state, then reads.
        x_test = x[:, train_len:]
        q, k, v, beta, (bs, nb) = attn._project(x_test)
        S = state["recurrent_state"]
        scale = attn.head_dim**-0.5
        outs = []
        for t in range(q.shape[1]):
            err = beta[:, t].unsqueeze(-1) * (
                v[:, t] - torch.einsum("bhk,bhkv->bhv", k[:, t], S)
            )
            S_t = S + torch.einsum("bhk,bhv->bhkv", k[:, t], err)
            outs.append(torch.einsum("bhk,bhkv->bhv", q[:, t] * scale, S_t))
        expected = attn._project_out(torch.stack(outs, dim=1), bs, nb)

    _assert_close(actual, expected)


def test_all_train_and_all_test_boundaries() -> None:
    """single_eval_pos at either extreme must not blow up (PerFeatureLayer
    allows 0 <= single_eval_pos <= seq_len)."""
    torch.manual_seed(0)
    attn = _build_attn(zero_init=False).eval()
    x = torch.randn(2, 5, 3, 8)

    with torch.no_grad():
        out_all_train = attn(x, single_eval_pos=x.shape[1])
        out_all_test = attn(x, single_eval_pos=0)

    assert out_all_train.shape == x.shape
    assert out_all_test.shape == x.shape
    assert torch.isfinite(out_all_train).all()
    assert torch.isfinite(out_all_test).all()


def test_test_positions_are_mutually_independent() -> None:
    """incontext_predict must not let test tokens see each other: permuting
    the test chunk should permute the outputs identically."""
    torch.manual_seed(0)
    attn = _build_attn(zero_init=False)
    attn.eval()

    train_len = 4
    test_len = 3
    x = torch.randn(2, train_len + test_len, 3, 8)

    with torch.no_grad():
        _, state = attn.incontext_fit(x[:, :train_len])
        out_test = attn.incontext_predict(x[:, train_len:], state)

        permutation = torch.tensor([2, 0, 1])
        out_test_permuted = attn.incontext_predict(x[:, train_len:][:, permutation], state)

    _assert_close(out_test_permuted, out_test[:, permutation])


def test_incontext_predict_handles_empty_test_chunk() -> None:
    torch.manual_seed(0)
    attn = _build_attn(zero_init=False)
    attn.eval()

    x = torch.randn(2, 4, 3, 8)
    with torch.no_grad():
        _, state = attn.incontext_fit(x)
        out_test = attn.incontext_predict(x[:, :0], state)

    assert out_test.shape == (2, 0, 3, 8)


def test_recurrent_state_matches_manual_delta_rule_reference() -> None:
    """Numeric ground-truth check of the CPU fallback recurrence against a
    direct, independently-written delta-rule loop (post-update readout,
    matching fla.ops.delta_rule.naive.delta_rule_recurrence's semantics)."""
    torch.manual_seed(0)
    attn = _build_attn(d_model=4, num_heads=1, qk_norm=None, zero_init=False)
    x = torch.randn(1, 5, 1, 4)
    attn.eval()

    q, k, v, beta, (batch_size, num_blocks) = attn._project(x)
    assert (batch_size, num_blocks) == (1, 1)

    head_dim = attn.head_dim
    scale = head_dim**-0.5
    state = torch.zeros(1, 1, head_dim, head_dim)
    expected_outputs = []
    for t in range(x.shape[1]):
        k_t, v_t, q_t, beta_t = k[:, t], v[:, t], q[:, t], beta[:, t]
        prediction = torch.einsum("bhk,bhkv->bhv", k_t, state)
        error = (v_t - prediction) * beta_t.unsqueeze(-1)
        state = state + torch.einsum("bhk,bhv->bhkv", k_t, error)
        expected_outputs.append(torch.einsum("bhk,bhkv->bhv", q_t * scale, state))
    expected = torch.stack(expected_outputs, dim=1)

    actual, _ = attn._delta_rule(q, k, v, beta, initial_state=None, output_final_state=False)
    _assert_close(actual, expected)


# --- causal linear attention row mixer -------------------------------------

def _build_linear(**kwargs) -> LinearRowAttention:
    defaults = dict(d_model=8, num_heads=2)
    defaults.update(kwargs)
    return LinearRowAttention(**defaults)


def test_linear_incontext_fit_predict_matches_eval_forward() -> None:
    torch.manual_seed(0)
    attn = _build_linear(zero_init=False).eval()
    train_len, x = 5, torch.randn(2, 11, 3, 8)
    with torch.no_grad():
        baseline = attn(x, single_eval_pos=train_len)
        out_fit, state = attn.incontext_fit(x[:, :train_len])
        out_predict = attn.incontext_predict(x[:, train_len:], state)
    _assert_close(torch.cat([out_fit, out_predict], dim=1), baseline)


def test_linear_closed_form_predict_matches_explicit_recurrence() -> None:
    """The closed-form test read must equal one explicit linear-attention
    write+read per test token against a private copy of the state."""
    torch.manual_seed(0)
    attn = _build_linear(zero_init=False).eval()
    train_len, x = 4, torch.randn(2, 9, 3, 8)
    with torch.no_grad():
        _, state = attn.incontext_fit(x[:, :train_len])
        actual = attn.incontext_predict(x[:, train_len:], state)
        q, k, v, (bs, nb) = attn._project(x[:, train_len:])
        S, Z = state["recurrent_state"], state["k_sum"]
        outs = []
        for t in range(q.shape[1]):
            S_t = S + torch.einsum("bhk,bhv->bhkv", k[:, t], v[:, t])
            Z_t = Z + k[:, t]
            num = torch.einsum("bhk,bhkv->bhv", q[:, t], S_t)
            denom = torch.einsum("bhk,bhk->bh", q[:, t], Z_t).unsqueeze(-1)
            outs.append(num / (denom + attn.eps))
        expected = attn._project_out(torch.stack(outs, dim=1), bs, nb)
    _assert_close(actual, expected)


def test_linear_test_positions_are_mutually_independent() -> None:
    torch.manual_seed(0)
    attn = _build_linear(zero_init=False).eval()
    train_len, x = 4, torch.randn(2, 8, 3, 8)
    with torch.no_grad():
        _, state = attn.incontext_fit(x[:, :train_len])
        out = attn.incontext_predict(x[:, train_len:], state)
        perm = torch.tensor([2, 0, 3, 1])
        out_perm = attn.incontext_predict(x[:, train_len:][:, perm], state)
    _assert_close(out_perm, out[:, perm])


def test_linear_chunking_matches_unchunked_fit() -> None:
    torch.manual_seed(0)
    a = _build_linear(zero_init=False, chunk_size=None).eval()
    b = _build_linear(zero_init=False, chunk_size=2).eval()
    b.load_state_dict(a.state_dict())
    x = torch.randn(2, 9, 3, 8)
    with torch.no_grad():
        out_a, _ = a.incontext_fit(x)
        out_b, _ = b.incontext_fit(x)
    _assert_close(out_b, out_a)


def test_linear_bf16_input_returns_bf16_output() -> None:
    torch.manual_seed(0)
    attn = _build_linear(zero_init=False).to(torch.bfloat16).eval()
    x = torch.randn(2, 7, 3, 8, dtype=torch.bfloat16)
    with torch.no_grad():
        out = attn(x, single_eval_pos=4)
    assert out.dtype == torch.bfloat16


# --- regressions ------------------------------------------------------------

@pytest.mark.parametrize("build", [_build_attn, _build_linear])
@pytest.mark.parametrize("train_len", [0, 1, 2, 3])
def test_short_context_does_not_crash(build, train_len: int) -> None:
    """FLA's chunk_linear_attn rejects seq_len < num_heads as a suspected
    head-first input, so short contexts must fall back to the PyTorch path."""
    torch.manual_seed(0)
    attn = build(zero_init=False).eval()
    x = torch.randn(2, 6, 3, 8)
    with torch.no_grad():
        out = attn(x, single_eval_pos=train_len)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("build", [_build_attn, _build_linear])
def test_cached_state_dtype_mismatch_is_tolerated(build) -> None:
    """The CUDA kernels can return an fp32 state for bf16 activations; predict
    must align it rather than dying in einsum."""
    torch.manual_seed(0)
    attn = build(zero_init=False).to(torch.bfloat16).eval()
    x = torch.randn(2, 9, 3, 8, dtype=torch.bfloat16)
    with torch.no_grad():
        _, state = attn.incontext_fit(x[:, :5])
        state = {k: v.float() for k, v in state.items()}  # simulate fp32 state
        out = attn.incontext_predict(x[:, 5:], state)
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out).all()
