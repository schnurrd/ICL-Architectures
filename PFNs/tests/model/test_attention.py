import pytest
import torch

from pfns.model.multi_head_attention import MultiHeadAttention

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Testing attention on {device=}.")
n_batch = 7
nhead = 4
n_seq_q = 534
n_seq_kv = 316
embed_dim = 128

dtype = torch.float16 if device == "cuda" else torch.float32

x_q = torch.normal(
    torch.tensor(0.0),
    torch.tensor(1.0),
    size=(n_batch, n_seq_q, embed_dim),
)
x_kv = torch.normal(
    torch.tensor(0.0),
    torch.tensor(1.0),
    size=(n_batch, n_seq_kv, embed_dim),
)
x_q = x_q.to(device, dtype)
x_kv = x_kv.to(device, dtype)


def test_attention():
    att_ref = torch.nn.MultiheadAttention(
        embed_dim,
        nhead,
        batch_first=True,
        bias=False,
        device=device,
        dtype=dtype,
    )
    att_test = MultiHeadAttention(
        input_size=embed_dim,
        output_size=embed_dim,
        d_k=embed_dim // nhead,
        d_v=embed_dim // nhead,
        nhead=nhead,
        device=device,
        dtype=dtype,
    )

    att_test.load_state_dict(
        MultiHeadAttention.convert_torch_nn_multihead_attention_state_dict(
            att_ref.state_dict(), nhead
        )
    )

    y, _ = att_ref(x_q, x_kv, x_kv)
    y_ = att_test(x_q, x_kv)
    assert torch.sqrt(torch.nn.functional.mse_loss(y, y_)) < 5e-5

    x_q_ = x_q.clone()
    y__ = att_test(x_q_, x_kv, add_input=True)
    assert torch.sqrt(torch.nn.functional.mse_loss(y + x_q, y__)) < 5e-5

    x_q_ = x_q.clone()
    with torch.no_grad():
        y__ = att_test(
            x_q_,
            x_kv,
            add_input=True,
            allow_inplace=True,
            save_peak_mem_factor=7,
        )
    assert torch.sqrt(torch.nn.functional.mse_loss(y + x_q, y__)) < 5e-5

    # Multiquery.
    share_kv_across_n_heads = 2
    att_multi_test = MultiHeadAttention(
        input_size=embed_dim,
        output_size=embed_dim,
        d_k=embed_dim // nhead,
        d_v=embed_dim // nhead,
        nhead=nhead,
        device=device,
        dtype=dtype,
        share_kv_across_n_heads=share_kv_across_n_heads,
    )
    w_kv = (
        att_multi_test.w_kv.unsqueeze(2)
        .expand(-1, -1, share_kv_across_n_heads, -1, -1)
        .reshape(2, nhead, embed_dim // nhead, embed_dim)
    )
    state_dict_to_load = {
        "_w_qkv": torch.cat([att_multi_test.w_q.unsqueeze(0), w_kv], dim=0),
        "_w_out": att_multi_test.w_out,
    }
    att_multi_ref = MultiHeadAttention(
        input_size=embed_dim,
        output_size=embed_dim,
        d_k=embed_dim // nhead,
        d_v=embed_dim // nhead,
        nhead=nhead,
        device=device,
        dtype=dtype,
    )
    att_multi_ref.load_state_dict(state_dict_to_load)
    y = att_multi_ref(x_q, x_kv)
    y_ = att_multi_test(x_q, x_kv)
    assert torch.sqrt(torch.nn.functional.mse_loss(y, y_)) < 5e-5


def test_attention_caching():
    att_test = MultiHeadAttention(
        input_size=embed_dim,
        output_size=embed_dim,
        d_k=embed_dim // nhead,
        d_v=embed_dim // nhead,
        nhead=nhead,
        device=device,
        dtype=dtype,
    )
    y = att_test(x_q, x_kv, cache_kv=True)
    y_ = att_test(x_q, use_cached_kv=True)
    assert torch.sqrt(torch.nn.functional.mse_loss(y, y_)) < 5e-5

    # gradients should fail for train part
    x_q_ = x_q.clone().requires_grad_(True)
    with pytest.raises(AssertionError):
        att_test(x_q_, x_kv, cache_kv=True)

    # gradients should not fail for test part
    x_q_ = x_q.clone().requires_grad_(True)
    y_ = att_test(x_q_, x_kv, use_cached_kv=True)
    y_.mean().backward()
    assert x_q_.grad is not None
    grads_with_cache = x_q_.grad.clone()

    # compute gradients without caching
    x_q_ = x_q.clone().requires_grad_(True)
    y_ = att_test(x_q_, x_kv)
    y_.mean().backward()
    assert x_q_.grad is not None

    grads_without_cache = x_q_.grad.clone()
    assert torch.isclose(grads_with_cache, grads_without_cache).all()


def test_attention_caching_with_rope_position_offsets():
    att_test = MultiHeadAttention(
        input_size=embed_dim,
        output_size=embed_dim,
        d_k=embed_dim // nhead,
        d_v=embed_dim // nhead,
        nhead=nhead,
        device=device,
        dtype=dtype,
        use_rope=True,
    )
    train_len = 211
    test_len = 73
    train_x = x_kv[:, :train_len]
    test_x = x_q[:, :test_len]

    baseline = att_test(
        test_x,
        train_x,
        q_position_offset=train_len,
        k_position_offset=0,
    )
    _ = att_test(
        train_x,
        train_x,
        cache_kv=True,
        q_position_offset=0,
        k_position_offset=0,
    )
    cached = att_test(test_x, use_cached_kv=True)

    torch.testing.assert_close(cached, baseline, rtol=1e-4, atol=1e-4)


def test_attention_caching_with_rope_pairwise_positions():
    att_test = MultiHeadAttention(
        input_size=embed_dim,
        output_size=embed_dim,
        d_k=embed_dim // nhead,
        d_v=embed_dim // nhead,
        nhead=nhead,
        device=device,
        dtype=dtype,
        use_rope=True,
    )
    train_len = 210
    test_len = 73
    train_x = x_kv[:, :train_len]
    test_x = x_q[:, :test_len]

    baseline = att_test(
        test_x,
        train_x,
        q_position_offset=train_len,
        k_position_offset=0,
        rope_pairwise_positions=True,
    )
    _ = att_test(
        train_x,
        train_x,
        cache_kv=True,
        q_position_offset=0,
        k_position_offset=0,
        rope_pairwise_positions=True,
    )
    cached = att_test(
        test_x,
        use_cached_kv=True,
        rope_pairwise_positions=True,
    )

    torch.testing.assert_close(cached, baseline, rtol=1e-4, atol=1e-4)


def test_rope_pairwise_positions_match_reference():
    att_test = MultiHeadAttention(
        input_size=embed_dim,
        output_size=embed_dim,
        d_k=embed_dim // nhead,
        d_v=embed_dim // nhead,
        nhead=nhead,
        device=device,
        dtype=torch.float32,
        use_rope=True,
    )
    x = torch.randn(2, 17, nhead, embed_dim // nhead, device=device, dtype=torch.float32)
    position_offset = 11
    y = att_test._apply_rope(
        x,
        position_offset=position_offset,
        rope_pairwise_positions=True,
    )

    seq_len = x.shape[-3]
    half_dim = x.shape[-1] // 2
    t = torch.arange(
        position_offset,
        position_offset + x.shape[1],
        device=device,
        dtype=torch.long,
    )
    t = torch.div(t, 2, rounding_mode="floor")
    freqs = torch.outer(t.to(dtype=torch.float32), att_test.inv_freq)
    cos = freqs.cos().to(dtype=x.dtype).view(1, seq_len, 1, half_dim)
    sin = freqs.sin().to(dtype=x.dtype).view(1, seq_len, 1, half_dim)
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    y_ref = torch.stack(
        (
            x_even * cos - x_odd * sin,
            x_even * sin + x_odd * cos,
        ),
        dim=-1,
    ).flatten(-2)

    torch.testing.assert_close(y, y_ref, rtol=1e-6, atol=1e-6)


def test_rope_inv_freq_buffer_matches_reference():
    rope_base = 17_777.0
    d_k = embed_dim // nhead
    att_test = MultiHeadAttention(
        input_size=embed_dim,
        output_size=embed_dim,
        d_k=d_k,
        d_v=d_k,
        nhead=nhead,
        device=device,
        dtype=torch.float32,
        use_rope=True,
        rope_base=rope_base,
    )

    # inv_freq is cached as a non-persistent buffer (not saved in state_dict).
    assert "inv_freq" in dict(att_test.named_buffers())
    assert "inv_freq" not in att_test.state_dict()

    x = torch.randn(3, 11, nhead, d_k, device=device, dtype=torch.float32)
    position_offset = 5
    y = att_test._apply_rope(x, position_offset=position_offset)

    seq_len = x.shape[-3]
    half_dim = x.shape[-1] // 2
    t = torch.arange(
        position_offset,
        position_offset + seq_len,
        device=x.device,
        dtype=torch.float32,
    )
    inv_freq_ref = 1.0 / (
        rope_base
        ** (
            torch.arange(0, x.shape[-1], 2, device=x.device, dtype=torch.float32)
            / x.shape[-1]
        )
    )
    freqs = torch.outer(t, inv_freq_ref)
    cos = freqs.cos().to(dtype=x.dtype).view(1, seq_len, 1, half_dim)
    sin = freqs.sin().to(dtype=x.dtype).view(1, seq_len, 1, half_dim)
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    y_ref = torch.stack(
        (
            x_even * cos - x_odd * sin,
            x_even * sin + x_odd * cos,
        ),
        dim=-1,
    ).flatten(-2)

    torch.testing.assert_close(y, y_ref, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    test_attention()
    test_attention_caching()
