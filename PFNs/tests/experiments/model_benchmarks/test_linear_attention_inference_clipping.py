import torch

from pfns.experiments.model_benchmarks.linear_attention_inference_clipping import (
    normalize_inference_clip_experiments,
)
from pfns.model.attention_utils import compute_kv_state_5d
from pfns.model.linear_attention import LinearAttention


def test_normalize_inference_clip_experiments_defaults_to_baseline() -> None:
    assert normalize_inference_clip_experiments(None) == {"baseline": {}}


def test_normalize_inference_clip_experiments_drops_none_values() -> None:
    assert normalize_inference_clip_experiments(
        {
            "clip": {
                "state_max": 4.0,
                "unused": None,
            }
        }
    ) == {"clip": {"state_max": 4.0}}


def test_linear_attention_auto_reference_uses_current_dataset_prefix_per_head() -> None:
    layer = LinearAttention(
        d_model=2,
        num_heads=2,
        dim_mlp_hidden=4,
        dropout=0.0,
        feature_dim=1,
    )
    layer.eval()
    layer.hidden_state_kv_over_ksum_reference = "auto"
    layer.hidden_state_kv_over_ksum_reference_seqlen = 2

    k = torch.ones((1, 3, 1, 2, 1), dtype=torch.float32)
    v = torch.tensor(
        [[[[[2.0], [5.0]]], [[[2.0], [5.0]]], [[[20.0], [50.0]]]]],
        dtype=torch.float32,
    )

    reference_ratio = layer._runtime_kv_over_ksum_reference(k, v)
    assert isinstance(reference_ratio, torch.Tensor)
    assert reference_ratio.shape == (1, 1, 2)
    assert torch.allclose(reference_ratio[0, 0], torch.tensor([2.0, 5.0]))

    kv_state, k_sum = compute_kv_state_5d(k, v)
    clipped_kv_state, clipped_k_sum = layer._clip_hidden_state_for_prediction(
        kv_state,
        k_sum,
        reference_ratio=reference_ratio,
    )
    clipped_ratio = clipped_kv_state.float().square().sum(dim=(-2, -1)).sqrt()
    clipped_ratio = clipped_ratio / clipped_k_sum.float().square().sum(dim=-1).sqrt().clamp_min(1e-12)
    assert torch.allclose(clipped_ratio[0, 0], torch.tensor([2.0, 5.0]))


def test_causal_linear_attention_items_resolves_auto_reference_from_current_dataset() -> None:
    layer = LinearAttention(
        d_model=2,
        num_heads=2,
        dim_mlp_hidden=4,
        dropout=0.0,
        feature_dim=1,
    )
    layer.eval()
    layer.hidden_state_kv_over_ksum_reference = "auto"
    layer.hidden_state_kv_over_ksum_reference_seqlen = 2
    layer.hidden_state_kv_over_ksum_reference_apply = "pre_attention"

    q = torch.ones((1, 3, 1, 2, 1), dtype=torch.float32)
    k = torch.ones((1, 3, 1, 2, 1), dtype=torch.float32)
    v = torch.tensor(
        [[[[[2.0], [5.0]]], [[[2.0], [5.0]]], [[[20.0], [50.0]]]]],
        dtype=torch.float32,
    )

    attn_auto, kv_auto, ksum_auto = layer._causal_linear_attention_items(q, k, v)
    reference_ratio = layer._runtime_kv_over_ksum_reference(k, v)
    attn_ref, kv_ref, ksum_ref = layer._causal_linear_attention_items(
        q,
        k,
        v,
        reference_ratio=reference_ratio,
    )

    assert torch.allclose(attn_auto, attn_ref)
    assert torch.allclose(kv_auto, kv_ref)
    assert torch.allclose(ksum_auto, ksum_ref)


