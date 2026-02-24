import pytest
import torch

from pfns.model.rebased_feature_map import BasedFeatureMap

def test_based_feature_map_matches_polynomial_kernel() -> None:
    torch.manual_seed(0)
    head_dim = 16
    q = torch.randn(4, 9, head_dim)
    k = torch.randn(4, 9, head_dim)
    feature_map = BasedFeatureMap()

    phi_q = feature_map(q)
    phi_k = feature_map(k)
    induced = (phi_q * phi_k).sum(dim=-1)

    dot = (q * k).sum(dim=-1)
    expected = 1.0 + dot + 0.5 * dot.square()
    torch.testing.assert_close(induced, expected, rtol=1e-5, atol=1e-6)

