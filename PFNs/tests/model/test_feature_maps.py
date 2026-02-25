import math
from contextlib import contextmanager

import pytest
import torch
import torch.utils.checkpoint

from pfns.model.rebased_feature_map import BasedFeatureMap, RebasedFeatureMap


def _based_kernel(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    dot = (q * k).sum(dim=-1)
    return 1.0 + dot + 0.5 * dot.square()


@contextmanager
def _checkpoint_no_reentrant():
    checkpoint_module = torch.utils.checkpoint
    original_checkpoint = checkpoint_module.checkpoint

    def checkpoint_wrapper(function, *args, **kwargs):
        kwargs.setdefault("use_reentrant", False)
        return original_checkpoint(function, *args, **kwargs)

    checkpoint_module.checkpoint = checkpoint_wrapper
    try:
        yield
    finally:
        checkpoint_module.checkpoint = original_checkpoint


@pytest.mark.parametrize("dense", [True, False])
def test_based_feature_map_kernel_and_shape(dense: bool) -> None:
    torch.manual_seed(0)
    head_dim = 16
    q = torch.randn(4, 9, head_dim)
    k = torch.randn(4, 9, head_dim)
    feature_map = BasedFeatureMap(dense=dense)

    phi_q = feature_map(q)
    phi_k = feature_map(k)
    expected_dim = (
        1 + head_dim + head_dim * head_dim
        if dense
        else 1 + head_dim + (head_dim * (head_dim + 1)) // 2
    )
    assert phi_q.shape[-1] == expected_dim
    assert phi_k.shape[-1] == expected_dim

    induced = (phi_q * phi_k).sum(dim=-1)
    expected = _based_kernel(q, k)
    torch.testing.assert_close(induced, expected, rtol=1e-5, atol=1e-6)


def test_rebased_dense_matches_reduced_kernel_and_shape() -> None:
    torch.manual_seed(1)
    head_dim = 32
    q = torch.randn(3, 11, 2, head_dim)
    k = torch.randn(3, 11, 2, head_dim)

    reduced = RebasedFeatureMap(
        head_dim=head_dim,
        use_gamma=False,
        use_beta=False,
        normalize=True,
        dense=False,
    )
    dense = RebasedFeatureMap(
        head_dim=head_dim,
        use_gamma=False,
        use_beta=False,
        normalize=True,
        dense=True,
    )

    phi_q_reduced = reduced(q)
    phi_k_reduced = reduced(k)
    phi_q_dense = dense(q)
    phi_k_dense = dense(k)

    assert phi_q_reduced.shape[-1] == head_dim * (head_dim + 1) // 2
    assert phi_k_reduced.shape[-1] == head_dim * (head_dim + 1) // 2
    assert phi_q_dense.shape[-1] == head_dim * head_dim
    assert phi_k_dense.shape[-1] == head_dim * head_dim

    induced_reduced = (phi_q_reduced * phi_k_reduced).sum(dim=-1)
    induced_dense = (phi_q_dense * phi_k_dense).sum(dim=-1)
    torch.testing.assert_close(induced_reduced, induced_dense, rtol=1e-5, atol=1e-6)


def test_rebased_reduced_matches_fla_forward_and_gradients() -> None:
    feature_map_module = pytest.importorskip("fla.modules.feature_map")
    fla_map_cls = feature_map_module.RebasedFeatureMap

    torch.manual_seed(2)
    head_dim = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        pytest.skip("FLA feature-map parity check requires CUDA in this environment.")

    local = RebasedFeatureMap(
        head_dim=head_dim,
        use_gamma=True,
        use_beta=True,
        normalize=True,
        dense=False,
    ).to(device)
    reference = fla_map_cls(head_dim, True, True, True).to(device)

    with torch.no_grad():
        local_state = local.state_dict()
        for key, value in reference.state_dict().items():
            if key in local_state and local_state[key].shape == value.shape:
                local_state[key].copy_(value)

    x_local = torch.randn(3, 11, 2, head_dim, device=device, requires_grad=True)
    x_ref = x_local.detach().clone().requires_grad_(True)
    y_local = local(x_local)
    with _checkpoint_no_reentrant():
        y_ref = reference(x_ref)
    torch.testing.assert_close(y_local, y_ref, rtol=1e-4, atol=1e-5)

    upstream_grad = torch.randn_like(y_local)
    y_local.backward(upstream_grad)
    with _checkpoint_no_reentrant():
        y_ref.backward(upstream_grad)
    assert x_local.grad is not None
    assert x_ref.grad is not None
    torch.testing.assert_close(x_local.grad, x_ref.grad, rtol=1e-4, atol=1e-5)
