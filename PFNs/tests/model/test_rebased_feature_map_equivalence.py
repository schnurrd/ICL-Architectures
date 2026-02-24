from contextlib import contextmanager

import pytest
import torch
import torch.utils.checkpoint

feature_map_module = pytest.importorskip("fla.modules.feature_map")
FLARebasedFeatureMap = feature_map_module.RebasedFeatureMap

from pfns.model.rebased_feature_map import RebasedFeatureMap as LocalRebasedFeatureMap


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


def _copy_common_state(dst: torch.nn.Module, src: torch.nn.Module) -> None:
    with torch.no_grad():
        dst_state = dst.state_dict()
        for key, value in src.state_dict().items():
            if key in dst_state and dst_state[key].shape == value.shape:
                dst_state[key].copy_(value)


@pytest.mark.parametrize(
    "use_gamma,use_beta,normalize",
    [
        (True, True, True),
        (True, False, True),
        (False, True, True),
        (False, False, False),
    ],
)
def test_local_rebased_feature_map_matches_fla(
    use_gamma: bool,
    use_beta: bool,
    normalize: bool,
) -> None:
    torch.manual_seed(0)
    feature_dim = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and use_gamma and use_beta and normalize:
        pytest.skip("FLA's layer_norm path for this combo requires CUDA in this environment.")

    local = LocalRebasedFeatureMap(
        head_dim=feature_dim,
        use_gamma=use_gamma,
        use_beta=use_beta,
        normalize=normalize,
    ).to(device)
    reference = FLARebasedFeatureMap(
        feature_dim,
        use_gamma,
        use_beta,
        normalize,
    ).to(device)
    _copy_common_state(local, reference)

    x_local = torch.randn(3, 11, 2, feature_dim, device=device, requires_grad=True)
    x_ref = x_local.detach().clone().requires_grad_(True)

    combo_supported = bool(use_gamma) or bool(normalize)
    if not combo_supported:
        with pytest.raises(RuntimeError, match="Not supported combination"):
            local(x_local)
        with _checkpoint_no_reentrant():
            with pytest.raises(RuntimeError, match="Not supported combination"):
                reference(x_ref)
        return

    try:
        y_local = local(x_local)
        with _checkpoint_no_reentrant():
            y_ref = reference(x_ref)
    except RuntimeError as exc:
        if device.type == "cpu" and "cuda" in str(exc).lower():
            pytest.skip("FLA RebasedFeatureMap requires CUDA in this environment.")
        raise

    torch.testing.assert_close(y_local, y_ref, rtol=1e-4, atol=1e-5)

    upstream_grad = torch.randn_like(y_local)
    y_local.backward(upstream_grad)
    with _checkpoint_no_reentrant():
        y_ref.backward(upstream_grad)
    assert x_local.grad is not None
    assert x_ref.grad is not None
    torch.testing.assert_close(x_local.grad, x_ref.grad, rtol=1e-4, atol=1e-5)
