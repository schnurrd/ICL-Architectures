import pytest
import torch

from pfns.model.fla_patches import _maybe_patch_shortconv_forward_pytorch
from fla.modules.convolution import ShortConvolution

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="ShortConvolution patch tests require CUDA.",
)



def test_shortconv_patch_matches_original_forward():
    torch.manual_seed(3)
    device = torch.device("cuda")
    B, T, D = 2, 1, 4
    dtype = torch.float32
    output_final_state = False

    for W in (2, 3, 4):
        for activation in (None, "silu"):
            for use_bias in (False, True):
                for use_residual in (False, True):
                    conv = ShortConvolution(
                        hidden_size=D,
                        kernel_size=W,
                        bias=use_bias,
                        activation=activation,
                        backend="cuda",
                    ).to(device=device, dtype=dtype)

                    x = torch.randn(B, T, D, device=device, dtype=dtype)
                    residual = (
                        torch.randn(B, T, D, device=device, dtype=dtype)
                        if use_residual
                        else None
                    )
                    cache = torch.randn(B, D, W, device=device, dtype=dtype)

                    y_orig, _ = conv(
                        x,
                        residual=residual,
                        cache=cache.clone(),
                        output_final_state=output_final_state,
                    )

                    with _maybe_patch_shortconv_forward_pytorch(True):
                        cache_in = cache.clone()
                        y_patch, _ = conv(
                            x,
                            residual=residual,
                            cache=cache_in,
                            output_final_state=output_final_state,
                        )

                    rtol, atol = 1e-6, 1e-6
                    torch.testing.assert_close(y_patch, y_orig, rtol=rtol, atol=atol)


def test_shortconv_patch_decode_path_is_differentiable():
    torch.manual_seed(7)
    device = torch.device("cuda")
    B, T, D, W = 6, 1, 4, 4

    conv = ShortConvolution(
        hidden_size=D,
        kernel_size=W,
        bias=True,
        activation="silu",
        backend="cuda",
    ).to(device=device, dtype=torch.float32)

    x = torch.randn(B, T, D, device=device, dtype=torch.float32, requires_grad=True)
    residual = torch.randn(B, T, D, device=device, dtype=torch.float32, requires_grad=True)
    cache = torch.randn(B // 3, D, W, device=device, dtype=torch.float32, requires_grad=True)

    with _maybe_patch_shortconv_forward_pytorch(True):
        y, _ = conv(
            x,
            residual=residual,
            cache=cache,
            output_final_state=False,
        )

    assert y.requires_grad
    y.sum().backward()
    assert x.grad is not None
    assert residual.grad is not None
    assert cache.grad is not None
