from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def flatten_diag_outer_product_off1(
    x: torch.Tensor,
    y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    z = torch.einsum("...i,...j->...ij", x, y)
    n = z.size(-1)
    indices = torch.triu_indices(n, n, 1, device=z.device)
    diag_idx = torch.arange(0, n, device=z.device)
    return z[..., indices[0], indices[1]], z[..., diag_idx, diag_idx]


def _apply_pre_map_transform(
    x: torch.Tensor,
    *,
    head_dim: int,
    use_gamma: bool | None,
    use_beta: bool | None,
    normalize: bool | None,
    gamma: torch.Tensor | None,
    beta: torch.Tensor | None,
) -> torch.Tensor:
    if normalize:
        return F.layer_norm(x, (head_dim,), gamma, beta)
    if use_gamma and use_beta:
        assert beta is not None and gamma is not None
        return torch.addcmul(beta, x, gamma)
    if use_gamma:
        assert gamma is not None
        return x.mul(gamma)
    if use_beta:
        assert beta is not None
        return x.add(beta)
    raise RuntimeError(
        "Not supported combination of `use_gamma`, `use_beta` and "
        f"`normalize`, which is currently set as "
        f"(`{use_gamma}`, `{use_beta}`, `{normalize}`)"
    )


class RebasedFeatureMap(nn.Module):
    """Torch-native implementation matching FLA's RebasedFeatureMap behavior."""

    def __init__(
        self,
        head_dim: int,
        use_gamma: bool | None = True,
        use_beta: bool | None = True,
        normalize: bool | None = True,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.use_gamma = use_gamma
        self.use_beta = use_beta
        self.normalize = normalize

        self.gamma = None
        self.beta = None
        if use_gamma:
            self.gamma = nn.Parameter(torch.ones(head_dim))
        if use_beta:
            self.beta = nn.Parameter(torch.zeros(head_dim))

    def forward(
        self,
        x: torch.Tensor,
        flatten: bool = True,
    ) -> torch.Tensor:
        x = _apply_pre_map_transform(
            x,
            head_dim=self.head_dim,
            use_gamma=self.use_gamma,
            use_beta=self.use_beta,
            normalize=self.normalize,
            gamma=self.gamma,
            beta=self.beta,
        )

        if not flatten:
            return x

        x2_1, x2_2 = flatten_diag_outer_product_off1(x, x)
        return torch.cat(
            [
                x2_2 * self.head_dim ** -0.5,
                x2_1 * (2 / self.head_dim) ** 0.5,
            ],
            dim=-1,
        )


class BasedFeatureMap(nn.Module):
    """Polynomial Based feature map: 1 + q^T k + (q^T k)^2 / 2."""

    def __init__(self) -> None:
        super().__init__()
        self.inv_sqrt2 = 1.0 / math.sqrt(2.0)

    def forward(
        self,
        x: torch.Tensor,
        flatten: bool = True,
    ) -> torch.Tensor:
        if not flatten:
            return x

        x2 = (x.unsqueeze(-1) * x.unsqueeze(-2)).flatten(start_dim=-2)
        ones = torch.ones_like(x[..., :1])
        return torch.cat([ones, x, x2 * self.inv_sqrt2], dim=-1)
