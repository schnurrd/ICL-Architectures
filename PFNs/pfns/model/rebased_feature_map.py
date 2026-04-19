from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def _dense_pairwise_flat(x: torch.Tensor) -> torch.Tensor:
    return (x.unsqueeze(-1) * x.unsqueeze(-2)).flatten(start_dim=-2)


class _UpperTriCache(nn.Module):
    def _diag_and_offdiag_products(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        diag = x.square()
        head_dim = x.shape[-1]
        if head_dim <= 1:
            return diag, x[..., :0]

        offdiag_chunks = [
            x[..., row : row + 1] * x[..., row + 1 :]
            for row in range(head_dim - 1)
        ]
        offdiag = torch.cat(offdiag_chunks, dim=-1)
        return diag, offdiag


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


class RebasedFeatureMap(_UpperTriCache):
    """Torch-native implementation matching FLA's RebasedFeatureMap behavior."""

    def __init__(
        self,
        head_dim: int,
        use_gamma: bool | None = True,
        use_beta: bool | None = True,
        normalize: bool | None = True,
        dense: bool = False,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.use_gamma = use_gamma
        self.use_beta = use_beta
        self.normalize = normalize
        self.dense = dense

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

        if self.dense:
            # Dense basis includes all d^2 pairwise terms; scaled to match the
            # compressed basis kernel value.
            return _dense_pairwise_flat(x) * self.head_dim ** -0.5

        x2_2, x2_1 = self._diag_and_offdiag_products(x)
        return torch.cat(
            [
                x2_2 * self.head_dim ** -0.5,
                x2_1 * (2 / self.head_dim) ** 0.5,
            ],
            dim=-1,
        )


class BasedFeatureMap(_UpperTriCache):
    """Polynomial Based feature map: 1 + q^T k + (q^T k)^2 / 2."""

    def __init__(
        self,
        dense: bool = False,
    ) -> None:
        super().__init__()
        self.dense = dense
        self.inv_sqrt2 = 1.0 / math.sqrt(2.0)

    def forward(
        self,
        x: torch.Tensor,
        flatten: bool = True,
    ) -> torch.Tensor:
        if not flatten:
            return x

        if self.dense:
            x2 = _dense_pairwise_flat(x)
            ones = torch.ones_like(x[..., :1])
            return torch.cat([ones, x, x2 * self.inv_sqrt2], dim=-1)

        # Exact basis with d(d+1)/2 quadratic terms.
        x2_diag, x2_offdiag = self._diag_and_offdiag_products(x)
        x2 = torch.cat([x2_diag, x2_offdiag * math.sqrt(2.0)], dim=-1)
        ones = torch.ones_like(x[..., :1])
        return torch.cat([ones, x, x2 * self.inv_sqrt2], dim=-1)
