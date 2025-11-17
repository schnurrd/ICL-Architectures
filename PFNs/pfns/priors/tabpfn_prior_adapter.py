"""
Utility wrappers to use the standalone TabPFN v1 prior implementations with the
PFNs training loop.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable

import torch

from pfns.priors.prior import Batch, PriorConfig

try:
    from tabpfn_prior.tabpfn_prior import TabPFNPriorDataLoader
except ModuleNotFoundError as exc:
    raise ImportError(
        "The tabpfn_prior package is required. Install it via "
        "`pip install -e prior-repos/tabpfn-v1-prior`."
    ) from exc


def _clone_hyperparameters(hparams: dict[str, Any]) -> dict[str, Any]:
    try:
        return deepcopy(hparams)
    except TypeError:
        return dict(hparams)


@dataclass(frozen=True)
class TabPFNPriorConfig(PriorConfig):

    prior_type: str = "prior_bag"
    max_num_classes: int = 10
    prior_config: dict[str, Any] | None = None
    flexible: bool = True
    differentiable: bool = False

    def create_get_batch_method(self) -> Callable[..., Batch]:
        loader = TabPFNPriorDataLoader(
            prior_type=self.prior_type,
            num_steps=1,
            batch_size=1,
            num_datapoints_max=2,
            num_features=2,
            max_num_classes=self.max_num_classes,
            device=torch.device("cpu"),
            prior_config=self.prior_config,
            flexible=self.flexible,
            differentiable=self.differentiable,
        )

        base_hparams = _clone_hyperparameters(loader.prior_hyperparameters)
        get_batch_fn = loader.get_batch_fn

        def get_batch(
            *,
            batch_size: int,
            seq_len: int,
            num_features: int,
            single_eval_pos: int | None,
            device: str | torch.device = "cpu",
            n_targets_per_input: int = 1,
            **kwargs,
        ) -> Batch:
            if single_eval_pos is None:
                single_eval_pos = max(1, int(0.8 * seq_len))

            hparams = _clone_hyperparameters(base_hparams)
            hparams["seq_len_used"] = seq_len
            hparams["num_features_used"] = (lambda nf=num_features: nf)
            hparams["max_num_classes"] = self.max_num_classes
            hparams["num_classes"] = self.max_num_classes

            batch_tuple = get_batch_fn(
                batch_size=batch_size,
                seq_len=seq_len,
                num_features=num_features,
                hyperparameters=hparams,
                device=torch.device(device),
                num_outputs=n_targets_per_input,
                single_eval_pos=single_eval_pos,
            )

            converted = loader._tabpfn_to_ours(batch_tuple, single_eval_pos)
            return Batch(
                x=converted["x"],
                y=converted["y"],
                target_y=converted["target_y"],
                single_eval_pos=converted["single_eval_pos"],
            )

        return get_batch


__all__ = ["TabPFNPriorConfig"]
