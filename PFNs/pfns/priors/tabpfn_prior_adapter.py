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
from pfns.utils import normalize_by_used_features_f

try:
    from tabpfn_prior import build_tabpfn_prior
except ModuleNotFoundError as exc:
    raise ImportError(
        "The tabpfn_prior package is required. Install it via "
        "`pip install -e prior-repos/tabpfn-v1-prior`."
    ) from exc


@dataclass(frozen=True)
class TabPFNPriorConfig(PriorConfig):

    prior_type: str = "mlp"
    max_num_classes: int = 10
    prior_config: dict[str, Any] | None = None
    flexible: bool = True
    differentiable: bool = False
    max_num_features: int = 20

    def create_get_batch_method(self) -> Callable[..., Batch]:
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

            assert (
                n_targets_per_input == 1
            ), "TabPFNPriorConfig only supports n_targets_per_input=1"

            assert num_features <= self.max_num_features, (
                f"num_features ({num_features}) cannot be larger than "
                f"max_num_features ({self.max_num_features})"
            )
            
            batch_iterator = iter(
                build_tabpfn_prior(
                    prior_type=self.prior_type,
                    num_steps=1,
                    batch_size=batch_size,
                    num_datapoints_max=seq_len,
                    num_features=num_features,
                    max_num_classes=self.max_num_classes,
                    device=device,
                    prior_config=self.prior_config,
                    flexible=self.flexible,
                    differentiable=self.differentiable,
                    **kwargs,
                )
            )

            batch = next(batch_iterator)  # get a single batch from the prior
            
            # Normalize by used features should like note be necessary anymore
            #x = normalize_by_used_features_f(
            #    batch["x"], num_features, self.max_num_features
            #)
            return Batch(
                x=batch["x"],
                y=batch["y"],
                target_y=batch["target_y"],
                single_eval_pos=single_eval_pos,  # we ignore the single_eval_pos from the prior
            )

        return get_batch


__all__ = ["TabPFNPriorConfig"]
