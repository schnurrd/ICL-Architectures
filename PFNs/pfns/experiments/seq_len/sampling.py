from __future__ import annotations

import time
from typing import Any

import torch

from pfns.priors.tabpfn_prior_adapter import TabPFNPriorConfig


class ClassCoverageBatchGenerator:
    """Iterably sample batches that contain all classes in train and eval slices."""

    @staticmethod
    def patch_class_sampler_to_max_num_classes() -> None:
        import tabpfn_prior.priors.flexible_categorical as fc

        def class_sampler_f(_min_: int, max_: int):
            def sampler() -> int:
                return max_

            return sampler

        fc.class_sampler_f = class_sampler_f

    @classmethod
    def create_prior_get_batch(
        cls,
        *,
        num_classes: int,
        num_features: int,
        prior_type: str = "mlp",
        device: str = "cuda",
        force_max_num_classes: bool = False,
        prior_overrides: dict[str, Any] | None = None,
    ):
        if force_max_num_classes:
            cls.patch_class_sampler_to_max_num_classes()

        prior_kwargs = {
            "prior_type": prior_type,
            "max_num_classes": num_classes,
            "max_num_features": num_features,
            "flexible": True,
            "differentiable": True,
            "return_categorical_mask": True,
            "nan_handling": True,
            "device": device,
        }
        if prior_overrides:
            prior_kwargs.update(prior_overrides)

        prior = TabPFNPriorConfig(**prior_kwargs)
        return prior.create_get_batch_method()

    def __init__(
        self,
        *,
        num_batches: int,
        largest_seqlen: int,
        smallest_seqlen: int,
        num_features: int,
        num_classes: int,
        number_of_test_samples: int,
        max_attempts: int = 1000,
        prior_type: str = "mlp",
        prior_device: str = "cuda",
        force_max_num_classes: bool = False,
        prior_overrides: dict[str, Any] | None = None,
    ) -> None:
        self.get_batch = self.create_prior_get_batch(
            num_classes=num_classes,
            num_features=num_features,
            prior_type=prior_type,
            device=prior_device,
            force_max_num_classes=force_max_num_classes,
            prior_overrides=prior_overrides,
        )
        self.num_batches = num_batches
        self.largest_seqlen = largest_seqlen
        self.smallest_seqlen = smallest_seqlen
        self.num_features = num_features
        self.num_classes = num_classes
        self.number_of_test_samples = number_of_test_samples
        self.max_attempts = max_attempts

    def _has_full_class_coverage(self, batch: Any) -> bool:
        train_class_count = torch.unique(batch.y[:, : self.smallest_seqlen]).numel()
        eval_class_count = torch.unique(
            batch.y[
                :,
                self.largest_seqlen : self.largest_seqlen + self.number_of_test_samples,
            ]
        ).numel()
        return (
            train_class_count >= self.num_classes
            and eval_class_count >= self.num_classes
        )

    def sample_one(self) -> tuple[Any, float]:
        start_time = time.perf_counter()
        for _ in range(self.max_attempts):
            batch = self.get_batch(
                batch_size=1,
                seq_len=self.largest_seqlen + self.number_of_test_samples,
                num_features=self.num_features,
                single_eval_pos=self.largest_seqlen,
                n_targets_per_input=1,
            )
            if self._has_full_class_coverage(batch):
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                return batch, float(elapsed_ms)

        raise RuntimeError("Failed to sample batch with full class coverage.")

    def __iter__(self):
        for _ in range(self.num_batches):
            yield self.sample_one()
