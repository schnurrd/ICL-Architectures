from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import random
import time
from typing import Any

import numpy as np
import torch

from pfns.priors.associative_recall import generate_associative_recall_batch
from pfns.priors.tabpfn_prior_adapter import TabPFNPriorConfig

SEQ_LEN_TASK_VARIANTS: tuple[str, ...] = ("tabular_prior", "associative_recall")


def _set_data_generation_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class BenchmarkBatch:
    x: torch.Tensor
    y: torch.Tensor
    target_y: torch.Tensor
    categorical_mask: torch.Tensor | None = None
    style: torch.Tensor | None = None
    y_style: torch.Tensor | None = None


def _resolve_tabular_prior_task_kwargs(
    task_kwargs: dict[str, Any] | None,
) -> dict[str, Any]:
    options = dict(task_kwargs or {})
    if options.pop("only_numerical_features", False):
        options.setdefault("prior_overrides", {}).setdefault(
            "prior_config", {}
        )["categorical_feature_p"] = 0.0
    return options


class ClassCoverageBatchGenerator:
    """Iterably sample batches that contain all classes in train and eval slices."""

    @staticmethod
    @contextmanager
    def patch_class_sampler_to_max_num_classes():
        import tabpfn_prior.priors.flexible_categorical as fc

        original_class_sampler_f = fc.class_sampler_f

        def class_sampler_f(_min_: int, max_: int):
            def sampler() -> int:
                return max_

            return sampler

        fc.class_sampler_f = class_sampler_f
        try:
            yield
        finally:
            fc.class_sampler_f = original_class_sampler_f

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
        get_batch = prior.create_get_batch_method()

        if not force_max_num_classes:
            return get_batch

        def get_batch_with_forced_num_classes(**kwargs):
            with cls.patch_class_sampler_to_max_num_classes():
                return get_batch(**kwargs)

        return get_batch_with_forced_num_classes

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
        data_generation_seed: int | None = None,
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
        self.data_generation_seed = (
            int(data_generation_seed) if data_generation_seed is not None else None
        )
        self._sample_index = 0

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
        if self.data_generation_seed is not None:
            _set_data_generation_seed(self.data_generation_seed + self._sample_index)
            self._sample_index += 1
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


class AssociativeRecallBatchGenerator:
    """Sample associative-recall key-value batches with shared API.

    Queries are sampled from the smallest train prefix so they remain answerable for
    every evaluated sequence length in the sweep.
    """

    def __init__(
        self,
        *,
        num_batches: int,
        largest_seqlen: int,
        smallest_seqlen: int,
        num_features: int,
        num_classes: int,
        number_of_test_samples: int,
        batch_device: str = "cpu",
        data_generation_seed: int | None = None,
    ) -> None:
        if min(num_batches, smallest_seqlen, num_features, number_of_test_samples) < 1:
            raise ValueError(
                "num_batches, smallest_seqlen, num_features, and "
                "number_of_test_samples must be >= 1."
            )
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2.")
        if largest_seqlen < smallest_seqlen:
            raise ValueError("largest_seqlen must be >= smallest_seqlen.")

        self.num_batches = num_batches
        self.largest_seqlen = largest_seqlen
        self.smallest_seqlen = smallest_seqlen
        self.num_features = num_features
        self.num_classes = num_classes
        self.number_of_test_samples = number_of_test_samples
        self.batch_device = batch_device
        self.data_generation_seed = (
            int(data_generation_seed) if data_generation_seed is not None else None
        )
        self._sample_index = 0

    def sample_one(self) -> tuple[BenchmarkBatch, float]:
        start_time = time.perf_counter()
        if self.data_generation_seed is not None:
            _set_data_generation_seed(self.data_generation_seed + self._sample_index)
            self._sample_index += 1
        sampled = generate_associative_recall_batch(
            batch_size=1,
            largest_seqlen=self.largest_seqlen,
            smallest_seqlen=self.smallest_seqlen,
            num_features=self.num_features,
            num_classes=self.num_classes,
            number_of_test_samples=self.number_of_test_samples,
            batch_device=self.batch_device,
        )

        batch = BenchmarkBatch(
            x=sampled.x,
            y=sampled.y,
            target_y=sampled.target_y,
            categorical_mask=sampled.categorical_mask,
            style=sampled.style,
            y_style=sampled.y_style,
        )
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        return batch, float(elapsed_ms)

    def __iter__(self):
        for _ in range(self.num_batches):
            yield self.sample_one()


def create_seq_len_batch_generator(
    *,
    task_variant: str,
    num_batches: int,
    largest_seqlen: int,
    smallest_seqlen: int,
    num_features: int,
    num_classes: int,
    number_of_test_samples: int,
    default_device: str,
    task_kwargs: dict[str, Any] | None = None,
):
    options = dict(task_kwargs or {})
    if task_variant == "tabular_prior":
        options = _resolve_tabular_prior_task_kwargs(options)
        generator_cls = ClassCoverageBatchGenerator
        options.setdefault("prior_device", default_device)
    elif task_variant == "associative_recall":
        generator_cls = AssociativeRecallBatchGenerator
        options.setdefault("batch_device", default_device)
    else:
        available = ", ".join(SEQ_LEN_TASK_VARIANTS)
        raise ValueError(
            f"Unknown task_variant {task_variant!r}. Available variants: {available}"
        )

    return generator_cls(
        num_batches=num_batches,
        largest_seqlen=largest_seqlen,
        smallest_seqlen=smallest_seqlen,
        num_features=num_features,
        num_classes=num_classes,
        number_of_test_samples=number_of_test_samples,
        **options,
    )
