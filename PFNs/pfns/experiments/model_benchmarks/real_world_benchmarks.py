from __future__ import annotations

from pfns.datasets.tabular_datasets import get_benchmark_suite_dids
from pfns.datasets.tabular_datasets import open_cc_dids as OPENCC_BENCHMARK

BENCHMARK_CHOICES = [
    "opencc",
    "openml_large_dataset",
    "tabarena_full",
    "tabarena_medium",
]


def get_real_world_benchmark_dataset_ids(benchmark: str) -> list[int]:
    if benchmark == "opencc":
        return list(OPENCC_BENCHMARK)
    if benchmark == "openml_large_dataset":
        return [1461]
    if benchmark == "tabarena_full":
        return get_benchmark_suite_dids(
            suite_id=457,
            max_features=None,
        )
    if benchmark == "tabarena_medium":
        return get_benchmark_suite_dids(
            suite_id=457,
            min_samples=10_000,
            max_samples=None,
            max_features=None,
        )

    supported = ", ".join(BENCHMARK_CHOICES)
    raise ValueError(f"Benchmark must be one of: {supported}.")


def get_real_world_benchmark_dataset_count(benchmark: str) -> int:
    return len(get_real_world_benchmark_dataset_ids(benchmark))
