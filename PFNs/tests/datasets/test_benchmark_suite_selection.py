from __future__ import annotations

import json

import pandas as pd
import pytest

from pfns.datasets import tabular_datasets
from pfns.experiments.model_benchmarks import real_world_benchmarks

SUITE_ID = 457

SUITE_TABLE = pd.DataFrame(
    [
        # did, classes, rows, features
        (10, 2, 150_000, 11),
        (20, 2, 129_880, 22),
        (30, 3, 78_053, 12),
        (40, 2, 76_000, 171),
        (50, 2, 71_518, 48),
        (60, 2, 45_211, 14),
        (70, 0, 999_999, 10),  # regression task: never eligible, however large
    ],
    columns=["did", "NumberOfClasses", "NumberOfInstances", "NumberOfFeatures"],
)


@pytest.fixture
def suite_cache(tmp_path, monkeypatch):
    """Point the dataset cache at tmp_path and pre-seed the suite membership."""
    monkeypatch.setenv("OPENML_LOCAL_CACHE_DIRECTORY", str(tmp_path))
    (tmp_path / f"openml_suite_{SUITE_ID}_dids.json").write_text(
        json.dumps({"dids": SUITE_TABLE["did"].tolist()})
    )
    monkeypatch.setattr(
        tabular_datasets, "load_openml_list_cached", lambda dids: SUITE_TABLE.copy()
    )
    return tmp_path


def test_largest_n_keeps_only_the_biggest_datasets(suite_cache):
    dids = tabular_datasets.get_benchmark_suite_dids(
        suite_id=SUITE_ID, max_features=None, largest_n=5
    )

    assert dids == [10, 20, 30, 40, 50]


def test_largest_n_is_applied_after_the_other_filters(suite_cache):
    """The 3 largest *among datasets that pass the feature cap*, not overall."""
    dids = tabular_datasets.get_benchmark_suite_dids(
        suite_id=SUITE_ID, max_features=20, largest_n=3
    )

    assert dids == [10, 30, 60]


def test_largest_n_ignores_non_classification_datasets(suite_cache):
    dids = tabular_datasets.get_benchmark_suite_dids(
        suite_id=SUITE_ID, max_features=None, largest_n=1
    )

    assert dids == [10]


def test_largest_n_caches_separately_from_the_unfiltered_selection(suite_cache):
    all_dids = tabular_datasets.get_benchmark_suite_dids(
        suite_id=SUITE_ID, max_features=None
    )
    largest = tabular_datasets.get_benchmark_suite_dids(
        suite_id=SUITE_ID, max_features=None, largest_n=5
    )

    assert len(all_dids) == 6
    assert largest == [10, 20, 30, 40, 50]
    # The pre-`largest_n` cache name must stay untouched, so caches written by
    # earlier runs keep resolving to the full selection.
    assert (suite_cache / f"benchmark_suite_{SUITE_ID}_dids_all_all_all.json").exists()
    assert (
        suite_cache / f"benchmark_suite_{SUITE_ID}_dids_all_all_all_largest_5.json"
    ).exists()


def test_largest_n_must_be_positive(suite_cache):
    with pytest.raises(ValueError, match="largest_n"):
        tabular_datasets.get_benchmark_suite_dids(suite_id=SUITE_ID, largest_n=0)


def test_tabarena_largest_5_benchmark_selects_five_datasets(suite_cache):
    assert "tabarena_largest_5" in real_world_benchmarks.BENCHMARK_CHOICES

    dids = real_world_benchmarks.get_real_world_benchmark_dataset_ids(
        "tabarena_largest_5"
    )

    assert dids == [10, 20, 30, 40, 50]
