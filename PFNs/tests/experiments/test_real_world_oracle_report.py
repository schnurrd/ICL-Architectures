from __future__ import annotations

import pandas as pd
import pytest

from pfns.experiments.model_benchmarks.io import save_dataframe_bundle
from pfns.experiments.model_benchmarks.real_world_oracle_report import (
    ArmBundle,
    build_delta_table,
    build_metric_table,
    build_split_table,
    check_comparable,
    find_arm_bundles,
    paired_stats,
)

BASE_EXPERIMENT = {
    "benchmark": "tabarena_full",
    "max_samples": 1_000_000,
    "max_features": 20,
    "max_classes": 10,
    "n_splits": 2,
    "batch_size_inference": 32,
    "n_ensemble_configurations": 10,
    "preprocess_transforms": ["none", "power"],
    "sample_order_permutation": True,
}


def make_results(
    model: str,
    *,
    datasets: dict[str, tuple[int, list[float]]],
) -> pd.DataFrame:
    rows = []
    for dataset, (num_rows, roc_aucs) in datasets.items():
        for split, roc_auc in enumerate(roc_aucs):
            rows.append(
                {
                    "split": split,
                    "accuracy": roc_auc - 0.05,
                    "roc_auc": roc_auc,
                    "log_loss": 1.0 - roc_auc,
                    "ece": 0.01,
                    "fit_time": 0.0,
                    "predict_time": 1.0,
                    "dataset_num_rows": num_rows,
                    "model": model,
                    "dataset": dataset,
                }
            )
    return pd.DataFrame(rows)


def write_bundle(root, name: str, results: pd.DataFrame, experiment: dict | None = None):
    experiment = experiment or BASE_EXPERIMENT
    return save_dataframe_bundle(
        dataframes={"results": results, "summary": None, "per_dataset": None},
        bundle_dir=root / "real_world" / name,
        experiment=experiment,
        run_metadata={**experiment, "device": "cuda:0"},
    )


@pytest.fixture
def output_root(tmp_path):
    write_bundle(
        tmp_path,
        "run_baseline",
        make_results(
            "baseline",
            datasets={"big": (100, [0.80, 0.82]), "small": (50, [0.70, 0.72])},
        ),
    )
    write_bundle(
        tmp_path,
        "run_oracle",
        make_results(
            "oracle",
            datasets={"big": (100, [0.85, 0.89]), "small": (50, [0.71, 0.71])},
        ),
    )
    return tmp_path


def test_find_arm_bundles_keys_on_model_column(output_root):
    arms = find_arm_bundles(["baseline", "oracle"], output_root=output_root)

    assert set(arms) == {"baseline", "oracle"}
    assert arms["oracle"].bundle_dir.name == "run_oracle"
    assert arms["oracle"].experiment["n_splits"] == 2


def test_find_arm_bundles_prefers_the_newest_bundle(output_root):
    """A rerun of the same model must win over the older bundle."""
    newer = make_results("baseline", datasets={"big": (100, [0.99, 0.99])})
    bundle = write_bundle(output_root, "run_baseline_rerun", newer)
    # `save_dataframe_bundle` writes with the current mtime, but make the
    # ordering explicit rather than relying on filesystem timestamp resolution.
    for path in bundle.rglob("*"):
        path.touch()
    bundle.touch()

    arms = find_arm_bundles(["baseline"], output_root=output_root)

    assert arms["baseline"].bundle_dir.name == "run_baseline_rerun"
    assert arms["baseline"].results["roc_auc"].tolist() == [0.99, 0.99]


def test_find_arm_bundles_reports_only_what_exists(output_root):
    arms = find_arm_bundles(["baseline", "missing_model"], output_root=output_root)

    assert set(arms) == {"baseline"}


def test_build_split_table_restricts_to_shared_datasets(output_root):
    arms = find_arm_bundles(["baseline", "oracle"], output_root=output_root)
    arms["oracle"] = ArmBundle(
        model="oracle",
        bundle_dir=arms["oracle"].bundle_dir,
        results=arms["oracle"].results[arms["oracle"].results["dataset"] == "big"],
        experiment=arms["oracle"].experiment,
    )

    table = build_split_table(arms)

    assert set(table["dataset"]) == {"big"}
    assert set(table["model"]) == {"baseline", "oracle"}


def test_build_metric_table_sorts_by_dataset_size(output_root):
    arms = find_arm_bundles(["baseline", "oracle"], output_root=output_root)

    table = build_metric_table(build_split_table(arms))

    assert list(table.index) == ["big", "small"]
    assert table.loc["big", "oracle"] == pytest.approx(0.87)
    assert table.loc["small", "baseline"] == pytest.approx(0.71)


def test_build_delta_table_pairs_within_fold(output_root):
    arms = find_arm_bundles(["baseline", "oracle"], output_root=output_root)

    deltas = build_delta_table(
        build_split_table(arms), arm="oracle", baseline="baseline"
    )

    big = deltas.set_index("dataset").loc["big"]
    assert big["delta"] == pytest.approx(0.06)  # mean of (0.05, 0.07)
    assert big["n_folds"] == 2
    small = deltas.set_index("dataset").loc["small"]
    assert small["delta"] == pytest.approx(0.0)  # mean of (0.01, -0.01)


def test_paired_stats_uses_both_pairing_units(output_root):
    arms = find_arm_bundles(["baseline", "oracle"], output_root=output_root)

    stats = paired_stats(build_split_table(arms), arm="oracle", baseline="baseline")

    assert stats["by_fold"]["n_pairs"] == 4
    assert stats["by_dataset"]["n_pairs"] == 2
    assert stats["by_fold"]["mean_delta"] == pytest.approx(0.03)
    assert stats["by_dataset"]["mean_delta"] == pytest.approx(0.03)


def test_check_comparable_flags_differing_settings(output_root):
    write_bundle(
        output_root,
        "run_mismatched",
        make_results("mismatched", datasets={"big": (100, [0.5, 0.5])}),
        experiment={**BASE_EXPERIMENT, "n_ensemble_configurations": 1},
    )
    arms = find_arm_bundles(
        ["baseline", "oracle", "mismatched"], output_root=output_root
    )

    warnings = check_comparable(arms)

    assert len(warnings) == 1
    assert warnings[0].startswith("mismatched:")
    assert "n_ensemble_configurations=1" in warnings[0]
