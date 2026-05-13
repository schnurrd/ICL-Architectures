from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pfns.experiments.model_benchmarks.analysis import (
    add_normalized_comparison_metrics,
    compute_global_normalization_constants,
)


def test_add_normalized_comparison_metrics_appends_direction_aware_metrics() -> None:
    metric_df = pd.DataFrame(
        {
            "model": ["A", "B", "A", "B", "A", "B", "A", "B"],
            "metric": ["acc", "acc", "ce", "ce", "acc", "acc", "ce", "ce"],
            "seqlen": [128, 128, 128, 128, 256, 256, 256, 256],
            "rep": [0, 0, 0, 0, 0, 0, 0, 0],
            "value": [0.8, 0.6, 0.2, 0.4, 0.3, 0.3, 0.7, 0.7],
        }
    )

    out = add_normalized_comparison_metrics(
        metric_df,
        metric_keys=["acc", "ce"],
        higher_is_better_metrics={"acc"},
        group_cols=("seqlen", "rep"),
    )

    normalized = out[out["metric"].isin({"normalized_acc", "normalized_ce"})].copy()
    assert len(normalized) == 8

    seq128 = normalized[normalized["seqlen"] == 128].sort_values(["metric", "model"]).reset_index(drop=True)
    assert seq128.loc[0, "metric"] == "normalized_acc"
    assert seq128.loc[0, "model"] == "A"
    assert seq128.loc[0, "value"] == pytest.approx(1.0)
    assert seq128.loc[1, "value"] == pytest.approx(0.0)
    assert seq128.loc[2, "metric"] == "normalized_ce"
    assert seq128.loc[2, "model"] == "A"
    assert seq128.loc[2, "value"] == pytest.approx(1.0)
    assert seq128.loc[3, "value"] == pytest.approx(0.0)

    seq256 = normalized[normalized["seqlen"] == 256].sort_values(["metric", "model"]).reset_index(drop=True)
    assert np.allclose(seq256["value"].to_numpy(dtype=float), 0.5)


def test_add_normalized_comparison_metrics_rejects_duplicate_comparison_slices() -> None:
    metric_df = pd.DataFrame(
        {
            "model": ["A", "A"],
            "metric": ["acc", "acc"],
            "seqlen": [128, 128],
            "rep": [0, 0],
            "value": [0.8, 0.81],
        }
    )

    with pytest.raises(RuntimeError, match="requires one row per comparison slice"):
        add_normalized_comparison_metrics(
            metric_df,
            metric_keys=["acc"],
            higher_is_better_metrics={"acc"},
            group_cols=("seqlen", "rep"),
        )


def test_compute_global_normalization_constants_ignores_high_x_collapse_for_min() -> None:
    metric_df = pd.DataFrame(
        {
            "model": ["A", "B", "A", "B"],
            "metric": ["roc_auc", "roc_auc", "roc_auc", "roc_auc"],
            "seqlen": [128, 128, 1024, 1024],
            "rep": [0, 0, 0, 0],
            "value": [0.7, 0.6, 0.95, 0.2],
        }
    )

    constants = compute_global_normalization_constants(
        metric_df,
        metric_keys=["roc_auc"],
        higher_is_better_metrics={"roc_auc"},
    )

    assert constants == {"roc_auc": {"min": 0.6, "max": 0.95}}


def test_compute_global_normalization_constants_can_use_x_window_for_min() -> None:
    metric_df = pd.DataFrame(
        {
            "model": ["A", "B", "A", "B", "A", "B"],
            "metric": [
                "roc_auc",
                "roc_auc",
                "roc_auc",
                "roc_auc",
                "roc_auc",
                "roc_auc",
            ],
            "seqlen": [250, 250, 1000, 1000, 2000, 2000],
            "rep": [0, 0, 0, 0, 0, 0],
            "value": [0.7, 0.6, 0.55, 0.8, 0.95, 0.2],
        }
    )

    constants = compute_global_normalization_constants(
        metric_df,
        metric_keys=["roc_auc"],
        higher_is_better_metrics={"roc_auc"},
        lower_bound_reference_max_x=1000,
    )

    assert constants == {"roc_auc": {"min": 0.55, "max": 0.95}}


def test_compute_global_normalization_constants_can_group_by_rep() -> None:
    metric_df = pd.DataFrame(
        {
            "model": ["A", "B", "A", "B", "A", "B", "A", "B"],
            "metric": ["roc_auc"] * 8,
            "seqlen": [250, 250, 2000, 2000, 250, 250, 2000, 2000],
            "rep": [0, 0, 0, 0, 1, 1, 1, 1],
            "value": [0.7, 0.6, 0.95, 0.2, 0.8, 0.75, 0.85, 0.7],
        }
    )

    constants = compute_global_normalization_constants(
        metric_df,
        metric_keys=["roc_auc"],
        higher_is_better_metrics={"roc_auc"},
        group_cols=("rep",),
        lower_bound_reference_max_x=1000,
    )

    assert constants == {
        "roc_auc": {
            0: {"min": 0.6, "max": 0.95},
            1: {"min": 0.75, "max": 0.85},
        }
    }


def test_compute_global_normalization_constants_uses_lowest_x_per_group() -> None:
    metric_df = pd.DataFrame(
        {
            "model": ["A", "B", "A", "B", "A", "B"],
            "metric": ["roc_auc"] * 6,
            "seqlen": [250, 250, 1000, 1000, 2000, 2000],
            "rep": [0, 0, 1, 1, 0, 1],
            "value": [0.7, 0.6, 0.8, 0.75, 0.95, 0.9],
        }
    )

    constants = compute_global_normalization_constants(
        metric_df,
        metric_keys=["roc_auc"],
        higher_is_better_metrics={"roc_auc"},
        group_cols=("rep",),
    )

    assert constants == {
        "roc_auc": {
            0: {"min": 0.6, "max": 0.95},
            1: {"min": 0.75, "max": 0.9},
        }
    }


def test_add_normalized_comparison_metrics_uses_grouped_constants() -> None:
    metric_df = pd.DataFrame(
        {
            "model": ["A", "B", "A", "B"],
            "metric": ["roc_auc", "roc_auc", "roc_auc", "roc_auc"],
            "seqlen": [250, 250, 250, 250],
            "rep": [0, 0, 1, 1],
            "value": [0.775, 0.6, 0.8, 0.75],
        }
    )

    out = add_normalized_comparison_metrics(
        metric_df,
        metric_keys=["roc_auc"],
        higher_is_better_metrics={"roc_auc"},
        group_cols=("rep",),
        normalization_scope="group",
        normalization_constants={
            "roc_auc": {
                0: {"min": 0.6, "max": 0.95},
                1: {"min": 0.75, "max": 0.85},
            }
        },
        normalized_prefix="normalized_global_",
    )

    normalized = (
        out[out["metric"] == "normalized_global_roc_auc"]
        .sort_values(["rep", "model"])
        .reset_index(drop=True)
    )

    assert normalized.loc[0, "value"] == pytest.approx(0.5)
    assert normalized.loc[1, "value"] == pytest.approx(0.0)
    assert normalized.loc[2, "value"] == pytest.approx(0.5)
    assert normalized.loc[3, "value"] == pytest.approx(0.0)


def test_add_normalized_comparison_metrics_uses_fixed_lowest_x_global_constants() -> None:
    metric_df = pd.DataFrame(
        {
            "model": ["A", "B", "A", "B"],
            "metric": ["roc_auc", "roc_auc", "roc_auc", "roc_auc"],
            "seqlen": [128, 128, 1024, 1024],
            "rep": [0, 0, 0, 0],
            "value": [0.7, 0.6, 0.95, 0.2],
        }
    )

    constants = {"roc_auc": {"min": 0.6, "max": 0.95}}
    out = add_normalized_comparison_metrics(
        metric_df,
        metric_keys=["roc_auc"],
        higher_is_better_metrics={"roc_auc"},
        group_cols=("rep",),
        normalization_scope="group",
        normalization_constants=constants,
        normalized_prefix="normalized_global_",
    )

    normalized = (
        out[out["metric"] == "normalized_global_roc_auc"]
        .sort_values(["model", "seqlen"])
        .reset_index(drop=True)
    )

    assert normalized.loc[0, "value"] == pytest.approx((0.7 - 0.6) / 0.35)
    assert normalized.loc[1, "value"] == pytest.approx(1.0)
    assert normalized.loc[2, "value"] == pytest.approx(0.0)
    assert normalized.loc[3, "value"] == pytest.approx((0.2 - 0.6) / 0.35)


def test_add_normalized_comparison_metrics_can_require_fixed_constants() -> None:
    metric_df = pd.DataFrame(
        {
            "model": ["A", "B"],
            "metric": ["roc_auc", "roc_auc"],
            "seqlen": [128, 128],
            "rep": [0, 0],
            "value": [0.7, 0.6],
        }
    )

    with pytest.raises(RuntimeError, match="Missing required normalization constants"):
        add_normalized_comparison_metrics(
            metric_df,
            metric_keys=["roc_auc"],
            higher_is_better_metrics={"roc_auc"},
            group_cols=("rep",),
            normalization_scope="group",
            normalization_constants={},
            require_normalization_constants=True,
            normalized_prefix="normalized_global_",
        )
