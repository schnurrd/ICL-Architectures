from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pfns.experiments.model_benchmarks.analysis import add_normalized_comparison_metrics


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
