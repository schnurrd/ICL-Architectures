from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .constants import DEFAULT_BUCKET_BINS, DEFAULT_BUCKET_LABELS


def nested_metric_table_to_long_df(
    nested_table: dict[str, dict[str, dict[int, list[float]]]],
    *,
    x_column: str = "seqlen",
    value_column: str = "value",
) -> pd.DataFrame:
    """Convert nested per-model/per-metric result dicts to a long dataframe."""
    rows: list[dict[str, Any]] = []
    for model, metrics in nested_table.items():
        for metric, by_x in metrics.items():
            for x, values in by_x.items():
                for rep, v in enumerate(values):
                    if v is None or (isinstance(v, float) and np.isnan(v)):
                        continue
                    rows.append(
                        {
                            "model": model,
                            "metric": metric,
                            x_column: int(x),
                            "rep": int(rep),
                            value_column: float(v),
                        }
                    )

    if not rows:
        return pd.DataFrame(columns=["model", "metric", x_column, "rep", value_column])
    return pd.DataFrame(rows).sort_values(["metric", "model", x_column, "rep"])


def long_df_to_nested_metric_table(
    df: pd.DataFrame,
    *,
    x_column: str = "seqlen",
    value_column: str = "value",
) -> dict[str, dict[str, dict[int, list[float]]]]:
    """Rebuild nested per-model/per-metric structure from long dataframe rows."""
    if df.empty:
        return {}

    table: dict[str, dict[str, dict[int, list[float]]]] = {}
    for (model, metric, x), group in df.sort_values(
        ["model", "metric", x_column, "rep"]
    ).groupby(["model", "metric", x_column], observed=True):
        table.setdefault(model, {}).setdefault(metric, {})[int(x)] = group[
            value_column
        ].astype(float).tolist()
    return table


def add_numeric_buckets(
    df: pd.DataFrame,
    *,
    value_col: str = "seqlen",
    bucket_col: str = "bucket",
    bins: list[float] | None = None,
    labels: list[str] | None = None,
) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    out = df.copy()
    out[bucket_col] = pd.cut(
        out[value_col],
        bins=bins or DEFAULT_BUCKET_BINS,
        labels=labels or DEFAULT_BUCKET_LABELS,
        include_lowest=True,
        right=True,
    )
    return out


def compute_mean_rank_tables(
    metric_df: pd.DataFrame,
    *,
    x_col: str = "seqlen",
    higher_is_better_metrics: set[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Compute mean ranks overall, by bucket, and by x-axis value."""
    if metric_df.empty:
        empty = pd.DataFrame(columns=["metric", "model", "rank"])
        by_x = pd.DataFrame(columns=["metric", x_col, "model", "rank"])
        return {
            "overall_ranks": empty,
            "bucket_ranks": pd.DataFrame(columns=["metric", "bucket", "model", "rank"]),
            "x_ranks": by_x,
        }

    higher_is_better_metrics = higher_is_better_metrics or {"acc", "roc_auc"}
    df = add_numeric_buckets(metric_df, value_col=x_col).dropna(subset=["value"]).copy()

    def with_rank(base: pd.DataFrame, scope: list[str]) -> pd.DataFrame:
        is_max = base["metric"].isin(higher_is_better_metrics)
        base.loc[is_max, "sort_val"] = -base.loc[is_max, "value"]
        base.loc[~is_max, "sort_val"] = base.loc[~is_max, "value"]
        base["rank"] = base.groupby(scope, observed=True)["sort_val"].rank(method="average")
        return base

    overall = with_rank(
        df.groupby(["metric", "rep", "model"], observed=True)["value"].mean().reset_index(),
        ["metric", "rep"],
    )
    overall_ranks = overall.groupby(["metric", "model"], observed=True)["rank"].mean().reset_index()

    bucket = with_rank(
        df.groupby(["metric", "bucket", "rep", "model"], observed=True)["value"].mean().reset_index(),
        ["metric", "bucket", "rep"],
    )
    bucket_ranks = bucket.groupby(["metric", "bucket", "model"], observed=True)["rank"].mean().reset_index()

    by_x = with_rank(
        df.groupby(["metric", x_col, "rep", "model"], observed=True)["value"].mean().reset_index(),
        ["metric", x_col, "rep"],
    )
    x_ranks = by_x.groupby(["metric", x_col, "model"], observed=True)["rank"].mean().reset_index()

    out = {
        "overall_ranks": overall_ranks,
        "bucket_ranks": bucket_ranks,
        "x_ranks": x_ranks,
    }
    return out
