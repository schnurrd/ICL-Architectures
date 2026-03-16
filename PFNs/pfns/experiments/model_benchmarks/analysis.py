from __future__ import annotations

from collections.abc import Iterable
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


def add_normalized_comparison_metrics(
    metric_df: pd.DataFrame,
    *,
    metric_keys: Iterable[str],
    higher_is_better_metrics: Iterable[str],
    group_cols: Iterable[str],
    comparison_col: str = "model",
    metric_col: str = "metric",
    value_col: str = "value",
    neutral_value: float = 0.5,
    normalized_prefix: str = "normalized_",
) -> pd.DataFrame:
    """Append min-max-normalized comparison metrics to a long-form metric dataframe.

    For each `(metric, *group_cols)` slice, values are normalized across `comparison_col`.
    Lower-is-better metrics are sign-flipped before normalization so that larger normalized
    values always indicate better performance.
    """
    if metric_df.empty:
        return metric_df.copy()

    metric_keys = list(dict.fromkeys(str(metric) for metric in metric_keys))
    higher_is_better_metrics = {str(metric) for metric in higher_is_better_metrics}
    group_cols = list(dict.fromkeys(str(col) for col in group_cols))

    required_cols = {comparison_col, metric_col, value_col, *group_cols}
    missing_cols = sorted(required_cols - set(metric_df.columns))
    if missing_cols:
        raise RuntimeError(
            f"Metric dataframe is missing required columns for normalization: {missing_cols}"
        )

    raw_metric_df = metric_df[metric_df[metric_col].astype(str).isin(metric_keys)].copy()
    if raw_metric_df.empty:
        return metric_df.copy()

    subset_cols = [metric_col, comparison_col, *group_cols]
    duplicate_mask = raw_metric_df.duplicated(subset=subset_cols, keep=False)
    if duplicate_mask.any():
        duplicate_preview = raw_metric_df.loc[duplicate_mask, subset_cols].head().to_dict("records")
        raise RuntimeError(
            "Normalization requires one row per comparison slice. "
            f"Found duplicates for columns {subset_cols}: {duplicate_preview}"
        )

    def _normalize_group(group: pd.DataFrame) -> pd.DataFrame:
        normalized_group = group.copy()
        raw_scores = normalized_group[value_col].to_numpy(dtype=float, copy=True)
        finite_mask = np.isfinite(raw_scores)
        metric_name = str(normalized_group[metric_col].iloc[0])
        scores = raw_scores if metric_name in higher_is_better_metrics else -raw_scores
        normalized = np.full_like(scores, np.nan, dtype=float)
        finite_scores = scores[finite_mask]
        if finite_scores.size:
            score_min = float(finite_scores.min())
            score_max = float(finite_scores.max())
            normalized[finite_mask] = (
                float(neutral_value)
                if np.isclose(score_min, score_max)
                else (finite_scores - score_min) / (score_max - score_min)
            )

        normalized_group[value_col] = normalized
        normalized_group[metric_col] = (
            normalized_prefix + normalized_group[metric_col].astype(str)
        )
        return normalized_group.dropna(subset=[value_col])

    normalized_metric_df = pd.concat(
        [
            _normalize_group(group)
            for _, group in raw_metric_df.groupby(
                [metric_col, *group_cols],
                observed=True,
                sort=True,
            )
        ],
        ignore_index=True,
    )
    if normalized_metric_df.empty:
        return metric_df.copy()

    return pd.concat([metric_df.copy(), normalized_metric_df], ignore_index=True)


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
