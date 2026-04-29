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
    q: int | None = None,
    bins: list[float] | None = None,
    labels: list[str] | None = None,
) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    out = df.copy()
    if q is not None:
        if bins is not None:
            raise ValueError("Pass either q or bins to add_numeric_buckets, not both.")
        effective_q = max(1, min(int(q), out[value_col].dropna().nunique()))
        out[bucket_col] = pd.qcut(
            out[value_col],
            q=effective_q,
            duplicates="drop",
        )
        categories = out[bucket_col].cat.categories
        if labels is None:
            labels = [
                f"{_format_bucket_value(interval.left)}-{_format_bucket_value(interval.right)}"
                for interval in categories
            ]
        if len(labels) != len(categories):
            raise ValueError("labels must match the number of quantile buckets.")
        out[bucket_col] = out[bucket_col].cat.rename_categories(labels)
    else:
        out[bucket_col] = pd.cut(
            out[value_col],
            bins=bins or DEFAULT_BUCKET_BINS,
            labels=labels or DEFAULT_BUCKET_LABELS,
            include_lowest=True,
            right=True,
    )
    return out


def _format_bucket_value(value: float) -> str:
    if pd.isna(value):
        return "NA"
    value = float(value)
    abs_value = abs(value)
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs_value >= 1_000:
        return f"{value / 1_000:.1f}k"
    if value.is_integer():
        return str(int(value))
    return f"{value:.0f}"


def compute_global_normalization_constants(
    metric_df: pd.DataFrame,
    *,
    metric_keys: Iterable[str],
    higher_is_better_metrics: Iterable[str],
    group_cols: Iterable[str] = (),
    x_col: str = "seqlen",
    metric_col: str = "metric",
    value_col: str = "value",
    lower_bound_reference_max_x: float | None = None,
) -> dict[str, Any]:
    """Compute fixed min/max constants for global normalized metrics."""
    metric_keys = list(dict.fromkeys(str(metric) for metric in metric_keys))
    higher_is_better_metrics = {str(metric) for metric in higher_is_better_metrics}
    grouping_cols = list(dict.fromkeys(str(col) for col in group_cols))
    grouping_keys = [metric_col, *grouping_cols]
    required_cols = {metric_col, value_col, x_col, *grouping_cols}
    missing_cols = sorted(required_cols - set(metric_df.columns))
    if missing_cols:
        raise RuntimeError(
            f"Metric dataframe is missing required columns for normalization constants: {missing_cols}"
        )

    df = metric_df.loc[
        metric_df[metric_col].astype(str).isin(metric_keys),
        [metric_col, value_col, x_col, *grouping_cols],
    ].copy()
    if df.empty:
        return {}

    df[metric_col] = df[metric_col].astype(str)
    df["_x"] = pd.to_numeric(df[x_col], errors="coerce")
    df["_oriented"] = pd.to_numeric(df[value_col], errors="coerce")
    df.loc[~df[metric_col].isin(higher_is_better_metrics), "_oriented"] *= -1
    df = df[np.isfinite(df["_oriented"]) & np.isfinite(df["_x"])]
    if df.empty:
        return {}

    grouped = df.groupby(grouping_keys, observed=True)
    if lower_bound_reference_max_x is None:
        lower_df = df[np.isclose(df["_x"], grouped["_x"].transform("min"))]
    else:
        lower_df = df[df["_x"] <= float(lower_bound_reference_max_x)]

    bounds = pd.concat(
        [
            lower_df.groupby(grouping_keys, observed=True)["_oriented"].min().rename("min"),
            grouped["_oriented"].max().rename("max"),
        ],
        axis=1,
    ).dropna(subset=["min", "max"])

    constants: dict[str, Any] = {}
    for index, row in bounds.iterrows():
        index_tuple = index if isinstance(index, tuple) else (index,)
        metric = str(index_tuple[0])
        bound = {"min": float(row["min"]), "max": float(row["max"])}
        if not grouping_cols:
            constants[metric] = bound
            continue
        group_key = index_tuple[1] if len(grouping_cols) == 1 else index_tuple[1:]
        constants.setdefault(metric, {})[group_key] = bound
    return constants


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
    normalization_scope: str = "comparison",
    normalization_constants: dict[str, Any] | None = None,
    require_normalization_constants: bool = False,
) -> pd.DataFrame:
    """Append min-max-normalized comparison metrics to a long-form metric dataframe."""
    if metric_df.empty:
        return metric_df.copy()

    metric_keys = list(dict.fromkeys(str(metric) for metric in metric_keys))
    higher_is_better_metrics = {str(metric) for metric in higher_is_better_metrics}
    normalization_group_cols = list(dict.fromkeys(str(col) for col in group_cols))
    normalization_scope = str(normalization_scope)
    if normalization_scope not in {"comparison", "group"}:
        raise ValueError(
            "normalization_scope must be either 'comparison' or 'group'."
        )

    required_cols = {comparison_col, metric_col, value_col, *normalization_group_cols}
    missing_cols = sorted(required_cols - set(metric_df.columns))
    if missing_cols:
        raise RuntimeError(
            f"Metric dataframe is missing required columns for normalization: {missing_cols}"
        )

    raw_metric_df = metric_df[metric_df[metric_col].astype(str).isin(metric_keys)].copy()
    if raw_metric_df.empty:
        return metric_df.copy()

    if normalization_scope == "comparison":
        # In comparison mode we normalize across `comparison_col` within each
        # normalization slice, so each comparison target must appear only once.
        uniqueness_cols = [metric_col, comparison_col, *normalization_group_cols]
        duplicate_mask = raw_metric_df.duplicated(subset=uniqueness_cols, keep=False)
        if duplicate_mask.any():
            duplicate_preview = raw_metric_df.loc[duplicate_mask, uniqueness_cols].head().to_dict("records")
            raise RuntimeError(
                "Normalization requires one row per comparison slice. "
                f"Found duplicates for columns {uniqueness_cols}: {duplicate_preview}"
            )

    grouping_keys = [metric_col, *normalization_group_cols]

    def _normalize_slice(slice_df: pd.DataFrame) -> pd.DataFrame:
        """Normalize one `(metric, *group_cols)` slice to [0, 1].

        The slice can represent:
        - one sequence length across models (`normalization_scope="comparison"`)
        - one dataset or repetition across sequence lengths (`normalization_scope="group"`)
        """
        normalized_slice = slice_df.copy()
        original_values = normalized_slice[value_col].to_numpy(dtype=float, copy=True)
        finite_mask = np.isfinite(original_values)

        metric_name = str(normalized_slice[metric_col].iloc[0])
        oriented_values = (
            original_values
            if metric_name in higher_is_better_metrics
            else -original_values
        )

        normalized_values = np.full_like(oriented_values, np.nan, dtype=float)
        finite_oriented_values = oriented_values[finite_mask]
        if finite_oriented_values.size:
            metric_constants = None
            if normalization_constants is not None:
                metric_constants = normalization_constants.get(metric_name)
                if (
                    isinstance(metric_constants, dict)
                    and "min" not in metric_constants
                ):
                    group_values = tuple(
                        normalized_slice[col].iloc[0]
                        for col in normalization_group_cols
                    )
                    group_key = (
                        group_values[0] if len(group_values) == 1 else group_values
                    )
                    metric_constants = metric_constants.get(group_key)
            if metric_constants is None:
                if require_normalization_constants:
                    group_values = {
                        col: normalized_slice[col].iloc[0]
                        for col in normalization_group_cols
                    }
                    raise RuntimeError(
                        "Missing required normalization constants for metric "
                        f"{metric_name!r} and group {group_values}."
                    )
                slice_min = float(finite_oriented_values.min())
                slice_max = float(finite_oriented_values.max())
            else:
                slice_min = float(metric_constants["min"])
                slice_max = float(metric_constants["max"])
            normalized_values[finite_mask] = (
                float(neutral_value)
                if np.isclose(slice_min, slice_max)
                else (finite_oriented_values - slice_min) / (slice_max - slice_min)
            )

        normalized_slice[value_col] = normalized_values
        normalized_slice[metric_col] = (
            normalized_prefix + normalized_slice[metric_col].astype(str)
        )
        return normalized_slice.dropna(subset=[value_col])

    normalized_metric_df = pd.concat(
        [
            _normalize_slice(group)
            for _, group in raw_metric_df.groupby(
                grouping_keys,
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
