from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


def ci95_halfwidth(values: pd.Series) -> float:
    """
    Return two-sided 95% CI half-width for the sample mean.
    """
    n = int(values.shape[0])
    if n <= 1:
        return 0.0
    std = float(values.std(ddof=1))
    sem = float(std / (n ** 0.5))
    t_crit = float(student_t.ppf(0.975, df=n - 1))
    return float(t_crit * sem)


def summarize_diff(diff: pd.Series) -> dict[str, float | int | bool] | None:
    n = int(diff.shape[0])
    if n == 0:
        return None

    mean_gain = float(diff.mean())
    std_gain = float(diff.std(ddof=1)) if n > 1 else 0.0
    sem_gain = float(std_gain / (n ** 0.5)) if n > 1 else 0.0
    ci95 = ci95_halfwidth(diff)
    ci95_low = mean_gain - ci95
    ci95_high = mean_gain + ci95

    return {
        "mean_gain": mean_gain,
        "std_gain": std_gain,
        "sem_gain": sem_gain,
        "ci95": ci95,
        "ci95_low": ci95_low,
        "ci95_high": ci95_high,
        "n_pairs": n,
        "ci95_excludes_zero": (n > 1) and ((ci95_low > 0.0) or (ci95_high < 0.0)),
    }


def paired_gain(series_a: pd.Series, series_b: pd.Series, *, higher_better: bool) -> pd.Series:
    """Return paired gain (positive is better for `series_a`)."""
    if higher_better:
        return (series_a - series_b).dropna()
    return (series_b - series_a).dropna()


def build_metric_wide_table(
    *,
    comparison_results: pd.DataFrame,
    metric_col: str,
    compare_col: str,
    pair_cols: Iterable[str],
    target_labels: Iterable[str] | None = None,
) -> pd.DataFrame:
    """
    Build a wide paired table for any comparison axis (settings/backbones/maskings/...).

    Rows are defined by `pair_cols`, columns by `compare_col`, and values by mean `metric_col`.
    """
    pair_cols = list(pair_cols)
    required_cols = pair_cols + [compare_col, metric_col]
    missing_cols = [c for c in required_cols if c not in comparison_results.columns]
    if missing_cols:
        raise RuntimeError(f"Missing required columns for wide table: {missing_cols}")

    metric_by_pair = (
        comparison_results.groupby(pair_cols + [compare_col], observed=True)[metric_col]
        .mean()
        .reset_index()
    )
    metric_wide = metric_by_pair.pivot_table(
        index=pair_cols,
        columns=compare_col,
        values=metric_col,
        observed=True,
    )
    if target_labels is not None:
        target_labels = list(dict.fromkeys(target_labels))
        metric_wide = metric_wide.reindex(columns=target_labels)
    return metric_wide


def get_complete_paired_rows(
    *,
    metric_wide: pd.DataFrame,
    target_labels: Iterable[str],
    empty_error_message: str | None = None,
) -> pd.DataFrame:
    """Filter a wide table to complete paired rows for all requested labels."""
    target_labels = list(dict.fromkeys(target_labels))
    metric_wide_complete = metric_wide.dropna(subset=target_labels)
    if metric_wide_complete.empty:
        raise RuntimeError(
            empty_error_message
            or (
                "No complete paired rows found across all requested comparisons. "
                "Evaluate more rows or reduce target_labels."
            )
        )
    return metric_wide_complete


def compute_reference_gain_analysis(
    *,
    metric_wide_complete: pd.DataFrame,
    target_labels: Iterable[str],
    reference_label: str,
    higher_better: bool,
    compare_col: str = "comparison",
) -> dict[str, Any]:
    """
    Compute reference-vs-others paired gain summaries for arbitrary comparisons.
    """
    target_labels = list(dict.fromkeys(target_labels))
    if reference_label not in target_labels:
        raise RuntimeError(
            f"reference_label={reference_label!r} must be in target_labels={target_labels}"
        )
    comparison_labels = [c for c in target_labels if c != reference_label]
    if not comparison_labels:
        raise RuntimeError("No comparison labels remain after selecting reference_label.")

    label_means = metric_wide_complete[target_labels].mean(axis=0)
    gain_records = []
    gain_long_frames = []
    reference_values = metric_wide_complete[reference_label]
    for label in comparison_labels:
        diff = paired_gain(
            metric_wide_complete[label],
            reference_values,
            higher_better=higher_better,
        )
        summary_stats = summarize_diff(diff)
        if summary_stats is None:
            continue
        summary_stats[compare_col] = label
        summary_stats["share_pairs_better"] = float((diff > 0).mean())
        gain_records.append(summary_stats)
        gain_long_frames.append(
            diff.rename("gain").to_frame().assign(**{compare_col: label}).reset_index()
        )

    if not gain_records:
        raise RuntimeError("No paired gains were computed.")

    gain_summary = (
        pd.DataFrame(gain_records)
        .sort_values("mean_gain", ascending=False)
        .reset_index(drop=True)
    )
    gain_long = (
        pd.concat(gain_long_frames, ignore_index=True)
        if gain_long_frames
        else pd.DataFrame(columns=[compare_col, "gain"])
    )

    return {
        "label_means": label_means,
        "reference_label": reference_label,
        "comparison_labels": comparison_labels,
        "gain_summary": gain_summary,
        "gain_long": gain_long,
    }


def compute_pairwise_gain_matrices(
    *,
    metric_wide_complete: pd.DataFrame,
    target_labels: Iterable[str],
    higher_better: bool,
) -> dict[str, pd.DataFrame]:
    """Compute pairwise mean gain and uncertainty matrices across any target labels."""
    target_labels = list(dict.fromkeys(target_labels))
    pairwise_mean = pd.DataFrame(index=target_labels, columns=target_labels, dtype=float)
    pairwise_ci95 = pd.DataFrame(index=target_labels, columns=target_labels, dtype=float)
    pairwise_n = pd.DataFrame(index=target_labels, columns=target_labels, dtype=float)
    pairwise_sig = pd.DataFrame(index=target_labels, columns=target_labels, dtype=bool)

    for row_label in target_labels:
        for col_label in target_labels:
            diff = paired_gain(
                metric_wide_complete[row_label],
                metric_wide_complete[col_label],
                higher_better=higher_better,
            )
            summary_stats = summarize_diff(diff)
            if summary_stats is None:
                pairwise_mean.loc[row_label, col_label] = np.nan
                pairwise_ci95.loc[row_label, col_label] = np.nan
                pairwise_n.loc[row_label, col_label] = np.nan
                pairwise_sig.loc[row_label, col_label] = False
                continue

            pairwise_mean.loc[row_label, col_label] = float(summary_stats["mean_gain"])
            pairwise_ci95.loc[row_label, col_label] = float(summary_stats["ci95"])
            pairwise_n.loc[row_label, col_label] = float(summary_stats["n_pairs"])
            pairwise_sig.loc[row_label, col_label] = bool(summary_stats["ci95_excludes_zero"])

    return {
        "pairwise_mean": pairwise_mean,
        "pairwise_ci95": pairwise_ci95,
        "pairwise_n": pairwise_n,
        "pairwise_sig": pairwise_sig,
    }


def plot_gain_barh(
    *,
    gain_summary: pd.DataFrame,
    compare_col: str,
    reference_label: str,
    metric_label: str,
    unit: str,
    error_bars: str = "ci95",
):
    """Horizontal bar plot for reference-based paired gains."""
    import matplotlib.pyplot as plt

    if error_bars not in {"std", "ci95"}:
        raise ValueError("error_bars must be 'std' or 'ci95'.")
    err_col = "std_gain" if error_bars == "std" else "ci95"

    plot_df = gain_summary.sort_values("mean_gain", ascending=True)
    bar_colors = [
        "#2ca02c" if low > 0 else ("#d62728" if high < 0 else "#7f7f7f")
        for low, high in zip(plot_df["ci95_low"], plot_df["ci95_high"])
    ]

    fig, ax = plt.subplots(figsize=(8.5, max(3.8, 0.65 * len(plot_df))), dpi=130)
    ax.barh(
        plot_df[compare_col],
        plot_df["mean_gain"],
        xerr=plot_df[err_col],
        color=bar_colors,
        alpha=0.9,
        ecolor="#202020",
        capsize=3,
    )
    ax.axvline(0.0, color="black", linewidth=1.0, alpha=0.8)
    ax.set_xlabel(f"Gain vs {reference_label} on {metric_label} (positive is better)")
    ax.set_title(f"Comparison gain with {error_bars} intervals (unit={unit})")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return fig, ax


def plot_gain_boxplot(
    *,
    gain_long: pd.DataFrame,
    compare_col: str,
    ordered_labels: Iterable[str],
    metric_label: str,
    reference_label: str,
    unit: str,
):
    """Boxplot of paired gain distributions against a reference label."""
    import matplotlib.pyplot as plt

    ordered_labels = list(dict.fromkeys(ordered_labels))
    box_data = [
        gain_long.loc[gain_long[compare_col] == label, "gain"].to_numpy()
        for label in ordered_labels
    ]

    fig, ax = plt.subplots(figsize=(8.5, max(3.8, 0.65 * len(ordered_labels))), dpi=130)
    ax.boxplot(
        box_data,
        labels=ordered_labels,
        vert=False,
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": "#9ecae1", "alpha": 0.65},
        medianprops={"color": "#08306b", "linewidth": 1.5},
    )
    means = [float(np.mean(arr)) if len(arr) else np.nan for arr in box_data]
    ax.scatter(means, np.arange(1, len(ordered_labels) + 1), color="#08306b", s=22, label="Mean gain")
    ax.axvline(0.0, color="black", linewidth=1.0, alpha=0.8)
    ax.set_xlabel(f"Paired gain on {metric_label} (positive is better)")
    ax.set_title(f"Gain distribution by comparison vs {reference_label} (unit={unit})")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig, ax


def run_comparison_analysis(
    *,
    comparison_results: pd.DataFrame,
    metric_col: str,
    metric_label: str,
    compare_col: str,
    target_labels: Iterable[str],
    pair_cols: Iterable[str],
    higher_better: bool,
    reference_label: str,
    unit: str,
    error_bars: str = "ci95",
    comparison_label: str = "comparison",
    include_boxplot: bool = True,
    include_pairwise_tables: bool = True,
    include_cd_diagram: bool = True,
    wilcoxon_alpha: float = 0.05,
    empty_error_message: str | None = None,
) -> dict[str, Any]:
    """
    End-to-end comparison analysis runner for settings/backbones/maskings/etc.

    Returns tables, summaries, and optional matplotlib figures.
    """
    target_labels = list(dict.fromkeys(target_labels))
    pair_cols = list(pair_cols)
    if len(target_labels) < 2:
        raise RuntimeError("Need at least two labels in target_labels for comparison analysis.")
    if unit not in {"dataset", "split"}:
        raise ValueError("unit must be 'dataset' or 'split'.")

    metric_wide = build_metric_wide_table(
        comparison_results=comparison_results,
        metric_col=metric_col,
        compare_col=compare_col,
        pair_cols=pair_cols,
        target_labels=target_labels,
    )
    metric_wide_complete = get_complete_paired_rows(
        metric_wide=metric_wide,
        target_labels=target_labels,
        empty_error_message=empty_error_message,
    )

    gain_payload = compute_reference_gain_analysis(
        metric_wide_complete=metric_wide_complete,
        target_labels=target_labels,
        reference_label=reference_label,
        higher_better=higher_better,
        compare_col=compare_col,
    )

    figures: dict[str, Any] = {}
    figures["gain_barh"] = plot_gain_barh(
        gain_summary=gain_payload["gain_summary"],
        compare_col=compare_col,
        reference_label=gain_payload["reference_label"],
        metric_label=metric_label,
        unit=unit,
        error_bars=error_bars,
    )

    if include_boxplot and not gain_payload["gain_long"].empty:
        figures["gain_boxplot"] = plot_gain_boxplot(
            gain_long=gain_payload["gain_long"],
            compare_col=compare_col,
            ordered_labels=gain_payload["gain_summary"][compare_col].tolist(),
            metric_label=metric_label,
            reference_label=gain_payload["reference_label"],
            unit=unit,
        )

    pairwise_payload: dict[str, pd.DataFrame] | None = None
    if include_pairwise_tables:
        pairwise_payload = compute_pairwise_gain_matrices(
            metric_wide_complete=metric_wide_complete,
            target_labels=target_labels,
            higher_better=higher_better,
        )

    if include_cd_diagram:
        figures["wilcoxon_cd"] = plot_wilcoxon_cd_diagram(
            target_labels=target_labels,
            metric_wide_complete=metric_wide_complete,
            higher_better=higher_better,
            alpha=wilcoxon_alpha,
            comparison_label=comparison_label,
            title=f"Wilcoxon/Holm comparison diagram ({metric_label}, unit={unit})",
        )

    return {
        "target_labels": target_labels,
        "metric_wide": metric_wide,
        "metric_wide_complete": metric_wide_complete,
        "n_complete_pairs": int(metric_wide_complete.shape[0]),
        "gain": gain_payload,
        "pairwise": pairwise_payload,
        "figures": figures,
    }

def plot_wilcoxon_cd_diagram(
    *,
    target_labels: Iterable[str],
    metric_wide_complete: pd.DataFrame,
    higher_better: bool,
    alpha: float = 0.05,
    title: str = "Wilcoxon/Holm comparison diagram",
    comparison_label: str = "comparison",
):
    """
    Draw a Wilcoxon/Holm CD diagram for arbitrary comparisons.
    """
    from pfns.experiments.model_benchmarks.wilcoxon_cd_diagram import (
        graph_ranks,
        wilcoxon_holm_from_wide,
    )

    target_labels = list(dict.fromkeys(target_labels))
    p_values, mean_ranks, _ = wilcoxon_holm_from_wide(
        metric_wide_complete=metric_wide_complete,
        target_labels=target_labels,
        higher_better=higher_better,
        alpha=alpha,
    )

    ordered = mean_ranks.reindex(target_labels).dropna().sort_values(ascending=False)
    names = ordered.index.to_numpy()
    avranks = ordered.to_numpy(dtype=float).tolist()

    fig, ax = graph_ranks(
        avranks=avranks,
        names=names,
        p_values=p_values,
        reverse=True,
        width=9,
        labels=False,
    )
    ax.set_title(title, y=0.98)
    ax.text(
        0.5,
        -0.10,
        (
            f"Black bars: non-significant {comparison_label}s (Holm-adjusted Wilcoxon, p >= {alpha:.2f}). "
            f"Ranks: 1 = best, {len(target_labels)} = worst."
        ),
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10,
    )
    return fig, ax
