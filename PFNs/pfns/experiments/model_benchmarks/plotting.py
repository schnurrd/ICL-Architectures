from __future__ import annotations

from typing import Any, Literal

import math

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .comparison_analysis import ci95_halfwidth
from .constants import DEFAULT_COLORS, DEFAULT_LINESTYLES, DEFAULT_MARKERS
from .model_registry import get_all_models


def _registry_display_name_map() -> dict[str, str]:
    return {
        model_name: str(model_config.get("display_name", model_name))
        for model_name, model_config in get_all_models().items()
    }


def resolve_display_name_map(df: pd.DataFrame | None = None) -> dict[str, str]:
    display_name_map = _registry_display_name_map().copy()

    if df is None or "display_name" not in df.columns or "model" not in df.columns:
        return display_name_map

    display_df = df.loc[df["display_name"].notna(), ["model", "display_name"]].drop_duplicates(
        subset=["model"]
    )
    display_name_map.update(
        {
            str(row["model"]): str(row["display_name"])
            for _, row in display_df.iterrows()
            if str(row["display_name"]).strip()
        }
    )
    return display_name_map


def build_model_style_map(
    model_names: list[str],
    *,
    colors: list[str] | None = None,
    markers: list[str] | None = None,
    linestyles: list[Any] | None = None,
) -> dict[str, tuple[str, Any, str]]:
    colors = colors or DEFAULT_COLORS
    markers = markers or DEFAULT_MARKERS
    linestyles = linestyles or DEFAULT_LINESTYLES

    return {
        name: (
            markers[i % len(markers)],
            linestyles[i % len(linestyles)],
            colors[i % len(colors)],
        )
        for i, name in enumerate(model_names)
    }


def _compute_dodged_positions(
    x_values: np.ndarray,
    *,
    model_index: int,
    model_count: int,
    log_x: bool,
    dodge_strength: float = 0.045,
) -> np.ndarray:
    if model_count <= 1:
        return x_values.astype(float, copy=True)

    slot = model_index - (model_count - 1) / 2.0
    if log_x:
        return x_values.astype(float, copy=True) * (1.0 + slot * dodge_strength)

    unique_x = np.unique(x_values.astype(float, copy=False))
    if unique_x.size <= 1:
        base_width = max(abs(float(unique_x[0])) * 0.08, 1.0) if unique_x.size == 1 else 1.0
    else:
        diffs = np.diff(np.sort(unique_x))
        positive_diffs = diffs[diffs > 0.0]
        base_width = float(positive_diffs.min()) if positive_diffs.size > 0 else 1.0
    return x_values.astype(float, copy=True) + slot * base_width * dodge_strength


def _compute_violin_widths(
    x_values: np.ndarray,
    *,
    model_count: int,
    log_x: bool,
    width_frac: float = 0.18,
) -> np.ndarray:
    x_values = x_values.astype(float, copy=True)
    if x_values.size == 0:
        return x_values

    if log_x:
        widths = x_values * (width_frac / max(1, model_count))
        return np.maximum(widths, np.finfo(float).tiny)

    if x_values.size == 1:
        return np.array([1.0], dtype=float)

    sorted_x = np.sort(np.unique(x_values))
    left_gaps = np.diff(sorted_x, prepend=sorted_x[0])
    right_gaps = np.diff(sorted_x, append=sorted_x[-1])
    local_span = np.minimum(
        np.where(left_gaps > 0.0, left_gaps, np.inf),
        np.where(right_gaps > 0.0, right_gaps, np.inf),
    )
    finite_span = local_span[np.isfinite(local_span)]
    fallback_span = float(finite_span.min()) if finite_span.size > 0 else 1.0
    local_span = np.where(np.isfinite(local_span), local_span, fallback_span)
    width_lookup = {float(x): float(span * width_frac / max(1, model_count)) for x, span in zip(sorted_x, local_span)}
    return np.array([width_lookup[float(x)] for x in x_values], dtype=float)


def _half_violin_side(*, model_index: int, model_count: int) -> Literal["left", "right"]:
    if model_count <= 1:
        return "right"
    return "left" if model_index < (model_count / 2.0) else "right"


def _clip_violin_to_half(
    violin: dict[str, Any],
    *,
    positions: np.ndarray,
    side: Literal["left", "right"],
) -> None:
    for body, center in zip(violin["bodies"], positions):
        for path in body.get_paths():
            vertices = path.vertices
            if side == "left":
                vertices[:, 0] = np.minimum(vertices[:, 0], center)
            else:
                vertices[:, 0] = np.maximum(vertices[:, 0], center)


def _compute_strip_positions(
    center: float,
    values: np.ndarray,
    *,
    width: float,
    log_x: bool,
) -> np.ndarray:
    if values.size <= 1:
        return np.array([center], dtype=float)

    offsets = np.linspace(-0.5, 0.5, values.size, dtype=float)
    ordered_offsets = np.empty_like(offsets)
    ordered_offsets[np.argsort(values, kind="mergesort")] = offsets

    if log_x:
        return center * (1.0 + ordered_offsets * width)
    return center + ordered_offsets * width


def _plot_individual_runs_for_model(
    *,
    ax: Any,
    sub: pd.DataFrame,
    x_col: str,
    value_col: str,
    rep_col: str,
    model_label: str | None,
    marker: str,
    linestyle: Any,
    color: str | None,
    run_alpha: float,
    log_x: bool,
    distribution_style: Literal["none", "half_violin", "strip"],
    model_index: int,
    model_count: int,
) -> bool:
    unique_x = np.sort(sub[x_col].dropna().unique().astype(float))
    if unique_x.size == 0:
        return False

    plot_x = _compute_dodged_positions(
        unique_x,
        model_index=model_index,
        model_count=model_count,
        log_x=log_x,
    )
    x_position_lookup = {float(base_x): float(curr_x) for base_x, curr_x in zip(unique_x, plot_x)}

    for _, run_df in sub.groupby(rep_col, observed=True, sort=True):
        run_df = run_df.sort_values(x_col)
        if run_df.empty:
            continue
        run_x = run_df[x_col].to_numpy(dtype=float, copy=True)
        ax.plot(
            np.array([x_position_lookup[float(x)] for x in run_x], dtype=float),
            run_df[value_col],
            label="_nolegend_",
            linestyle=linestyle,
            color=color,
            linewidth=0.55,
            marker=None,
            alpha=run_alpha,
            zorder=2,
        )

    grouped = list(sub.groupby(x_col, observed=True, sort=True))
    distribution = (
        sub.groupby(x_col, observed=True)[value_col]
        .agg(
            median="median",
            q25=lambda values: values.quantile(0.25),
            q75=lambda values: values.quantile(0.75),
        )
        .reset_index()
        .sort_values(x_col)
    )
    if distribution.empty:
        return False

    if distribution_style == "half_violin":
        violin = ax.violinplot(
            dataset=[group[value_col].to_numpy(dtype=float, copy=False) for _, group in grouped],
            positions=plot_x,
            widths=_compute_violin_widths(
                plot_x,
                model_count=model_count,
                log_x=log_x,
            ),
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        _clip_violin_to_half(
            violin,
            positions=plot_x,
            side=_half_violin_side(
                model_index=model_index,
                model_count=model_count,
            ),
        )
        for body in violin["bodies"]:
            body.set_facecolor(color)
            body.set_edgecolor("none")
            body.set_alpha(0.14)
            body.set_zorder(3)
    elif distribution_style == "strip":
        strip_widths = _compute_violin_widths(
            plot_x,
            model_count=model_count,
            log_x=log_x,
        )
        for x_idx, (_, group) in enumerate(grouped):
            values = group[value_col].to_numpy(dtype=float, copy=False)
            strip_x = _compute_strip_positions(
                float(plot_x[x_idx]),
                values,
                width=float(strip_widths[x_idx]) * 0.7,
                log_x=log_x,
            )
            ax.scatter(
                strip_x,
                values,
                s=8.0,
                color=color,
                alpha=min(0.22, max(0.08, run_alpha * 3.0)),
                linewidths=0.0,
                zorder=3,
            )

    ax.plot(
        plot_x,
        distribution["median"],
        label=model_label,
        linestyle=linestyle,
        color=color,
        linewidth=2.6,
        marker=marker,
        markersize=7,
        alpha=0.95,
        zorder=4,
    )
    return True


def plot_curves_from_df(
    df: pd.DataFrame,
    *,
    specs: list[tuple[str, str]],
    style_map: dict[str, tuple[str, Any, str]],
    x_col: str = "seqlen",
    value_col: str = "value",
    x_label: str = "In-context samples",
    title_suffix: str = "",
    show_std: bool = False,
    error_bars: Literal["std", "ci95"] | None = None,
    error_style: Literal["bars", "band"] = "bars",
    plot_mode: Literal["aggregate", "individual_runs"] = "aggregate",
    rep_col: str = "rep",
    run_alpha: float = 0.35,
    distribution_style: Literal["none", "half_violin", "strip"] = "half_violin",
    log_x: bool = False,
    log_y: bool = False,
    invert_y: bool = False,
    model_legend_layout: Literal["bottom", "right"] = "bottom",
    figsize: tuple[float, float] | None = None,
    dpi: int = 400,
):
    """Generic plotting function used by notebook-level plot wrappers."""
    if df.empty:
        print("No data to plot.")
        return None, None
    if model_legend_layout not in {"bottom", "right"}:
        raise ValueError("model_legend_layout must be 'bottom' or 'right'.")
    if error_bars not in {None, "std", "ci95"}:
        raise ValueError("error_bars must be one of None, 'std', or 'ci95'.")
    if error_style not in {"bars", "band"}:
        raise ValueError("error_style must be 'bars' or 'band'.")
    if plot_mode not in {"aggregate", "individual_runs"}:
        raise ValueError("plot_mode must be 'aggregate' or 'individual_runs'.")
    if distribution_style not in {"none", "half_violin", "strip"}:
        raise ValueError("distribution_style must be 'none', 'half_violin', or 'strip'.")
    if plot_mode == "individual_runs" and rep_col not in df.columns:
        raise RuntimeError(
            f"plot_mode='individual_runs' requires a '{rep_col}' column in the dataframe."
        )
    if not 0.0 < run_alpha <= 1.0:
        raise ValueError("run_alpha must be in the interval (0, 1].")

    if plot_mode == "individual_runs":
        show_std = False
        error_bars = None

    display_name_map = resolve_display_name_map(df)
    sns.set_theme(style="whitegrid", font_scale=1.2)
    if figsize is None:
        # Scale the figure with the number of panels
        panel_width = 7.0
        min_width = 8.0
        max_width = 24.0
        figsize = (
            min(max_width, max(min_width, panel_width * len(specs))),
            6.0,
        )
    fig, axes = plt.subplots(nrows=1, ncols=len(specs), figsize=figsize, dpi=dpi)
    fig.subplots_adjust(left=0.06, bottom=0.12, right=0.98, top=0.92, wspace=0.45)
    if len(specs) == 1:
        axes = [axes]

    # Fixed pre-training / generalization split styling.
    pretrain_max_x = 1_000.0
    pretrain_region_color = "#e3f2fd"
    generalization_region_color = "#fff3e0"
    region_alpha = 0.35
    boundary_color = "#546e7a"
    boundary_linestyle = "--"
    metric_keys = {metric_key for metric_key, _ in specs}
    show_split = bool(metric_keys.intersection({"acc", "ce", "roc_auc"}))

    for idx, (metric_key, metric_name) in enumerate(specs):
        ax = axes[idx]
        subset_metric = df[df["metric"] == metric_key]
        present_models = subset_metric["model"].astype(str).unique().tolist()
        model_names = [name for name in style_map if name in present_models]
        model_names.extend(name for name in present_models if name not in model_names)
        pretrain_boundary = float(pretrain_max_x)
        x_values = subset_metric[x_col].to_numpy(dtype=np.float64, copy=False)
        finite_x_values = x_values[np.isfinite(x_values)]
        positive_x_values = finite_x_values[finite_x_values > 0.0]

        for model in model_names:
            sub = subset_metric[subset_metric["model"] == model]
            marker, linestyle, color = style_map.get(model, ("o", "-", None))
            model_label = display_name_map.get(str(model), str(model)) if idx == 0 else None
            if plot_mode == "individual_runs":
                plotted = _plot_individual_runs_for_model(
                    ax=ax,
                    sub=sub,
                    x_col=x_col,
                    value_col=value_col,
                    rep_col=rep_col,
                    model_label=model_label,
                    marker=marker,
                    linestyle=linestyle,
                    color=color,
                    run_alpha=run_alpha,
                    log_x=log_x,
                    distribution_style=distribution_style,
                    model_index=model_names.index(model),
                    model_count=len(model_names),
                )
                if not plotted:
                    continue
                continue

            agg = (
                sub.groupby(x_col, observed=True)[value_col]
                .agg(mean="mean", std="std", ci95=ci95_halfwidth)
                .reset_index()
                .sort_values(x_col)
            )
            if agg.empty:
                continue

            ax.plot(
                agg[x_col],
                agg["mean"],
                label=model_label,
                linestyle=linestyle,
                color=color,
                linewidth=2.5,
                marker=marker,
                markersize=8,
            )
            if show_std:
                std = agg["std"].fillna(0.0)
                ax.fill_between(
                    agg[x_col],
                    np.maximum(agg["mean"] - std, 0.0),
                    agg["mean"] + std,
                    alpha=0.2,
                    color=color,
                )
            if error_bars is not None:
                err = agg[error_bars].fillna(0.0).to_numpy(dtype=float)
                if np.any(err > 0.0):
                    mean_values = agg["mean"].to_numpy(dtype=float)
                    lower_values = mean_values - err
                    if log_y:
                        lower_values = np.maximum(lower_values, np.finfo(float).tiny)
                    if error_style == "band":
                        ax.fill_between(
                            agg[x_col],
                            lower_values,
                            mean_values + err,
                            alpha=0.12,
                            color=color,
                        )
                    else:
                        if log_y:
                            lower_err = np.minimum(
                                err,
                                np.maximum(mean_values - np.finfo(float).tiny, 0.0),
                            )
                            yerr = np.vstack([lower_err, err])
                        else:
                            yerr = err
                        ax.errorbar(
                            agg[x_col],
                            agg["mean"],
                            yerr=yerr,
                            fmt="none",
                            ecolor=color,
                            elinewidth=1.2,
                            capsize=3.0,
                            alpha=0.7,
                            zorder=3,
                        )

        ax.set_xlabel(x_label)
        ax.set_ylabel(metric_name)
        ax.set_title(f"{metric_name}{title_suffix}")
        ax.grid(True, which="both", ls="-", alpha=0.2)
        if log_x:
            ax.set_xscale("log")
            if positive_x_values.size > 0:
                ax.set_xlim(left=float(positive_x_values.min()), right=float(positive_x_values.max()))
        else:
            right_limit = float(finite_x_values.max()) if finite_x_values.size > 0 else None
            ax.set_xlim(left=0.0, right=right_limit)
        if log_y:
            ax.set_yscale("log")
        if invert_y:
            ax.invert_yaxis()

        if show_split:
            # Shade full visible axis range (not only where data points exist).
            x_left, x_right = ax.get_xlim()
            if x_right > x_left:
                pretrain_end = min(pretrain_boundary, x_right)
                if pretrain_end > x_left:
                    ax.axvspan(
                        x_left,
                        pretrain_end,
                        color=pretrain_region_color,
                        alpha=region_alpha,
                        zorder=0,
                    )

                if x_right > pretrain_boundary:
                    ax.axvspan(
                        max(pretrain_boundary, x_left),
                        x_right,
                        color=generalization_region_color,
                        alpha=region_alpha,
                        zorder=0,
                    )

                if x_left <= pretrain_boundary <= x_right:
                    ax.axvline(
                        pretrain_boundary,
                        color=boundary_color,
                        linestyle=boundary_linestyle,
                        linewidth=1.5,
                        alpha=0.9,
                    )

    if show_split:
        region_handles = [
            mpatches.Patch(
                facecolor=pretrain_region_color,
                alpha=region_alpha,
                edgecolor="none",
                label="Pre-training range (<=1k)",
            ),
            mpatches.Patch(
                facecolor=generalization_region_color,
                alpha=region_alpha,
                edgecolor="none",
                label="Generalization range (>1k)",
            ),
            mlines.Line2D(
                [],
                [],
                color=boundary_color,
                linestyle=boundary_linestyle,
                linewidth=1.5,
                label="Pre-training limit (1k)",
            ),
        ]
        range_legend = axes[0].legend(
            handles=region_handles,
            loc="best",
            fontsize=8,
            frameon=True,
            framealpha=0.9,
            edgecolor="#d0d0d0",
            borderaxespad=0.3,
            labelspacing=0.25,
            handletextpad=0.5,
        )
        axes[0].add_artist(range_legend)

    legend_handles, legend_labels = axes[0].get_legend_handles_labels()
    legend_model_count = len(legend_labels)
    if legend_model_count > 0:
        if model_legend_layout == "right":
            fig.subplots_adjust(right=0.82)
            axes[0].legend(
                legend_handles,
                legend_labels,
                fontsize=11,
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                ncol=1,
                borderaxespad=0.0,
                alignment="left",
            )
        else:
            legend_cols = min(max(1, legend_model_count), 4)
            legend_rows = max(1, math.ceil(legend_model_count / legend_cols))
            # Reserve enough room for x-labels + legend while keeping the legend close to the axes.
            bottom_margin = min(0.42, 0.14 + 0.055 * legend_rows)
            fig.subplots_adjust(bottom=bottom_margin)
            fig.legend(
                legend_handles,
                legend_labels,
                fontsize=11,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.03),
                ncol=legend_cols,
                borderaxespad=0.0,
                alignment="center",
            )
    for i in range(1, len(specs)):
        if axes[i].get_legend():
            axes[i].get_legend().remove()

    plt.show()
    return fig, axes
