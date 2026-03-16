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
            agg = (
                sub.groupby(x_col, observed=True)[value_col]
                .agg(mean="mean", std="std", ci95=ci95_halfwidth)
                .reset_index()
                .sort_values(x_col)
            )
            if agg.empty:
                continue

            marker, linestyle, color = style_map.get(model, ("o", "-", None))
            ax.plot(
                agg[x_col],
                agg["mean"],
                label=display_name_map.get(str(model), str(model)) if idx == 0 else None,
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
                            lower_err = np.minimum(err, np.maximum(mean_values - np.finfo(float).tiny, 0.0))
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
