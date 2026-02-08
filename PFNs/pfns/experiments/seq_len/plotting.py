from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .constants import DEFAULT_COLORS, DEFAULT_LINESTYLES, DEFAULT_MARKERS


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
    x_label: str = "Number of In-context Samples",
    title_suffix: str = "",
    show_std: bool = False,
    log_x: bool = False,
    invert_y: bool = False,
    figsize: tuple[int, int] = (24, 6),
    dpi: int = 300,
):
    """Generic plotting function used by notebook-level plot wrappers."""
    if df.empty:
        print("No data to plot.")
        return None, None

    sns.set_theme(style="whitegrid", font_scale=1.2)
    fig, axes = plt.subplots(nrows=1, ncols=len(specs), figsize=figsize, dpi=dpi)
    fig.subplots_adjust(left=0.06, bottom=0.2, right=0.98, top=0.92, wspace=0.25)
    if len(specs) == 1:
        axes = [axes]

    for idx, (metric_key, metric_name) in enumerate(specs):
        ax = axes[idx]
        subset_metric = df[df["metric"] == metric_key]
        for model in sorted(subset_metric["model"].unique()):
            sub = subset_metric[subset_metric["model"] == model]
            agg = (
                sub.groupby(x_col, observed=True)[value_col]
                .agg(["mean", "std"])
                .reset_index()
                .sort_values(x_col)
            )
            if agg.empty:
                continue

            marker, linestyle, color = style_map.get(model, ("o", "-", None))
            ax.plot(
                agg[x_col],
                agg["mean"],
                label=model if idx == 0 else None,
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

        ax.set_xlabel(x_label)
        ax.set_ylabel(metric_name)
        ax.set_title(f"{metric_name}{title_suffix}")
        ax.grid(True, which="both", ls="-", alpha=0.2)
        if log_x:
            ax.set_xscale("log")
        if invert_y:
            ax.invert_yaxis()

    axes[0].legend(fontsize=11, loc="upper left", bbox_to_anchor=(0, -0.2), ncol=3)
    for i in range(1, len(specs)):
        if axes[i].get_legend():
            axes[i].get_legend().remove()

    plt.show()
    return fig, axes
