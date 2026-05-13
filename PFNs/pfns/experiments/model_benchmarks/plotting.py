from __future__ import annotations

from collections.abc import Callable
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

PRETRAIN_MAX_X = 1_000.0
PLOT_FACE_COLOR = "#fbfbfb"
PRETRAIN_REGION_COLOR = "#edf3f7"
GENERALIZATION_REGION_COLOR = "#f7f3ea"
SPLIT_REGION_ALPHA = 0.55
SPLIT_BOUNDARY_COLOR = "#546e7a"
SPLIT_BOUNDARY_LINESTYLE = "--"

def _registry_display_name_map() -> dict[str, str]:
    return {
        model_name: str(model_config.get("display_name", model_name))
        for model_name, model_config in get_all_models().items()
    }


def _upper_bound_model_names() -> set[str]:
    return {
        model_name
        for model_name, model_config in get_all_models().items()
        if bool(model_config.get("oracle_evaluate_only_max_seqlen", False))
    }


def resolve_display_name_map(df: pd.DataFrame | None = None) -> dict[str, str]:
    display_name_map = _registry_display_name_map().copy()

    if df is None or "display_name" not in df.columns or "model" not in df.columns:
        return display_name_map

    display_df = df.loc[
        df["display_name"].notna(), ["model", "display_name"]
    ].drop_duplicates(subset=["model"])
    display_name_map.update(
        {
            str(row["model"]): str(row["display_name"])
            for _, row in display_df.iterrows()
            if str(row["display_name"]).strip()
        }
    )
    return display_name_map


def make_display_name_formatter(
    df: pd.DataFrame | None = None,
    *,
    display_name_map: dict[str, str] | None = None,
) -> Callable[[object], str]:
    """Return a label formatter backed by an explicit display-name map."""
    name_map = (
        resolve_display_name_map(df)
        if display_name_map is None
        else dict(display_name_map)
    )
    return lambda label: name_map.get(str(label), str(label))


def build_model_legend_name_map(
    df: pd.DataFrame | None = None,
    *,
    append_max_hidden_state_size: bool = False,
    hidden_state_size_df: pd.DataFrame | None = None,
    hidden_state_size_col: str = "context_size_mb",
) -> dict[str, str]:
    display_name_map = resolve_display_name_map(df)
    if (
        not append_max_hidden_state_size
        or hidden_state_size_df is None
        or hidden_state_size_df.empty
    ):
        return display_name_map
    if (
        "model" not in hidden_state_size_df.columns
        or hidden_state_size_col not in hidden_state_size_df.columns
    ):
        return display_name_map

    size_df = hidden_state_size_df[["model", hidden_state_size_col]].copy()
    size_df = size_df.dropna(subset=["model", hidden_state_size_col])
    if size_df.empty:
        return display_name_map

    max_size_by_model = (
        size_df.groupby("model", observed=True)[hidden_state_size_col].max().to_dict()
    )
    for model_name, max_size in max_size_by_model.items():
        base_name = display_name_map.get(str(model_name), str(model_name))
        display_name_map[str(model_name)] = f"{base_name} ({float(max_size):.1f} MB)"
    return display_name_map


def _uses_pretraining_split(metric_keys: set[str]) -> bool:
    split_metrics = {"acc", "ce", "roc_auc"}
    return any(
        metric_key in split_metrics
        or any(metric_key.endswith(f"_{base_metric}") for base_metric in split_metrics)
        for metric_key in metric_keys
    )


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
        base_width = (
            max(abs(float(unique_x[0])) * 0.08, 1.0) if unique_x.size == 1 else 1.0
        )
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
    width_lookup = {
        float(x): float(span * width_frac / max(1, model_count))
        for x, span in zip(sorted_x, local_span)
    }
    return np.array([width_lookup[float(x)] for x in x_values], dtype=float)


def _half_violin_side(*, model_index: int, model_count: int) -> Literal["low", "high"]:
    if model_count <= 1:
        return "high"
    return "low" if model_index < (model_count / 2.0) else "high"


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


def create_panel_figure(
    *,
    panel_count: int,
    figsize: tuple[float, float] | None = None,
    dpi: int = 400,
    sharey: bool = False,
    panel_width: float = 7.0,
    panel_height: float = 6.0,
    min_width: float = 8.0,
    max_width: float = 24.0,
) -> tuple[Any, list[Any]]:
    if figsize is None:
        figsize = (
            min(max_width, max(min_width, panel_width * panel_count)),
            panel_height,
        )
    fig, axes = plt.subplots(
        nrows=1,
        ncols=panel_count,
        figsize=figsize,
        dpi=dpi,
        sharey=sharey,
    )
    if panel_count == 1:
        axes = [axes]
    else:
        axes = list(axes)
    return fig, axes


def apply_pretraining_split_background(
    ax: Any,
    *,
    boundary: float = PRETRAIN_MAX_X,
    boundary_label: str | None = None,
    pretrain_region_color: str = PRETRAIN_REGION_COLOR,
    generalization_region_color: str = GENERALIZATION_REGION_COLOR,
    region_alpha: float = SPLIT_REGION_ALPHA,
    boundary_color: str = SPLIT_BOUNDARY_COLOR,
    boundary_linestyle: str = SPLIT_BOUNDARY_LINESTYLE,
    boundary_linewidth: float = 1.5,
) -> bool:
    x_left, x_right = ax.get_xlim()
    if x_right <= x_left:
        return False

    pretrain_end = min(float(boundary), x_right)
    if pretrain_end > x_left:
        ax.axvspan(
            x_left,
            pretrain_end,
            color=pretrain_region_color,
            alpha=region_alpha,
            zorder=0,
        )

    if x_right > boundary:
        ax.axvspan(
            max(float(boundary), x_left),
            x_right,
            color=generalization_region_color,
            alpha=region_alpha,
            zorder=0,
        )

    if x_left <= boundary <= x_right:
        ax.axvline(
            float(boundary),
            color=boundary_color,
            linestyle=boundary_linestyle,
            linewidth=boundary_linewidth,
            alpha=0.9,
            label=boundary_label,
        )
    return True


def _format_boundary_label(boundary: float) -> str:
    if (
        boundary >= 1000.0
        and float(boundary).is_integer()
        and int(boundary) % 1000 == 0
    ):
        return f"{int(boundary / 1000)}k"
    if float(boundary).is_integer():
        return f"{int(boundary):,}"
    return f"{boundary:g}"


def add_pretraining_split_legend(
    ax: Any,
    *,
    boundary: float = PRETRAIN_MAX_X,
    pretrain_region_color: str = PRETRAIN_REGION_COLOR,
    generalization_region_color: str = GENERALIZATION_REGION_COLOR,
    region_alpha: float = SPLIT_REGION_ALPHA,
    boundary_color: str = SPLIT_BOUNDARY_COLOR,
    boundary_linestyle: str = SPLIT_BOUNDARY_LINESTYLE,
) -> Any:
    boundary_label = _format_boundary_label(boundary)
    region_handles = [
        mpatches.Patch(
            facecolor=pretrain_region_color,
            alpha=region_alpha,
            edgecolor="none",
            label=f"Pre-training range (<={boundary_label})",
        ),
        mpatches.Patch(
            facecolor=generalization_region_color,
            alpha=region_alpha,
            edgecolor="none",
            label=f"Generalisation range (>{boundary_label})",
        ),
        mlines.Line2D(
            [],
            [],
            color=boundary_color,
            linestyle=boundary_linestyle,
            linewidth=1.5,
            label=f"Pre-training limit ({boundary_label})",
        ),
    ]
    range_legend = ax.legend(
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
    ax.add_artist(range_legend)
    return range_legend


def apply_shared_legend_layout(
    fig: Any,
    axes: list[Any],
    *,
    layout: Literal["bottom", "right"] = "bottom",
    fontsize: int = 11,
) -> None:
    legend_handles, legend_labels = axes[0].get_legend_handles_labels()
    legend_model_count = len(legend_labels)
    if legend_model_count > 0:
        if layout == "right":
            fig.subplots_adjust(right=0.82)
            axes[0].legend(
                legend_handles,
                legend_labels,
                fontsize=fontsize,
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                ncol=1,
                borderaxespad=0.0,
                alignment="left",
            )
        else:
            legend_cols = min(max(1, legend_model_count), 4)
            legend_rows = max(1, math.ceil(legend_model_count / legend_cols))
            bottom_margin = min(0.42, 0.14 + 0.055 * legend_rows)
            fig.subplots_adjust(bottom=bottom_margin)
            fig.legend(
                legend_handles,
                legend_labels,
                fontsize=fontsize,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.026),
                ncol=legend_cols,
                borderaxespad=0.0,
                alignment="center",
            )
    for i in range(1, len(axes)):
        if axes[i].get_legend():
            axes[i].get_legend().remove()


def plot_grouped_runs_with_distribution(
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
    distribution_alpha: float | None,
    distribution_width_frac: float,
    show_run_lines: bool,
    log_x: bool,
    distribution_style: Literal["none", "half_violin", "strip"],
    model_index: int,
    model_count: int,
    summary_stat: Literal["mean", "median"] = "median",
    render_as_upper_bound: bool = False,
    line_width: float = 2.2,
    marker_size: float = 5.2,
    distribution_zorder: int = 3,
    summary_zorder: int = 4,
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
    x_position_lookup = {
        float(base_x): float(curr_x) for base_x, curr_x in zip(unique_x, plot_x)
    }

    if show_run_lines and not render_as_upper_bound:
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
            mean="mean",
            median="median",
            q25=lambda values: values.quantile(0.25),
            q75=lambda values: values.quantile(0.75),
        )
        .reset_index()
        .sort_values(x_col)
    )
    if distribution.empty:
        return False

    if render_as_upper_bound:
        pass
    elif distribution_style == "half_violin":
        violin = ax.violinplot(
            dataset=[
                group[value_col].to_numpy(dtype=float, copy=False)
                for _, group in grouped
            ],
            positions=plot_x,
            widths=_compute_violin_widths(
                plot_x,
                model_count=model_count,
                log_x=log_x,
                width_frac=distribution_width_frac,
            ),
            showmeans=False,
            showmedians=False,
            showextrema=False,
            side=_half_violin_side(
                model_index=model_index,
                model_count=model_count,
            ),
        )
        for body in violin["bodies"]:
            body.set_facecolor(color)
            body.set_edgecolor("none")
            body.set_alpha(0.14 if distribution_alpha is None else distribution_alpha)
            body.set_zorder(distribution_zorder)
    elif distribution_style == "strip":
        strip_widths = _compute_violin_widths(
            plot_x,
            model_count=model_count,
            log_x=log_x,
            width_frac=distribution_width_frac,
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
                alpha=(
                    min(0.22, max(0.08, run_alpha * 3.0))
                    if distribution_alpha is None
                    else distribution_alpha
                ),
                linewidths=0.0,
                zorder=distribution_zorder,
            )

    ax.plot(
        plot_x,
        distribution[summary_stat],
        label=model_label,
        linestyle=":" if render_as_upper_bound else linestyle,
        color=color,
        linewidth=line_width,
        marker=None if render_as_upper_bound else marker,
        markersize=marker_size,
        alpha=0.95,
        zorder=summary_zorder,
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
    distribution_alpha: float | None = None,
    distribution_width_frac: float = 0.18,
    show_run_lines: bool = True,
    distribution_style: Literal["none", "half_violin", "strip"] = "half_violin",
    log_x: bool = False,
    log_y: bool = False,
    invert_y: bool = False,
    model_legend_layout: Literal["bottom", "right"] = "bottom",
    append_max_hidden_state_size: bool = False,
    hidden_state_size_df: pd.DataFrame | None = None,
    hidden_state_size_col: str = "context_size_mb",
    show_pretraining_split: bool | None = None,
    pretrain_boundary: float = PRETRAIN_MAX_X,
    figsize: tuple[float, float] | None = None,
    dpi: int = 400,
    show: bool = True,
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
        raise ValueError(
            "distribution_style must be 'none', 'half_violin', or 'strip'."
        )
    if plot_mode == "individual_runs" and rep_col not in df.columns:
        raise RuntimeError(
            f"plot_mode='individual_runs' requires a '{rep_col}' column in the dataframe."
        )
    if not 0.0 < run_alpha <= 1.0:
        raise ValueError("run_alpha must be in the interval (0, 1].")
    if distribution_alpha is not None and not 0.0 < distribution_alpha <= 1.0:
        raise ValueError("distribution_alpha must be in the interval (0, 1].")
    if not 0.0 < distribution_width_frac:
        raise ValueError("distribution_width_frac must be positive.")

    if plot_mode == "individual_runs":
        show_std = False
        error_bars = None

    display_name_map = build_model_legend_name_map(
        df,
        append_max_hidden_state_size=append_max_hidden_state_size,
        hidden_state_size_df=hidden_state_size_df,
        hidden_state_size_col=hidden_state_size_col,
    )
    upper_bound_model_names = _upper_bound_model_names()
    sns.set_theme(style="whitegrid", font_scale=1.2)
    fig, axes = create_panel_figure(
        panel_count=len(specs),
        figsize=figsize,
        dpi=dpi,
    )
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.06, bottom=0.12, right=0.98, top=0.92, wspace=0.45)

    # Fixed pre-training / generalization split styling.
    metric_keys = {metric_key for metric_key, _ in specs}
    show_split = (
        _uses_pretraining_split(metric_keys)
        if show_pretraining_split is None
        else show_pretraining_split
    )

    for idx, (metric_key, metric_name) in enumerate(specs):
        ax = axes[idx]
        ax.set_facecolor(PLOT_FACE_COLOR)
        subset_metric = df[df["metric"] == metric_key]
        present_models = subset_metric["model"].astype(str).unique().tolist()
        model_names = [name for name in style_map if name in present_models]
        model_names.extend(name for name in present_models if name not in model_names)
        panel_pretrain_boundary = float(pretrain_boundary)
        x_values = subset_metric[x_col].to_numpy(dtype=np.float64, copy=False)
        finite_x_values = x_values[np.isfinite(x_values)]
        positive_x_values = finite_x_values[finite_x_values > 0.0]

        for model in model_names:
            sub = subset_metric[subset_metric["model"] == model]
            marker, linestyle, color = style_map.get(model, ("o", "-", None))
            model_label = (
                display_name_map.get(str(model), str(model)) if idx == 0 else None
            )
            render_as_upper_bound = model in upper_bound_model_names
            if plot_mode == "individual_runs":
                plotted = plot_grouped_runs_with_distribution(
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
                    distribution_alpha=distribution_alpha,
                    distribution_width_frac=distribution_width_frac,
                    show_run_lines=show_run_lines,
                    log_x=log_x,
                    distribution_style=distribution_style,
                    model_index=model_names.index(model),
                    model_count=len(model_names),
                    summary_stat="median",
                    render_as_upper_bound=render_as_upper_bound,
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
                linestyle=":" if render_as_upper_bound else linestyle,
                color=color,
                linewidth=2.25,
                marker=None if render_as_upper_bound else marker,
                markersize=5.4,
                alpha=0.96,
            )
            if show_std and not render_as_upper_bound:
                std = agg["std"].fillna(0.0)
                ax.fill_between(
                    agg[x_col],
                    np.maximum(agg["mean"] - std, 0.0),
                    agg["mean"] + std,
                    alpha=0.12,
                    color=color,
                )
            if error_bars is not None and not render_as_upper_bound:
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
                            alpha=0.09,
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
        ax.grid(True, which="major", ls="-", alpha=0.18)
        ax.grid(True, which="minor", ls="-", alpha=0.08)
        for spine in ax.spines.values():
            spine.set_color("#d7d7d7")
            spine.set_linewidth(0.9)
        if log_x:
            ax.set_xscale("log")
            if positive_x_values.size > 0:
                ax.set_xlim(
                    left=float(positive_x_values.min()),
                    right=float(positive_x_values.max()),
                )
        else:
            right_limit = (
                float(finite_x_values.max()) if finite_x_values.size > 0 else None
            )
            ax.set_xlim(left=0.0, right=right_limit)
        if log_y:
            ax.set_yscale("log")
        if invert_y:
            ax.invert_yaxis()

        if show_split:
            apply_pretraining_split_background(ax, boundary=panel_pretrain_boundary)

    if show_split:
        add_pretraining_split_legend(axes[0], boundary=pretrain_boundary)

    apply_shared_legend_layout(fig, axes, layout=model_legend_layout, fontsize=11)

    if show:
        plt.show()
    return fig, axes
