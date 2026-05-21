from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import pandas as pd

from .wilcoxon_cd_diagram import (
    DEFAULT_GROUP_LETTERS,
    wilcoxon_holm_significance_letters,
)

DEFAULT_LATEX_RANK_COLORS = {1: "yellow!35", 2: "gray!25", 3: "orange!25"}


def _latex_line_index(lines: Sequence[str], prefix: str) -> int | None:
    return next(
        (i for i, line in enumerate(lines) if line.lstrip().startswith(prefix)),
        None,
    )


def _is_caption_or_label(line: str) -> bool:
    return line.lstrip().startswith((r"\caption{", r"\label{"))


def latex_escape(value: object) -> str:
    value = " ".join(str(value).replace("\n", " ").split())
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in value)


def format_latex_number(value: object, *, precision: int = 4) -> str:
    return "-" if pd.isna(value) else f"{float(value):.{precision}f}"


def format_ranked_latex_value(
    value: object,
    rank: object,
    *,
    marker: str = "",
    rank_colors: Mapping[int, str] | None = None,
    precision: int = 4,
) -> str:
    formatted = format_latex_number(value, precision=precision)
    if marker:
        formatted = f"{formatted}{marker}"
    color = None
    if pd.notna(rank):
        color = (rank_colors or DEFAULT_LATEX_RANK_COLORS).get(int(rank))
    return rf"\cellcolor{{{color}}}{formatted}" if color else formatted


def tighten_latex_table_spacing(
    latex_table: str,
    *,
    tabcolsep: str = "3pt",
    arraystretch: str = "1.0",
) -> str:
    """Reduce column padding without scaling the table font."""
    lines = latex_table.splitlines()
    begin_idx = _latex_line_index(lines, r"\begin{tabular}")
    if begin_idx is None:
        return latex_table

    wrapped = lines[:begin_idx]
    wrapped.extend(
        [
            r"\centering",
            rf"\setlength{{\tabcolsep}}{{{tabcolsep}}}",
            rf"\renewcommand{{\arraystretch}}{{{arraystretch}}}",
        ]
    )
    wrapped.extend(lines[begin_idx:])
    return "\n".join(wrapped)


def move_latex_caption_and_label_to_bottom(
    latex_table: str,
    *,
    caption_skip: str = "0.5em",
) -> str:
    lines = latex_table.splitlines()
    caption_label_lines = [line for line in lines if _is_caption_or_label(line)]
    if not caption_label_lines:
        return latex_table

    lines = [line for line in lines if not _is_caption_or_label(line)]
    end_tabular_idx = _latex_line_index(lines, r"\end{tabular}")
    if end_tabular_idx is None:
        return latex_table

    return "\n".join(
        lines[: end_tabular_idx + 1]
        + [rf"\vspace{{{caption_skip}}}"]
        + caption_label_lines
        + lines[end_tabular_idx + 1 :]
    )


def insert_latex_midrules_after_data_rows(
    latex_table: str,
    row_counts: Sequence[int],
) -> str:
    split_points = set()
    running_total = 0
    for count in row_counts[:-1]:
        running_total += int(count)
        if running_total > 0:
            split_points.add(running_total)
    if not split_points:
        return latex_table

    lines = latex_table.splitlines()
    midrule_idx = _latex_line_index(lines, r"\midrule")
    if midrule_idx is None:
        return latex_table

    output = lines[: midrule_idx + 1]
    seen_rows = 0
    for idx, line in enumerate(lines[midrule_idx + 1 :], start=midrule_idx + 1):
        output.append(line)
        if line.lstrip().startswith(r"\bottomrule"):
            output.extend(lines[idx + 1 :])
            break
        if not line.strip() or line.lstrip().startswith("\\"):
            continue
        seen_rows += 1
        if seen_rows in split_points:
            output.append(r"\midrule")
            split_points.discard(seen_rows)
            if not split_points:
                output.extend(lines[idx + 1 :])
                break
    return "\n".join(output)


def metric_label_higher_is_better(metric_label: object) -> bool:
    label = str(metric_label)
    return not any(token in label for token in ("\\downarrow", "downarrow", "↓"))


def normalize_setting_pair_scores(
    complete_pair_table: pd.DataFrame,
    *,
    target_labels: Sequence[str],
    higher_is_better: bool,
) -> pd.DataFrame:
    """Convert raw paired scores to per-row min-max scores."""
    values = complete_pair_table[list(target_labels)]
    row_min = values.min(axis=1)
    row_max = values.max(axis=1)
    denominator = (row_max - row_min).mask(lambda value: value == 0.0)
    if higher_is_better:
        scores = values.sub(row_min, axis=0).div(denominator, axis=0)
    else:
        scores = values.rsub(row_max, axis=0).div(denominator, axis=0)
    return scores.fillna(0.5)


def complete_metric_wide_scores(
    comparison_df: pd.DataFrame,
    metric_col: str,
    *,
    compare_col: str,
    pair_cols: Sequence[str],
    target_labels: Sequence[str],
    normalize: bool = False,
    higher_is_better: bool = True,
) -> pd.DataFrame:
    """Build a complete wide metric table, optionally normalized per paired row."""
    pair_cols = list(pair_cols)
    target_labels = list(target_labels)
    metric_by_pair = (
        comparison_df.groupby(pair_cols + [compare_col], observed=True)[metric_col]
        .mean()
        .reset_index()
    )
    wide = metric_by_pair.pivot_table(
        index=pair_cols,
        columns=compare_col,
        values=metric_col,
        observed=True,
    ).reindex(columns=target_labels)
    complete = wide.dropna(subset=target_labels)
    if normalize:
        complete = normalize_setting_pair_scores(
            complete[list(target_labels)],
            target_labels=target_labels,
            higher_is_better=higher_is_better,
        )
    return complete[list(target_labels)]


def significance_markers_from_scores(
    metric_scores: pd.DataFrame,
    *,
    target_labels: Sequence[str],
    alpha: float = 0.05,
    higher_is_better: bool = True,
    enabled: bool = True,
    letters: str = DEFAULT_GROUP_LETTERS,
    average_index_level: str | None = None,
) -> pd.Series:
    """Return LaTeX significance markers for a wide score table."""
    if not enabled:
        return pd.Series("", index=list(target_labels), dtype=object)
    test_scores = metric_scores
    if (
        average_index_level is not None
        and isinstance(metric_scores.index, pd.MultiIndex)
        and average_index_level in metric_scores.index.names
    ):
        test_scores = metric_scores.groupby(level=average_index_level).mean()
    return wilcoxon_holm_significance_letters(
        test_scores[list(target_labels)],
        target_labels=target_labels,
        alpha=alpha,
        higher_better=higher_is_better,
        letters=letters,
    )


def build_setting_metric_tables(
    *,
    benchmark_results: Mapping[str, pd.DataFrame],
    benchmark_labels: Mapping[str, str],
    metrics: Sequence[tuple[str, str, bool]],
    target_labels: Sequence[str],
    compare_col: str,
    pair_cols: Sequence[str],
    prepare_comparison_results: Callable[[pd.DataFrame], pd.DataFrame],
    metric_label_fn: Callable[[str, bool], str],
    sort_metric: str,
    normalize: bool = False,
    add_significance_markers: bool = True,
    significance_alpha: float = 0.05,
    group_letters: str = DEFAULT_GROUP_LETTERS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Build numeric, rank, and Wilcoxon/Holm marker tables for settings."""
    numeric_columns: list[tuple[str, str]] = []
    numeric_parts: list[pd.Series] = []
    rank_parts: list[pd.Series] = []
    significance_parts: list[pd.Series] = []
    n_pairs_by_benchmark: dict[str, int] = {}
    higher_is_better_by_column: dict[tuple[str, str], bool] = {}

    for preset_name, benchmark_label in benchmark_labels.items():
        preset_results = benchmark_results.get(preset_name, pd.DataFrame())
        if preset_results.empty:
            continue

        comparison_df = prepare_comparison_results(preset_results)
        for metric_col, _metric_label, higher_is_better in metrics:
            if metric_col not in comparison_df.columns:
                continue
            metric_scores = complete_metric_wide_scores(
                comparison_df,
                metric_col,
                compare_col=compare_col,
                pair_cols=pair_cols,
                target_labels=target_labels,
                normalize=normalize,
                higher_is_better=higher_is_better,
            )
            if metric_scores.empty:
                continue
            metric_means = metric_scores[list(target_labels)].mean(axis=0)
            metric_ranks = metric_means.rank(
                ascending=False if normalize else not higher_is_better,
                method="min",
            )
            significance_markers = significance_markers_from_scores(
                metric_scores,
                target_labels=target_labels,
                alpha=significance_alpha,
                higher_is_better=True if normalize else higher_is_better,
                enabled=add_significance_markers,
                letters=group_letters,
                average_index_level="dataset",
            )
            column = (benchmark_label, metric_label_fn(metric_col, normalize))
            higher_is_better_by_column[column] = higher_is_better
            numeric_columns.append(column)
            numeric_parts.append(metric_means.rename(column))
            rank_parts.append(metric_ranks.rename(column))
            significance_parts.append(significance_markers.rename(column))
            n_pairs_by_benchmark.setdefault(
                benchmark_label, int(metric_scores.shape[0])
            )

    if not numeric_parts:
        raise RuntimeError("No compatible setting results were found.")

    numeric_table = pd.concat(numeric_parts, axis=1).reindex(target_labels)
    rank_table = pd.concat(rank_parts, axis=1).reindex(target_labels)
    significance_table = pd.concat(significance_parts, axis=1).reindex(target_labels)
    numeric_table.columns = pd.MultiIndex.from_tuples(numeric_columns)
    rank_table.columns = pd.MultiIndex.from_tuples(numeric_columns)
    significance_table.columns = pd.MultiIndex.from_tuples(numeric_columns)

    sort_col = (
        benchmark_labels.get("openml", next(iter(benchmark_labels.values()))),
        metric_label_fn(sort_metric, normalize),
    )
    if sort_col not in numeric_table.columns:
        sort_col = numeric_table.columns[0]
    sort_higher_is_better = True if normalize else higher_is_better_by_column[sort_col]
    sorted_index = numeric_table.sort_values(
        sort_col,
        ascending=not sort_higher_is_better,
    ).index
    return (
        numeric_table.loc[sorted_index],
        rank_table.loc[sorted_index],
        significance_table.loc[sorted_index],
        n_pairs_by_benchmark,
    )


def render_setting_average_performance_latex(
    numeric_table: pd.DataFrame,
    rank_table: pd.DataFrame,
    significance_table: pd.DataFrame,
    *,
    caption: str,
    label: str,
    rank_colors: Mapping[int, str] | None = None,
    tabcolsep: str = "4pt",
    arraystretch: str = "1.0",
) -> str:
    """Render a compact training-setup LaTeX table."""
    metric_labels = list(numeric_table.columns)
    benchmark_labels = list(dict.fromkeys(col[0] for col in metric_labels))
    n_metric_cols = len(metric_labels)
    lines = [
        r"\begin{table}",
        r"\centering",
        rf"\setlength{{\tabcolsep}}{{{tabcolsep}}}",
        rf"\renewcommand{{\arraystretch}}{{{arraystretch}}}",
        rf"\begin{{tabular}}{{l{'c' * n_metric_cols}}}",
        r"\toprule",
        " & "
        + " & ".join(
            rf"\multicolumn{{{sum(col[0] == benchmark for col in metric_labels)}}}{{c}}{{{benchmark}}}"
            for benchmark in benchmark_labels
        )
        + r" \\",
        "Training setup & " + " & ".join(col[1] for col in metric_labels) + r" \\",
        r"\midrule",
    ]

    for setting_label, row in numeric_table.iterrows():
        values = []
        for metric_label in metric_labels:
            values.append(
                format_ranked_latex_value(
                    row[metric_label],
                    rank_table.loc[setting_label, metric_label],
                    marker=str(significance_table.loc[setting_label, metric_label]),
                    rank_colors=rank_colors,
                )
            )
        lines.append(" & ".join([latex_escape(setting_label), *values]) + r" \\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{0.5em}",
            caption,
            label,
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def make_real_world_metric_tables(
    results: pd.DataFrame,
    *,
    metric_labels: Mapping[str, str],
    compute_per_dataset_stats: Callable[[pd.DataFrame], pd.DataFrame | None],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a mean-only model table from split-level results."""
    per_dataset_stats = compute_per_dataset_stats(results)
    if per_dataset_stats is None or per_dataset_stats.empty:
        return pd.DataFrame(), pd.DataFrame()

    available_metrics = [
        metric
        for metric in metric_labels
        if f"{metric}_mean" in per_dataset_stats.columns
    ]
    if not available_metrics:
        return pd.DataFrame(), per_dataset_stats

    grouped = per_dataset_stats.groupby("model", sort=False)
    return (
        pd.DataFrame(
            {
                metric_labels[metric]: grouped[f"{metric}_mean"].mean()
                for metric in available_metrics
            }
        ),
        per_dataset_stats,
    )


def make_real_world_significance_table(
    per_dataset_stats: pd.DataFrame,
    *,
    metric_labels: Mapping[str, str],
    benchmark_label: str,
    target_models: Sequence[str],
    output_models: Sequence[str] | None = None,
    alpha: float = 0.05,
    letters: str = DEFAULT_GROUP_LETTERS,
) -> pd.DataFrame:
    """Build Wilcoxon/Holm marker columns for one real-world benchmark."""
    markers_by_metric: dict[str, pd.Series] = {}
    available_models = set(per_dataset_stats["model"])
    target_models = [model for model in target_models if model in available_models]
    output_models = (
        list(output_models) if output_models is not None else list(target_models)
    )
    for metric, metric_label in metric_labels.items():
        value_col = f"{metric}_mean"
        if value_col not in per_dataset_stats.columns or len(target_models) < 2:
            markers_by_metric[metric_label] = pd.Series(
                "", index=output_models, dtype=object
            )
            continue
        score_wide = per_dataset_stats.pivot_table(
            index="dataset",
            columns="model",
            values=value_col,
            observed=True,
        ).reindex(columns=target_models)
        markers_by_metric[metric_label] = significance_markers_from_scores(
            score_wide,
            target_labels=target_models,
            alpha=alpha,
            higher_is_better=metric_label_higher_is_better(metric_label),
            letters=letters,
        ).reindex(output_models, fill_value="")
    table = pd.DataFrame(markers_by_metric).reindex(output_models)
    table.columns = pd.MultiIndex.from_tuples(
        [(benchmark_label, metric_label) for metric_label in table.columns]
    )
    return table


def sort_models_by_reference_metric(
    table: pd.DataFrame,
    *,
    benchmark_label: str,
    metric_label: str,
    baseline_model_names: set[str],
    unranked_model_names: set[str],
) -> list[str]:
    """Order ranked models, unranked models, then baselines by one reference metric."""
    sort_col = (benchmark_label, metric_label)
    if sort_col not in table.columns:
        return list(table.index)
    reverse = metric_label_higher_is_better(metric_label)

    def sort_subset(model_names: list[str]) -> list[str]:
        return sorted(
            model_names,
            key=lambda model: table.loc[model, sort_col],
            reverse=reverse,
        )

    ranked_models = [
        model
        for model in table.index
        if model not in baseline_model_names and model not in unranked_model_names
    ]
    unranked_models = [model for model in table.index if model in unranked_model_names]
    baseline_models = [model for model in table.index if model in baseline_model_names]
    return (
        sort_subset(ranked_models)
        + sort_subset(unranked_models)
        + sort_subset(baseline_models)
    )


def apply_real_world_latex_cell_formatting(
    numeric_table: pd.DataFrame,
    *,
    significance_table: pd.DataFrame | None = None,
    baseline_model_names: set[str] | None = None,
    unranked_model_names: set[str] | None = None,
) -> pd.DataFrame:
    """Apply numeric formatting, rank emphasis, baseline bolding, and markers."""
    formatted = numeric_table.map(format_latex_number)

    baseline_model_names = baseline_model_names or set()
    unranked_model_names = unranked_model_names or set()

    if significance_table is not None:
        if isinstance(significance_table.columns, pd.MultiIndex) and (
            significance_table.columns.nlevels > formatted.columns.nlevels
        ):
            significance_table = significance_table.copy()
            significance_table.columns = significance_table.columns.droplevel(
                list(
                    range(
                        significance_table.columns.nlevels - formatted.columns.nlevels
                    )
                )
            )
        aligned_markers = significance_table.reindex(
            index=formatted.index,
            columns=formatted.columns,
        ).fillna("")
        for row_label in formatted.index:
            for col in formatted.columns:
                marker = str(aligned_markers.loc[row_label, col])
                if marker:
                    formatted.loc[row_label, col] = (
                        f"{formatted.loc[row_label, col]}{marker}"
                    )

    ranked_models = [
        model
        for model in formatted.index
        if model not in baseline_model_names and model not in unranked_model_names
    ]
    baseline_models = [
        model for model in formatted.index if model in baseline_model_names
    ]

    for col in formatted.columns:
        metric_label = col[-1] if isinstance(col, tuple) else str(col)
        ascending = not metric_label_higher_is_better(metric_label)

        if ranked_models:
            ranked_values = numeric_table.loc[ranked_models, col]
            ranks = ranked_values.rank(ascending=ascending, method="min")
            for model, rank in ranks.items():
                if pd.isna(rank):
                    continue
                rank = int(rank)
                if rank == 1:
                    formatted.loc[model, col] = (
                        rf"\textbf{{{formatted.loc[model, col]}}}"
                    )
                elif rank == 2:
                    formatted.loc[model, col] = (
                        rf"\underline{{{formatted.loc[model, col]}}}"
                    )

        if baseline_models:
            baseline_values = numeric_table.loc[baseline_models, col]
            baseline_ranks = baseline_values.rank(ascending=ascending, method="min")
            for model, rank in baseline_ranks.items():
                if pd.notna(rank) and int(rank) == 1:
                    formatted.loc[model, col] = (
                        rf"\textbf{{{formatted.loc[model, col]}}}"
                    )
    return formatted


def render_combined_real_world_latex_table(
    display_table: pd.DataFrame,
    *,
    caption: str,
    label: str,
    row_counts: Sequence[int],
    tabcolsep: str,
    arraystretch: str,
) -> str:
    """Render the combined real-world summary table."""
    latex_table = display_table.to_latex(
        multicolumn=True,
        multicolumn_format="c",
        na_rep="-",
        escape=False,
        caption=caption,
        label=label,
        column_format="l" + "c" * len(display_table.columns),
        index_names=False,
    )
    latex_table = tighten_latex_table_spacing(
        latex_table,
        tabcolsep=tabcolsep,
        arraystretch=arraystretch,
    )
    latex_table = insert_latex_midrules_after_data_rows(latex_table, row_counts)
    return move_latex_caption_and_label_to_bottom(latex_table)
