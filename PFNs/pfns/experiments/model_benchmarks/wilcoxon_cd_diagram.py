"""
Wilcoxon/Holm and critical-difference style plotting utilities.

Inspired by: https://github.com/hfawaz/cd-diagram (GPL-3.0).
This module is an original PFNs implementation.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon


PairwiseResult = tuple[str, str, float, bool]

# used for significance groups
DEFAULT_GROUP_LETTERS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _unique_labels(labels: Iterable[object]) -> list[str]:
    return list(dict.fromkeys(str(label) for label in labels))


def _require_labels(
    frame: pd.DataFrame,
    labels: Sequence[str],
    *,
    context: str,
) -> None:
    missing = [label for label in labels if label not in frame.columns]
    if missing:
        raise RuntimeError(f"Missing labels in {context}: {missing}")


def _complete_score_matrix(
    frame: pd.DataFrame,
    labels: Sequence[str],
    *,
    higher_better: bool,
) -> pd.DataFrame:
    frame = frame.rename(columns=str)
    _require_labels(frame, labels, context="score table")
    score_matrix = frame.reindex(columns=labels).dropna(subset=labels)
    if score_matrix.empty:
        raise RuntimeError(
            "No complete paired rows available for Wilcoxon/Holm analysis."
        )

    score_matrix = score_matrix.astype(float)
    if not higher_better:
        score_matrix = -score_matrix
    return score_matrix.reset_index(drop=True)


def _wilcoxon_p_value_from_arrays(
    values_a: Sequence[float],
    values_b: Sequence[float],
    *,
    alternative: str = "two-sided",
    zero_method: str = "pratt",
) -> float:
    values_a = np.asarray(values_a, dtype=np.float64)
    values_b = np.asarray(values_b, dtype=np.float64)
    if values_a.size < 2:
        return float("nan")
    if np.all(values_a == values_b):
        return 1.0

    try:
        return float(
            wilcoxon(
                values_a,
                values_b,
                alternative=alternative,
                zero_method=zero_method,
            ).pvalue
        )
    except ValueError:
        return float("nan")


def _raw_pairwise_p_values(
    score_matrix: pd.DataFrame,
    labels: Sequence[str],
    *,
    alternative: str = "two-sided",
) -> dict[tuple[str, str], float]:
    raw_p_values: dict[tuple[str, str], float] = {}
    for label_a, label_b in combinations(labels, 2):
        paired = score_matrix[[label_a, label_b]].dropna()
        raw_p_values[(label_a, label_b)] = _wilcoxon_p_value_from_arrays(
            paired[label_a],
            paired[label_b],
            alternative=alternative,
        )
    return raw_p_values


def _find_maximal_cliques(
    nodes: Sequence[str],
    adjacency: dict[str, set[str]],
) -> list[set[str]]:
    """
    Return maximal cliques for an undirected graph using Bron-Kerbosch.

    Nodes are expected to be small in number for CD diagrams, so an exact search
    is practical and avoids drawing redundant pairwise bars.
    """
    node_set = set(nodes)
    cliques: list[set[str]] = []

    def bron_kerbosch(r: set[str], p: set[str], x: set[str]) -> None:
        if not p and not x:
            if len(r) >= 2:
                cliques.append(set(r))
            return

        pivot_candidates = p | x
        pivot = (
            max(pivot_candidates, key=lambda n: len(adjacency[n]))
            if pivot_candidates
            else None
        )
        if pivot is None:
            expand = list(p)
        else:
            expand = list(p - adjacency[pivot])

        for node in expand:
            bron_kerbosch(r | {node}, p & adjacency[node], x & adjacency[node])
            p.remove(node)
            x.add(node)

    bron_kerbosch(set(), node_set, set())
    return cliques


def _non_significant_adjacency(
    names: Sequence[str],
    p_values: Sequence[PairwiseResult],
) -> dict[str, set[str]]:
    adjacency: dict[str, set[str]] = {name: set() for name in names}
    for name_a, name_b, _p_raw, significant in p_values:
        if significant or name_a not in adjacency or name_b not in adjacency:
            continue
        adjacency[name_a].add(name_b)
        adjacency[name_b].add(name_a)
    return adjacency


def _non_significant_rank_intervals(
    rank_by_name: Mapping[str, float],
    p_values: Sequence[PairwiseResult],
) -> list[tuple[float, float]]:
    adjacency = _non_significant_adjacency(list(rank_by_name), p_values)
    intervals: list[tuple[float, float]] = []
    for clique in _find_maximal_cliques(list(rank_by_name), adjacency):
        ranks = [rank_by_name[name] for name in clique]
        rank_min, rank_max = min(ranks), max(ranks)
        if (rank_max - rank_min) > 1e-8:
            intervals.append((rank_min, rank_max))
    return sorted(
        set(intervals),
        key=lambda item: (item[0], -(item[1] - item[0])),
    )


def _pack_intervals_into_lanes(
    intervals: Sequence[tuple[float, float]],
    *,
    interval_pad: float,
) -> list[tuple[float, float, int]]:
    lane_right_edges: list[float] = []
    packed: list[tuple[float, float, int]] = []
    for x_lo, x_hi in intervals:
        lane = 0
        while (
            lane < len(lane_right_edges)
            and x_lo <= lane_right_edges[lane] + interval_pad
        ):
            lane += 1
        if lane == len(lane_right_edges):
            lane_right_edges.append(x_hi)
        else:
            lane_right_edges[lane] = max(lane_right_edges[lane], x_hi)
        packed.append((x_lo, x_hi, lane))
    return packed


def holm_adjust_p_values(p_values: Mapping[object, float] | pd.Series) -> pd.Series:
    """
    Return Holm step-down adjusted p-values.

    The input index/keys are preserved. NaN p-values are ignored for the
    multiplicity count and remain NaN in the output.
    """
    p_values = pd.Series(p_values, dtype=float)
    adjusted = pd.Series(index=p_values.index, dtype=float)
    valid = p_values.dropna().sort_values(kind="stable")
    running_max = 0.0
    m = int(valid.shape[0])
    for i, (label, p_value) in enumerate(valid.items()):
        adjusted_p = min((m - i) * float(p_value), 1.0)
        running_max = max(running_max, adjusted_p)
        adjusted.loc[label] = running_max
    return adjusted.reindex(p_values.index)


def pairwise_wilcoxon_holm(
    metric_scores: pd.DataFrame,
    *,
    target_labels: Iterable[str],
    alpha: float = 0.05,
    higher_better: bool = True,
    alternative: str = "two-sided",
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Run all pairwise paired Wilcoxon tests and Holm adjustment.

    Returns a symmetric boolean significance matrix and the Holm-adjusted
    p-values indexed by `(label_a, label_b)` tuples.
    """
    labels = _unique_labels(target_labels)
    if len(labels) < 2:
        raise RuntimeError("Need at least two labels for pairwise Wilcoxon/Holm.")

    metric_scores = metric_scores.rename(columns=str)
    _require_labels(metric_scores, labels, context="metric_scores")

    means = metric_scores[labels].mean(axis=0)
    labels = list(means.sort_values(ascending=not higher_better).index)
    raw_p_values = _raw_pairwise_p_values(
        metric_scores,
        labels,
        alternative=alternative,
    )

    adjusted_p_values = holm_adjust_p_values(raw_p_values)
    significantly_different = pd.DataFrame(False, index=labels, columns=labels)
    for (label_a, label_b), adjusted_p_value in adjusted_p_values.items():
        is_significant = pd.notna(adjusted_p_value) and adjusted_p_value <= alpha
        significantly_different.loc[label_a, label_b] = is_significant
        significantly_different.loc[label_b, label_a] = is_significant
    return significantly_different, adjusted_p_values


def _validate_compact_significance_letters(
    significantly_different: pd.DataFrame,
    markers: pd.Series,
) -> None:
    """Verify that compact letters exactly encode pairwise non-significance."""
    if not markers.map(bool).all():
        missing = markers.index[~markers.map(bool)].tolist()
        raise RuntimeError(f"Missing significance letters for labels: {missing}")

    for label_a, label_b in combinations(markers.index, 2):
        share_letter = bool(set(markers.loc[label_a]) & set(markers.loc[label_b]))
        are_not_significantly_different = not bool(
            significantly_different.loc[label_a, label_b]
        )
        if share_letter != are_not_significantly_different:
            relation = (
                "not significantly different"
                if are_not_significantly_different
                else "significantly different"
            )
            raise RuntimeError(
                "Significance letter mismatch for "
                f"{label_a!r} and {label_b!r}: pair is {relation}, "
                f"letters are {markers.loc[label_a]!r} and {markers.loc[label_b]!r}."
            )


def compact_significance_letters(
    significantly_different: pd.DataFrame,
    *,
    target_labels: Iterable[object],
    letters: str = DEFAULT_GROUP_LETTERS,
) -> pd.Series:
    """
    Assign compact display letters for a pairwise significance matrix.

    Labels sharing a letter are not significantly different. The assignment is
    deterministic and intended for table display; it is not an optimization
    routine for the minimum possible number of letters.
    """
    labels = _unique_labels(target_labels)
    ordered_labels = list(significantly_different.index)
    groups: list[list[str]] = []

    for label in ordered_labels:
        placed = False
        for group in groups:
            if all(
                not bool(significantly_different.loc[label, other]) for other in group
            ):
                group.append(label)
                placed = True
        if not placed:
            groups.append([label])

    for i, label_a in enumerate(ordered_labels[:-1]):
        for label_b in ordered_labels[i + 1 :]:
            if bool(significantly_different.loc[label_a, label_b]):
                continue
            if any(label_a in group and label_b in group for group in groups):
                continue
            for group in groups:
                can_add_b = label_a in group and all(
                    not bool(significantly_different.loc[label_b, other])
                    for other in group
                )
                can_add_a = label_b in group and all(
                    not bool(significantly_different.loc[label_a, other])
                    for other in group
                )
                if can_add_b:
                    group.append(label_b)
                    break
                if can_add_a:
                    group.append(label_a)
                    break
            else:
                groups.append([label_a, label_b])

    letter_by_label = {label: "" for label in labels}
    for group_idx, group in enumerate(groups):
        if group_idx >= len(letters):
            raise ValueError("Not enough letters for significance groups.")
        letter = letters[group_idx]
        for label in dict.fromkeys(group):
            letter_by_label[label] += letter

    markers = pd.Series(letter_by_label, index=labels, dtype=object)
    _validate_compact_significance_letters(significantly_different, markers)
    return markers


def wilcoxon_holm_significance_letters(
    metric_scores: pd.DataFrame,
    *,
    target_labels: Iterable[object],
    alpha: float = 0.05,
    higher_better: bool = True,
    letters: str = DEFAULT_GROUP_LETTERS,
) -> pd.Series:
    """Return LaTeX superscript letters from paired Wilcoxon/Holm tests."""
    labels = _unique_labels(target_labels)
    significantly_different, _ = pairwise_wilcoxon_holm(
        metric_scores,
        target_labels=labels,
        alpha=alpha,
        higher_better=higher_better,
    )
    return compact_significance_letters(
        significantly_different,
        target_labels=labels,
        letters=letters,
    ).map(lambda value: rf"$^{{{value}}}$" if value else "")


def wilcoxon_holm_from_wide(
    *,
    metric_wide_complete: pd.DataFrame,
    target_labels: Iterable[object],
    higher_better: bool,
    alpha: float = 0.05,
) -> tuple[list[PairwiseResult], pd.Series, float]:
    """
    Pairwise Wilcoxon tests with Holm correction from a wide paired table.

    Returns `(p_values, average_ranks, n_pairs)` where:
    - `p_values` entries are `(label_a, label_b, p_raw, significant_holm)`.
    - `average_ranks` uses rank 1 as best and is sorted descending for CD plotting.
    """
    labels = _unique_labels(target_labels)
    if len(labels) < 2:
        raise RuntimeError("Need at least two labels for Wilcoxon/Holm analysis.")

    score_matrix = _complete_score_matrix(
        metric_wide_complete,
        labels,
        higher_better=higher_better,
    )

    friedman_reject = True
    if len(labels) >= 3:
        friedman_p_value = float(
            friedmanchisquare(
                *(score_matrix[label].to_numpy(dtype=np.float64) for label in labels)
            )[1]
        )
        friedman_reject = friedman_p_value < alpha

    raw_p_values = _raw_pairwise_p_values(score_matrix, labels)
    adjusted_p_values = (
        holm_adjust_p_values(raw_p_values)
        if friedman_reject
        else pd.Series(index=raw_p_values, dtype=float)
    )
    p_values: list[PairwiseResult] = [
        (
            label_a,
            label_b,
            float(p_value),
            pd.notna(adjusted_p_values.loc[(label_a, label_b)])
            and adjusted_p_values.loc[(label_a, label_b)] <= alpha,
        )
        for (label_a, label_b), p_value in raw_p_values.items()
    ]

    average_ranks = (
        score_matrix.rank(axis=1, method="average", ascending=False)
        .mean(axis=0)
        .sort_values(ascending=False)
    )
    return p_values, average_ranks, float(score_matrix.shape[0])


def graph_ranks(
    avranks: Sequence[float],
    names: Sequence[str],
    p_values: Sequence[PairwiseResult],
    *,
    reverse: bool = True,
    width: float = 9.0,
    labels: bool = False,
):
    """
    Draw a critical-difference style rank diagram.

    Black horizontal bars encode maximal groups of models that are not
    significantly different after Holm correction.
    """
    if len(avranks) != len(names):
        raise RuntimeError("avranks and names must have matching lengths.")
    if len(avranks) == 0:
        raise RuntimeError("Need at least one rank to draw the diagram.")

    import matplotlib.pyplot as plt

    rank_values = np.asarray(avranks, dtype=np.float64)
    rank_names = np.asarray(names, dtype=object)

    lowv = float(min(1, int(np.floor(rank_values.min()))))
    highv = float(max(len(rank_values), int(np.ceil(rank_values.max()))))

    width = float(width)
    n_methods = len(rank_values)

    side_pad = 0.14
    label_pad = 0.06
    x_axis_min = side_pad
    x_axis_max = 1.0 - side_pad
    x_left_anchor = label_pad
    x_right_anchor = 1.0 - label_pad
    rank_line_y = 0.74
    left_step = 0.08
    right_step = 0.08
    label_fontsize = max(11, int(19 - 0.9 * max(n_methods - 4, 0)))
    tick_fontsize = max(11, label_fontsize - 1)

    def rank_to_x(rank: float) -> float:
        span = max(highv - lowv, 1.0)
        frac = (rank - lowv) / span
        if reverse:
            frac = 1.0 - frac
        return x_axis_min + frac * (x_axis_max - x_axis_min)

    rank_by_name = {
        str(name): float(rank) for name, rank in zip(rank_names, rank_values)
    }
    order_by_x = np.argsort(
        np.array([rank_to_x(float(rank)) for rank in rank_values], dtype=np.float64)
    )
    left_count = int(np.ceil(n_methods / 2))
    left_idx = order_by_x[:left_count]
    right_idx = order_by_x[left_count:]

    rank_intervals = _non_significant_rank_intervals(rank_by_name, p_values)
    n_intervals = len(rank_intervals)
    fig_height = max(4.8, 2.6 + 0.34 * n_methods + 0.12 * min(n_intervals, 8))
    fig, ax = plt.subplots(figsize=(width, fig_height), dpi=400)
    ax.set_axis_off()

    ax.plot(
        [x_axis_min, x_axis_max],
        [rank_line_y, rank_line_y],
        color="black",
        linewidth=2.0,
    )
    tick_values = list(np.arange(lowv, highv, 0.5)) + [highv]
    for tick in tick_values:
        x = rank_to_x(float(tick))
        is_integer = float(tick).is_integer()
        tick_len = 0.030 if is_integer else 0.020
        ax.plot(
            [x, x], [rank_line_y, rank_line_y + tick_len], color="black", linewidth=1.2
        )
        if is_integer:
            ax.text(
                x,
                rank_line_y + tick_len + 0.016,
                str(int(tick)),
                ha="center",
                va="bottom",
                fontsize=tick_fontsize,
            )

    label_y_start = rank_line_y - 0.18
    for side_idx, anchor, text_dx, ha, step in (
        (left_idx, x_left_anchor, -0.012, "right", left_step),
        (right_idx, x_right_anchor, 0.012, "left", right_step),
    ):
        for idx, model_idx in enumerate(side_idx):
            name = str(rank_names[model_idx])
            rank = float(rank_values[model_idx])
            x_rank = rank_to_x(rank)
            y = label_y_start - idx * step
            horizontal = [anchor, x_rank] if ha == "right" else [x_rank, anchor]
            ax.plot([x_rank, x_rank], [rank_line_y, y], color="#222222", linewidth=1.6)
            ax.plot(horizontal, [y, y], color="#222222", linewidth=1.6)
            ax.text(
                anchor + text_dx,
                y,
                name,
                ha=ha,
                va="center",
                fontsize=label_fontsize,
            )
            if labels:
                ax.text(
                    x_rank,
                    y - 0.018,
                    f"{rank:.3f}",
                    ha="center",
                    va="top",
                    fontsize=9,
                    color="#444444",
                )

    interval_tuples: list[tuple[float, float]] = []
    for rank_lo, rank_hi in rank_intervals:
        x_values = [rank_to_x(rank_lo), rank_to_x(rank_hi)]
        x_lo, x_hi = min(x_values), max(x_values)
        # Skip degenerate bars when tied ranks map to the same x-coordinate.
        if (x_hi - x_lo) <= 1e-8:
            continue
        interval_tuples.append((x_lo, x_hi))

    interval_tuples = sorted(
        set(interval_tuples),
        key=lambda item: (item[0], -(item[1] - item[0])),
    )

    base_interval_y = rank_line_y - 0.055
    lane_gap = 0.034
    interval_pad = 0.004
    for x_lo, x_hi, lane in _pack_intervals_into_lanes(
        interval_tuples,
        interval_pad=interval_pad,
    ):
        y = base_interval_y - lane * lane_gap
        ax.plot(
            [x_lo, x_hi], [y, y], color="black", linewidth=5.0, solid_capstyle="butt"
        )

    lowest_label_y = min(
        label_y_start - max(len(left_idx) - 1, 0) * left_step,
        (
            label_y_start - max(len(right_idx) - 1, 0) * right_step
            if len(right_idx)
            else label_y_start
        ),
    )
    y_min = lowest_label_y - 0.09
    y_max = rank_line_y + 0.12
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(y_min, y_max)
    return fig, ax
