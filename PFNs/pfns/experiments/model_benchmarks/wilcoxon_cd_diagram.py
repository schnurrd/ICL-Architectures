from __future__ import annotations

"""
Wilcoxon/Holm and critical-difference style plotting utilities.

Inspired by: https://github.com/hfawaz/cd-diagram (GPL-3.0).
This module is an original PFNs implementation.
"""

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon


PairwiseResult = tuple[str, str, float, bool]


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
        pivot = max(pivot_candidates, key=lambda n: len(adjacency[n])) if pivot_candidates else None
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


def wilcoxon_holm_from_wide(
    *,
    metric_wide_complete: pd.DataFrame,
    target_labels: Iterable[str],
    higher_better: bool,
    alpha: float = 0.05,
) -> tuple[list[PairwiseResult], pd.Series, float]:
    """
    Pairwise Wilcoxon tests with Holm correction from a wide paired table.

    Returns `(p_values, average_ranks, n_pairs)` where:
    - `p_values` entries are `(label_a, label_b, p_raw, significant_holm)`.
    - `average_ranks` uses rank 1 as best and is sorted descending for CD plotting.
    """
    labels = list(dict.fromkeys(target_labels))
    if len(labels) < 2:
        raise RuntimeError("Need at least two labels for Wilcoxon/Holm analysis.")

    missing = [label for label in labels if label not in metric_wide_complete.columns]
    if missing:
        raise RuntimeError(f"Missing target labels in metric_wide_complete: {missing}")

    score_matrix = metric_wide_complete.reindex(columns=labels).dropna(subset=labels)
    if score_matrix.empty:
        raise RuntimeError("No complete paired rows available for Wilcoxon/Holm analysis.")

    score_matrix = score_matrix.astype(float)
    if not higher_better:
        score_matrix = -score_matrix
    score_matrix = score_matrix.reset_index(drop=True)

    friedman_reject = True
    if len(labels) >= 3:
        friedman_p_value = float(
            friedmanchisquare(*(score_matrix[label].to_numpy(dtype=np.float64) for label in labels))[1]
        )
        friedman_reject = friedman_p_value < alpha

    p_values: list[PairwiseResult] = []
    for i, label_a in enumerate(labels[:-1]):
        perf_a = score_matrix[label_a].to_numpy(dtype=np.float64)
        for label_b in labels[i + 1 :]:
            perf_b = score_matrix[label_b].to_numpy(dtype=np.float64)
            try:
                p_value = float(wilcoxon(perf_a, perf_b, zero_method="pratt")[1])
            except ValueError:
                # All differences are exactly zero.
                p_value = 1.0
            p_values.append((label_a, label_b, p_value, False))

    if friedman_reject and p_values:
        sorted_idx = sorted(range(len(p_values)), key=lambda idx: p_values[idx][2])
        for rank, idx in enumerate(sorted_idx):
            threshold = float(alpha / (len(p_values) - rank))
            if p_values[idx][2] <= threshold:
                label_a, label_b, p_raw, _ = p_values[idx]
                p_values[idx] = (label_a, label_b, p_raw, True)
            else:
                break

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

    rank_by_name = {str(name): float(rank) for name, rank in zip(rank_names, rank_values)}
    order_by_x = np.argsort(np.array([rank_to_x(float(rank)) for rank in rank_values], dtype=np.float64))
    left_count = int(np.ceil(n_methods / 2))
    left_idx = order_by_x[:left_count]
    right_idx = order_by_x[left_count:]

    non_sig_adj: dict[str, set[str]] = {name: set() for name in rank_by_name}
    for name_a, name_b, _p_raw, significant in p_values:
        if significant or name_a not in rank_by_name or name_b not in rank_by_name:
            continue
        non_sig_adj[name_a].add(name_b)
        non_sig_adj[name_b].add(name_a)

    non_sig_cliques = _find_maximal_cliques(list(rank_by_name), non_sig_adj)
    n_intervals = len(non_sig_cliques)
    fig_height = max(4.8, 2.6 + 0.34 * n_methods + 0.12 * min(n_intervals, 8))
    fig, ax = plt.subplots(figsize=(width, fig_height), dpi=130)
    ax.set_axis_off()

    ax.plot([x_axis_min, x_axis_max], [rank_line_y, rank_line_y], color="black", linewidth=2.0)
    tick_values = list(np.arange(lowv, highv, 0.5)) + [highv]
    for tick in tick_values:
        x = rank_to_x(float(tick))
        is_integer = float(tick).is_integer()
        tick_len = 0.030 if is_integer else 0.020
        ax.plot([x, x], [rank_line_y, rank_line_y + tick_len], color="black", linewidth=1.2)
        if is_integer:
            ax.text(
                x,
                rank_line_y + tick_len + 0.016,
                str(int(tick)),
                ha="center",
                va="bottom",
                fontsize=tick_fontsize,
            )

    left_y_start = rank_line_y - 0.18
    right_y_start = rank_line_y - 0.18
    for idx, model_idx in enumerate(left_idx):
        name = str(rank_names[model_idx])
        rank = float(rank_values[model_idx])
        x_rank = rank_to_x(rank)
        y = left_y_start - idx * left_step
        ax.plot([x_rank, x_rank], [rank_line_y, y], color="#222222", linewidth=1.6)
        ax.plot([x_left_anchor, x_rank], [y, y], color="#222222", linewidth=1.6)
        ax.text(x_left_anchor - 0.012, y, name, ha="right", va="center", fontsize=label_fontsize)
        if labels:
            ax.text(x_rank, y - 0.018, f"{rank:.3f}", ha="center", va="top", fontsize=9, color="#444444")

    for idx, model_idx in enumerate(right_idx):
        name = str(rank_names[model_idx])
        rank = float(rank_values[model_idx])
        x_rank = rank_to_x(rank)
        y = right_y_start - idx * right_step
        ax.plot([x_rank, x_rank], [rank_line_y, y], color="#222222", linewidth=1.6)
        ax.plot([x_rank, x_right_anchor], [y, y], color="#222222", linewidth=1.6)
        ax.text(x_right_anchor + 0.012, y, name, ha="left", va="center", fontsize=label_fontsize)
        if labels:
            ax.text(x_rank, y - 0.018, f"{rank:.3f}", ha="center", va="top", fontsize=9, color="#444444")

    interval_tuples: list[tuple[float, float]] = []
    for clique in non_sig_cliques:
        x_values = [rank_to_x(rank_by_name[name]) for name in clique]
        x_lo, x_hi = min(x_values), max(x_values)
        # Skip degenerate bars when tied ranks map to the same x-coordinate.
        if (x_hi - x_lo) <= 1e-8:
            continue
        interval_tuples.append((x_lo, x_hi))

    interval_tuples = sorted(
        set(interval_tuples),
        key=lambda item: (item[0], -(item[1] - item[0])),
    )

    lane_right_edges: list[float] = []
    base_interval_y = rank_line_y - 0.055
    lane_gap = 0.034
    interval_pad = 0.004
    for x_lo, x_hi in interval_tuples:
        lane = 0
        while lane < len(lane_right_edges) and x_lo <= lane_right_edges[lane] + interval_pad:
            lane += 1
        if lane == len(lane_right_edges):
            lane_right_edges.append(x_hi)
        else:
            lane_right_edges[lane] = max(lane_right_edges[lane], x_hi)
        y = base_interval_y - lane * lane_gap
        ax.plot([x_lo, x_hi], [y, y], color="black", linewidth=5.0, solid_capstyle="butt")

    lowest_label_y = min(
        left_y_start - max(len(left_idx) - 1, 0) * left_step,
        right_y_start - max(len(right_idx) - 1, 0) * right_step if len(right_idx) else left_y_start,
    )
    y_min = lowest_label_y - 0.09
    y_max = rank_line_y + 0.12
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(y_min, y_max)
    return fig, ax
