"""Paired statistics shared by the benchmark analyses.

Model arms are evaluated on the same fixed datasets, so comparisons between
them are paired and use the paired Wilcoxon signed-rank test -- the same test
the sequence-length and real-world tables use, with the same Holm correction
for multiplicity.

Two things beyond a plain significance test are needed to support the claims
these analyses make, and each has its own function:

`holm_correct`
    Many arms times many lengths is a family of tests. Uncorrected p-values
    would overstate the evidence.

`equivalence_test`
    "Indistinguishable from the control" is a claim of *absence*, and a
    non-significant p-value does not establish it. TOST does, against an
    explicit margin.

"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata, wilcoxon

ALPHA = 0.05


@dataclass(frozen=True)
class PairedTest:
    """A paired comparison of two arms over the same datasets."""

    n_pairs: int
    mean_delta: float
    median_delta: float
    ci_low: float
    ci_high: float
    p_value: float
    rank_biserial: float
    """Matched-pairs rank-biserial correlation in [-1, 1].

    The effect size that belongs with a signed-rank test: the normalised
    difference between the summed positive and negative ranks. |r| near 1 means
    the sign of the difference is nearly consistent across datasets, which is
    the relevant notion here -- a small mean shift that points the same way on
    every dataset is strong evidence, and Cohen's d would not say so.
    """

    def as_record(self) -> dict[str, float | int]:
        return {
            "n_pairs": self.n_pairs,
            "mean_delta": self.mean_delta,
            "median_delta": self.median_delta,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "p_value": self.p_value,
            "rank_biserial": self.rank_biserial,
        }


def bootstrap_ci(
    deltas: np.ndarray,
    *,
    num_resamples: int = 10_000,
    seed: int = 0,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean paired difference."""
    if deltas.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(deltas, size=(num_resamples, deltas.size), replace=True).mean(
        axis=1
    )
    tail = (1.0 - confidence) / 2.0 * 100.0
    return float(np.percentile(means, tail)), float(np.percentile(means, 100.0 - tail))


def rank_biserial_correlation(deltas: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation of the non-zero differences.

    Ties in |delta| take average ranks, as the signed-rank test itself does.
    Breaking them arbitrarily instead would make perfectly symmetric
    differences report a spurious non-zero effect.
    """
    nonzero = np.asarray(deltas, dtype=float)
    nonzero = nonzero[nonzero != 0]
    if nonzero.size == 0:
        return 0.0
    ranks = rankdata(np.abs(nonzero))
    total = ranks.sum()
    if total == 0:
        return 0.0
    positive = ranks[nonzero > 0].sum()
    negative = ranks[nonzero < 0].sum()
    return float((positive - negative) / total)


def paired_test(
    arm_values: np.ndarray,
    control_values: np.ndarray,
    *,
    seed: int = 0,
) -> PairedTest:
    """Paired Wilcoxon test of ``arm - control`` with CI and effect size."""
    deltas = np.asarray(arm_values, dtype=float) - np.asarray(
        control_values, dtype=float
    )
    ci_low, ci_high = bootstrap_ci(deltas, seed=seed)
    if deltas.size < 2 or np.allclose(deltas, 0.0):
        # `wilcoxon` raises on an all-zero difference vector. That is a genuine
        # outcome -- the arms agree exactly -- not an error.
        p_value = 1.0
    else:
        p_value = float(wilcoxon(deltas).pvalue)
    return PairedTest(
        n_pairs=int(deltas.size),
        mean_delta=float(deltas.mean()) if deltas.size else float("nan"),
        median_delta=float(np.median(deltas)) if deltas.size else float("nan"),
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_value,
        rank_biserial=rank_biserial_correlation(deltas),
    )


def holm_correct(p_values: list[float], *, alpha: float = ALPHA) -> tuple[
    list[float], list[bool]
]:
    """Holm-Bonferroni step-down correction.

    Returns the adjusted p-values (in the input order) and whether each is
    rejected at ``alpha``. Adjusted values are made monotone so that a test can
    never end up with a smaller adjusted p-value than a more significant one.
    """
    finite = [(i, p) for i, p in enumerate(p_values) if np.isfinite(p)]
    adjusted = [float("nan")] * len(p_values)
    if not finite:
        return adjusted, [False] * len(p_values)

    order = sorted(finite, key=lambda item: item[1])
    m = len(order)
    running = 0.0
    for rank, (index, p) in enumerate(order):
        scaled = min(1.0, (m - rank) * p)
        running = max(running, scaled)
        adjusted[index] = running
    return adjusted, [
        bool(np.isfinite(a) and a < alpha) for a in adjusted
    ]


def equivalence_test(
    arm_values: np.ndarray,
    control_values: np.ndarray,
    *,
    margin: float,
) -> dict[str, float | bool]:
    """TOST: are the two arms equivalent to within +/- ``margin``?

    A non-significant difference only means the data failed to resolve one. To
    assert that an arm is *the same as* the truncation ceiling -- the claim that
    a "correction" achieved nothing beyond discarding the context -- both
    one-sided tests have to reject, which is what this returns as
    ``equivalent``.

    Args:
        margin: Largest difference still considered practically irrelevant, in
            the metric's own units. Choose it before seeing the data; the
            smallest effect the paper would care about is a defensible choice.
    """
    deltas = np.asarray(arm_values, dtype=float) - np.asarray(
        control_values, dtype=float
    )
    if deltas.size < 2:
        return {
            "margin": margin,
            "p_lower": float("nan"),
            "p_upper": float("nan"),
            "p_tost": float("nan"),
            "equivalent": False,
        }

    # Wilcoxon on the shifted differences gives each one-sided test.
    def one_sided(shifted: np.ndarray, alternative: str) -> float:
        if np.allclose(shifted, 0.0):
            return 1.0
        return float(wilcoxon(shifted, alternative=alternative).pvalue)

    p_lower = one_sided(deltas + margin, "greater")  # H1: delta > -margin
    p_upper = one_sided(deltas - margin, "less")  # H1: delta <  margin
    p_tost = max(p_lower, p_upper)
    return {
        "margin": float(margin),
        "p_lower": p_lower,
        "p_upper": p_upper,
        "p_tost": p_tost,
        "equivalent": bool(p_tost < ALPHA),
    }
