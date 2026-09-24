"""Tests for the paired statistics shared by the benchmark analyses."""

from __future__ import annotations

import numpy as np
import pytest

from pfns.experiments.model_benchmarks.paired_stats import (
    bootstrap_ci,
    equivalence_test,
    holm_correct,
    paired_test,
    rank_biserial_correlation,
)


# --------------------------------------------------------------------------
# Holm correction
# --------------------------------------------------------------------------


def test_holm_matches_hand_computation() -> None:
    adjusted, rejected = holm_correct([0.001, 0.02, 0.04, 0.5])

    # m=4: 4*0.001, 3*0.02, 2*0.04, 1*0.5 with running max enforced.
    assert adjusted == pytest.approx([0.004, 0.06, 0.08, 0.5])
    assert rejected == [True, False, False, False]


def test_holm_is_monotone() -> None:
    """A less significant test can never receive a smaller adjusted p-value."""
    raw = [0.01, 0.011, 0.012, 0.013, 0.9]
    adjusted, _ = holm_correct(raw)
    finite = [a for a in adjusted if np.isfinite(a)]
    assert finite == sorted(finite)


def test_holm_preserves_input_order() -> None:
    adjusted, rejected = holm_correct([0.5, 0.001])
    assert adjusted[1] < adjusted[0]
    assert rejected == [False, True]


def test_holm_is_more_conservative_than_raw() -> None:
    raw = [0.03] * 5
    adjusted, rejected = holm_correct(raw)
    assert all(a >= 0.03 for a in adjusted)
    assert not any(rejected), "five p=0.03 tests should not all survive correction"


def test_holm_handles_nan() -> None:
    adjusted, rejected = holm_correct([0.001, float("nan")])
    assert np.isfinite(adjusted[0])
    assert np.isnan(adjusted[1])
    assert rejected == [True, False]


# --------------------------------------------------------------------------
# Effect size
# --------------------------------------------------------------------------


def test_rank_biserial_is_one_when_all_differences_positive() -> None:
    assert rank_biserial_correlation(np.array([0.1, 0.2, 0.3])) == pytest.approx(1.0)


def test_rank_biserial_is_minus_one_when_all_negative() -> None:
    assert rank_biserial_correlation(np.array([-0.1, -0.2])) == pytest.approx(-1.0)


def test_rank_biserial_is_zero_for_symmetric_differences() -> None:
    assert rank_biserial_correlation(
        np.array([0.1, -0.1, 0.2, -0.2])
    ) == pytest.approx(0.0)


def test_rank_biserial_rewards_consistency_not_magnitude() -> None:
    """A tiny shift that points the same way on every dataset is strong evidence;
    a large but inconsistent one is not. This is why rank-biserial rather than
    Cohen's d is reported alongside a signed-rank test."""
    consistent = np.full(20, 1e-4)
    inconsistent = np.array([1.0, -1.0] * 10)
    assert rank_biserial_correlation(consistent) == pytest.approx(1.0)
    assert abs(rank_biserial_correlation(inconsistent)) < 0.2


def test_rank_biserial_ignores_exact_ties() -> None:
    assert rank_biserial_correlation(
        np.array([0.0, 0.0, 0.1, 0.2])
    ) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Paired test
# --------------------------------------------------------------------------


def test_paired_test_recovers_a_known_shift() -> None:
    rng = np.random.default_rng(0)
    control = rng.normal(0.74, 0.01, 200)
    # A varying shift, as in real data: a perfectly constant one gives the
    # bootstrap zero variance and a degenerate zero-width interval.
    arm = control + rng.normal(0.02, 0.004, 200)

    result = paired_test(arm, control)

    assert result.n_pairs == 200
    assert result.mean_delta == pytest.approx(0.02, abs=0.002)
    assert result.p_value < 1e-10
    assert result.rank_biserial > 0.95
    assert result.ci_low < result.mean_delta < result.ci_high


def test_paired_test_on_identical_arms_is_not_significant() -> None:
    values = np.linspace(0.7, 0.8, 50)
    result = paired_test(values, values.copy())

    assert result.mean_delta == 0.0
    assert result.p_value == 1.0
    assert result.rank_biserial == 0.0


def test_bootstrap_ci_brackets_the_mean() -> None:
    rng = np.random.default_rng(1)
    deltas = rng.normal(0.01, 0.005, 500)
    low, high = bootstrap_ci(deltas)
    assert low < deltas.mean() < high
    assert high - low < 0.005


# --------------------------------------------------------------------------
# Equivalence (TOST)
# --------------------------------------------------------------------------


def test_tost_declares_equivalence_for_near_identical_arms() -> None:
    rng = np.random.default_rng(0)
    control = rng.normal(0.74, 0.01, 300)
    arm = control + rng.normal(0.0, 2e-4, 300)

    result = equivalence_test(arm, control, margin=0.005)

    assert result["equivalent"]
    assert result["p_tost"] < 0.05


def test_tost_refuses_equivalence_for_a_real_difference() -> None:
    rng = np.random.default_rng(0)
    control = rng.normal(0.74, 0.01, 300)
    arm = control + 0.02

    result = equivalence_test(arm, control, margin=0.005)

    assert not result["equivalent"]


def test_tost_is_not_merely_a_non_significant_p_value() -> None:
    """The distinction the analysis depends on: with few, noisy samples the
    difference test fails to reject *and* equivalence is not established. A
    plain p>0.05 would have been misreported as 'no difference'."""
    rng = np.random.default_rng(3)
    control = rng.normal(0.74, 0.05, 6)
    arm = control + rng.normal(0.01, 0.05, 6)

    difference = paired_test(arm, control)
    equivalence = equivalence_test(arm, control, margin=0.005)

    assert difference.p_value > 0.05
    assert not equivalence["equivalent"]


def test_tost_margin_controls_the_conclusion() -> None:
    rng = np.random.default_rng(0)
    control = rng.normal(0.74, 0.005, 300)
    arm = control + 0.004

    assert not equivalence_test(arm, control, margin=0.002)["equivalent"]
    assert equivalence_test(arm, control, margin=0.01)["equivalent"]
