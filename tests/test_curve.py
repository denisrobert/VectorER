"""Tests for the exact fixed-scorer P/R curve (:mod:`vectorer.scoring._curve`)."""

import numpy as np
import pytest

from vectorer import MatchWeightCurve, OperatingPoint, match_weight_curve
from vectorer.comparisons import make_comparison
from vectorer.scoring import FellegiSunterScorer


def _scorer(prior=0.01):
    return FellegiSunterScorer.from_comparisons(
        [make_comparison("jaro_winkler_at_thresholds", col_name="fn")],
        prior=prior,
    )


def _dataset():
    pairs = []
    for i in range(7):            # positives: identical names
        pairs.append(({"fn": f"name{i}"}, {"fn": f"name{i}"}, 1))
    for i in range(11):           # negatives: dissimilar names
        pairs.append(({"fn": f"name{i}"}, {"fn": f"other{i}"}, 0))
    return pairs


def test_curve_is_exact_against_brute_force():
    scorer = _scorer()
    pairs = _dataset()
    curve = match_weight_curve(scorer, pairs)

    lefts = [p[0] for p in pairs]
    rights = [p[1] for p in pairs]
    y = np.array([1 if p[2] else 0 for p in pairs])
    post = np.asarray(scorer.score_pairs(lefts, rights), dtype=float)

    best = 0.0
    for thr in np.unique(post):
        pred = post >= thr
        tp = int((pred & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum())
        fn = int((~pred & (y == 1)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        best = max(best, f1)

    assert curve.best_f1.f1 == pytest.approx(best, abs=1e-12)
    assert isinstance(curve, MatchWeightCurve)
    assert isinstance(curve.best_f1, OperatingPoint)
    assert curve.n_pairs == len(pairs)
    assert curve.n_positive == 7


def test_breakpoints_are_distinct_scores_plus_accept_none():
    scorer = _scorer()
    pairs = _dataset()
    curve = match_weight_curve(scorer, pairs)
    post = np.asarray(
        scorer.score_pairs([p[0] for p in pairs], [p[1] for p in pairs]), dtype=float
    )
    distinct = set(np.unique(post).tolist())
    thresholds = [pt.threshold for pt in curve.points]
    assert len(thresholds) == len(set(thresholds))          # no duplicates
    assert float("inf") in thresholds                       # accept-none endpoint
    assert set(t for t in thresholds if t != float("inf")) == distinct


def test_recall_monotone_as_threshold_decreases():
    curve = match_weight_curve(_scorer(), _dataset())
    # points are ordered by descending threshold: recall is nondecreasing.
    recalls = [pt.recall for pt in curve.points]
    assert all(recalls[i] >= recalls[i - 1] - 1e-12 for i in range(1, len(recalls)))
    thresholds = [pt.threshold for pt in curve.points]
    assert all(thresholds[i] <= thresholds[i - 1] for i in range(1, len(thresholds)))


def test_threshold_coordinate_helpers_are_consistent():
    op = OperatingPoint(threshold=0.8, precision=0.9, recall=0.7, f1=0.79,
                        tp=7, fp=1, fn=3)
    import math
    # total match weight (bits) == logit(tau)/ln2
    assert op.match_weight_bits() == pytest.approx(
        (math.log(0.8) - math.log(0.2)) / math.log(2.0), rel=1e-12
    )
    # pi-free kappa == logit(tau) - logit(prior)
    prior = 1e-4
    expected = (math.log(0.8) - math.log(0.2)) - (math.log(prior) - math.log(1 - prior))
    assert op.pi_free_weight(prior) == pytest.approx(expected, rel=1e-12)


def test_empty_pairs_raises_and_no_positives_is_zero():
    scorer = _scorer()
    with pytest.raises(ValueError, match="no pairs"):
        match_weight_curve(scorer, [])
    pairs = [({"fn": "a"}, {"fn": "b"}, 0), ({"fn": "c"}, {"fn": "d"}, 0)]
    curve = match_weight_curve(scorer, pairs)
    assert curve.best_f1.f1 == 0.0
    assert curve.n_positive == 0


def test_best_for_metric_selects_by_metric():
    curve = match_weight_curve(_scorer(), _dataset())
    assert curve.best_for("f1").f1 == curve.best_f1.f1
    assert curve.best_for("precision").precision >= curve.best_f1.precision
    assert curve.best_for("recall").recall >= curve.best_f1.recall
