"""Exact precision/recall curve of a *fixed* scorer over a single threshold axis.

Once ``m/u`` and the prior are fixed, the decision depends on one scalar only
(the posterior threshold ``tau`` — equivalently, since they are monotone
transforms of one another, the match-weight threshold ``kappa``; see
``.docs/calibration.md`` §1).  Because the score is a function of the
comparison-level pattern it takes finitely many values on any labelled set, so
the precision/recall/F1 curve is an exact **step function** whose breakpoints
are the achieved scores.

:func:`match_weight_curve` therefore builds the whole curve in one pass
(``O(n log n)``) and returns every attainable operating point plus the exact
F1-optimal one — no ``tau``/``pi`` grid, no per-point refitting.  Use it
instead of sweeping ``(pi, tau)`` when ``m/u`` are fixed; to explore *model*
variation, retrain and rebuild the curve.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

__all__ = ["OperatingPoint", "MatchWeightCurve", "match_weight_curve"]

_LN2 = math.log(2.0)


@dataclass(frozen=True)
class OperatingPoint:
    """One attainable operating point of a fixed scorer.

    ``threshold`` is the posterior threshold ``tau`` (accept iff
    ``posterior >= threshold``); ``.match_weight_bits()`` and
    ``.pi_free_weight(prior)`` give the equivalent thresholds on the total
    match weight and on the prior-free ``W`` of ``calibration.md`` §1.
    """

    threshold: float
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int

    def match_weight_bits(self) -> float:
        """Equivalent threshold on the total match weight (bits, incl. prior)."""
        t = min(max(float(self.threshold), 1e-15), 1.0 - 1e-15)
        return (math.log(t) - math.log1p(-t)) / _LN2

    def pi_free_weight(self, prior: float) -> float:
        """Equivalent threshold on the prior-free ``W = sum log(m/u)``."""
        p = min(max(float(prior), 1e-15), 1.0 - 1e-15)
        return self.match_weight_bits() * _LN2 - (math.log(p) - math.log1p(-p))


@dataclass
class MatchWeightCurve:
    """The exact P/R/F1 curve of a fixed scorer (step function)."""

    points: list[OperatingPoint] = field(default_factory=list)
    best_f1: OperatingPoint | None = None
    n_pairs: int = 0
    n_positive: int = 0

    def best_for(self, metric: str = "f1") -> OperatingPoint:
        """Best point under ``metric`` in {"f1", "precision", "recall"}."""
        if not self.points:
            raise ValueError("empty curve")
        key = {"f1": "f1", "precision": "precision", "recall": "recall"}[metric]
        return max(self.points, key=lambda p: getattr(p, key))


def match_weight_curve(scorer, pairs: Iterable[tuple[dict, dict, object]]) -> MatchWeightCurve:
    """Exact P/R/F1 curve of ``scorer`` over labelled ``pairs``.

    ``pairs`` yields ``(left_record, right_record, label)`` with a truthy label
    for a true match.  The scorer must be **fixed** (``m/u`` and prior already
    fitted); the curve is then exact and needs no threshold grid.  Returns a
    :class:`MatchWeightCurve` whose ``points`` are the distinct attainable
    operating points (plus the accept-none point at ``tau = inf``) and whose
    ``best_f1`` is the exact F1 optimum.
    """
    lefts: list[dict] = []
    rights: list[dict] = []
    labels: list[int] = []
    for left, right, label in pairs:
        lefts.append(left)
        rights.append(right)
        labels.append(1 if label else 0)
    if not lefts:
        raise ValueError("no pairs supplied")

    y = np.asarray(labels, dtype=np.int64)
    n = y.size
    n_pos = int(y.sum())
    posterior = np.asarray(scorer.score_pairs(lefts, rights), dtype=np.float64)

    order = np.argsort(-posterior, kind="mergesort")
    p_sorted = posterior[order]
    y_sorted = y[order]

    # Accept-nothing endpoint first (highest threshold), then the distinct
    # attained scores in descending order -> ``points`` is sorted by threshold.
    points: list[OperatingPoint] = [
        OperatingPoint(float("inf"), 0.0, 0.0, 0.0, 0, 0, n_pos)
    ]
    tp = 0
    fp = 0
    i = 0
    while i < n:
        thr = p_sorted[i]
        j = i
        while j < n and p_sorted[j] == thr:
            if y_sorted[j] == 1:
                tp += 1
            else:
                fp += 1
            j += 1
        fn = n_pos - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / n_pos if n_pos else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        points.append(
            OperatingPoint(float(thr), precision, recall, f1, tp, fp, fn)
        )
        i = j

    best = max(points, key=lambda p: p.f1)
    return MatchWeightCurve(points=points, best_f1=best, n_pairs=n, n_positive=n_pos)
