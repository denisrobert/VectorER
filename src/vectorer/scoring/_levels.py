"""Level-assignment and m/u level helpers (vectorized NumPy).

These are the pure, stateless building blocks the WeightTable and scorer
consume: highest-priority level assignment per pair, per-level ``log(m/u)``
bayes factors, and the m/u level defaults / proportions used by EM training.
(The content-equality and sigmoid primitives live in :mod:`._math`.)
"""

from __future__ import annotations

import numpy as np

from ..comparisons import ComparisonSpec, PairValues
from ._constants import _LOG_CLIP


def _assign_levels(spec: ComparisonSpec, pv: PairValues) -> np.ndarray:
    """Highest-priority level index per pair (CASE semantics, vectorized).

    The spec's optional ``prescore`` step computes the shared score arrays once
    per batch; every level's test then reads them from the cache.

    **Required level-list contract** (violating it silently mis-labels
    unmatched pairs):

    * Levels must be ordered by **decreasing agreement** (best match first);
      the first level whose ``test`` returns true owns the pair ("CASE"
      priority).
    * The **final level must be the ELSE fallback** -- a level with
      ``test=None`` (see ``comparisons._else_level``).  Every pair that fails
      all prior tests is assigned this last index.  If the last level carried a
      real test, pairs matching *nothing* would be assigned the same index as
      pairs matching that test, conflating the two.
    """
    n = pv.n
    cache = spec.prescore(pv) if spec.prescore is not None else {}
    assigned = np.full(n, -1, dtype=np.int64)
    for index, level in enumerate(spec.levels):
        if level.test is None:
            continue  # ELSE level: fallback
        need = (assigned == -1) & level.test(pv, cache)
        assigned[need] = index
    assigned[assigned == -1] = len(spec.levels) - 1
    return assigned


def _level_log_bayes_factors(spec: ComparisonSpec) -> np.ndarray:
    """Per-level ``log(m/u)`` (null level -> 0), clipped for stability."""
    out = []
    for level in spec.levels:
        if level.is_null or level.m is None:
            out.append(0.0)
        elif level.m <= 0:
            out.append(-_LOG_CLIP)
        elif level.u is None or level.u <= 0:
            out.append(_LOG_CLIP)
        else:
            log_bf = np.log(level.m) - np.log(level.u)
            out.append(float(np.clip(log_bf, -_LOG_CLIP, _LOG_CLIP)))
    return np.asarray(out, dtype=np.float64)


def _level_proportions(
    spec: ComparisonSpec, assigned: np.ndarray
) -> list[float]:
    """Relative frequency of each level over ``assigned`` (u estimate)."""
    counts = np.bincount(assigned, minlength=len(spec.levels)).astype(np.float64)
    total = max(float(counts.sum()), 1e-12)
    return np.clip(counts / total, 1e-8, None).tolist()


def _level_defaults(spec: ComparisonSpec) -> list[float]:
    return [lv.m if (lv.m is not None and not lv.is_null) else 1e-8 for lv in spec.levels]