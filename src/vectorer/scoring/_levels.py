"""Level-assignment and m/u level helpers (vectorized NumPy).

These are the pure, stateless building blocks the WeightTable and scorer
consume: content equality, the sigmoid, highest-priority level assignment per
pair, per-level ``log(m/u)`` bayes factors, and the m/u level defaults /
proportions used by EM training.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..comparisons import ComparisonSpec, PairValues
from ._constants import _LOG_CLIP


def _values_equal(a: Any, b: Any) -> bool:
    """Content equality for record values (scalars, lists, tuples, arrays).

    ``None`` equals only ``None``; lists/tuples compare element-wise; arrays
    compare with ``array_equal``; everything else uses ``==``.
    """
    if a is b:
        return True
    if a is None or b is None:
        return False
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        if not (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)):
            return False
        return a.shape == b.shape and bool(np.array_equal(a, b))
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        if not (isinstance(a, (list, tuple)) and isinstance(b, (list, tuple))):
            return False
        return len(a) == len(b) and all(_values_equal(x, y) for x, y in zip(a, b))
    try:
        return bool(a == b)
    except Exception:
        return False


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def _assign_levels(spec: ComparisonSpec, pv: PairValues) -> np.ndarray:
    """Highest-priority level index per pair (CASE semantics, vectorized).

    The spec's optional ``prescore`` step computes the shared score arrays once
    per batch; every level's test then reads them from the cache.
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