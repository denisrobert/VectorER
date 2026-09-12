"""Small numeric/equality primitives shared across the scoring package.

These are stateless helpers that the level-assignment, inference and training
modules all consume: the numerically-stable sigmoid and content equality for
record values.
"""

from __future__ import annotations

from typing import Any

import numpy as np


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