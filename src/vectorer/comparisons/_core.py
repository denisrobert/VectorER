"""Core comparison building blocks: pairs plumbing, levels, spec, default m/u.

Everything here is shared by every comparison family.  It defines the
:class:`PairValues` batch wrapper, the :class:`Level` and :class:`ComparisonSpec`
shapes, the default ``m/u`` scheme, and the small toolkit of level/prescore
builders the family modules compose (``build_spec``, ``_else_level``,
``_null_test``, ``_exact_test``, ``_cached_threshold_test``,
``_cached_epoch_diff_test``, ``_and_test``, ``_jw_prescore``,
``_distance_prescore``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

import numpy as np

from .. import sim


# ---------------------------------------------------------------------------
# Value plumbing shared by level predicates
# ---------------------------------------------------------------------------


class PairValues:
    """All pairs' values for a set of fields, exposed as object arrays.

    ``left(field)`` / ``right(field)`` return the length-N object arrays of the
    ``_l`` / ``_r`` values of ``field`` (elements may be ``None`` for missing).
    """

    __slots__ = ("_left", "_right", "n")

    def __init__(self, left: dict, right: dict) -> None:
        self._left = left
        self._right = right
        self.n = len(next(iter(left.values()))) if left else 0

    def left(self, field: str) -> np.ndarray:
        return self._left[field]

    def right(self, field: str) -> np.ndarray:
        return self._right[field]

    def __len__(self) -> int:
        return self.n


def _isnan(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.fromiter(
        (l is None or r is None for l, r in zip(left, right)),
        dtype=bool,
        count=len(left),
    )


# ---------------------------------------------------------------------------
# Levels and comparison specs
# ---------------------------------------------------------------------------

Cache = Optional[dict]


@dataclass
class Level:
    """One Fellegi-Sunter comparison level.

    ``test`` is a vectorized predicate ``(PairValues, cache) -> bool mask``
    (``None`` marks an *ELSE* catch-all).  ``is_null`` flags the always-BF-1
    null level.  Optional term-frequency adjustment applies when ``tf_column``
    is set (``u`` divided by ``max(tf_l, tf_r) ** tf_weight``, ``tf_min_u``
    floor), matching the standard term-frequency scheme.
    """

    label: str
    test: Optional[Callable[[PairValues, Cache], np.ndarray]] = None
    is_null: bool = False
    m: Optional[float] = None
    u: Optional[float] = None
    tf_column: Optional[str] = None
    tf_weight: float = 1.0
    tf_min_u: float = 0.0
    cvv: int = -1

    def prob_dict(self) -> dict:
        out = {"label_for_charts": self.label, "is_null_level": self.is_null}
        if not self.is_null:
            out["m_probability"] = self.m
            out["u_probability"] = self.u
        if self.tf_column is not None:
            out["tf_adjustment_column"] = self.tf_column
            out["tf_adjustment_weight"] = self.tf_weight
            out["tf_minimum_u_value"] = self.tf_min_u
        return out


@dataclass
class ComparisonSpec:
    """A built comparison: ordered levels + the columns it compares.

    ``prescore``, when set, computes the score arrays shared by the levels in
    one vectorized pass over the batch.  Level m/u defaults are assigned at
    build time using the standard default-value algorithm.

    **Required level ordering** (enforced by :func:`~vectorer.scoring.
    _levels._assign_levels` at scoring time -- a contract, not a convention):

    * Levels are ordered by **decreasing agreement** (null level first, then
      exact match, then progressively looser fuzzy levels, then a final
      fallback).  The first level whose ``test`` is true owns the pair.
    * The **last level must be the ELSE fallback** (``test=None``, the "All
      other comparisons" level).  Pairs matching no earlier level are assigned
      that final index.  If the last level instead had a real ``test``,
      unmatched pairs would be indistinguishable from pairs matching that
      level -- a silent mis-labelling of every unclassified pair.
    """

    output_column_name: str
    levels: list[Level]
    fields: tuple[str, ...] = ()
    prescore: Optional[Callable[[PairValues], dict]] = None

    def with_probabilities(self, level_probs: Sequence[tuple[Optional[float], Optional[float]]]) -> "ComparisonSpec":
        """Return a copy with overridden per-level ``(m, u)`` (index-aligned)."""
        new_levels = []
        for level, (m, u) in zip(self.levels, level_probs):
            new_levels.append(
                Level(
                    label=level.label,
                    test=level.test,
                    is_null=level.is_null,
                    m=level.m if m is None else m,
                    u=level.u if u is None else u,
                    tf_column=level.tf_column,
                    tf_weight=level.tf_weight,
                    tf_min_u=level.tf_min_u,
                    cvv=level.cvv,
                )
            )
        return ComparisonSpec(
            output_column_name=self.output_column_name,
            levels=new_levels,
            fields=self.fields,
            prescore=self.prescore,
        )


# ---------------------------------------------------------------------------
# Standard default m/u scheme
# ---------------------------------------------------------------------------


def _default_m_values(num_levels: int) -> list[float]:
    """Default m per non-null level, in level order (exact first -> ELSE last).

    The exact match level (the first non-null level) holds ``0.95``; the
    ``num_levels - 1`` remaining levels (fuzzy thresholds + the ELSE fallback)
    split ``0.05`` evenly.  ``apply_default_mu`` consumes this list in level
    order (index 0 = exact), which is why ``0.95`` is the *first* element.
    """
    split_remainder = 0.05 / (num_levels - 1)
    return [0.95] + [split_remainder] * (num_levels - 1)


def _default_u_values(num_levels: int) -> list[float]:
    """Default u per non-null level, in level order (exact first -> ELSE last).

    ``u`` is smallest for the exact level (a rare value agreeing is the
    strongest evidence) and largest for the ELSE fallback.  Weights step from
    the exact-match weight (+10) down to a strongly non-matching level (-5),
    so ``u = m / 2**weight`` decreases in level order.
    """
    m_vals = _default_m_values(num_levels)
    if num_levels == 2:
        match_weights = [10.0, -5.0]
    else:
        match_weights = [10.0] + list(np.linspace(3.0, -5.0, num_levels - 1))
    return [m / (2 ** w) for m, w in zip(m_vals, match_weights)]


def apply_default_mu(spec: ComparisonSpec) -> None:
    """Assign the standard default m/u to levels lacking explicit probabilities."""
    non_null = [lv for lv in spec.levels if not lv.is_null]
    if not non_null or len(spec.levels) <= 1:
        return
    k = len(non_null)
    ms = _default_m_values(k)
    us = _default_u_values(k)
    # ``slice_idx`` walks the default lists in level order (index 0 = exact,
    # last = ELSE), while ``cvv`` is the *reverse* ordinal -- the exact level is
    # the highest-value version (cvv = k-1), the ELSE catch-all the lowest
    # (cvv = 0).  The lists are naturally ordered (exact = 0.95 first) but
    # ``cvv`` preserves the historical descending convention.
    slice_idx = 0
    cvv = k - 1
    for lv in spec.levels:
        if lv.is_null:
            lv.cvv = -1
            continue
        lv.cvv = cvv
        if lv.m is None:
            lv.m = ms[slice_idx]
        if lv.u is None:
            lv.u = us[slice_idx]
        slice_idx += 1
        cvv -= 1


# ---------------------------------------------------------------------------
# Level-building helpers
# ---------------------------------------------------------------------------


def _else_level(label: str = "All other comparisons") -> dict:
    return {"label_for_charts": label}


def build_spec(
    output_column_name: str,
    level_dicts: Sequence[dict],
    fields: Sequence[str] = (),
    prescore: Optional[Callable[[PairValues], dict]] = None,
) -> ComparisonSpec:
    """Assemble a ComparisonSpec from ordered level dicts and apply defaults.

    ``level_dicts`` must be ordered by decreasing agreement and its **last
    entry must be the ELSE fallback** (``"test": None`` -- use
    :func:`_else_level`); the fallback receives every pair that matches no
    earlier level.  See :class:`ComparisonSpec` for the full ordering
    contract.
    """
    levels = [
        Level(
            label=spec.get("label_for_charts", ""),
            test=spec.get("test"),
            is_null=bool(spec.get("is_null_level", False)),
            m=spec.get("m_probability"),
            u=spec.get("u_probability"),
            tf_column=spec.get("tf_adjustment_column"),
            tf_weight=spec.get("tf_adjustment_weight", 1.0),
            tf_min_u=spec.get("tf_minimum_u_value", 0.0),
        )
        for spec in level_dicts
    ]
    spec = ComparisonSpec(
        output_column_name=output_column_name,
        levels=levels,
        fields=tuple(fields),
        prescore=prescore,
    )
    apply_default_mu(spec)
    return spec


def _exact_test(field: str) -> Callable[[PairValues, Cache], np.ndarray]:
    return lambda pv, cache=None: sim.exact_equals(pv.left(field), pv.right(field))


def _null_test(field: str) -> Callable[[PairValues, Cache], np.ndarray]:
    return lambda pv, cache=None: _isnan(pv.left(field), pv.right(field))


def _cached_threshold_test(
    key: str, op: str, value: float
) -> Callable[[PairValues, Cache], np.ndarray]:
    def test(pv: PairValues, cache: Cache = None) -> np.ndarray:
        scores = np.asarray(cache[key], dtype=np.float64)
        if op == ">=":
            return scores >= value
        return scores <= value
    return test


def _cached_epoch_diff_test(
    key_l: str, key_r: str, seconds: float
) -> Callable[[PairValues, Cache], np.ndarray]:
    def test(pv: PairValues, cache: Cache = None) -> np.ndarray:
        diff = np.abs(np.asarray(cache[key_l], dtype=np.float64)
                      - np.asarray(cache[key_r], dtype=np.float64))
        return np.isfinite(diff) & (diff <= seconds)
    return test


def _and_test(*tests: Callable[[PairValues, Cache], np.ndarray]) -> Callable[[PairValues, Cache], np.ndarray]:
    def test(pv: PairValues, cache: Cache = None) -> np.ndarray:
        out = tests[0](pv, cache)
        for t in tests[1:]:
            out = out & t(pv, cache)
        return out
    return test


def _jw_prescore(field: str, prefix: str = "jw") -> Callable[[PairValues], dict]:
    def prescore(pv: PairValues) -> dict:
        return {prefix: sim.jaro_winkler_similarity(pv.left(field), pv.right(field))}
    return prescore


def _distance_prescore(field: str, fn: Callable, key: str = "dist") -> Callable[[PairValues], dict]:
    def prescore(pv: PairValues) -> dict:
        return {key: fn(pv.left(field), pv.right(field))}
    return prescore