"""Extensible Fellegi-Sunter comparison set -- native, fully vectorized.

A full reimplementation of the comparison family that Splink exposes through
``splink.comparison_library``, with **no Splink / DuckDB / SQL dependency**:
each comparison is a list of *levels*, and every level is a vectorized NumPy
predicate evaluated over whole batches of pairs (see :mod:`vectorer.sim`).
The set is name-keyed through :class:`ComparisonRegistry` and covers the same
19 comparison options, so a model declared here behaves like the Splink model
of the same name.

Performance
-----------
Each comparison carries an optional *pre-score* step (:attr:`ComparisonSpec.prescore`)
that computes shared score arrays (e.g. one Jaro-Winkler pass instead of one per
threshold) once per batch; every level then reads those cached arrays.  Levels
are ordered by decreasing agreement, exactly as Splink's comparison levels are,
and the first matching level decides the pair's bayes factor.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import numpy as np

from . import sim

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
    floor), matching Splink exactly.
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
    build time using Splink's default-value algorithm.
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
                    m=m if m is not None else level.m,
                    u=u if u is not None else level.u,
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
# Splink-equivalent default m/u
# ---------------------------------------------------------------------------


def _default_m_values(num_levels: int) -> list[float]:
    split_remainder = 0.05 / (num_levels - 1)
    return [split_remainder] * (num_levels - 1) + [0.95]


def _default_u_values(num_levels: int) -> list[float]:
    m_vals = _default_m_values(num_levels)
    if num_levels == 2:
        match_weights = [-5]
    else:
        match_weights = list(np.linspace(-5.0, 3.0, num_levels - 1))
    match_weights = match_weights + [10.0]
    return [m / (2 ** w) for m, w in zip(m_vals, match_weights)]


def apply_default_mu(spec: ComparisonSpec) -> None:
    """Assign Splink's default m/u to levels lacking explicit probabilities."""
    non_null = [lv for lv in spec.levels if not lv.is_null]
    if not non_null or len(spec.levels) <= 1:
        return
    k = len(non_null)
    ms = _default_m_values(k)
    us = _default_u_values(k)
    cvv = k - 1
    for lv in spec.levels:
        if lv.is_null:
            lv.cvv = -1
            continue
        lv.cvv = cvv
        if lv.m is None:
            lv.m = ms[cvv]
        if lv.u is None:
            lv.u = us[cvv]
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
    """Assemble a ComparisonSpec from ordered level dicts and apply defaults."""
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


# ---------------------------------------------------------------------------
# Comparison factories (one per Splink comparison option)
# ---------------------------------------------------------------------------

_UNITS_TO_SECONDS = {
    "year": sim.SECONDS_PER_YEAR,
    "month": sim.SECONDS_PER_MONTH,
    "week": 604800.0,
    "day": 86400.0,
    "hour": 3600.0,
    "minute": 60.0,
    "second": 1.0,
}

_DISTANCE_FUNCTIONS = {
    "levenshtein": sim.levenshtein_distance,
    "damerau_levenshtein": sim.damerau_levenshtein_distance,
    "jaro_winkler": sim.jaro_winkler_similarity,
    "jaro": sim.jaro_similarity,
    "jaccard": sim.jaccard,
}

_POSTCODE_FULL = r"^[A-Za-z]{1,2}[0-9][A-Za-z0-9]?\s[0-9][A-Za-z]{2}$"


def exact_match_spec(col_name: str, term_frequency_adjustments: bool = False) -> ComparisonSpec:
    levels = [
        {"label_for_charts": f"{col_name} is NULL", "is_null_level": True, "test": _null_test(col_name)},
        {"label_for_charts": f"Exact match on {col_name}", "test": _exact_test(col_name),
         "tf_adjustment_column": col_name if term_frequency_adjustments else None},
        _else_level(),
    ]
    return build_spec(
        output_column_name=col_name,
        level_dicts=levels,
        fields=(col_name,),
    )


def jaro_winkler_at_thresholds_spec(
    col_name: str,
    score_threshold_or_thresholds: Sequence[float] = (0.9, 0.7),
) -> ComparisonSpec:
    thresholds = sorted(score_threshold_or_thresholds, reverse=True)
    levels = [
        {"label_for_charts": f"{col_name} is NULL", "is_null_level": True, "test": _null_test(col_name)},
        {"label_for_charts": f"Exact match on {col_name}", "test": _exact_test(col_name)},
    ]
    for t in thresholds:
        levels.append({
            "label_for_charts": f"Jaro-Winkler distance of {col_name} >= {t}",
            "test": _cached_threshold_test("jw", ">=", t),
        })
    levels.append(_else_level())
    return build_spec(
        output_column_name=col_name, level_dicts=levels, fields=(col_name,),
        prescore=_jw_prescore(col_name),
    )


def jaro_at_thresholds_spec(
    col_name: str,
    score_threshold_or_thresholds: Sequence[float] = (0.9, 0.7),
) -> ComparisonSpec:
    thresholds = sorted(score_threshold_or_thresholds, reverse=True)
    levels = [
        {"label_for_charts": f"{col_name} is NULL", "is_null_level": True, "test": _null_test(col_name)},
        {"label_for_charts": f"Exact match on {col_name}", "test": _exact_test(col_name)},
    ]
    for t in thresholds:
        levels.append({
            "label_for_charts": f"Jaro distance of {col_name} >= {t}",
            "test": _cached_threshold_test("jaro", ">=", t),
        })
    levels.append(_else_level())
    return build_spec(
        output_column_name=col_name, level_dicts=levels, fields=(col_name,),
        prescore=_distance_prescore(col_name, sim.jaro_similarity, "jaro"),
    )


def _edit_distance_spec(
    col_name: str,
    fn: Callable,
    thresholds: Sequence[int],
    distance_name: str,
) -> ComparisonSpec:
    levels = [
        {"label_for_charts": f"{col_name} is NULL", "is_null_level": True, "test": _null_test(col_name)},
        {"label_for_charts": f"Exact match on {col_name}", "test": _exact_test(col_name)},
    ]
    for t in sorted(thresholds):
        levels.append({
            "label_for_charts": f"{distance_name} distance of {col_name} <= {t}",
            "test": _cached_threshold_test("dist", "<=", t),
        })
    levels.append(_else_level())
    return build_spec(
        output_column_name=col_name, level_dicts=levels, fields=(col_name,),
        prescore=_distance_prescore(col_name, fn),
    )


def levenshtein_at_thresholds_spec(
    col_name: str,
    distance_threshold_or_thresholds: Sequence[int] = (1, 2),
) -> ComparisonSpec:
    return _edit_distance_spec(
        col_name, sim.levenshtein_distance,
        list(distance_threshold_or_thresholds), "Levenshtein",
    )


def damerau_levenshtein_at_thresholds_spec(
    col_name: str,
    distance_threshold_or_thresholds: Sequence[int] = (1, 2),
) -> ComparisonSpec:
    return _edit_distance_spec(
        col_name, sim.damerau_levenshtein_distance,
        list(distance_threshold_or_thresholds), "Damerau-Levenshtein",
    )


def jaccard_at_thresholds_spec(
    col_name: str,
    score_threshold_or_thresholds: Sequence[float] = (0.9, 0.7),
) -> ComparisonSpec:
    thresholds = sorted(score_threshold_or_thresholds, reverse=True)
    levels = [
        {"label_for_charts": f"{col_name} is NULL", "is_null_level": True, "test": _null_test(col_name)},
        {"label_for_charts": f"Exact match on {col_name}", "test": _exact_test(col_name)},
    ]
    for t in thresholds:
        levels.append({
            "label_for_charts": f"Jaccard distance of {col_name} >= {t}",
            "test": _cached_threshold_test("jac", ">=", t),
        })
    levels.append(_else_level())
    return build_spec(
        output_column_name=col_name, level_dicts=levels, fields=(col_name,),
        prescore=_distance_prescore(col_name, sim.jaccard, "jac"),
    )


def cosine_similarity_at_thresholds_spec(
    col_name: str,
    score_threshold_or_thresholds: Sequence[float] = (0.9, 0.7),
) -> ComparisonSpec:
    thresholds = sorted(score_threshold_or_thresholds, reverse=True)
    levels = [
        {"label_for_charts": f"{col_name} is NULL", "is_null_level": True, "test": _null_test(col_name)},
    ]
    for t in thresholds:
        levels.append({
            "label_for_charts": f"Cosine similarity of {col_name} >= {t}",
            "test": _cached_threshold_test("cos", ">=", t),
        })
    levels.append(_else_level())
    return build_spec(
        output_column_name=col_name, level_dicts=levels, fields=(col_name,),
        prescore=_distance_prescore(col_name, sim.cosine_similarity, "cos"),
    )


def array_intersect_at_sizes_spec(
    col_name: str,
    size_threshold_or_thresholds: Sequence[int] = (1,),
) -> ComparisonSpec:
    levels = [
        {"label_for_charts": f"{col_name} is NULL", "is_null_level": True, "test": _null_test(col_name)},
    ]
    for size in sorted(size_threshold_or_thresholds):
        levels.append({
            "label_for_charts": f"Array intersection size >= {size} on {col_name}",
            "test": _cached_threshold_test("inter", ">=", size),
        })
    levels.append(_else_level())
    return build_spec(
        output_column_name=col_name, level_dicts=levels, fields=(col_name,),
        prescore=_distance_prescore(col_name, sim.array_intersect_size, "inter"),
    )


def date_of_birth_comparison_spec(
    col_name: str,
    input_is_string: bool = True,
    datetime_thresholds: Sequence = (1, 1, 10),
    datetime_metrics: Sequence[str] = ("month", "year", "year"),
    datetime_format: Optional[str] = None,
    invalid_dates_as_null: bool = True,
) -> ComparisonSpec:
    fmt = datetime_format or "%Y-%m-%d"

    def prescore(pv: PairValues) -> dict:
        left = pv.left(col_name)
        right = pv.right(col_name)
        e_l = sim._epoch_seconds(left, fmt)
        e_r = sim._epoch_seconds(right, fmt)
        return {
            "el": e_l,
            "er": e_r,
            "valid": np.isfinite(e_l) & np.isfinite(e_r),
            "dl": sim.damerau_levenshtein_distance(left, right),
        }

    levels = [
        {
            "label_for_charts": f"transformed {col_name} is NULL",
            "is_null_level": True,
            "test": lambda pv, cache=None: ~np.asarray(cache["valid"], dtype=bool),
        },
        {"label_for_charts": f"Exact match on date of birth {col_name}", "test": _exact_test(col_name)},
        {
            "label_for_charts": f"DamerauLevenshtein distance <= 1 on {col_name}",
            "test": _cached_threshold_test("dl", "<=", 1),
        },
    ]
    for metric, threshold in zip(datetime_metrics, datetime_thresholds):
        seconds = float(threshold) * _UNITS_TO_SECONDS[metric]
        levels.append({
            "label_for_charts": f"Abs difference of {col_name} <= {threshold} {metric}",
            "test": _cached_epoch_diff_test("el", "er", seconds),
        })
    levels.append(_else_level())
    return build_spec(
        output_column_name=col_name, level_dicts=levels, fields=(col_name,),
        prescore=prescore,
    )


def email_comparison_spec(col_name: str) -> ComparisonSpec:
    def prescore(pv: PairValues) -> dict:
        left = pv.left(col_name)
        right = pv.right(col_name)
        lparts = [sim.email_parts(str(v)) if v is not None else None for v in left]
        rparts = [sim.email_parts(str(v)) if v is not None else None for v in right]
        uname_present = np.fromiter(
            (l is not None and r is not None for l, r in zip(lparts, rparts)),
            dtype=bool, count=pv.n,
        )
        uname_equal = np.fromiter(
            (l is not None and r is not None and l == r for l, r in zip(lparts, rparts)),
            dtype=bool, count=pv.n,
        )
        uname_jw = sim.jaro_winkler_similarity(lparts, rparts)
        return {
            "jw": sim.jaro_winkler_similarity(left, right),
            "uname_equal": uname_equal & uname_present,
            "uname_jw": uname_jw,
            "uname_present": uname_present,
        }

    def uname_jw_test(threshold: float):
        def test(pv: PairValues, cache=None) -> np.ndarray:
            return (np.asarray(cache["uname_jw"]) >= threshold) & np.asarray(cache["uname_present"])
        return test

    levels = [
        {"label_for_charts": f"{col_name} is NULL", "is_null_level": True, "test": _null_test(col_name)},
        {"label_for_charts": f"Exact match on {col_name}", "test": _exact_test(col_name)},
        {"label_for_charts": f"Exact match on username of {col_name}",
         "test": lambda pv, cache=None: np.asarray(cache["uname_equal"])},
        {"label_for_charts": f"Jaro-Winkler distance of {col_name} >= 0.88",
         "test": _cached_threshold_test("jw", ">=", 0.88)},
        {"label_for_charts": f"Jaro-Winkler >0.88 on username of {col_name}", "test": uname_jw_test(0.88)},
        _else_level(),
    ]
    return build_spec(
        output_column_name=col_name, level_dicts=levels, fields=(col_name,),
        prescore=prescore,
    )


def name_comparison_spec(
    col_name: str,
    jaro_winkler_thresholds: Sequence[float] = (0.92, 0.88, 0.7),
    dmeta_col_name: Optional[str] = None,
    dmeta_fn: Optional[Callable[[str], str]] = None,
) -> ComparisonSpec:
    thresholds = sorted(jaro_winkler_thresholds, reverse=True)
    levels = [
        {"label_for_charts": f"{col_name} is NULL", "is_null_level": True, "test": _null_test(col_name)},
        {"label_for_charts": f"Exact match on {col_name}", "test": _exact_test(col_name)},
    ]
    if dmeta_col_name is not None and dmeta_fn is not None:
        def dmeta_test(pv: PairValues, cache=None) -> np.ndarray:
            out = np.empty(pv.n, dtype=bool)
            left, right = pv.left(col_name), pv.right(col_name)
            for i in range(pv.n):
                if left[i] is None or right[i] is None:
                    out[i] = False
                else:
                    try:
                        out[i] = dmeta_fn(str(left[i])) == dmeta_fn(str(right[i]))
                    except Exception:
                        out[i] = False
            return out
        levels.append({"label_for_charts": f"Double metaphone {dmeta_col_name} match", "test": dmeta_test})
    for t in thresholds:
        levels.append({
            "label_for_charts": f"Jaro-Winkler distance of {col_name} >= {t}",
            "test": _cached_threshold_test("jw", ">=", t),
        })
    levels.append(_else_level())
    return build_spec(
        output_column_name=col_name, level_dicts=levels, fields=(col_name,),
        prescore=_jw_prescore(col_name),
    )


def forename_surname_comparison_spec(
    forename_col_name: str,
    surname_col_name: str,
    jaro_winkler_thresholds: Sequence[float] = (0.92, 0.88),
) -> ComparisonSpec:
    thresholds = sorted(jaro_winkler_thresholds, reverse=True)

    def prescore(pv: PairValues) -> dict:
        return {
            "f_jw": sim.jaro_winkler_similarity(
                pv.left(forename_col_name), pv.right(forename_col_name)),
            "s_jw": sim.jaro_winkler_similarity(
                pv.left(surname_col_name), pv.right(surname_col_name)),
        }

    levels = [
        {
            "label_for_charts": f"({forename_col_name} is NULL) AND ({surname_col_name} is NULL)",
            "is_null_level": True,
            "test": lambda pv, cache=None: _isnan(
                pv.left(forename_col_name), pv.right(forename_col_name)
            ) & _isnan(pv.left(surname_col_name), pv.right(surname_col_name)),
        },
        {
            "label_for_charts": f"(Exact match on {forename_col_name}) AND (Exact match on {surname_col_name})",
            "test": _and_test(_exact_test(forename_col_name), _exact_test(surname_col_name)),
        },
        {
            "label_for_charts": f"Match on reversed cols: {forename_col_name} and {surname_col_name} (both directions)",
            "test": lambda pv, cache=None: (
                sim.exact_equals(pv.left(forename_col_name), pv.right(surname_col_name))
                & sim.exact_equals(pv.left(surname_col_name), pv.right(forename_col_name))
            ),
        },
    ]
    for t in thresholds:
        levels.append({
            "label_for_charts": f"(Jaro-Winkler distance of {forename_col_name} >= {t}) AND (Jaro-Winkler distance of {surname_col_name} >= {t})",
            "test": _and_test(
                _cached_threshold_test("f_jw", ">=", t),
                _cached_threshold_test("s_jw", ">=", t),
            ),
        })
    levels.append({"label_for_charts": f"Exact match on {surname_col_name}", "test": _exact_test(surname_col_name)})
    levels.append({"label_for_charts": f"Exact match on {forename_col_name}", "test": _exact_test(forename_col_name)})
    levels.append(_else_level())
    return build_spec(
        output_column_name=f"{forename_col_name}_{surname_col_name}",
        level_dicts=levels,
        fields=(forename_col_name, surname_col_name),
        prescore=prescore,
    )


def postcode_comparison_spec(
    col_name: str,
    invalid_postcodes_as_null: bool = False,
    lat_col: Optional[str] = None,
    long_col: Optional[str] = None,
    km_thresholds: Sequence[float] = (1, 10, 100),
) -> ComparisonSpec:
    def prescore(pv: PairValues) -> dict:
        left = pv.left(col_name)
        right = pv.right(col_name)
        parts_l = [sim.postcode_parts(str(v)) if v is not None else (None, None, None) for v in left]
        parts_r = [sim.postcode_parts(str(v)) if v is not None else (None, None, None) for v in right]
        out = {"parts_l": parts_l, "parts_r": parts_r, "null": _isnan(left, right)}
        if invalid_postcodes_as_null:
            invalid = np.fromiter(
                (
                    l is None or r is None
                    or not re.fullmatch(_POSTCODE_FULL, str(l).strip())
                    or not re.fullmatch(_POSTCODE_FULL, str(r).strip())
                    for l, r in zip(left, right)
                ),
                dtype=bool, count=pv.n,
            )
            out["null"] = out["null"] | invalid
        if lat_col is not None and long_col is not None and km_thresholds:
            out["km"] = sim.haversine_km(
                pv.left(lat_col), pv.left(long_col),
                pv.right(lat_col), pv.right(long_col),
            )
        return out

    def part_equal(part_index: int):
        def test(pv: PairValues, cache=None) -> np.ndarray:
            out = np.zeros(pv.n, dtype=bool)
            parts_l = cache["parts_l"]
            parts_r = cache["parts_r"]
            for i in range(pv.n):
                l = parts_l[i][part_index]
                r = parts_r[i][part_index]
                if l is not None and r is not None and l == r:
                    out[i] = True
            return out
        return test

    levels = [
        {"label_for_charts": f"{col_name} is NULL", "is_null_level": True,
         "test": lambda pv, cache=None: np.asarray(cache["null"], dtype=bool)},
        {"label_for_charts": f"Exact match on full {col_name}", "test": _exact_test(col_name)},
        {"label_for_charts": f"Exact match on sector of {col_name}", "test": part_equal(0)},
        {"label_for_charts": f"Exact match on district of {col_name}", "test": part_equal(1)},
        {"label_for_charts": f"Exact match on area of {col_name}", "test": part_equal(2)},
    ]
    if lat_col is not None and long_col is not None and km_thresholds:
        for threshold in sorted(km_thresholds):
            levels.append({
                "label_for_charts": f"Distance in km less than {threshold}",
                "test": _cached_threshold_test("km", "<=", float(threshold)),
            })
    levels.append(_else_level())
    fields = (col_name,) + (() if lat_col is None else (lat_col, long_col))
    return build_spec(
        output_column_name=col_name, level_dicts=levels, fields=fields, prescore=prescore,
    )


def distance_in_km_at_thresholds_spec(
    lat_col: str,
    long_col: str,
    km_thresholds: Sequence[float] = (1, 10, 100),
) -> ComparisonSpec:
    def prescore(pv: PairValues) -> dict:
        return {
            "km": sim.haversine_km(
                pv.left(lat_col), pv.left(long_col),
                pv.right(lat_col), pv.right(long_col),
            ),
            "null": _isnan(pv.left(lat_col), pv.right(lat_col))
            | _isnan(pv.left(long_col), pv.right(long_col)),
        }

    levels = [
        {
            "label_for_charts": f"({lat_col} is NULL) OR ({long_col} is NULL)",
            "is_null_level": True,
            "test": lambda pv, cache=None: np.asarray(cache["null"], dtype=bool),
        },
    ]
    for threshold in sorted(km_thresholds):
        levels.append({
            "label_for_charts": f"Distance in km less than {threshold}",
            "test": _cached_threshold_test("km", "<=", float(threshold)),
        })
    levels.append(_else_level())
    return build_spec(
        output_column_name=f"{lat_col}_{long_col}",
        level_dicts=levels,
        fields=(lat_col, long_col),
        prescore=prescore,
    )


def _resolve_distance_function(distance_function_name) -> Callable:
    if callable(distance_function_name):
        return distance_function_name
    name = distance_function_name.lower()
    if name not in _DISTANCE_FUNCTIONS:
        raise ValueError(
            f"unknown distance function {distance_function_name!r}; choose from "
            f"{sorted(_DISTANCE_FUNCTIONS)} or pass a callable"
        )
    return _DISTANCE_FUNCTIONS[name]


def distance_function_at_thresholds_spec(
    col_name: str,
    distance_function_name: str,
    distance_threshold_or_thresholds: Sequence[float],
    higher_is_more_similar: bool = True,
) -> ComparisonSpec:
    fn = _resolve_distance_function(distance_function_name)
    op = ">=" if higher_is_more_similar else "<="
    levels = [
        {"label_for_charts": f"{col_name} is NULL", "is_null_level": True, "test": _null_test(col_name)},
    ]
    for t in sorted(distance_threshold_or_thresholds, reverse=higher_is_more_similar):
        levels.append({
            "label_for_charts": f"Distance of {col_name} {op} {t}",
            "test": _cached_threshold_test("score", op, t),
        })
    levels.append(_else_level())
    return build_spec(
        output_column_name=col_name, level_dicts=levels, fields=(col_name,),
        prescore=_distance_prescore(col_name, fn, "score"),
    )


def pairwise_string_distance_function_at_thresholds_spec(
    col_name: str,
    distance_function_name: str,
    distance_threshold_or_thresholds: Sequence[float],
) -> ComparisonSpec:
    fn = _resolve_distance_function(distance_function_name)
    distance_like = distance_function_name.lower() in ("levenshtein", "damerau_levenshtein")
    op = "<=" if distance_like else ">="

    def prescore(pv: PairValues) -> dict:
        return {
            "inter": sim.array_intersect_size(pv.left(col_name), pv.right(col_name)),
            "max": sim.pairwise_max_similarity(pv.left(col_name), pv.right(col_name), fn),
        }

    levels = [
        {"label_for_charts": f"{col_name} is NULL", "is_null_level": True, "test": _null_test(col_name)},
        {"label_for_charts": f"Array intersection size >= 1 on {col_name}",
         "test": _cached_threshold_test("inter", ">=", 1)},
    ]
    for t in sorted(distance_threshold_or_thresholds, reverse=not distance_like):
        levels.append({
            "label_for_charts": f"Max {distance_function_name} distance of {col_name} {op} {t}",
            "test": _cached_threshold_test("max", op, t),
        })
    levels.append(_else_level())
    return build_spec(
        output_column_name=col_name, level_dicts=levels, fields=(col_name,),
        prescore=prescore,
    )


def _abs_difference_spec(
    col_name: str,
    metrics: Sequence[str],
    thresholds: Sequence[float],
    datetime_format: Optional[str],
    label: str,
) -> ComparisonSpec:
    def prescore(pv: PairValues) -> dict:
        e_l = sim._epoch_seconds(pv.left(col_name), datetime_format)
        e_r = sim._epoch_seconds(pv.right(col_name), datetime_format)
        return {
            "el": e_l,
            "er": e_r,
            "valid": np.isfinite(e_l) & np.isfinite(e_r),
        }

    levels = [
        {
            "label_for_charts": f"transformed {col_name} is NULL",
            "is_null_level": True,
            "test": lambda pv, cache=None: ~np.asarray(cache["valid"], dtype=bool),
        },
        {"label_for_charts": f"Exact match on {col_name}", "test": _exact_test(col_name)},
    ]
    for metric, threshold in zip(metrics, thresholds):
        seconds = float(threshold) * _UNITS_TO_SECONDS[metric]
        levels.append({
            "label_for_charts": f"Abs {label} of {col_name} <= {threshold} {metric}",
            "test": _cached_epoch_diff_test("el", "er", seconds),
        })
    levels.append(_else_level())
    return build_spec(
        output_column_name=col_name, level_dicts=levels, fields=(col_name,),
        prescore=prescore,
    )


def absolute_date_difference_at_thresholds_spec(
    col_name: str,
    input_is_string: bool = True,
    metrics: Sequence[str] = ("day",),
    thresholds: Sequence[float] = (1,),
    datetime_format: Optional[str] = None,
    invalid_dates_as_null: bool = True,
) -> ComparisonSpec:
    if len(metrics) != len(thresholds):
        raise ValueError("metrics and thresholds must have the same length")
    return _abs_difference_spec(col_name, metrics, thresholds, datetime_format, "date difference")


def absolute_time_difference_at_thresholds_spec(
    col_name: str,
    input_is_string: bool = True,
    metrics: Sequence[str] = ("minute",),
    thresholds: Sequence[float] = (60,),
    datetime_format: Optional[str] = None,
    invalid_dates_as_null: bool = True,
) -> ComparisonSpec:
    if len(metrics) != len(thresholds):
        raise ValueError("metrics and thresholds must have the same length")
    fmt = datetime_format or "%Y-%m-%dT%H:%M:%SZ"
    return _abs_difference_spec(col_name, metrics, thresholds, fmt, "time difference")


# ---------------------------------------------------------------------------
# Custom comparison (declarative conditions, no SQL engine)
# ---------------------------------------------------------------------------

_NULL_CONDITION = re.compile(
    r'^\s*"(?P<c1>[^"]+_l)"\s+IS\s+NULL\s+OR\s+"(?P<c2>[^"]+_r)"\s+IS\s+NULL\s*$',
    re.I,
)
_EQ_CONDITION = re.compile(
    r'^\s*"(?P<c1>[^"]+_l)"\s*=\s*"(?P<c2>[^"]+_r)"\s*$'
)


def custom_comparison_spec(
    output_column_name: str,
    comparison_levels: Sequence[dict],
) -> ComparisonSpec:
    """Build a comparison from user-supplied level dicts (no SQL engine).

    Each level dict may carry:

    * ``test`` -- a callable ``(PairValues, cache) -> bool mask`` used verbatim,
    * ``sql_condition`` -- one of the **recognized** declarative conditions:
      ``"ELSE"``, ``'"<col>"_l" IS NULL OR "<col>"_r" IS NULL'`` (null level),
      ``'"<col>"_l" = "<col>"_r'`` (exact match level).

    Any other ``sql_condition`` string is rejected: this framework evaluates
    comparisons natively and does not run SQL.
    """
    if not output_column_name:
        raise ValueError("custom_comparison requires an output_column_name")
    fields: list[str] = []
    seen: set[str] = set()
    levels: list[dict] = []
    for raw in comparison_levels:
        level = dict(raw)
        test = level.get("test")
        condition = level.get("sql_condition", "")
        if test is None and condition:
            stripped = condition.strip()
            if stripped.upper() == "ELSE":
                test = None
                level["is_null_level"] = False
            else:
                null_match = _NULL_CONDITION.fullmatch(stripped)
                exact_match = _EQ_CONDITION.fullmatch(stripped)
                if null_match:
                    col = _column_of_condition(stripped)
                    test = _null_test(col)
                    level["is_null_level"] = True
                elif exact_match:
                    col = exact_match.group("c1")[:-2]
                    test = _exact_test(col)
                    _collect_field(fields, seen, col)
                else:
                    raise ValueError(
                        "custom_comparison supports only the declarative conditions "
                        "'ELSE', '\"col_l\" IS NULL OR \"col_r\" IS NULL' and "
                        "'\"col_l\" = \"col_r\"' (this framework evaluates "
                        "comparisons natively and does not run SQL); provide a "
                        "'test' callable for anything more expressive. Got "
                        f"sql_condition: {stripped!r}"
                    )
        level["test"] = test
        levels.append(level)
    return build_spec(
        output_column_name=output_column_name,
        level_dicts=levels,
        fields=tuple(fields),
    )


def _column_of_condition(condition: str) -> str:
    m = _NULL_CONDITION.fullmatch(condition)
    if not m:
        raise ValueError(f"cannot parse null condition {condition!r}")
    return m.group("c1")[:-2]


def _collect_field(fields: list[str], seen: set[str], field: str) -> None:
    if field and field not in seen:
        seen.add(field)
        fields.append(field)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegistryEntry:
    factory: Callable[..., ComparisonSpec]
    fields: tuple[str, ...]
    defaults: dict = field(default_factory=dict)
    description: str = ""


class ComparisonRegistry:
    """Name-keyed registry of Fellegi-Sunter comparison builders."""

    def __init__(self) -> None:
        self._entries: dict[str, RegistryEntry] = {}

    def register(
        self,
        name: str,
        factory: Callable[..., ComparisonSpec],
        fields: Sequence[str] = ("col_name",),
        defaults: Optional[dict] = None,
        description: str = "",
    ) -> None:
        name = name.strip().lower()
        if not re.fullmatch(r"[a-z0-9_]+", name):
            raise ValueError(f"comparison names must be lowercase snake_case, got {name!r}")
        self._entries[name] = RegistryEntry(
            factory=factory,
            fields=tuple(fields),
            defaults=dict(defaults or {}),
            description=description,
        )

    def __contains__(self, name: str) -> bool:
        return name.strip().lower() in self._entries

    def names(self) -> list[str]:
        return sorted(self._entries)

    def fields_of(self, name: str) -> tuple[str, ...]:
        return self._entries[name.strip().lower()].fields

    def available(self) -> dict[str, str]:
        return {name: entry.description for name, entry in sorted(self._entries.items())}

    def make(self, name: str, **kwargs: Any) -> "Comparison":
        name = name.strip().lower()
        if name not in self._entries:
            raise KeyError(f"unknown comparison {name!r}; available: {self.names()}")
        entry = self._entries[name]
        params = dict(entry.defaults)
        params.update(kwargs)
        record_fields = tuple(params[f] for f in entry.fields if f in params)
        return Comparison(name=name, params=params, fields=record_fields)

    def construct(self, comparison: "Comparison") -> ComparisonSpec:
        return self._entries[comparison.name].factory(**comparison.params)


REGISTRY = ComparisonRegistry()


_DESCRIPTIONS = {
    "exact_match": "Exact match on a column (with optional term-frequency adjustment).",
    "jaro_winkler_at_thresholds": "Jaro-Winkler similarity at score thresholds.",
    "jaro_at_thresholds": "Jaro similarity at score thresholds.",
    "levenshtein_at_thresholds": "Levenshtein distance at distance thresholds.",
    "damerau_levenshtein_at_thresholds": "Damerau-Levenshtein distance at distance thresholds.",
    "jaccard_at_thresholds": "Jaccard similarity on array columns at score thresholds.",
    "cosine_similarity_at_thresholds": "Cosine similarity on array columns at score thresholds.",
    "date_of_birth_comparison": "Date-of-birth comparison (month/year + year thresholds).",
    "email_comparison": "Email comparison (exact + fuzzy + domain levels).",
    "name_comparison": "Full-name comparison with jaro-winkler (and optional metaphone) levels.",
    "forename_surname_comparison": "Forename/surname pair comparison.",
    "postcode_comparison": "Postcode comparison with optional lat/long distance levels.",
    "distance_in_km_at_thresholds": "Haversine distance in km between lat/long columns.",
    "distance_function_at_thresholds": "Arbitrary distance function at thresholds.",
    "pairwise_string_distance_function_at_thresholds": "Max cross-pair string distance between array columns.",
    "absolute_date_difference_at_thresholds": "Absolute date difference at thresholds (metric-aware).",
    "absolute_time_difference_at_thresholds": "Absolute time (timestamp) difference at thresholds.",
    "array_intersect_at_sizes": "Array intersection count at size thresholds.",
    "custom_comparison": "User-supplied levels (declarative conditions or vectorized test callables).",
}


def _register_built_ins() -> None:
    R = REGISTRY

    def add(name: str, factory: Callable, fields=("col_name",), defaults=None) -> None:
        R.register(
            name,
            factory,
            fields=fields,
            defaults=defaults,
            description=_DESCRIPTIONS.get(name, ""),
        )

    add("exact_match", exact_match_spec, defaults={"term_frequency_adjustments": False})
    add("jaro_winkler_at_thresholds", jaro_winkler_at_thresholds_spec,
        defaults={"score_threshold_or_thresholds": [0.9, 0.7]})
    add("jaro_at_thresholds", jaro_at_thresholds_spec,
        defaults={"score_threshold_or_thresholds": [0.9, 0.7]})
    add("levenshtein_at_thresholds", levenshtein_at_thresholds_spec,
        defaults={"distance_threshold_or_thresholds": [1, 2]})
    add("damerau_levenshtein_at_thresholds", damerau_levenshtein_at_thresholds_spec,
        defaults={"distance_threshold_or_thresholds": [1, 2]})
    add("jaccard_at_thresholds", jaccard_at_thresholds_spec,
        defaults={"score_threshold_or_thresholds": [0.9, 0.7]})
    add("cosine_similarity_at_thresholds", cosine_similarity_at_thresholds_spec,
        defaults={"score_threshold_or_thresholds": [0.9, 0.8, 0.7]})
    add("date_of_birth_comparison", date_of_birth_comparison_spec,
        defaults={"input_is_string": True})
    add("email_comparison", email_comparison_spec)
    add("name_comparison", name_comparison_spec,
        defaults={"jaro_winkler_thresholds": [0.92, 0.88, 0.7]})
    add("forename_surname_comparison", forename_surname_comparison_spec,
        fields=("forename_col_name", "surname_col_name"),
        defaults={"jaro_winkler_thresholds": [0.92, 0.88]})
    add("postcode_comparison", postcode_comparison_spec,
        defaults={"invalid_postcodes_as_null": False, "km_thresholds": [1, 10, 100]})
    add("distance_in_km_at_thresholds", distance_in_km_at_thresholds_spec,
        fields=("lat_col", "long_col"),
        defaults={"km_thresholds": [1, 10, 100]})
    add("distance_function_at_thresholds", distance_function_at_thresholds_spec,
        defaults={"higher_is_more_similar": True})
    add("pairwise_string_distance_function_at_thresholds",
        pairwise_string_distance_function_at_thresholds_spec)
    add("absolute_date_difference_at_thresholds",
        absolute_date_difference_at_thresholds_spec,
        defaults={"input_is_string": True, "metrics": ["day", "day", "day"],
                  "thresholds": [1, 7, 30]})
    add("absolute_time_difference_at_thresholds",
        absolute_time_difference_at_thresholds_spec,
        defaults={"input_is_string": True, "metrics": ["minute", "minute", "minute"],
                  "thresholds": [1, 60, 1440]})
    add("array_intersect_at_sizes", array_intersect_at_sizes_spec,
        defaults={"size_threshold_or_thresholds": [1]})
    R.register(
        "custom_comparison",
        custom_comparison_spec,
        fields=(),
        description=_DESCRIPTIONS["custom_comparison"],
    )


_register_built_ins()


def comparison_catalog() -> dict[str, dict]:
    """Catalog of every registered comparison (name -> fields + description)."""
    return {
        name: {"fields": entry.fields, "description": entry.description}
        for name, entry in sorted(REGISTRY._entries.items())
    }


def register_comparison(
    name: str,
    factory: Callable[..., ComparisonSpec],
    fields: Sequence[str] = ("col_name",),
    defaults: Optional[dict] = None,
    description: str = "",
) -> None:
    """Register a custom comparison builder (see :meth:`ComparisonRegistry.register`)."""
    REGISTRY.register(name, factory, fields=fields, defaults=defaults, description=description)


def available_comparisons() -> dict[str, str]:
    """All comparison options currently available (name -> description)."""
    return REGISTRY.available()


# ---------------------------------------------------------------------------
# Declared comparisons
# ---------------------------------------------------------------------------


class Comparison:
    """A declared comparison: registry name + constructor kwargs.

    Instances are light and JSON-serializable; :meth:`spec` builds the
    :class:`ComparisonSpec` (levels + default m/u) that the scorer consumes.
    """

    def __init__(self, name: str, params: dict, fields: tuple[str, ...] = ()) -> None:
        self.name = name
        self.params = dict(params)
        self.fields = tuple(fields)
        self._spec = None

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> Optional["Comparison"]:
        if not data:
            return None
        return cls(
            name=data["type"],
            params=dict(data["params"]),
            fields=tuple(data.get("fields", ())),
        )

    @classmethod
    def from_resolved(cls, resolved: dict) -> "Comparison":
        """Rebuild a comparison from ``{"type", "params", "levels"}``.

        ``levels`` override the (possibly trained) per-level m/u; level tests
        are re-derived from ``type`` + ``params`` via the registry.
        """
        comparison = cls(
            name=resolved["type"],
            params=dict(resolved.get("params", {})),
            fields=tuple(resolved.get("fields", ())),
        )
        spec = REGISTRY.construct(comparison)
        overrides = resolved.get("levels")
        if overrides:
            if len(overrides) != len(spec.levels):
                raise ValueError(
                    f"comparison {resolved['type']!r} has {len(spec.levels)} levels "
                    f"but {len(overrides)} were supplied"
                )
            for level, level_dict in zip(spec.levels, overrides):
                if not level.is_null:
                    m = level_dict.get("m_probability")
                    u = level_dict.get("u_probability")
                    if m is not None:
                        level.m = float(m)
                    if u is not None:
                        level.u = float(u)
                level.tf_column = level_dict.get("tf_adjustment_column", level.tf_column)
                level.tf_weight = level_dict.get("tf_adjustment_weight", level.tf_weight)
                level.tf_min_u = level_dict.get("tf_minimum_u_value", level.tf_min_u)
        comparison._spec = spec
        return comparison

    @classmethod
    def from_spec(cls, spec: ComparisonSpec) -> "Comparison":
        """Wrap a raw ``ComparisonSpec`` for use by the scorer.

        Persistence / calibration of such a comparison requires it to be
        re-declarable through the registry, so prefer ``make_comparison``.
        """
        comparison = cls(
            name="custom_comparison",
            params={"output_column_name": spec.output_column_name},
            fields=spec.fields,
        )
        comparison._spec = spec
        return comparison

    def to_dict(self) -> dict:
        return {"type": self.name, "params": dict(self.params), "fields": list(self.fields)}

    def spec(self) -> ComparisonSpec:
        """Build (and cache) the ComparisonSpec with default m/u applied."""
        if self._spec is None:
            self._spec = REGISTRY.construct(self)
        return self._spec

    def resolved(self) -> dict:
        """Serializable spec with the current (possibly trained) level probabilities."""
        spec = self.spec()
        return {
            "type": self.name,
            "params": dict(self.params),
            "levels": [level.prob_dict() for level in spec.levels],
        }

    def output_column_name(self) -> str:
        return self.spec().output_column_name

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, Comparison)
            and self.name == other.name
            and self.params == other.params
            and self.fields == other.fields
        )

    def __hash__(self) -> int:
        return hash((self.name, tuple(sorted(self.params.items())), self.fields))

    def __repr__(self) -> str:
        return f"Comparison({self.name}, params={self.params})"


def make_comparison(name: str, **kwargs: Any) -> Comparison:
    return REGISTRY.make(name, **kwargs)


def make_comparisons(specs: Sequence[Any]) -> list[Comparison]:
    """Build a list of :class:`Comparison` from specs.

    Each spec may be a ``Comparison`` (passed through), a dict
    ``{"type": ..., **kwargs}``, or a registered name string.
    """
    comparisons: list[Comparison] = []
    for spec in specs:
        if isinstance(spec, Comparison):
            comparisons.append(spec)
            continue
        if isinstance(spec, dict):
            spec = dict(spec)
            name = spec.pop("type")
            comparisons.append(REGISTRY.make(name, **spec))
            continue
        if isinstance(spec, str):
            comparisons.append(REGISTRY.make(spec))
            continue
        raise TypeError(f"cannot make a comparison from {type(spec).__name__}")
    return comparisons


def comparison_set(comparisons: Sequence[Any]) -> list[ComparisonSpec]:
    """Resolve declared comparisons to the specs the scorer evaluates.

    For ``Comparison`` objects this builds (and caches) each ``.spec()``.
    ``ComparisonSpec`` instances pass through unchanged.
    """
    out = []
    for item in comparisons:
        if isinstance(item, Comparison):
            out.append(item.spec())
        elif isinstance(item, ComparisonSpec):
            out.append(item)
        elif isinstance(item, dict):
            out.append(comparison_from_dict(item).spec())
        else:
            raise TypeError(f"expected Comparison / ComparisonSpec / dict, got {type(item).__name__}")
    return out


def comparison_to_dict(comparison: Any) -> dict:
    """Serialize a ``Comparison`` or raw spec dict."""
    if isinstance(comparison, Comparison):
        return comparison.to_dict()
    if isinstance(comparison, dict):
        return comparison
    raise TypeError("comparison_to_dict expects a Comparison")


def comparison_from_dict(data: Optional[dict]) -> Optional[Comparison]:
    return Comparison.from_dict(data)


def comparison_fields(comparisons: Sequence[Comparison]) -> list[str]:
    """Ordered list of record columns the comparison set reads."""
    fields: list[str] = []
    seen: set[str] = set()
    for comparison in comparisons:
        for field in comparison.fields:
            if field not in seen:
                seen.add(field)
                fields.append(field)
    return fields