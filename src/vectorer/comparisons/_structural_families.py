"""Structural/identifier comparison families.

Email, full name, forename/surname, postcode and lat/long-distance.  These
share the parts-parsing style (split the structured value, then test exact
parts / fuzzy scores from a cached prescore) rather than the pure
thresholded-distance skeleton of :mod:`._distance_families`.
"""

from __future__ import annotations

import re
from typing import Callable, Optional, Sequence

import numpy as np

from .. import sim
from ._core import (
    ComparisonSpec,
    PairValues,
    build_spec,
    _and_test,
    _cached_threshold_test,
    _else_level,
    _exact_test,
    _isnan,
    _jw_prescore,
    _null_test,
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
    country: str = "UK",
) -> ComparisonSpec:
    """Postcode comparison (exact + sector/district/area part levels).

    ``country`` selects the postcode format, defaulting to ``"UK"`` for
    compatibility with Splink's own ``postcode_uk`` comparison ("SW1A 1AA"
    outward/inward format).  Set ``country="CA"`` for Canadian postcodes
    ("M5A 1A1" FSA + LDU format).  See :data:`vectorer.sim.POSTCODE_PATTERNS`
    for the supported formats.
    """
    full_pattern = sim.POSTCODE_PATTERNS.get(country.strip().lower())
    if full_pattern is None:
        raise ValueError(
            f"unsupported postcode country {country!r}; choose from "
            f"{sorted(k.upper() for k in sim.POSTCODE_PATTERNS)}"
        )

    def prescore(pv: PairValues) -> dict:
        left = pv.left(col_name)
        right = pv.right(col_name)
        parts_l = [sim.postcode_parts(str(v), country) if v is not None else (None, None, None) for v in left]
        parts_r = [sim.postcode_parts(str(v), country) if v is not None else (None, None, None) for v in right]
        out = {"parts_l": parts_l, "parts_r": parts_r, "null": _isnan(left, right)}
        if invalid_postcodes_as_null:
            full = full_pattern["full"]
            invalid = np.fromiter(
                (
                    l is None or r is None
                    or re.fullmatch(full, str(l).strip()) is None
                    or re.fullmatch(full, str(r).strip()) is None
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