"""Thresholded distance/similarity comparison families.

The most numerous family group, all sharing the same skeleton -- a null level,
an exact level, one level per threshold reading a cached score array, and an
ELSE fallback -- over a single ``_distance_prescore`` pass (one Jaro/JW/edit/
jaccard/cosine/intersection computation shared by every threshold level).
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

from .. import sim
from ._core import (
    ComparisonSpec,
    build_spec,
    _cached_threshold_test,
    _distance_prescore,
    _else_level,
    _exact_test,
    _null_test,
)

_DISTANCE_FUNCTIONS = {
    "levenshtein": sim.levenshtein_distance,
    "damerau_levenshtein": sim.damerau_levenshtein_distance,
    "jaro_winkler": sim.jaro_winkler_similarity,
    "jaro": sim.jaro_similarity,
    "jaccard": sim.jaccard,
}


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
    from ._core import _jw_prescore

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

    def prescore(pv) -> dict:
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