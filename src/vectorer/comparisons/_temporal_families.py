"""Temporal comparison families: dates and date-of-birth.

All three share the same skeleton -- a null level for missing/invalid values,
an exact level, one level per (metric, threshold) reading cached epoch-second
arrays, and an ELSE fallback -- over ``sim._epoch_seconds`` prescores.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .. import sim
from ._core import (
    ComparisonSpec,
    build_spec,
    _cached_epoch_diff_test,
    _cached_threshold_test,
    _else_level,
    _exact_test,
)

_UNITS_TO_SECONDS = {
    "year": sim.SECONDS_PER_YEAR,
    "month": sim.SECONDS_PER_MONTH,
    "week": 604800.0,
    "day": 86400.0,
    "hour": 3600.0,
    "minute": 60.0,
    "second": 1.0,
}


def date_of_birth_comparison_spec(
    col_name: str,
    input_is_string: bool = True,
    datetime_thresholds: Sequence = (1, 1, 10),
    datetime_metrics: Sequence[str] = ("month", "year", "year"),
    datetime_format: Optional[str] = None,
    invalid_dates_as_null: bool = True,
) -> ComparisonSpec:
    fmt = datetime_format or "%Y-%m-%d"

    def prescore(pv) -> dict:
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


def _abs_difference_spec(
    col_name: str,
    metrics: Sequence[str],
    thresholds: Sequence[float],
    datetime_format: Optional[str],
    label: str,
) -> ComparisonSpec:
    def prescore(pv) -> dict:
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