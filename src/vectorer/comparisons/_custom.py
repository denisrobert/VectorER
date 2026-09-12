"""Custom comparison + the time-decay wrapper family.

``custom_comparison_spec`` builds from user-supplied level dicts (recognized
declarative conditions, otherwise a ``test`` callable).  ``time_decay_wrapper``
wraps any inner comparison so its levels are weighted by a time-distance band
-- the *product* comparison that implements time-decayed matching.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Optional, Sequence

import numpy as np

from ._core import (
    ComparisonSpec,
    PairValues,
    build_spec,
    _exact_test,
    _null_test,
)

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
# Time-decay wrapper
# ---------------------------------------------------------------------------

DAY = 86400.0


def _time_band(lo_days: float, hi_days: float, lo_s: float, hi_s: float):
    """Return a ``(PairValues, cache) -> bool mask`` that matches rows whose
    cached ``tdiff`` (seconds) lies in ``[lo, hi]``."""
    def band(pv, cache):
        return np.isfinite(cache["tdiff"]) & (cache["tdiff"] >= lo_s) & (cache["tdiff"] <= hi_s)
    band.__name__ = f"time_band_{lo_days:g}_{hi_days:g}d"
    return band


def time_decay_wrapper(
    inner_spec: Any,
    time_col: str = "event_date",
    bands: Optional[Sequence[tuple[float, float, float]]] = None,
    keep_inner_null: bool = False,
) -> ComparisonSpec:
    """Wrap ``inner_spec`` so its levels are weighted by a time-distance band.

    Builds a *product* comparison: every non-null level of ``inner_spec`` is
    crossed with each time band ``(lo_days, hi_days, weight)``.  The level's
    ``m`` is multiplied by the band's weight and then renormalized so the
    comparison's m-probabilities sum to 1 (m is the same-entity distribution);
    ``u`` is left unchanged (it is the chance-level probability, largely
    time-independent).  This implements time-decayed matching: an exact
    address close in time gets more evidence than the same address far apart.

    A leading null level fires whenever the timestamp is missing/invalid, so a
    missing date never reads as "a huge time gap".  With ``keep_inner_null``,
    the inner comparison's own null level (e.g. a missing address) is also
    preserved on top of the time-null level.
    """
    from .. import sim

    # Accept either a ComparisonSpec or a declaration (Comparison / dict).
    # ``Comparison`` is imported lazily to avoid a module cycle (`_registry`
    # imports this module; `_comparison` imports `_registry`).
    from ._comparison import Comparison, comparison_from_dict

    if isinstance(inner_spec, ComparisonSpec):
        pass
    elif isinstance(inner_spec, Comparison):
        inner_spec = inner_spec.spec()
    elif isinstance(inner_spec, dict):
        inner_spec = comparison_from_dict(inner_spec).spec()
    else:
        # Fallback: if it has .spec() (a Comparison-like), use it.
        spec_method = getattr(inner_spec, "spec", None)
        if spec_method is not None:
            inner_spec = spec_method()
        else:
            raise TypeError(
                "time_decay_wrapper expects a ComparisonSpec, a Comparison, or a "
                f"comparison dict; got {type(inner_spec).__name__}"
            )

    if bands is None:
        bands = [(0, 30, 1.0), (30, 365, 0.3), (365, 10 ** 9, 0.05)]
    time_band_secs = [(lo * DAY, hi * DAY, w) for (lo, hi, w) in bands]

    def prescore(pv):
        out = {"tdiff": sim.absolute_seconds_difference(pv.left(time_col), pv.right(time_col))}
        inner_prescore = inner_spec.prescore
        if inner_prescore is not None:
            out.update(inner_prescore(pv))
        return out

    levels = []
    # 1. Time-null: no valid timestamp -> no evidence.
    levels.append({
        "label_for_charts": f"{time_col} missing or invalid",
        "is_null_level": True,
        "test": lambda pv, cache: ~np.isfinite(cache["tdiff"]),
    })
    # 2. (optional) preserve the inner null (e.g. field missing)
    if keep_inner_null:
        for inner_lv in inner_spec.levels:
            if inner_lv.is_null:
                levels.append({
                    "label_for_charts": inner_lv.label,
                    "is_null_level": True,
                    "test": inner_lv.test,
                })

    # 3. Product of inner non-null levels x time bands; m decayed + renormalized.
    inner_non_null = [lv for lv in inner_spec.levels if not lv.is_null and lv.test is not None]
    product = []
    for inner_lv in inner_non_null:
        for (lo_s, hi_s, w) in time_band_secs:
            band_test = _time_band(lo_s / DAY, hi_s / DAY, lo_s, hi_s)
            def combined(pv, cache, inner=inner_lv.test, band=band_test):
                return inner(pv, cache) & band(pv, cache)
            product.append({
                "label_for_charts": f"{inner_lv.label} & t in [{lo_s/DAY:.0f},{hi_s/DAY:.0f}]d",
                "test": combined,
                "m_probability": (inner_lv.m if inner_lv.m is not None else 0.0) * w,
                "u_probability": inner_lv.u,
            })
    total_m = sum(entry["m_probability"] for entry in product)
    if total_m > 0:
        for entry in product:
            entry["m_probability"] = entry["m_probability"] / total_m

    levels += product

    # 4. ELSE catch-all (inner's else, or a fresh one).
    else_lvs = [lv for lv in inner_spec.levels if lv.test is None]
    if else_lvs:
        levels.append({"label_for_charts": else_lvs[0].label, "test": None})
    else:
        levels.append({"label_for_charts": "All other comparisons", "test": None})

    return build_spec(
        output_column_name=f"{inner_spec.output_column_name}_decayed",
        level_dicts=levels,
        fields=tuple(inner_spec.fields) + (time_col,),
        prescore=prescore,
    )


def time_decayed_comparison_builder(
    comparison: Any,
    time_col: str = "event_date",
    bands: Optional[Sequence[tuple[float, float, float]]] = None,
    col_name: Optional[str] = None,
    keep_inner_null: bool = False,
) -> ComparisonSpec:
    """Named-buildable wrapper: wrap ``comparison`` (a spec, Comparison, or a
    registered comparison name) with time-band decay.  ``time_col`` names the
    timestamp field present in the scored records; ``col_name`` is forwarded to
    ``comparison`` when it is a registered name string and requires one."""
    from ._comparison import make_comparison as _make

    if isinstance(comparison, str):
        kwargs = {"col_name": col_name} if col_name else {}
        comparison = _make(comparison, **kwargs)
    return time_decay_wrapper(
        comparison,
        time_col=time_col,
        bands=bands,
        keep_inner_null=keep_inner_null,
    )