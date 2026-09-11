"""Bridging Splink-trained settings into a native FellegiSunterScorer."""

from __future__ import annotations

from typing import Any, Optional, Sequence

from ..comparisons import Comparison, make_comparisons
from ._constants import DEFAULT_PRIOR, DEFAULT_THRESHOLD
from ._scorer import FellegiSunterScorer


def import_splink_scorer(
    splink_settings: dict,
    native_comparisons: Sequence[Any],
    *,
    threshold: float = DEFAULT_THRESHOLD,
    idempotent: bool = True,
    base_records: Optional[Sequence[dict]] = None,
) -> FellegiSunterScorer:
    """Build a runnable ``FellegiSunterScorer`` from m/u (and TF) trained by *Splink*.

    Splink-trains ``m_probability``/``u_probability`` (and optional
    ``tf_adjustment_weight`` / ``tf_minimum_u_value``) per comparison level.
    This framework evaluates the same comparison *family* natively (see
    :mod:`vectorer.comparisons`), but the level predicates are not Splink SQL
    -- so Splink's settings JSON cannot be loaded byte-for-byte.  This helper
    bridges that gap: for each native comparison you supply, it finds Splink's
    matching comparison by ``output_column_name`` and transfers the per-level
    ``m``/``u`` and TF metadata onto the native levels, preserving the level
    ordering (null -> exact -> fuzzy -> else).

    Parameters
    ----------
    splink_settings:
        A Splink settings dict (e.g. the JSON produced by
        ``splink.linker.Linker.misc.save_model_to_json()``).  Its
        ``comparisons`` are read for per-level ``m_probability``,
        ``u_probability`` and TF fields; ``probability_two_random_records_match``
        is used as the prior.
    native_comparisons:
        The framework's comparison set over the **same columns and thresholds**
        as the Splink model (``make_comparison`` objects).  Levels are matched
        by ``output_column_name``.
    threshold, idempotent, base_records:
        Forwarded to :meth:`FellegiSunterScorer.from_settings` (``base_records``
        rebuilds the term-frequency value tables from your population).

    Returns
    -------
    A scorer with Splink's trained parameters, usable by every mode (batch,
    link, incremental) via ``scorer=``.
    """
    if not isinstance(splink_settings.get("comparisons"), list):
        raise ValueError(
            "splink_settings['comparisons'] must be a list of comparison dicts "
            "(e.g. from Linker.misc.save_model_to_json())"
        )

    # Index Splink's trained comparisons by their output column name.
    splink_by_col: dict[str, dict] = {}
    for entry in splink_settings["comparisons"]:
        if isinstance(entry, dict):
            col = entry.get("output_column_name")
            if col:
                splink_by_col[col] = entry
            continue
        # Allow resolved Splink comparison objects too.
        try:
            obj = entry.get_comparison("duckdb")
            splink_by_col[obj.output_column_name] = {
                "output_column_name": obj.output_column_name,
                "comparison_levels": [
                    {
                        "sql_condition": lv.sql_condition,
                        "m_probability": getattr(lv, "m_probability", None),
                        "u_probability": getattr(lv, "u_probability", None),
                        "tf_adjustment_weight": getattr(lv, "_tf_adjustment_weight", None),
                        "tf_minimum_u_value": getattr(lv, "_tf_minimum_u_value", None),
                        "tf_adjustment_column": getattr(
                            getattr(lv, "_tf_adjustment_column", None), "input_name", None
                        ),
                    }
                    for lv in obj.comparison_levels
                ],
            }
        except Exception:
            raise ValueError(
                "splink_settings['comparisons'] entries must be dicts or Splink "
                "comparison objects"
            ) from None

    resolved_comparisons: list[Comparison] = []
    for native in make_comparisons(list(native_comparisons)):
        spec = native.spec()
        splink = splink_by_col.get(spec.output_column_name)
        if splink is None:
            raise ValueError(
                f"no Splink-trained comparison matches native comparison "
                f"'{spec.output_column_name}'. The native comparison set must "
                f"use the same output column names as the Splink model."
            )
        splink_levels = splink.get("comparison_levels") or []
        if len(splink_levels) != len(spec.levels):
            raise ValueError(
                f"comparison '{spec.output_column_name}': Splink trained "
                f"{len(splink_levels)} levels but the native comparison has "
                f"{len(spec.levels)}. Align thresholds/columns between the two "
                f"models (level order must match: null -> exact -> fuzzy -> else)."
            )
        overrides = []
        for level, sl in zip(spec.levels, splink_levels):
            if level.is_null:
                overrides.append({})
                continue
            entry = {}
            m = sl.get("m_probability") if isinstance(sl, dict) else getattr(sl, "m_probability", None)
            u = sl.get("u_probability") if isinstance(sl, dict) else getattr(sl, "u_probability", None)
            if m is not None:
                entry["m_probability"] = float(m)
            if u is not None:
                entry["u_probability"] = float(u)
            tfw = sl.get("tf_adjustment_weight") if isinstance(sl, dict) else getattr(sl, "_tf_adjustment_weight", None)
            tfmu = sl.get("tf_minimum_u_value") if isinstance(sl, dict) else getattr(sl, "_tf_minimum_u_value", None)
            tfc = sl.get("tf_adjustment_column") if isinstance(sl, dict) else getattr(sl, "_tf_adjustment_column", None)
            if tfc is not None:
                entry["tf_adjustment_column"] = getattr(tfc, "input_name", tfc)
            if tfw is not None:
                entry["tf_adjustment_weight"] = float(tfw)
            if tfmu is not None:
                entry["tf_minimum_u_value"] = float(tfmu)
            overrides.append(entry)
        resolved = native.resolved()
        resolved["levels"] = overrides
        resolved_comparisons.append(Comparison.from_resolved(resolved))

    prior = float(
        splink_settings.get("probability_two_random_records_match", DEFAULT_PRIOR)
    )
    return FellegiSunterScorer.from_settings(
        {
            "comparisons": [c.resolved() for c in resolved_comparisons],
            "probability_two_random_records_match": prior,
            "idempotent": idempotent,
        },
        threshold=threshold,
        base_records=base_records,
        idempotent=idempotent,
    )