"""Comparison registry: name-keyed registration of every comparison family.

Holds :class:`ComparisonRegistry`, the module-level :data:`REGISTRY`, the
built-in descriptions, and the single ``_register_built_ins`` manifest that
wires every family factory into the registry under its public name.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

from ._core import ComparisonSpec
from ._distance_families import (
    array_intersect_at_sizes_spec,
    cosine_similarity_at_thresholds_spec,
    damerau_levenshtein_at_thresholds_spec,
    distance_function_at_thresholds_spec,
    exact_match_spec,
    jaccard_at_thresholds_spec,
    jaro_at_thresholds_spec,
    jaro_winkler_at_thresholds_spec,
    levenshtein_at_thresholds_spec,
    pairwise_string_distance_function_at_thresholds_spec,
)
from ._temporal_families import (
    absolute_date_difference_at_thresholds_spec,
    absolute_time_difference_at_thresholds_spec,
    date_of_birth_comparison_spec,
)
from ._structural_families import (
    distance_in_km_at_thresholds_spec,
    email_comparison_spec,
    forename_surname_comparison_spec,
    name_comparison_spec,
    postcode_comparison_spec,
)
from ._custom import custom_comparison_spec, time_decayed_comparison_builder


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
        # Imported lazily to avoid a module cycle (`_comparison` imports this
        # module for REGISTRY, and this method needs the Comparison class).
        from ._comparison import Comparison

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
    "time_decayed_comparison": "Wraps another comparison, weighting its levels by a time-distance band.",
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
        defaults={"invalid_postcodes_as_null": False, "km_thresholds": [1, 10, 100],
                  "country": "UK"})
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
    R.register(
        "time_decayed_comparison",
        time_decayed_comparison_builder,
        fields=(),
        description=_DESCRIPTIONS["time_decayed_comparison"],
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