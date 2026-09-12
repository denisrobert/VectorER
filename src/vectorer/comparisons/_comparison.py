"""Comparison objects: declared comparisons, construction and serialization.

A :class:`Comparison` is a light, JSON-serializable declaration (registry name
+ constructor kwargs) whose :meth:`spec` builds the :class:`ComparisonSpec` the
scorer evaluates.  The module also provides the public constructors
``make_comparison`` / ``make_comparisons`` and the declaration helpers
``comparison_set`` / ``comparison_to_dict`` / ``comparison_from_dict`` /
``comparison_fields``.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from ._core import ComparisonSpec
from ._registry import REGISTRY


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