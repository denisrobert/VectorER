"""WeightTable: compiled comparison specs + TF tables + log bayes factors.

The scorer consumes a ``WeightTable``; rebuilding the table after
calibration/EM substitutes the newly trained ``m/u``.
"""

from __future__ import annotations

from typing import Optional, Sequence

from ..comparisons import ComparisonSpec
from ._constants import DEFAULT_PRIOR
from ._levels import _level_log_bayes_factors


class WeightTable:
    """Compiled comparison specs + term-frequency tables + log bayes factors."""

    def __init__(
        self,
        specs: Sequence[ComparisonSpec],
        prior: float = DEFAULT_PRIOR,
        base_records: Optional[Sequence[dict]] = None,
    ) -> None:
        self.prior = float(prior)
        self.specs = list(specs)
        self.fields: list[str] = []
        seen: set[str] = set()
        for spec in self.specs:
            for field in spec.fields:
                if field not in seen:
                    seen.add(field)
                    self.fields.append(field)
        self.log_bf = [_level_log_bayes_factors(spec) for spec in self.specs]
        self._tf_tables: dict[str, Optional[dict]] = {}
        self._tf_spec = {}  # id(spec) -> (tf_column, level_index, u_exact, weight, min_u)
        for spec in self.specs:
            for index, level in enumerate(spec.levels):
                if level.tf_column is not None:
                    self._tf_spec[id(spec)] = (
                        level.tf_column,
                        index,
                        level.u if level.u is not None else (level.m if level.m else 1.0),
                        level.tf_weight,
                        level.tf_min_u,
                    )
        if base_records is not None:
            for col in {entry[0] for entry in self._tf_spec.values()}:
                self._tf_tables[col] = _build_tf_table(col, base_records)

    def per_spec_tf(self, spec: ComparisonSpec) -> Optional[tuple]:
        return self._tf_spec.get(id(spec))


def _build_tf_table(col: str, base_records: Sequence[dict]) -> Optional[dict]:
    """value -> relative frequency over the reference population."""
    counts: dict = {}
    total = 0
    for record in base_records:
        value = record.get(col)
        if value is None:
            continue
        counts[value] = counts.get(value, 0) + 1
        total += 1
    if not counts:
        return None
    return {value: count / total for value, count in counts.items()}