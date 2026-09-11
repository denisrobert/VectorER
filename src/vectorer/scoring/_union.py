"""Union-Class lift for the Swoosh Union Class match function.

Records whose compared fields may hold ``set``/``frozenset`` values (a union
of alternative values) are scored under the existential lift: every pair is
expanded over its value combinations and the **maximum** posterior (and its
match weight) is returned -- ``M(r1, r2) = true iff some value pair matches``.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


class UnionClassMixin:
    """Union-Class (existential) scoring methods.

    Abstract: mixed into :class:`~vectorer.scoring.FellegiSunterScorer`; relies
    on the scalar scoring methods (``_scalar_posterior_pairs`` /
    ``_scalar_weight_pairs``) and ``self.table`` provided by the concrete class.
    """

    @staticmethod
    def _is_union_value(value: Any) -> bool:
        """A ``set``/``frozenset`` field value marks a union of alternatives.

        List/tuple values are NOT union values: they are comparison-column
        values (embedding vectors, tag lists) for the list-aware comparisons.
        """
        return isinstance(value, (set, frozenset))

    def _needs_union_pairs(
        self, left_records: Sequence[dict], right_records: Sequence[dict]
    ) -> bool:
        for left, right in zip(left_records, right_records):
            for field in self.table.fields:
                if self._is_union_value(left.get(field)) or self._is_union_value(right.get(field)):
                    return True
        return False

    def _union_expand(
        self, left: dict, right: dict, fields: Sequence[str]
    ) -> list[tuple[dict, dict]]:
        """Expand a ``(left, right)`` record pair whose compared fields may hold
        frozensets (union of alternative values) into scalar ``(left, right)``
        pairs covering every value combination.

        An empty value set is treated as missing (``None``), matching the null
        level semantics.  A pair with no set-valued compared field is returned
        unchanged as a single scalar pair.
        """
        set_fields = [
            field
            for field in fields
            if self._is_union_value(left.get(field)) or self._is_union_value(right.get(field))
        ]
        if not set_fields:
            return [(left, right)]
        expansions: list[tuple[dict, dict]] = [(left, right)]
        for field in set_fields:
            lv = left.get(field)
            rv = right.get(field)
            l_vals = list(lv) if self._is_union_value(lv) else [lv]
            r_vals = list(rv) if self._is_union_value(rv) else [rv]
            l_vals = l_vals if l_vals else [None]
            r_vals = r_vals if r_vals else [None]
            next_expansions: list[tuple[dict, dict]] = []
            for lrec, rrec in expansions:
                for x in l_vals:
                    for y in r_vals:
                        nl = dict(lrec)
                        nl[field] = x
                        nr = dict(rrec)
                        nr[field] = y
                        next_expansions.append((nl, nr))
            expansions = next_expansions
        return expansions

    def _score_union_pairs(
        self, left_records: Sequence[dict], right_records: Sequence[dict]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Posterior + match weight under the union (existential) lift.

        Every pair is expanded over its set-valued compared fields and scored
        scalarly; the **maximum** posterior (and its match weight) over the
        value combinations is returned -- the Union-Class ``M(r1,r2) = true iff
        some value pair matches`` semantics.
        """
        n = len(left_records)
        posterior = np.zeros(n, dtype=np.float64)
        weight = np.full(n, -np.inf, dtype=np.float64)
        for idx in range(n):
            rows = self._union_expand(left_records[idx], right_records[idx], self.table.fields)
            if not rows:
                continue
            lefts = [r[0] for r in rows]
            rights = [r[1] for r in rows]
            scores = self._scalar_posterior_pairs(lefts, rights)
            best = int(np.argmax(scores))
            posterior[idx] = float(scores[best])
            weight[idx] = float(self._scalar_weight_pairs([lefts[best]], [rights[best]])[0])
        return posterior, weight