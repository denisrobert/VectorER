"""Scalar inference core + the public scoring API.

The inference methods evaluate ``(query, candidate)`` / ``(left, right)`` pairs
end-to-end: pair-value extraction, the ``log(prior odds) + sum(log m/u)``
total-bayes core (with term-frequency multipliers), and the public
``score`` / ``score_batch`` / ``score_pairs`` / ``match_weight_*`` entry
points.  The Union-Class existence lift is provided by
:class:`~vectorer.scoring._union.UnionClassMixin`; the scalar primitives here
are its building blocks.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..comparisons import PairValues
from ._constants import _LOG_CLIP
from ._levels import _assign_levels
from ._math import _sigmoid, _values_equal


class InferenceMixin:
    """Scalar inference + pair-value extraction.

    Abstract: mixed into :class:`~vectorer.scoring.FellegiSunterScorer`; reads
    ``self.table`` / ``self.comparisons`` / ``self.idempotent`` provided by the
    concrete class and delegates to the Union-Class methods for set-valued
    records.
    """

    # -- pair value extraction ----------------------------------------------

    def _candidate_pair_values(self, left: dict, candidates: Sequence[dict]) -> PairValues:
        n = len(candidates)
        left_arrays = {}
        for field in self.table.fields:
            arr = np.empty(n, dtype=object)
            arr.fill(left.get(field))  # fill references the same value (lists included)
            left_arrays[field] = arr
        right_arrays = {
            field: np.array([c.get(field) for c in candidates], dtype=object)
            for field in self.table.fields
        }
        return PairValues(left_arrays, right_arrays)

    def _record_pair_values(
        self, left_records: Sequence[dict], right_records: Sequence[dict]
    ) -> PairValues:
        left_arrays = {
            field: np.array([r.get(field) for r in left_records], dtype=object)
            for field in self.table.fields
        }
        right_arrays = {
            field: np.array([r.get(field) for r in right_records], dtype=object)
            for field in self.table.fields
        }
        return PairValues(left_arrays, right_arrays)

    # -- core evaluation ----------------------------------------------------

    def _log_total_bayes(self, pv: PairValues) -> np.ndarray:
        """``log(prior odds) + sum over comparisons of log(level m/u)``."""
        total = np.zeros(pv.n, dtype=np.float64)
        for spec in self.table.specs:
            assigned = _assign_levels(spec, pv)
            is_null = np.array(
                [lv.is_null for lv in spec.levels], dtype=bool
            )[assigned]
            total += np.where(
                is_null,
                0.0,
                np.take(self.table.log_bf[self.table.specs.index(spec)], assigned),
            )
            tf = self.table.per_spec_tf(spec)
            if tf is not None:
                total += self._tf_log_multiplier(spec, tf, assigned, pv)
        combined = np.clip(self._log_prior_bf + total, -_LOG_CLIP, _LOG_CLIP)
        return combined

    def _tf_log_multiplier(self, spec, tf, assigned: np.ndarray, pv: PairValues) -> np.ndarray:
        col, level_index, u_exact, weight, min_u = tf
        tf_table = self.table._tf_tables.get(col)
        if tf_table is None:
            return np.zeros(assigned.shape, dtype=np.float64)
        left = pv.left(col)
        right = pv.right(col)
        tf_l = np.array([tf_table.get(v, np.nan) for v in left], dtype=np.float64)
        tf_r = np.array([tf_table.get(v, np.nan) for v in right], dtype=np.float64)
        present_l = np.isfinite(tf_l)
        present_r = np.isfinite(tf_r)
        present = present_l | present_r
        divisor = np.where(present_l & present_r, np.maximum(tf_l, tf_r),
                           np.where(present_l, tf_l, tf_r))
        if min_u > 0:
            divisor = np.maximum(divisor, float(min_u))
        divisor = np.maximum(divisor, 1e-12)
        exact = assigned == level_index
        ratio = u_exact / divisor if u_exact is not None else np.ones_like(divisor)
        log_mult = np.where(exact & present, float(weight) * np.log(ratio), 0.0)
        return log_mult

    def _combined_bayes(self, pv: PairValues) -> np.ndarray:
        return np.exp(self._log_total_bayes(pv))

    # -- scalar core (no union expansion) -----------------------------------

    def _scalar_posterior_batch(self, left: dict, candidates: Sequence[dict]) -> np.ndarray:
        pv = self._candidate_pair_values(left, candidates)
        posterior = _sigmoid(self._log_total_bayes(pv))
        if self.idempotent:
            posterior = self._apply_idempotence(posterior, pv)
        return posterior

    def _scalar_weight_batch(self, left: dict, candidates: Sequence[dict]) -> np.ndarray:
        return self._log_total_bayes(self._candidate_pair_values(left, candidates)) / np.log(2.0)

    def _scalar_posterior_pairs(
        self, left_records: Sequence[dict], right_records: Sequence[dict]
    ) -> np.ndarray:
        pv = self._record_pair_values(left_records, right_records)
        posterior = _sigmoid(self._log_total_bayes(pv))
        if self.idempotent:
            posterior = self._apply_idempotence(posterior, pv)
        return posterior

    def _scalar_weight_pairs(
        self, left_records: Sequence[dict], right_records: Sequence[dict]
    ) -> np.ndarray:
        return self._log_total_bayes(self._record_pair_values(left_records, right_records)) / np.log(2.0)

    # -- inference ----------------------------------------------------------

    def _self_equal_mask(self, pv: PairValues) -> np.ndarray:
        """True where a pair's compared fields are content-identical.

        Compares only ``table.fields`` (the columns the comparison set reads),
        so two records that differ on non-compared attributes still count as
        identical for matching -- the matcher is defined on the compared
        columns.  ``None``-vs-scalar is never equal.
        """
        mask = np.ones(pv.n, dtype=bool)
        for field in self.table.fields:
            if not mask.any():
                break
            l = pv.left(field)
            r = pv.right(field)
            for i in np.flatnonzero(mask):
                if not _values_equal(l[i], r[i]):
                    mask[i] = False
        return mask

    def score(self, left: dict, right: dict) -> float:
        """Match probability for one (query, candidate) pair."""
        return float(self.score_batch(left, [right])[0])

    def score_and_weight_batch(
        self, left: dict, candidates: Sequence[dict]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Posterior + match weight per candidate from a single model evaluation."""
        if not candidates:
            return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
        if self._needs_union_pairs([left] * len(candidates), candidates):
            return self._score_union_pairs([left] * len(candidates), candidates)
        pv = self._candidate_pair_values(left, candidates)
        log_total = self._log_total_bayes(pv)
        posterior = _sigmoid(log_total)
        if self.idempotent:
            posterior = self._apply_idempotence(posterior, pv)
        return posterior, log_total / np.log(2.0)

    def _apply_idempotence(self, posterior: np.ndarray, pv: PairValues) -> np.ndarray:
        """Force posterior 1.0 for content-identical pairs (reflexivity)."""
        equal = self._self_equal_mask(pv)
        if equal.any():
            posterior = posterior.copy()
            posterior[equal] = 1.0
        return posterior

    def score_batch(self, left: dict, candidates: Sequence[dict]) -> np.ndarray:
        """Posterior per candidate, aligned with ``candidates``.

        Set-valued compared fields (a Union-Class record) are scored under the
        existence lift: the maximum posterior over the cross value pairs.
        """
        if not candidates:
            return np.asarray([], dtype=np.float64)
        if self._needs_union_pairs([left] * len(candidates), candidates):
            posterior, _ = self._score_union_pairs([left] * len(candidates), candidates)
            return posterior
        return self._scalar_posterior_batch(left, candidates)

    def match_weight_batch(self, left: dict, candidates: Sequence[dict]) -> np.ndarray:
        """``match_weight = log2(total bayes factor)`` per candidate."""
        if not candidates:
            return np.asarray([], dtype=np.float64)
        if self._needs_union_pairs([left] * len(candidates), candidates):
            _, weight = self._score_union_pairs([left] * len(candidates), candidates)
            return weight
        return self._scalar_weight_batch(left, candidates)

    def score_pairs(self, left_records: Sequence[dict], right_records: Sequence[dict]) -> np.ndarray:
        """Vectorised posterior for equal-length ``left`` / ``right`` sequences.

        Set-valued compared fields (Union-Class records) are scored under the
        existence lift (max over cross value pairs).
        """
        if not left_records:
            return np.asarray([], dtype=np.float64)
        if self._needs_union_pairs(left_records, right_records):
            posterior, _ = self._score_union_pairs(left_records, right_records)
            return posterior
        return self._scalar_posterior_pairs(left_records, right_records)

    def match_weight_pairs(self, left_records: Sequence[dict], right_records: Sequence[dict]) -> np.ndarray:
        """Vectorised match weights for equal-length sequences."""
        if not left_records:
            return np.asarray([], dtype=np.float64)
        if self._needs_union_pairs(left_records, right_records):
            _, weight = self._score_union_pairs(left_records, right_records)
            return weight
        return self._scalar_weight_pairs(left_records, right_records)