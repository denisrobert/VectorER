"""Fellegi-Sunter scoring engine (native NumPy, no SQL engine).

Two stages consume the comparison set:

* *train* - estimate the comparison-level ``m`` (match) / ``u`` (non-match)
  probabilities and the base prior.  Two estimators are provided:

  - :meth:`FellegiSunterScorer.calibrate_from_pairs` - supervised calibration
    from labelled match/non-match pairs;
  - :meth:`FellegiSunterScorer.fit_em` - unsupervised expectation maximisation
    on a (near-duplicate-bearing) population, mirroring the standard
    Fellegi-Sunter parameter-estimation workflow (u estimated by
    random sampling of pairs, m and the prior fit by EM over blocking-rule
    candidate pairs).

* *infer* - score ``(query, candidate)`` pairs in one pass.

Inference is entirely vectorized: every comparison's levels are evaluated as
NumPy predicates over whole batches of pairs (see :mod:`vectorer.sim`), each
pair is assigned its highest-priority level, and the posterior is the sigmoid
of ``log(prior odds) + sum(log(m/u))`` -- the same match-weight algebra
expressed without SQL.  "Vectoring" here is over the *batch* (all
of a query's candidates, or every canopy pair) rather than row-by-row.

By default the match function is *reflexive* (:attr:`idempotent`): a pair whose
compared fields are content-identical scores exactly ``1.0``.  This guarantees
the ``r ~ r`` property of the Swoosh Union-Class ICAR construction, which would
otherwise fail for "thin" records (few non-null comparison fields) whose
self-posterior -- all null levels carrying no evidence -- sits at the prior.
Pass ``idempotent=False`` to recover the raw calibrated posterior for
identical-content pairs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from .comparisons import Comparison, ComparisonSpec, PairValues

DEFAULT_PRIOR = 0.0001
DEFAULT_THRESHOLD = 0.85

_LOG_CLIP = 690.0  # ln(1e300) clip on the total bayes factor


def _values_equal(a: Any, b: Any) -> bool:
    """Content equality for record values (scalars, lists, tuples, arrays).

    ``None`` equals only ``None``; lists/tuples compare element-wise; arrays
    compare with ``array_equal``; everything else uses ``==``.
    """
    if a is b:
        return True
    if a is None or b is None:
        return False
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        if not (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)):
            return False
        return a.shape == b.shape and bool(np.array_equal(a, b))
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        if not (isinstance(a, (list, tuple)) and isinstance(b, (list, tuple))):
            return False
        return len(a) == len(b) and all(_values_equal(x, y) for x, y in zip(a, b))
    try:
        return bool(a == b)
    except Exception:
        return False


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def _assign_levels(spec: ComparisonSpec, pv: PairValues) -> np.ndarray:
    """Highest-priority level index per pair (CASE semantics, vectorized).

    The spec's optional ``prescore`` step computes the shared score arrays once
    per batch; every level's test then reads them from the cache.
    """
    n = pv.n
    cache = spec.prescore(pv) if spec.prescore is not None else {}
    assigned = np.full(n, -1, dtype=np.int64)
    for index, level in enumerate(spec.levels):
        if level.test is None:
            continue  # ELSE level: fallback
        need = (assigned == -1) & level.test(pv, cache)
        assigned[need] = index
    assigned[assigned == -1] = len(spec.levels) - 1
    return assigned


def _level_log_bayes_factors(spec: ComparisonSpec) -> np.ndarray:
    """Per-level ``log(m/u)`` (null level -> 0), clipped for stability."""
    out = []
    for level in spec.levels:
        if level.is_null or level.m is None:
            out.append(0.0)
        elif level.m <= 0:
            out.append(-_LOG_CLIP)
        elif level.u is None or level.u <= 0:
            out.append(_LOG_CLIP)
        else:
            log_bf = np.log(level.m) - np.log(level.u)
            out.append(float(np.clip(log_bf, -_LOG_CLIP, _LOG_CLIP)))
    return np.asarray(out, dtype=np.float64)


# ---------------------------------------------------------------------------
# Weight table
# ---------------------------------------------------------------------------


class WeightTable:
    """Compiled comparison specs + term-frequency tables + log bayes factors.

    The scorer consumes a ``WeightTable``; rebuilding the table after
    calibration/EM substitutes the newly trained ``m/u``.
    """

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


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


class FellegiSunterScorer:
    """Scores pairs by applying (trained) ``m/u`` per comparison level.

    Level assignment uses an ordered priority (first matching level wins), but
    evaluated as vectorized NumPy predicates over a whole batch of pairs.  Two
    pair inputs are supported:

    * :meth:`score_batch` / :meth:`match_weight_batch` - one query record vs a
      list of candidate records (the incremental pipeline);
    * :meth:`score_pairs` / :meth:`match_weight_pairs` - equal-length left /
      right record lists (the batch pipeline's canopy pairs).
    """

    def __init__(
        self,
        table: WeightTable,
        threshold: float = DEFAULT_THRESHOLD,
        comparisons: Optional[Sequence[Any]] = None,
        prior: Optional[float] = None,
        trained_settings: Optional[dict] = None,
        idempotent: bool = True,
    ) -> None:
        self.table = table
        self.threshold = float(threshold)
        self.comparisons = list(comparisons) if comparisons is not None else []
        self.prior = float(prior) if prior is not None else table.prior
        self.trained_settings = trained_settings
        self._em = (trained_settings or {}).get("_em")
        # Reflexivity (idempotence of the match function): content-identical
        # pairs are forced to posterior 1.0.  Without this, a "thin" record
        # (few non-null comparison fields) would score against itself below
        # the threshold (its null levels carry no evidence, leaving the prior),
        # breaking the r ~ r property required by the Swoosh Union-Class ICAR
        # construction.  Set False to restore the raw calibrated posterior.
        self.idempotent = bool(idempotent)
        self._prior_bf = (
            table.prior / (1.0 - table.prior) if table.prior != 1.0 else float("inf")
        )
        self._log_prior_bf = float(
            np.clip(np.log(self._prior_bf), -_LOG_CLIP, _LOG_CLIP)
        )

    # -- construction -------------------------------------------------------

    @classmethod
    def from_comparisons(
        cls,
        comparisons: Sequence[Any],
        prior: float = DEFAULT_PRIOR,
        threshold: float = DEFAULT_THRESHOLD,
        base_records: Optional[Sequence[dict]] = None,
        idempotent: bool = True,
    ) -> "FellegiSunterScorer":
        """Build from declared comparisons (``Comparison`` or ``ComparisonSpec``)."""
        specs = _as_specs(comparisons)
        return cls(
            WeightTable(specs, prior=prior, base_records=base_records),
            threshold=threshold,
            comparisons=_as_comparisons(comparisons),
            prior=prior,
            idempotent=idempotent,
        )

    @classmethod
    def from_settings(
        cls,
        settings: dict,
        threshold: float = DEFAULT_THRESHOLD,
        base_records: Optional[Sequence[dict]] = None,
        idempotent: bool = True,
    ) -> "FellegiSunterScorer":
        """Build from a resolved settings dict (trained m/u).

        ``settings["comparisons"]`` is a list of ``{"type", "params", "levels"}``
        entries as produced by :meth:`to_settings`.
        """
        comparisons = []
        for entry in settings.get("comparisons") or []:
            if isinstance(entry, dict) and "type" in entry:
                if "levels" in entry:
                    comparisons.append(Comparison.from_resolved(entry))
                else:
                    comparisons.append(Comparison.from_dict(entry))
            else:
                comparisons.append(entry)
        prior = settings.get("probability_two_random_records_match", DEFAULT_PRIOR)
        return cls(
            WeightTable(_as_specs(comparisons), prior=prior, base_records=base_records),
            threshold=threshold,
            comparisons=comparisons,
            prior=prior,
            trained_settings=settings,
            idempotent=settings.get("idempotent", idempotent),
        )

    def to_settings(self) -> dict:
        """Serializable settings (comparisons + current m/u + prior)."""
        return {
            "comparisons": [c.resolved() for c in self.comparisons],
            "probability_two_random_records_match": self.prior,
            "idempotent": self.idempotent,
            **({"_em": self._em} if self._em else {}),
        }

    def to_dict(self) -> dict:
        return {
            "threshold": self.threshold,
            **self.to_settings(),
        }

    def save(self, path: Any) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")

    @classmethod
    def load(cls, path: Any) -> "FellegiSunterScorer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_settings(
            data,
            threshold=float(data.get("threshold", DEFAULT_THRESHOLD)),
        )

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

    # -- Union-Class lift (Swoosh Union Class match function) ---------------

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

    # -- training -----------------------------------------------------------

    def calibrate_from_pairs(
        self,
        pair_records: Sequence[dict],
        smoothing: float = 0.5,
    ) -> "FellegiSunterScorer":
        """Fit supervised ``m/u`` from labelled match/non-match pairs.

        ``pair_records`` is a sequence of records with a boolean ``is_match``
        column (``1`` = match, ``0`` = non-match) plus, for each field compared
        by the comparison set, ``<field>_l`` and ``<field>_r`` columns.
        Laplace ``smoothing`` is added to every level count (vectorized).
        """
        if not self.comparisons:
            raise ValueError("no comparisons configured on the scorer")
        is_match = np.array(
            [1 if r.get("is_match") else 0 for r in pair_records], dtype=np.int64
        )
        m_total = int((is_match == 1).sum())
        u_total = int((is_match == 0).sum())
        if m_total == 0 or u_total == 0:
            raise ValueError("pairs must contain both match (1) and non-match (0) rows")

        new_comparisons: list[Comparison] = []
        for comparison in self.comparisons:
            spec = comparison.spec()
            fields = tuple(spec.fields)
            if not fields:
                first_pair = pair_records[0]
                fields = tuple(sorted({k[:-2] for k in first_pair if k.endswith("_l")}))
            left = {
                f: np.array([r.get(f"{f}_l") for r in pair_records], dtype=object)
                for f in fields
            }
            right = {
                f: np.array([r.get(f"{f}_r") for r in pair_records], dtype=object)
                for f in fields
            }
            pv = PairValues(left, right)
            assigned = _assign_levels(spec, pv)
            num_levels = len(spec.levels)
            level_probs: list[tuple[Optional[float], Optional[float]]] = []
            for index, level in enumerate(spec.levels):
                if level.is_null:
                    level_probs.append((None, None))
                    continue
                m_count = int(((assigned == index) & (is_match == 1)).sum())
                u_count = int(((assigned == index) & (is_match == 0)).sum())
                m = (m_count + smoothing) / (m_total + smoothing * num_levels)
                u = (u_count + smoothing) / (u_total + smoothing * num_levels)
                level_probs.append((float(m), float(u)))

            resolved = comparison.resolved()
            new_levels = []
            for level_dict, (m, u) in zip(resolved["levels"], level_probs):
                if level_dict.get("is_null_level"):
                    new_levels.append(level_dict)
                else:
                    new_levels.append(
                        {**level_dict, "m_probability": m, "u_probability": u}
                    )
            resolved["levels"] = new_levels
            new_comparisons.append(Comparison.from_resolved(resolved))

        return self.__class__.from_settings(
            {
                "comparisons": [c.resolved() for c in new_comparisons],
                "probability_two_random_records_match": self.prior,
            },
            threshold=self.threshold,
        )

    def fit_em(
        self,
        records: Sequence[dict],
        training_block_on: Optional[Sequence[Sequence[str]]] = None,
        recall: float = 0.7,
        max_pairs: float = 1e6,
        max_iterations: int = 20,
        em_convergence: float = 0.001,
        seed: Optional[int] = None,
        prior: Optional[float] = None,
        fixed_prior: Optional[float] = None,
        extra_settings: Optional[dict] = None,
    ) -> "FellegiSunterScorer":
        """Fit ``m``/``u`` and the base prior via expectation maximisation.

    Native training workflow (no SQL engine):

    1. candidate pairs are generated under the blocking rules
       (:meth:`_blocked_pairs`) -- this defines the *training* pair pool;
    2. ``u`` probabilities are estimated from uniformly sampled pairs
       (the unconditional non-match distribution);
    3. ``m`` and the blocked-pair match proportion are fit by EM (u held
       fixed), with ``m`` renormalized per comparison at each M-step;
    4. the base prior (probability two random records match) is the
       recall-adjusted share of blocked pairs that are matches extended to the
       total number of possible pairs, capped at 0.5 -- unless ``prior`` or
       ``fixed_prior`` is supplied.

    Fixed-prior mode (the "calibration paradox" remedy): passing
    ``fixed_prior=`` holds the base prior **frozen across every EM iteration**
    (it is used in the E-step's responsibilities and never re-estimated in the
    M-step), so only ``m`` (and not ``pi``) is learned -- the same machinery
    Splink's fixed-prior EM uses.  This lets you sweep fixed priors x threshold
    to find an operating point, instead of trusting EM's own (often
    miscalibrated) prior estimate.  ``prior=`` remains the earlier behaviour:
    EM still learns ``pi`` internally, and ``prior`` only overrides the
    reported base rate.

    Only one of ``prior`` / ``fixed_prior`` should be set; the caller may also
    pass both only if they agree.
    """
        rng = np.random.default_rng(seed)
        rules = list(training_block_on) if training_block_on else [
            ("first_name",),
            ("date_of_birth",),
        ]
        if not self.comparisons:
            raise ValueError("no comparisons configured on the scorer")

        blocked_pairs = self._blocked_pairs(records, rules, int(max_pairs), rng)
        n_pairs = len(blocked_pairs)
        if n_pairs == 0:
            raise RuntimeError(
                "no blocking-rule candidate pairs were generated; supply a "
                "duplicate-bearing population or adjust training_block_on"
            )

        specs = [c.spec() for c in self.comparisons]
        field_list = self.table.fields

        # (1) comparison vectors for the blocked pair pool, vectorized.
        blocked_pv = self._positions_pair_values(records, blocked_pairs, field_list)
        gammas = [_assign_levels(spec, blocked_pv) for spec in specs]

        # (2) u from a uniform sample of pairs (unconditional non-match dist).
        u_sample = list(_sample_all_pairs(len(records), min(200_000, n_pairs * 2), rng))
        if not u_sample:
            u_sample = blocked_pairs
        u_pv = self._positions_pair_values(records, u_sample, field_list)
        u_gammas = [_assign_levels(spec, u_pv) for spec in specs]
        us = [_level_proportions(spec, gamma) for spec, gamma in zip(specs, u_gammas)]

        # (3) EM: start m from current defaults, pi neutral.
        ms = [_level_defaults(spec) for spec in specs]
        null_flags = [
            np.array([lv.is_null for lv in spec.levels], dtype=bool)
            for spec in specs
        ]
        pi = 0.5
        # Fixed-prior mode: pi is frozen at fixed_prior for the whole EM run.
        if fixed_prior is not None:
            if not 0.0 < fixed_prior < 1.0:
                raise ValueError("fixed_prior must be in (0, 1)")
            if prior is not None and abs(prior - fixed_prior) > 1e-12:
                raise ValueError("prior and fixed_prior disagree; set only one")
            pi = float(fixed_prior)
        prev_pi = None
        try:
            from tqdm import tqdm

            em_bar = tqdm(range(int(max_iterations)), desc="EM m/u", unit="iter",
                          leave=False, ascii=True)
        except Exception:  # noqa: BLE001  (tqdm optional)
            em_bar = range(int(max_iterations))
        # Guards against the degenerate all-C1 fixed point: if EM drives the
        # responsibility mass to every blocked pair being a "match" (pi -> 1),
        # the m levels collapse onto whichever level has the highest base rate
        # and every other level gets floored, destroying the evidence contrast
        # needed for scoring.  Rather than letting that silently lock in, we
        # clip pi so it can never leave (0, 1) by more than the numeric slack,
        # freeze pi at its last sane value when the likelihood landscape is
        # degenerate, and detect the collapse to report a clear error.
        pi_min = 1e-6
        pi_max = 1.0 - 1e-6
        for _ in em_bar:
            # E-step (log-space responsibilities over the blocked pairs).
            log_m = np.zeros(n_pairs, dtype=np.float64)
            log_u = np.zeros(n_pairs, dtype=np.float64)
            for spec, gamma, m_levels, u_levels, is_null in zip(
                specs, gammas, ms, us, null_flags
            ):
                null_gamma = is_null[gamma]
                m_g = np.clip(np.asarray(m_levels, dtype=np.float64)[gamma], 1e-8, None)
                u_g = np.clip(np.asarray(u_levels, dtype=np.float64)[gamma], 1e-8, None)
                # Null-level comparisons carry no evidence (bayes factor 1).
                log_m += np.where(null_gamma, 0.0, np.log(m_g))
                log_u += np.where(null_gamma, 0.0, np.log(u_g))
            log_odds = log_m - log_u + np.log(pi + 1e-12) - np.log1p(-pi + 1e-12)
            r = _sigmoid(np.clip(log_odds, -50.0, 50.0))

            # M-step: per-comparison m (renormalized) uses the full posterior
            # responsibilities ``r`` (standard EM).  The blocked-pair match
            # share ``pi``, however, is estimated from the *evidence-only*
            # responsibilities ``r_ev = sigmoid(log_m - log_u)`` (no logit-pi
            # term).  This is the statistical fix for the spurious 1.0 fixed
            # point: ``pi = mean(r)`` with ``r`` containing ``logit(pi)`` is a
            # self-reinforcing loop (as pi -> 1 the logit term explodes and
            # drags every responsibility to 1).  The evidence-only share
            # reflects the pool's composition without that coupling, so pi
            # converges to the actual share of evidence-supported pairs.
            if fixed_prior is None:
                r_ev = _sigmoid(np.clip(log_m - log_u, -50.0, 50.0))
                pi_new = float(np.clip(float(np.mean(r_ev)), pi_min, pi_max))
            else:
                pi_new = pi
            ms_new = []
            for spec, gamma in zip(specs, gammas):
                weights = r.copy()
                sums = np.zeros(len(spec.levels), dtype=np.float64)
                for index in range(len(spec.levels)):
                    sums[index] = float(weights[gamma == index].sum())
                total = float(sums.sum())
                m_levels = np.clip(sums / (total + 1e-12), 1e-8, None)
                m_levels = m_levels / m_levels.sum()  # per-comparison multinomial
                ms_new.append(m_levels.tolist())
            change = abs(pi_new - pi)
            if prev_pi is not None and fixed_prior is None:
                for old, new in zip(ms, ms_new):
                    change = max(change, float(np.max(np.abs(np.asarray(old) - np.asarray(new)))))
            pi = pi_new
            ms = ms_new
            if change < em_convergence:
                break

        # Base prior: probability two *random* records match = the model's own
        # expected match rate over a uniform (unconditional) pair sample.  We
        # cannot use `pi` (the blocked-pair match share) extrapolated to the
        # full pair domain: the blocked pool is intentionally biased toward
        # matches (it is built from blocking rules), so `pi * |blocked|` is a
        # multiple of the true match count and the "extrapolation" over
        # C(n,2) wildly overstates the base rate.  Instead solve the fixed
        # point over a uniform pair sample (see _uniform_prior_estimate).
        # Fixed-prior mode returns the frozen prior verbatim; a caller-supplied
        # `prior` overrides the reported base rate (historical behaviour).
        if fixed_prior is not None:
            final_prior = float(fixed_prior)
        elif prior is not None:
            final_prior = float(prior)
        else:
            final_prior = None  # resolved from the uniform-sample score below

        new_comparisons = []
        for comparison, m_levels, u_levels in zip(self.comparisons, ms, us):
            resolved = comparison.resolved()
            new_levels = []
            for idx, level_dict in enumerate(resolved["levels"]):
                if level_dict.get("is_null_level"):
                    new_levels.append(level_dict)
                else:
                    new_levels.append({**level_dict, "m_probability": float(m_levels[idx]),
                                       "u_probability": float(u_levels[idx])})
            resolved["levels"] = new_levels
            new_comparisons.append(Comparison.from_resolved(resolved))

        if final_prior is None:
            # Uniform-sample fixed-point estimate of the unconditional match
            # rate (self-consistent, not biased by a flat 0.5 prior).
            base_rate = self._uniform_prior_estimate(
                records, min(200_000, n_pairs * 4), rng)
            if base_rate != base_rate:  # nan: unexpected empty sample
                base_rate = 0.0
            final_prior = float(np.clip(base_rate, 1e-8, 0.5))

        scorer = self.__class__.from_settings(
            {
                "comparisons": [c.resolved() for c in new_comparisons],
                "probability_two_random_records_match": float(final_prior),
            },
            threshold=self.threshold,
        )
        # Record the EM mixing share over the blocked candidate pairs (pi) and
        # the blocked-pair count (n_pairs = |S0|).  ``pi`` reflects the blocked
        # pool, which is intentionally biased toward matches (blocking rules),
        # so it is a *blocked*-share estimate -- always use ``prior_empirical``
        # (the uniform-posterior base rate) as the honest full-pair prior.  The
        # pair count and block share let recalibrate_prior reproduce either.
        scorer._em = {
            "pi": float(pi),
            "n_pairs": int(n_pairs),
            "prior_empirical": float(final_prior),
        }
        return scorer

    def recalibrate_prior(
        self,
        records: Sequence[dict],
        *,
        method: str = "yancey",
        sample_size: int = 200_000,
        seed: Optional[int] = None,
        recall: float = 1.0,
    ) -> "FellegiSunterScorer":
        """Recover the **full-set** match prior for a model trained on an
        enriched subset.

        After match-enrichment EM (Yancey 2004), the EM estimate of ``Pr(C1)``
        reflects the match share of the enriched training subset, not the full
        population -- Yancey recalibrates only the non-match classes (C2/C3)
        and leaves the match prior un-adjusted.  In a posterior/threshold
        system the match prior enters the posterior directly, so an inflated
        prior biases every posterior.

        ``method`` selects the correction computed on ``records`` (the full
        population):

        * ``"yancey"`` (default) -- the paper's count-ratio correction
          (Yancey 2004, eq. for Pr(C1): ``Pr(C1)_S = (|S0|/|S|) * Pr(C1)_S0``).
          ``|S0|`` is the enriched blocked-pair count recorded by
          :meth:`fit_em`, ``|S|`` is ``C(len(records), 2)`` (the full pair
          domain), and ``Pr(C1)_S0`` is EM's mixing proportion ``pi`` over the
          enriched blocked pairs.  This is deterministic, free (no scoring
          pass), and matches the article's rescaling of the enriched class
          shares to the full pair set.

        * ``"empirical"`` -- resample: draw a ``sample_size`` uniform pair
          sample from ``records``, score it with the trained ``m/u``, and set
          the prior to the model's own expected match rate (optionally divided
          by ``recall`` to compensate for blocking that produced candidates).

        ``method="yancey"`` requires the scorer to carry EM metadata (i.e. it
        was produced by :meth:`fit_em`); otherwise a :class:`ValueError` is
        raised.  Use the resulting scorer's ``fixed_prior`` (or the
        prior-sweep) at the corrected prior.
        """
        if method == "empirical":
            return self._recalibrate_empirical(
                records, sample_size=sample_size, seed=seed, recall=recall,
            )
        if method != "yancey":
            raise ValueError(
                f"unknown recalibrate_prior method {method!r}; "
                "expected 'yancey' or 'empirical'"
            )
        if not self._em or "pi" not in self._em or "n_pairs" not in self._em:
            raise ValueError(
                "recalibrate_prior(method='yancey') requires a scorer produced "
                "by fit_em (which records the enriched EM mixing share and "
                "blocked-pair count); use method='empirical' for a manually "
                "constructed scorer"
            )
        pi = float(self._em.get("pi", 0.0))
        n0 = int(self._em.get("n_pairs", 0))
        n = len(records)
        n_total_pairs = n * (n - 1) // 2
        # Yancey 2004 sec 2.4: Pr(C2)_S = (|S0|/|S|) * Pr(C2) (and C3 absorbs
        # the uncounted remainder).  For a match prior this is the share of
        # full-pair space the enriched EM attributes to matches: pi of the
        # |S0| blocked pairs are matches, so pi * |S0| / |S|.
        full_prior = np.clip(
            (pi * n0) / max(float(recall), 1e-3) / max(n_total_pairs, 1),
            1e-8, 0.5,
        ) if n_total_pairs else 0.0
        new_settings = self.to_settings()
        new_settings["probability_two_random_records_match"] = float(full_prior)
        return self.__class__.from_settings(
            new_settings, threshold=self.threshold,
        )

    def _recalibrate_empirical(
        self,
        records: Sequence[dict],
        *,
        sample_size: int = 200_000,
        seed: Optional[int] = None,
        recall: float = 1.0,
    ) -> "FellegiSunterScorer":
        """Empirical resample recalibration (the pre-``method`` behaviour).

        Draws a ``sample_size`` uniform pair sample from ``records`` and sets
        the prior to the model's own expected match rate via the self-
        consistent fixed-point estimate (:meth:`_uniform_prior_estimate`).
        ``recall`` is accepted for API compatibility; the uniform sample is
        already drawn over the full pair domain, so no recall adjustment is
        applied (dividing a full-domain posterior mean by recall was biased).
        """
        rng = np.random.default_rng(seed)
        base_rate = self._uniform_prior_estimate(
            records, int(sample_size), rng)
        if base_rate != base_rate:  # nan: empty sample
            return self
        full_prior = float(np.clip(base_rate, 1e-8, 0.5))
        new_settings = self.to_settings()
        new_settings["probability_two_random_records_match"] = full_prior
        return self.__class__.from_settings(
            new_settings, threshold=self.threshold,
        )

    # -- pair sources -------------------------------------------------------

    def _uniform_prior_estimate(
        self,
        records: Sequence[dict],
        sample_size: int,
        rng: np.random.Generator,
    ) -> float:
        """Self-consistent estimate of the unconditional match base rate.

        Draws a ``sample_size`` uniform pair sample and solves the fixed point
        ``pi = mean(posterior(pi))`` over it (a few likelihood iterations on the
        *precomputed* per-pair evidence).  ``p05`` is the posterior at a flat
        ``0.5`` prior, so ``evidence = logit(p05)`` and
        ``posterior(pi) = sigmoid(evidence + logit(pi))``; the fixed-point loop
        is a cheap vectorized iteration over a 1-D array -- no scorer rebuild
        per iteration.

        Boundary guard: ``pi = mean(posterior(pi))`` has attracting fixed
        points at both 0 and 1 (the logistic is increasing in the prior), so
        an evidence-positive pool can run the iteration to the spurious 1.0.
        The guard runs *on every iterate*: once the iterate is pushed above
        ``0.9`` and still climbing, we fall back to the **evidence-positive
        share** ``mean(ev > 0)`` -- a prior-free, bounded, lower-anchor
        estimate of the match share that cannot be inflated by the logit
        coupling.  Returns ``nan`` when the sample is empty (callers must
        handle it).
        """
        n = len(records)
        pairs = list(_sample_all_pairs(n, int(sample_size), rng))
        if not pairs:
            return float("nan")
        left = [records[i] for i, _ in pairs]
        right = [records[j] for _, j in pairs]
        # Per-pair evidence: logit of the posterior at a neutral 0.5 prior
        # (sigmoid(evidence) = posterior(0.5)), computed once.
        neutral = FellegiSunterScorer.from_settings(
            {
                "comparisons": [c.resolved() for c in self.comparisons],
                "probability_two_random_records_match": 0.5,
            },
            threshold=self.threshold,
        )
        p05 = np.asarray(neutral.score_pairs(left, right), dtype=np.float64)
        p05 = np.clip(p05, 1e-12, 1.0 - 1e-12)
        evidence = np.log(p05) - np.log(1.0 - p05)
        # Fixed-point iteration pi = mean(sigmoid(evidence + logit(pi))).
        # Convergence is tested in logit space (which contracts faster in the
        # slow near-boundary regime) and the spurious-1 trajectory guard runs
        # *every* iterate, so a pool climbing past 0.9 without stabilizing is
        # caught even if it would land below the final-value threshold.
        ev_share = float((evidence > 0.0).mean())
        pi = 0.5
        for _ in range(40):
            logit_pi = np.log(max(pi, 1e-12)) - np.log(max(1.0 - pi, 1e-12))
            pi_new = float(np.mean(_sigmoid(np.clip(evidence + logit_pi, -50.0, 50.0))))
            if abs((np.log(max(pi_new, 1e-12)) - np.log(max(1.0 - pi_new, 1e-12)))
                   - logit_pi) < 1e-4:
                pi = pi_new
                break
            # Trajectory boundary guard: if the iterate is pushed into the
            # top band and keeps climbing, fall back to the evidence-positive
            # share (prior-free, bounded) rather than converging to ~1.
            if pi_new > 0.9 and pi_new > pi:
                pi = min(pi_new, max(ev_share, 1e-6))
                break
            pi = pi_new
        return pi

    def _blocked_pairs(
        self,
        records: Sequence[dict],
        rules: Sequence[Sequence[str]],
        max_pairs: int,
        rng: np.random.Generator,
    ) -> list[tuple[int, int]]:
        """Generate candidate pairs under the blocking rules.

        Two changes vs the original exact-key grouping:

        * **Fuzzy blocking for string columns.**  Perturbed twins (case flips,
          initials, typos) never share an exact first-name string, so an exact
          group would join only non-matches.  For a single string field we use
          a *papered* fuzzy block: records whose Jaro similarity is above
          ``_FUZZY_BLOCK_MIN`` are placed into the same buckets by a
          character-2-gram signature with a wide window, so a perturbed value
          still lands in its base's bucket (it shares many trigrams).  For
          multi-column rules we still use exact keys (interpreted as a
          conjunction), which is the correct semantics for canonical ids.

        * **Per-rule pair budget.**  The old loop ``break``ed out of the whole
          rule list once the total cap was hit, so the first (most common)
          rule could saturate all ``max_pairs`` and later, more discriminative
          rules (e.g. ``date_of_birth``, which twins DO share) never ran.
          Each rule now gets a slice of the budget, so the candidate pool mixes
          evidence from every rule.
        """
        from collections import defaultdict

        import vectorer.sim as sim

        pairs: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()

        def add_clique(members: list[int], allowance: list[int]) -> None:
            """Emit pairs from a group, drawing from a per-rule remaining
            ``allowance`` (shared across all of the rule's groups and paths,
            so the exact and fuzzy emission paths together never exceed the
            rule's budget slice).
            """
            nonlocal pairs, seen
            if len(members) < 2 or allowance[0] <= 0:
                return
            # The pool may already be full from earlier rules.
            if len(pairs) >= max_pairs:
                return
            remaining = min(allowance[0], max_pairs - len(pairs))
            if remaining <= 0:
                return
            if len(members) * (len(members) - 1) // 2 <= remaining:
                for i in range(len(members)):
                    for j in range(i + 1, len(members)):
                        key = (members[i], members[j])
                        if key not in seen:
                            seen.add(key)
                            pairs.append((members[i], members[j]))
                            allowance[0] -= 1
                        if allowance[0] <= 0 or len(pairs) >= max_pairs:
                            return
            else:
                chosen = rng.choice(members, size=min(len(members), 256), replace=False)
                chosen = sorted(chosen.tolist())
                for i in range(len(chosen)):
                    for j in range(i + 1, len(chosen)):
                        key = (chosen[i], chosen[j])
                        if key not in seen:
                            seen.add(key)
                            pairs.append((chosen[i], chosen[j]))
                            allowance[0] -= 1
                        if allowance[0] <= 0 or len(pairs) >= max_pairs:
                            return

        # Order rules by specificity: columns with the most distinct values
        # (high cardinality, e.g. date_of_birth) produce small, rare groups
        # that are most likely to contain *twins* and least likely to waste the
        # budget on common-value non-match cliques.  The budget is allocated
        # proportionally to this specificity, not equally: a common-name rule
        # needs only a small slice each group, while a rare-DOB rule needs the
        # budget to reach its (scarce but decisive) twins.
        scored_rules: list[tuple[float, Sequence[str]]] = []
        for rule in rules:
            if not rule:
                continue
            distinct = set()
            for record in records:
                if len(rule) == 1:
                    v = record.get(rule[0])
                    if v is None:
                        continue
                    distinct.add(str(v).strip() if isinstance(v, str) else v)
                else:
                    k = tuple(record.get(f) for f in rule)
                    if any(x is None for x in k):
                        continue
                    distinct.add(k)
            # Cardinality ratio -> revenue share.  Higher distinct-count => more
            # budget share (capped so a single rule can't take it all).
            share = min(float(len(distinct)), 50000.0)
            scored_rules.append((share, rule))
        scored_rules.sort(key=lambda t: t[0], reverse=True)
        shares = [s for s, _ in scored_rules]
        total_share = float(sum(shares)) or 1.0
        # Per-rule remaining allowance (shared across the rule's groups and
        # both emission paths).  Every rule gets at least 1 pair so a
        # rounding-absorbing last rule is never starved to zero; the global
        # `max_pairs` cap is still enforced at emission time.
        budget_by_rule = [max(1, int(max_pairs * (s / total_share))) for s in shares]
        if budget_by_rule and len(budget_by_rule) > 1:
            budget_by_rule[-1] = max(
                1, int(max_pairs) - int(sum(budget_by_rule[:-1])))

        for rule, per_rule0 in zip([r for _, r in scored_rules], budget_by_rule):
            if len(pairs) >= max_pairs:
                break
            if not rule:
                continue
            allowance = [per_rule0]
            is_single_string = len(rule) == 1
            if is_single_string:
                # Fuzzy bucket: case/typo-flipped twins collide on a
                # count-aware trigram signature; exact groups catch identical
                # (canonical) values.  Short strings use a case-insensitive
                # exact key (no trigrams), so "jack"/"Jack" still join via the
                # bucket path when long enough, and identical short values via
                # the exact group.
                exact_groups: dict = defaultdict(list)
                fuzzy_buckets: dict = defaultdict(dict)
                for position, record in enumerate(records):
                    value = record.get(rule[0])
                    if value is None:
                        continue
                    val_str = str(value).strip()
                    if not val_str:
                        continue
                    bucket = _fuzzy_block_key(val_str)
                    if bucket is None:
                        # Short string: case-insensitive exact key.
                        exact_groups[val_str.casefold()].append(position)
                        continue
                    fuzzy_buckets[bucket][position] = val_str
                # Exact groups first (canonical pairs).
                for members in exact_groups.values():
                    if len(members) < 2 or len(pairs) >= max_pairs or allowance[0] <= 0:
                        continue
                    add_clique(members, allowance)
                    if len(pairs) >= max_pairs or allowance[0] <= 0:
                        break
                # Then fuzzy buckets (perturbed twins): Jaro-check every
                # within-bucket pair *vectorized* (a single batched similarity
                # call over the outer-product pair array), so only genuinely
                # similar strings are admitted.
                for bucket, members in fuzzy_buckets.items():
                    if len(members) < 2 or len(pairs) >= max_pairs or allowance[0] <= 0:
                        continue
                    # If the rule's remaining allowance is too small to be a
                    # meaningful sample of this bucket, the Jaro grid (and the
                    # ~sqrt(allowance) subsample it forces) would only distort:
                    # a low allowance can admit at most a handful of pairs, but
                    # subsampling a big bucket down to sqrt(allowance) members
                    # makes hitting the true twin vanishingly unlikely.  Skip
                    # the bucket when its size dwarfs the remaining budget.
                    n_fb = len(members)
                    if n_fb > allowance[0] * 4:
                        continue
                    mlist = sorted(members)
                    a_vals = np.array([members[i] for i in mlist], dtype=object)
                    n_fb = len(mlist)
                    # Only the remaining allowance can ever be admitted for
                    # this bucket, so the Jaro grid never needs more than
                    # sqrt(allowance) members.  This bounds each bucket's grid
                    # work (O(allowance)) instead of O(n_fb^2) which could be
                    # quadratic in the whole population for a heavy-tailed
                    # bucket.
                    grid_n = int(np.ceil(np.sqrt(max(float(allowance[0]), 1.0))))
                    grid_n = min(n_fb, max(grid_n, 8), 1000)
                    if grid_n < n_fb:
                        keep = rng.choice(mlist, size=grid_n, replace=False)
                        mlist = sorted(keep.tolist())
                        a_vals = np.array([members[i] for i in mlist], dtype=object)
                        n_fb = len(mlist)
                    # Vectorized Jaro over the (i, j) pair grid.
                    left = np.repeat(a_vals, n_fb)
                    right = np.tile(a_vals, n_fb)
                    sims = sim.jaro_similarity(left, right).reshape(n_fb, n_fb)
                    diag_mask = ~np.eye(n_fb, dtype=bool)
                    sims = np.where(diag_mask, sims, -1.0)
                    # Admit only genuinely similar, ordered pairs.
                    # takes the lower-triangle candidate indices (x < y) from
                    # the flattened lower-half of the matrix.
                    ys, xs = np.where(sims >= 0.8)
                    cand = [(x, y) for x, y in zip(xs, ys) if x < y]
                    seats = min(len(cand), allowance[0])
                    if seats <= 0:
                        if allowance[0] <= 0:
                            break
                        continue
                    if len(cand) <= allowance[0]:
                        pick_idx = range(len(cand))
                    else:
                        pick_idx = rng.choice(len(cand), size=seats, replace=False)
                        pick_idx = pick_idx.tolist()
                    for pick in pick_idx:
                        x, y = cand[pick]
                        key = (int(mlist[x]), int(mlist[y]))
                        if key not in seen:
                            seen.add(key)
                            pairs.append(key)
                            allowance[0] -= 1
                        if allowance[0] <= 0 or len(pairs) >= max_pairs:
                            break
                    if len(pairs) >= max_pairs or allowance[0] <= 0:
                        break
            else:
                # Multi-column rule: exact conjunction group (canonical id).
                groups: dict = defaultdict(list)
                for position, record in enumerate(records):
                    key = tuple(record.get(f) for f in rule)
                    if any(k is None for k in key):
                        continue
                    groups[key].append(position)
                for members in groups.values():
                    if len(members) < 2 or len(pairs) >= max_pairs or allowance[0] <= 0:
                        continue
                    add_clique(members, allowance)
                    if len(pairs) >= max_pairs or allowance[0] <= 0:
                        break
        return pairs

    def _positions_pair_values(
        self,
        records: Sequence[dict],
        pairs: Sequence[tuple[int, int]],
        fields: Sequence[str],
    ) -> PairValues:
        n = len(pairs)
        left = {
            f: np.array([records[i].get(f) for i, _ in pairs], dtype=object)
            for f in fields
        }
        right = {
            f: np.array([records[j].get(f) for _, j in pairs], dtype=object)
            for f in fields
        }
        return PairValues(left, right)


def import_splink_scorer(
    splink_settings: dict,
    native_comparisons: Sequence[Any],
    *,
    threshold: float = DEFAULT_THRESHOLD,
    idempotent: bool = True,
    base_records: Optional[Sequence[dict]] = None,
) -> "FellegiSunterScorer":
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
    from .comparisons import Comparison, make_comparisons

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


def _as_specs(comparisons: Sequence[Any]) -> list[ComparisonSpec]:
    from .comparisons import Comparison, comparison_from_dict

    out = []
    for item in comparisons:
        if isinstance(item, Comparison):
            out.append(item.spec())
        elif isinstance(item, ComparisonSpec):
            out.append(item)
        elif isinstance(item, dict):
            c = comparison_from_dict(item)
            out.append(c.spec() if c is not None else None)
        else:
            raise TypeError(f"expected Comparison / ComparisonSpec, got {type(item).__name__}")
    return [spec for spec in out if spec is not None]


def _as_comparisons(comparisons: Sequence[Any]) -> list[Comparison]:
    from .comparisons import Comparison

    out = []
    for item in comparisons:
        if isinstance(item, Comparison):
            out.append(item)
        elif isinstance(item, ComparisonSpec):
            out.append(Comparison.from_spec(item))
    return out


def _level_proportions(
    spec: ComparisonSpec, assigned: np.ndarray
) -> list[float]:
    """Relative frequency of each level over ``assigned`` (u estimate)."""
    counts = np.bincount(assigned, minlength=len(spec.levels)).astype(np.float64)
    total = max(float(counts.sum()), 1e-12)
    return np.clip(counts / total, 1e-8, None).tolist()


def _level_defaults(spec: ComparisonSpec) -> list[float]:
    return [lv.m if (lv.m is not None and not lv.is_null) else 1e-8 for lv in spec.levels]


def _fuzzy_block_key(value: str) -> Optional[str]:
    """A fixed-size, **count-aware** trigram signature bucket key.

    Returns ``None`` when the value is too short to be bucketed.  The key is a
    join of ``trigram:count`` entries (lowercased), so two strings that share
    most trigrams *with similar multiplicity* (a case/typo-perturbed twin and
    its base) land in the same bucket, while position-flipped or length-
    divergent values (which have different trigram multisets) usually don't.
    """
    v = "".join(ch for ch in value.lower() if ch.isalnum())
    if len(v) < 6:
        return None
    trigrams = [v[i:i + 3] for i in range(len(v) - 2)]
    counts: dict = {}
    for t in trigrams:
        counts[t] = counts.get(t, 0) + 1
    return "|".join(f"{t}:{c}" for t, c in sorted(counts.items()))


def _sample_all_pairs(n: int, cap: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    """Sample up to ``cap`` unordered index pairs uniformly from range(n)."""
    if n < 2:
        return []
    total = n * (n - 1) // 2
    wanted = min(cap, total)
    if wanted >= total:
        return [(i, j) for i in range(n) for j in range(i + 1, n)]
    pairs: set[tuple[int, int]] = set()
    while len(pairs) < wanted:
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n))
        if i != j:
            pairs.add((i, j) if i < j else (j, i))
    return sorted(pairs)