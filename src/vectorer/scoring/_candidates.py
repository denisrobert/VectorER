"""Candidate-pair generation for the training pool.

These helpers build the pair pool EM trains on:

* :func:`build_blocked_pairs` -- blocking-rule candidate generation with
  exact / fuzzy (trigram + Jaro) keys and specificity-ordered per-rule budgets;
* :func:`_sample_all_pairs` -- uniform unordered index-pair sampling (the
  non-match side of the u estimate);
* :func:`_positions_pair_values` -- ``PairValues`` from index pairs into
  ``records`` (vectorized field extraction for a training pool).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional, Sequence

import numpy as np

import vectorer.sim as sim

from ..comparisons import PairValues


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


def _positions_pair_values(
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


def build_blocked_pairs(
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
      a *papered* fuzzy block: records whose Jaro similarity is above 0.8 are
      placed into the same buckets by a count-aware trigram signature, so a
      perturbed value still lands near its base's bucket (it shares most
      trigrams).  For multi-column rules we still use exact keys
      (interpreted as a conjunction), which is the correct semantics for
      canonical ids.

    * **Per-rule pair budget.**  The old loop ``break``ed out of the whole
      rule list once the total cap was hit, so the first (most common) rule
      could saturate all ``max_pairs`` and later, more discriminative rules
      (e.g. ``date_of_birth``, which twins DO share) never ran.  Each rule now
      gets a budget slice proportional to its cardinality, so the candidate
      pool mixes evidence from every rule.
    """
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
    # (high cardinality, e.g. date_of_birth) produce small, rare groups that
    # are most likely to contain *twins* and least likely to waste the budget
    # on common-value non-match cliques.  The budget is allocated
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
    # Per-rule remaining allowance (shared across the rule's groups and both
    # emission paths).  Every rule gets at least 1 pair so a rounding-absorbing
    # last rule is never starved to zero; the global `max_pairs` cap is still
    # enforced at emission time.
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
            # Fuzzy bucket: case/typo-flipped twins collide on a count-aware
            # trigram signature; exact groups catch identical (canonical)
            # values.  Short strings use a case-insensitive exact key (no
            # trigrams), so "jack"/"Jack" still join via the bucket path when
            # long enough, and identical short values via the exact group.
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
                # makes hitting the true twin vanishingly unlikely.  Skip the
                # bucket when its size dwarfs the remaining budget.
                n_fb = len(members)
                if n_fb > allowance[0] * 4:
                    continue
                mlist = sorted(members)
                a_vals = np.array([members[i] for i in mlist], dtype=object)
                n_fb = len(mlist)
                # Only the remaining allowance can ever be admitted for this
                # bucket, so the Jaro grid never needs more than
                # sqrt(allowance) members.  This bounds each bucket's grid
                # work (O(allowance)) instead of O(n_fb^2) which could be
                # quadratic in the whole population for a heavy-tailed bucket.
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
                # takes the lower-triangle candidate indices (x < y) from the
                # flattened lower-half of the matrix.
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
class CandidatesMixin:
    """Candidate-pair generation entry points.

    Thin method wrappers over the module-level helpers so the scorer's
    training code and callers invoking `scorer._blocked_pairs(...)` /
    `scorer._positions_pair_values(...)` keep working unchanged.
    """

    def _blocked_pairs(
        self,
        records: Sequence[dict],
        rules: Sequence[Sequence[str]],
        max_pairs: int,
        rng: np.random.Generator,
    ) -> list[tuple[int, int]]:
        return build_blocked_pairs(records, rules, max_pairs, rng)

    def _positions_pair_values(
        self,
        records: Sequence[dict],
        pairs: Sequence[tuple[int, int]],
        fields: Sequence[str],
    ) -> PairValues:
        return _positions_pair_values(records, pairs, fields)
