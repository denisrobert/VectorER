"""Benchmark data loading.

Shared by the benchmark scripts.  Each benchmark normally synthesizes its own
population via its local generator; passing ``--data-file path/to/records.jsonl``
(or ``.json``) makes the benchmark use the supplied dataset instead of the
synthetic one.

File formats
------------
* ``.jsonl`` — one JSON object per line (most common for exports), or a
  whitespace-separated stream of objects.
* ``.json`` — either a JSON *list* of records, or a single object with a
  key under ``--data-key`` holding a list of records (e.g. ``{"people": [...]}``).

All records are plain mappings.  The benchmarks assume the compared columns
(``first_name``, ``last_name``, ``date_of_birth``, ``email``, ``address``) are
present, optionally ``None``; provide a real dataset with those fields.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Mapping, Sequence


def load_records(path: str, key: str | None = None) -> list[dict]:
    """Load records from a JSONL or JSON file.

    ``key`` selects the list inside a single-object JSON file.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"data-file not found: {p}")
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".jsonl" or p.suffix.lower() == ".ndjson":
        records: list[dict] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, list):
                records.extend(obj)
            elif isinstance(obj, dict):
                records.append(obj)
            else:
                raise ValueError(f"expected a dict or list per JSONL line in {p}, got {type(obj).__name__}")
        return records

    if p.suffix.lower() == ".json":
        obj = json.loads(text)
        if isinstance(obj, list):
            return [r for r in obj if isinstance(r, dict)]
        if isinstance(obj, dict):
            if key is None:
                raise ValueError(
                    f"{p} is a single object; pass --data-key to select the records "
                    f"list (available keys: {sorted(k for k, v in obj.items() if isinstance(v, list))})"
                )
            records = obj.get(key)
            if records is None:
                raise ValueError(f"no key {key!r} in {p}; available: {sorted(obj)}")
            return [r for r in records if isinstance(r, dict)]
        raise ValueError(f"{p} must contain a list of records or an object with a records list")

    raise ValueError(f"unsupported data-file extension {p.suffix!r}; use .jsonl or .json")


def build_unrelated_negatives(
    records: Sequence[Mapping[str, Any]],
    n_neg: int,
    seed: int,
    *,
    pool_factor: int = 20,
    pair_left_record: bool = False,
    rng: random.Random | None = None,
) -> list[dict] | list[tuple[dict, dict]]:
    """Generate fresh census-distributed identities disjoint from ``records``.

    The synthetic population's tiny name grid makes any two population records
    near-duplicates, so real index records cannot serve as non-match negatives:
    a working pipeline would (correctly) match them.  This instead generates
    brand-new people whose full identity ``(first_name, last_name, date_of_birth)``
    is disjoint from the population -- shifting candidate DOBs that collide with
    a population DOB by +5 days and nulling colliding emails, so no blocker or
    attribute comparison can spuriously match them.

    With ``pair_left_record`` set, each fresh negative is paired with a random
    population record and tuples ``(record_a, unrelated_record)`` are returned
    (for record-pair scoring where both sides must live in ``records``);
    otherwise the bare unrelated records are returned.
    """
    import datetime as _dt

    from generate_census_population import PopulationConfig as _PC
    from generate_census_population import generate as _gen_census

    _rng = rng if rng is not None else random.Random(seed)
    ref_names = {(r["first_name"], r["last_name"], r["date_of_birth"]) for r in records}
    ref_dobs = {r["date_of_birth"] for r in records}
    ref_emails = {r["email"] for r in records if r["email"]}
    pool = _gen_census(_PC(n=max(n_neg * pool_factor, 2000), seed=seed))
    out: list[dict] = []
    for cand in pool:
        if len(out) >= n_neg:
            break
        nc = dict(cand)
        if (nc["first_name"], nc["last_name"], nc["date_of_birth"]) in ref_names:
            continue
        if nc["date_of_birth"] in ref_dobs:
            nc["date_of_birth"] = (_dt.date.fromisoformat(nc["date_of_birth"])
                                   + _dt.timedelta(days=5)).isoformat()
        if nc.get("email") in ref_emails:
            nc["email"] = None
        out.append(nc)
    if pair_left_record:
        return [(records[_rng.randrange(len(records))], nc) for nc in out]
    return out


def require_compared_fields(
    records: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
    allow_none: bool = True,
) -> None:
    """Warn (not fail) if the dataset lacks a compared column the benchmarks expect."""
    missing = [f for f in fields if not any(f in r for r in records)]
    if missing:
        import sys

        print(
            f"  [warning] the supplied dataset has no columns named {missing}; "
            f"those comparisons will degrade to null (no evidence) levels.",
            file=sys.stderr,
        )


def build_weight_pool(
    records: Sequence[Mapping[str, Any]],
    max_pairs: int,
    seed: int,
) -> list[tuple[int, int]]:
    """A memory-bounded pair index pool for preliminary Yancey weighting.

    Mirrors the enrichment benchmark's ``(i, j)`` sliding-window pool but caps
    both the *total* pair count (so it scales to full 300k+ populations) and
    the window per row (so a huge ``records`` cannot materialize tens of
    millions of tuples).  The pool is shuffled so the top-weight keep in
    :func:`enrich_records` is a fair sample of the population's heaviest
    pairs.
    """
    import random

    rng = random.Random(seed)
    n = len(records)
    cap = max(1, int(max_pairs))
    pool: list[tuple[int, int]] = []
    if n < 2:
        return pool
    per_row = max(1, min(n - 1, max(1, cap // n) + 1))
    for i in range(n):
        hi = min(n, i + 1 + per_row)
        for j in range(i + 1, hi):
            pool.append((i, j))
            if len(pool) >= cap:
                break
        if len(pool) >= cap:
            break
    rng.shuffle(pool)
    return pool


def enrich_records(
    records: Sequence[Mapping[str, Any]],
    score_pool_idx: Sequence[tuple[int, int]],
    weights,
    keep_frac: float,
    seed: int,
) -> list[dict]:
    """Record-level Yancey match-enrichment: keep the records that participate
    in the highest-weight pairs (a weight-sorted keep of ``keep_frac`` of the
    pair pool's records)."""
    import numpy as np

    order = np.argsort(weights)[::-1]
    n_keep = max(1, int(len(order) * keep_frac))
    keep_idx = set()
    for k in order[:n_keep]:
        i, j = score_pool_idx[k]
        keep_idx.add(i)
        keep_idx.add(j)
    return [records[i] for i in sorted(keep_idx)]