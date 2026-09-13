"""Global term-frequency pre-reduction.

Merges per-shard term-frequency counters so every machine computes with the
same TF weights regardless of how the data was sharded.
"""

from __future__ import annotations


def merge_tf_counters(chunks) -> dict:
    """Stream-reduce per-value term-frequency counters across shards.

    ``chunks`` yields dicts ``{value: count}`` (one per machine/shard).  The
    merged dict is the **global** term-frequency table, identical regardless of
    sharding -- so all machines share the same TF weights.
    """
    from collections import Counter

    total: Counter = Counter()
    for chunk in chunks:
        total.update(chunk)
    return dict(total)


def build_global_tf_tables(shard_record_iterables, fields) -> dict:
    """Count per-value frequencies of ``fields`` across record shards.

    ``shard_record_iterables`` is an iterable of record-lists (one per shard);
    returns ``{field: {value: relative_frequency}}`` using the global counts.
    Pass the returned tables' value populations to
    ``FellegiSunterScorer.from_comparisons(..., base_records=pop)`` (or feed the
    global counts directly) so TF adjustments are identical on every machine.
    """
    from collections import Counter

    col_counters: dict[str, Counter] = {f: Counter() for f in fields}
    for shard in shard_record_iterables:
        for record in shard:
            for f in fields:
                v = record.get(f)
                if v is not None:
                    col_counters[f][v] += 1
    tables: dict[str, dict] = {}
    for f, counter in col_counters.items():
        total = sum(counter.values()) or 1
        tables[f] = {v: c / total for v, c in counter.items()}
    return tables