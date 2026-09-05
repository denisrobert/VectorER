"""Distributed batch ER.

An additive distributed executor around the framework's public stage hooks.
The single-process ``BatchPipeline.run`` is NOT touched; this module produces
the *same* cluster assignment (given the same data, geometry and scorer) by
executing the same stages across workers:

- parse/embed: shard map
- canopy train: driver (sampled) + worker local assignment
- candidate pairs: emit, dedupe, own-each-by-pair-hash
- Fellegi-Sunter: parallel map over the owned pairs (only above-tau edges
  cross the wire)
- Swoosh closure: distributed connected components (exact) over the above-tau
  edges

Executors
---------
* ``multiprocessing`` (default; extra-free) -- :func:`distributed_batch_er`.
* Ray is straightforward to add by swapping the executor used internally; the
  seams (worker functions, pair ownership, closure) are backend-agnostic.

Why the result matches single-process: the candidate-pair set is identical
(same centroids, same multi-assignment when identical vectors/geometry are
used), FS scoring calls the same scorer on the same pairs, and the closure is
the same union-find over the above-tau edges.  Ownership by pair hash only
picks *which worker* computes each pair, never the value.

To reproduce the single-process canopy exactly with ``n_workers`` shards, the
vectors passed to the driver's :func:`train_canopy_centroids` must be the same
as local :func:`canopy_blocking` sees -- so by default the centroids are
trained on the **full** vector matrix (same as local), with an optional sample
size for very large inputs.

In addition, this module exposes the **streaming / multi-machine building
blocks** (v0.4.0 distribution plan):

* :func:`distributed_score_pairs` -- the FS scoring *map*: pairs owned by a
  deterministic balanced hash, each worker scores its chunk with the same
  serialized scorer and returns only the above-``tau`` rows (only those cross
  the wire).  With ``pair_positions=`` it returns proper ``ScoredPair`` edges.
* :func:`distributed_score_and_reduce` -- composes the scoring map with the
  closure reduce for a single streaming score-then-cluster call.
* :func:`streaming_distributed_closure` -- transitive closure over an iterator
  of edge chunks (streaming reduce, bounded memory).
* :func:`distributed_closure_reduce` -- multi-machine exact connected
  components: per-worker local union-find + a shared-node merge into
  min-position ids, bit-for-bit identical to the single-process closure.
* :func:`merge_tf_counters` / :func:`build_global_tf_tables` -- global
  term-frequency pre-reduction so TF weights stay consistent across machines.
* :func:`create_executor` / :class:`RayExecutor` -- backend abstraction
  (``process``, ``thread``, ``ray``).

Caveats (per the plan): G-Swoosh (`gswoosh`/`cluster_with_merger`), per-query
FS scoring (incremental / link-directed), and single-machine canopy training
remain single-process by design; use the transitive-closure mode for
distributed runs and the external-vector-DB adapter for the incremental store.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Callable, Optional, Sequence

import numpy as np

from .blocking import assign_canopies, train_canopy_centroids
from .clustering import Cluster, ClusterAssignment, ScoredPair, _DisjointSet, _build
from .records import to_record_dict
from .scoring import FellegiSunterScorer


def hash_pair(i: int, j: int, n_workers: int) -> int:
    """Deterministic owner slot for an unordered (i, j) record pair."""
    a, b = (i, j) if i <= j else (j, i)
    return hash((a, b)) % int(n_workers)


# ---------------------------------------------------------------------------
# Worker-side functions (module-level so multiprocessing can pickle them)
# ---------------------------------------------------------------------------


def _embed_shard(records, embed_dim: int, embed_seed: int) -> tuple[list[dict], np.ndarray]:
    """Parse + embed a shard using the deterministic hashing embedder, which
    round-trips workers without shipping a model object."""
    from .embeddings import CharacterHashingEmbedding

    embedder = CharacterHashingEmbedding(dimension=embed_dim, ngrams=(2, 3))
    del embed_seed
    parsed = [to_record_dict(r) for r in records]
    texts = [_serialize(r) for r in parsed]
    vecs = np.asarray(embedder.embed_many(texts), dtype="float32")
    return parsed, vecs


def _serialize(record: dict) -> str:
    return "\n".join(f"{k}: {v}" for k, v in record.items() if v is not None)


def _assign_shard(vectors, centroids, overlap_m):
    """Worker-side canopy assignment (local CanopyIndex)."""
    return assign_canopies(vectors, centroids, overlap_m)


def _score_shard(left_records, right_records, scorer_state, tau):
    """Score one worker's owned pairs; return (mask, probs, weights) where
    ``mask[i]`` is True iff pair i is at/above ``tau``.  The caller aligns the
    mask with the original pair list (``_score_shard`` filters nothing)."""
    scorer = _scorer_from_state(scorer_state)
    probs = scorer.score_pairs(left_records, right_records)
    weights = scorer.match_weight_pairs(left_records, right_records)
    mask = [float(p) >= tau for p in probs]
    return mask, list(map(float, probs)), list(map(float, weights))


def _score_worker(worker, pairs, all_records, scorer_state, tau):
    """Module-level process-picklable worker: score a worker's owned pairs and
    return the above-tau ScoredPairs with global positions attached."""
    left = [all_records[gi] for gi, gj in pairs]
    right = [all_records[gj] for gi, gj in pairs]
    mask, probs, weights = _score_shard(left, right, scorer_state, tau)
    return [
        ScoredPair(left_position=p[0], right_position=p[1],
                   probability=prob, match_weight=weight)
        for p, prob, weight, keep in zip(pairs, probs, weights, mask)
        if keep
    ]


def _scorer_from_state(state):
    from .scoring import FellegiSunterScorer

    if "settings" in state:
        return FellegiSunterScorer.from_settings(state["settings"])
    if "comparisons" in state:
        return FellegiSunterScorer.from_comparisons(state["comparisons"])
    raise ValueError("scorer_state must contain 'settings' or 'comparisons'")


def _scorer_state_of(scorer: FellegiSunterScorer) -> dict:
    return {"settings": scorer.to_settings()}


# ---------------------------------------------------------------------------
# Executor abstraction (Milestone A: backend-agnostic execution)
# ---------------------------------------------------------------------------


class RayExecutor:
    """A minimal Ray-backed executor exposing ``map(fn, items)``.

    Optional: requires the ``ray`` package (installed separately).  The worker
    callable must be a module-level (picklable) function, not a closure --
    closures are only safe on the thread/process executors.
    """

    def __init__(self, n_workers: int = 2, address: Optional[str] = None) -> None:
        import ray
        import os

        if not ray.is_initialized():
            env_addr = os.environ.get("RAY_ADDRESS")
            if address in (None, "auto") and env_addr:
                ray.init(address=env_addr)
            elif address in (None, "auto"):
                # No cluster specified and no RAY_ADDRESS: start a **local** one
                # (no GCS), so `--ray-address auto` works as a one-host demo of
                # the same code path; a real multi-node cluster is joined by
                # passing the head's ip:port (or setting RAY_ADDRESS).
                ray.init(num_cpus=int(n_workers))
            else:
                ray.init(address=address, num_cpus=int(n_workers))
        self.n_workers = int(n_workers)

    def map(self, fn, items):
        import ray

        @ray.remote
        def _call(f, item):  # noqa: E306  (remote fn must be module-scoped)
            return f(item)

        futures = [_call.remote(fn, item) for item in items]
        return list(ray.get(futures))


def create_executor(
    kind: str = "process",
    n_workers: int = 2,
    address: Optional[str] = None,
):
    """Create an executor backend by name.

    ``kind`` in {"process", "thread", "ray"}:
    * "process" -- ``ProcessPoolExecutor`` (default; extra-free).
    * "thread"  -- ``ThreadPoolExecutor``.
    * "ray"     -- :class:`RayExecutor` (requires the ``ray`` package).
    """
    if kind == "process":
        return ProcessPoolExecutor(max_workers=int(n_workers))
    if kind == "thread":
        return ThreadPoolExecutor(max_workers=int(n_workers))
    if kind == "ray":
        return RayExecutor(n_workers=n_workers, address=address)
    raise ValueError(f"unknown executor kind {kind!r}; use process, thread, or ray")


def _score_pairs_worker(indices, left_records, right_records, scorer_state, tau):
    """Module-level worker: score the ``indices`` slice of equal-length
    ``left``/``right`` lists, returning only ``(index, prob, weight)`` rows at
    or above ``tau``.  Used by :func:`distributed_score_pairs`."""
    scorer = _scorer_from_state(scorer_state)
    left = [left_records[i] for i in indices]
    right = [right_records[i] for i in indices]
    probs = scorer.score_pairs(left, right)
    weights = scorer.match_weight_pairs(left, right)
    return [
        (i, float(p), float(w))
        for i, p, w in zip(indices, probs, weights)
        if float(p) >= tau
    ]


def _owner_of_index(idx: int, n_workers: int) -> int:
    """Deterministic, process-stable owner slot for a pair *index* (unlike
    ``hash_pair`` which owns unordered record pairs).  Used to balance the
    scoring map across workers."""
    mixed = (idx * 2654435761) & 0xFFFFFFFF
    return mixed % int(n_workers)


def distributed_score_pairs(
    scorer: FellegiSunterScorer,
    left_records: Sequence[dict],
    right_records: Sequence[dict],
    *,
    tau: float,
    n_workers: int = 2,
    executor: Optional[Any] = None,
    pair_positions: Optional[Sequence[tuple[int, int]]] = None,
) -> list[Any]:
    """Score equal-length ``left``/``right`` pair lists in parallel (map).

    Pairs are owned by worker via :func:`_owner_of_index` (deterministic,
    balanced); each worker scores its chunk with the same serialized scorer and
    returns only the rows at/above ``tau`` -- so *only above-tau edges cross
    the wire*.  The returned list is not necessarily sorted; it carries the
    pair ``index`` so the caller can reconstruct alignment.

    With ``pair_positions`` (one ``(i, j)`` per row), the result is a list of
    above-tau ``ScoredPair`` objects carrying those **record positions**, ready
    for :func:`distributed_closure_reduce`.

    ``executor`` may be a ``concurrent.futures`` executor or
    :class:`RayExecutor`; otherwise a :class:`ProcessPoolExecutor` with
    ``n_workers`` is created.
    """
    n_pairs = len(left_records)
    if n_pairs == 0 or len(right_records) != n_pairs:
        raise ValueError("left_records and right_records must be equal-length, non-empty")
    scorer_state = _scorer_state_of(scorer)
    owned: dict[int, list[int]] = {w: [] for w in range(n_workers)}
    for idx in range(n_pairs):
        owned[_owner_of_index(idx, n_workers)].append(idx)

    from functools import partial

    worker = partial(_score_pairs_worker, left_records=left_records,
                     right_records=right_records, scorer_state=scorer_state,
                     tau=tau)
    if executor is not None:
        scored = list(executor.map(worker, [owned[w] for w in range(n_workers)]))
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            scored = list(ex.map(worker, [owned[w] for w in range(n_workers)]))
    rows = [row for chunk in scored for row in chunk]
    if pair_positions is None:
        return rows
    return [
        ScoredPair(left_position=pair_positions[idx][0],
                   right_position=pair_positions[idx][1],
                   probability=prob, match_weight=weight)
        for idx, prob, weight in rows
    ]


def distributed_score_and_reduce(
    scorer: FellegiSunterScorer,
    left_records: Sequence[dict],
    right_records: Sequence[dict],
    pair_positions: Sequence[tuple[int, int]],
    *,
    tau: float,
    n: int,
    n_workers: int = 2,
    executor: Optional[Any] = None,
    records: Optional[Sequence[Any]] = None,
) -> ClusterAssignment:
    """Score equal-length ``left``/``right`` pair lists in parallel and return
    the cluster assignment over the above-τ edges.

    Composes :func:`distributed_score_pairs` (streaming scoring map -- only
    above-τ edges cross the wire) with :func:`distributed_closure_reduce`
    (multi-machine connected components).  ``pair_positions[i]`` is the
    ``(left_pos, right_pos)`` record pair for row ``i``, and ``n`` is the
    number of records (the closure's node count).
    """
    edges = distributed_score_pairs(
        scorer, left_records, right_records,
        tau=tau, n_workers=n_workers, executor=executor,
        pair_positions=pair_positions,
    )
    return distributed_closure_reduce(
        edges, n, n_workers=n_workers, executor=executor, records=records,
    )


# ---------------------------------------------------------------------------
# Term-frequency pre-reduce (Milestone C)
# ---------------------------------------------------------------------------


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


def gather_canopy_sample(
    vector_shards: Sequence[np.ndarray],
    sample_size: int,
    seed: int = 42,
) -> np.ndarray:
    """Gather a **cross-machine sample** of vectors for canopy-centroid training.

    Instead of materializing the full vector matrix on one node, each shard
    contributes a proportional, deterministic random slice, and only the sample
    is concatenated.  Uses a seeded RNG so the sample is reproducible and
    identical regardless of shard boundaries.
    """
    total = sum(len(v) for v in vector_shards)
    if total <= int(sample_size):
        return np.vstack(vector_shards)
    rng = np.random.default_rng(seed)
    per_shard = int(sample_size) // max(1, len(vector_shards))
    slices = []
    for v in vector_shards:
        take = max(1, min(per_shard, len(v)))
        idx = rng.choice(len(v), take, replace=False)
        slices.append(np.asarray(v)[idx])
    return np.vstack(slices)


# ---------------------------------------------------------------------------
# Distributed batch ER
# ---------------------------------------------------------------------------


def distributed_batch_er(
    records: Sequence[Any],
    *,
    scorer: FellegiSunterScorer,
    n_canopies: int,
    overlap_m: int = 2,
    tau: float = 0.85,
    seed: int = 42,
    n_workers: int = 2,
    embed_dim: int = 384,
    sample_size: Optional[int] = 200_000,
    use_threads: bool = False,
    executor: Optional[Any] = None,
) -> ClusterAssignment:
    """Run the batch ER stages in parallel and return the cluster assignment.

    Parameters match :class:`~vectorer.batch.BatchPipeline` plus distribution
    knobs.  ``executor`` may be a ``concurrent.futures`` executor; otherwise a
    :class:`ProcessPoolExecutor` (or :class:`ThreadPoolExecutor` when
    ``use_threads``) with ``n_workers`` is created.

    Notes
    -----
    * The deterministic :class:`~vectorer.embeddings.CharacterHashingEmbedding`
      is used for the embedding stage (identical to the pipeline default), so
      results match ``build_batch_pipeline(embedder=...)``.
    * ``scorer`` is serialized to each worker via its settings -- the same
      m/u, prior and threshold as single-process.
    * The closure over the above-tau edges is the exact distributed union-find,
      equivalent to the single-process transitive closure.
    """
    n = len(records)
    if n == 0:
        return ClusterAssignment(node_cluster={}, clusters={}, n_pairs_evaluated=0, n_pairs_matched=0)

    # Contiguous shards so that the flattened ``all_records`` order matches the
    # original record order -- required for global ids to equal the positions
    # the single-process pipeline produces (and for identical results).
    boundaries = [n * w // n_workers for w in range(n_workers + 1)]
    shards = [records[boundaries[w]:boundaries[w + 1]] for w in range(n_workers)]
    bases = boundaries[:-1]

    # --- stage 1: parse + embed (map) -------------------------------------
    # Embedding via CharacterHashingEmbedding is pure numpy (releases the GIL),
    # so a thread pool is safe and identical to serial.  Only the FS scoring
    # stage (below) uses processes/threads as configured.
    def _run_embed():
        if executor is not None:
            return list(executor.map(
                lambda shard: _embed_shard(shard, embed_dim, seed), shards))
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            return list(ex.map(
                lambda shard: _embed_shard(shard, embed_dim, seed), shards))

    shard_data = _run_embed()
    parsed_shards = [d[0] for d in shard_data]
    vector_shards = [d[1] for d in shard_data]

    # --- stage 2: canopy train (driver) + assign (workers) ----------------
    # Train on a cross-machine SAMPLE (Milestone C) so the driver never
    # materializes the full vector matrix.  sample_size=None trains on the full
    # matrix (needed for bit-identical centroids on small data); otherwise the
    # sample is reproducible across shardings via gather_canopy_sample.
    if sample_size is None or not vector_shards:
        train_vectors = np.vstack(vector_shards) if vector_shards else np.zeros((0, 0))
        centroids = train_canopy_centroids(train_vectors, n_canopies, seed=seed, sample_size=None)
    else:
        sampled = gather_canopy_sample(vector_shards, sample_size=int(sample_size), seed=seed)
        centroids = train_canopy_centroids(sampled, n_canopies, seed=seed, sample_size=None)

    def _run_assign():
        if executor is not None:
            return list(executor.map(
                lambda v: _assign_shard(v, centroids, overlap_m), vector_shards))
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            return list(ex.map(
                lambda v: _assign_shard(v, centroids, overlap_m), vector_shards))

    canopies = _run_assign()

    # --- stage 3: build GLOBAL canopies across shards, emit pairs, hash-own --
    # Single-process canopies the whole dataset at once, so two matching records
    # that landed in different shards still share a centroid.  We reproduce that:
    # each shard's CanopyIndex carries per-record centroid assignments; the
    # driver groups global record ids by centroid across ALL shards and emits
    # every intra-centroid pair (dedup + hash-own).
    centroid_to_ids: dict[int, set[int]] = {}
    for w, canopy in enumerate(canopies):
        offset = bases[w]
        assignments = canopy.assignments  # (shard_local_n, overlap_m) of centroids
        for local_i, row in enumerate(assignments):
            gi = offset + local_i
            for centroid in row:
                if centroid >= 0:
                    centroid_to_ids.setdefault(int(centroid), set()).add(gi)

    pair_buckets: dict[int, list[tuple[int, int]]] = {w: [] for w in range(n_workers)}
    seen: set[tuple[int, int]] = set()
    for centroid, ids in centroid_to_ids.items():
        del centroid
        ids = sorted(ids)
        for k in range(len(ids)):
            for l2 in range(k + 1, len(ids)):
                a, b = ids[k], ids[l2]
                key = (a, b)
                if key in seen:
                    continue
                seen.add(key)
                owner = hash_pair(a, b, n_workers)
                pair_buckets[owner].append(key)

    # --- stage 4: score owned pairs in parallel (map, drop below-tau) ------
    scorer_state = _scorer_state_of(scorer)
    all_records = [r for shard in parsed_shards for r in shard]

    def _run_score_one(worker):
        pairs = pair_buckets[worker]
        if not pairs:
            return []
        left = [all_records[gi] for gi, gj in pairs]
        right = [all_records[gj] for gi, gj in pairs]
        mask, probs, weights = _score_shard(left, right, scorer_state, tau)
        return [
            ScoredPair(
                left_position=p[0], right_position=p[1],
                probability=prob, match_weight=weight,
            )
            for p, prob, weight, keep in zip(pairs, probs, weights, mask)
            if keep
        ]

    if executor is not None:
        # An external executor (thread or process) already owns scheduling; give
        # it the worker closure.  Thread executors share memory fine; process
        # executors must receive picklable callables -- provide the module fn.
        use_cls = type(executor)
        if use_cls is ThreadPoolExecutor:
            scored_lists = list(executor.map(_run_score_one, range(n_workers)))
        else:
            # ProcessPoolExecutor: _run_score_one closes over big objects; pass
            # only picklable args via a module-level worker.
            futures = [
                executor.submit(
                    _score_worker,
                    worker,
                    pair_buckets[worker],
                    all_records,
                    scorer_state,
                    tau,
                )
                for worker in range(n_workers)
            ]
            scored_lists = [f.result() for f in futures]
    else:
        if use_threads:
            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                scored_lists = list(ex.map(_run_score_one, range(n_workers)))
        else:
            all_records_sp = all_records
            with ProcessPoolExecutor(max_workers=n_workers) as ex:
                futures = [
                    ex.submit(
                        _score_worker,
                        w,
                        pair_buckets[w],
                        all_records_sp,
                        scorer_state,
                        tau,
                    )
                    for w in range(n_workers)
                ]
                scored_lists = [f.result() for f in futures]
    edges = [e for lst in scored_lists for e in lst]

    # --- stage 5: distributed closure over the above-tau edges -------------
    # Use the weighted reduce (Milestones B-C): the driver no longer holds all
    # edges in one union-find; each worker union-finds its partition and the
    # merge is exact/multi-machine.  With n_workers == 1 this degenerates to
    # the single union-find.
    return distributed_closure_reduce(edges, n, n_workers=n_workers, executor=executor)


# ---------------------------------------------------------------------------
# Distributed closure (exact connected components)
# ---------------------------------------------------------------------------


def distributed_closure(
    edges: Sequence[ScoredPair],
    n: int,
    n_workers: int = 1,
    records: Optional[Sequence[Any]] = None,
) -> ClusterAssignment:
    """Exact connected components over ScoredPair edges (transitive closure).

    Equivalent to the single-process ``SwooshClusterer.cluster`` on the same
    above-tau edges: union-find over all edges, deterministic min-position ids.
    """
    del n_workers  # the union-find is naturally order-independent
    ds = _DisjointSet(n)
    for e in edges:
        ds.union(e.left_position, e.right_position)
    node_cluster, grouped = _build(n, ds)
    clusters = {}
    for cid, pos in grouped.items():
        rep_pos = min(pos)
        clusters[cid] = Cluster(
            cluster_id=cid,
            member_positions=set(pos),
            representative_position=rep_pos,
            representative=records[rep_pos] if records is not None else None,
        )
    return ClusterAssignment(
        node_cluster=node_cluster,
        clusters=clusters,
        n_pairs_evaluated=len(edges),
        n_pairs_matched=len(edges),
    )


def _dense_assignment(node_cluster: dict[int, int], n: int,
                      records: Optional[Sequence[Any]], n_matched: int = 0) -> ClusterAssignment:
    """Build a ClusterAssignment from a node->cluster-id map.

    ``node_cluster`` must already map every node to its (min-position) cluster
    id.  Singletons are added automatically for untouched nodes.
    """
    groups: dict[int, list[int]] = {}
    for node in range(n):
        cid = node_cluster.get(node, node)
        groups.setdefault(cid, []).append(node)
    clusters = {}
    for cid, pos in groups.items():
        rep_pos = min(pos)
        clusters[cid] = Cluster(
            cluster_id=cid,
            member_positions=set(pos),
            representative_position=rep_pos,
            representative=records[rep_pos] if records is not None else None,
        )
    return ClusterAssignment(
        node_cluster={node: node_cluster.get(node, node) for node in range(n)},
        clusters=clusters,
        n_pairs_evaluated=n_matched,
        n_pairs_matched=n_matched,
    )


def streaming_distributed_closure(
    edge_chunks,
    n: int,
    records: Optional[Sequence[Any]] = None,
) -> ClusterAssignment:
    """Transitive closure over a **stream** of ScoredPair chunks (streaming reduce).

    Consumes an iterable of edge chunks (each a sequence of ``ScoredPair``)
    and union-finds incrementally, so peak memory is bounded by the largest
    chunk rather than the full edge set.  Identical to the single-process
    transitive closure given the same edges.
    """
    ds = _DisjointSet(n)
    n_matched = 0
    for chunk in edge_chunks:
        for e in chunk:
            ds.union(e.left_position, e.right_position)
            n_matched += 1
    node_cluster, _ = _build(n, ds)
    return _dense_assignment(node_cluster, n, records, n_matched)


def _local_component_map(edges, n: int) -> dict[int, int]:
    """Union-find one edge partition; return ``{touched_node: local_min_node}``."""
    ds = _DisjointSet(n)
    for e in edges:
        ds.union(e.left_position, e.right_position)
    groups: dict[int, list[int]] = {}
    for e in edges:
        for node in (e.left_position, e.right_position):
            r = ds.find(node)
            groups.setdefault(r, []).append(node)
    comp_min = {r: min(nodes) for r, nodes in groups.items()}
    return {node: comp_min[ds.find(node)] for e in edges for node in (e.left_position, e.right_position)}


def distributed_closure_reduce(
    edges: Sequence[ScoredPair],
    n: int,
    n_workers: int = 1,
    executor: Optional[Any] = None,
    records: Optional[Sequence[Any]] = None,
) -> ClusterAssignment:
    """Exact connected components across **machines** (distributed reduce).

    Edges are partitioned by :func:`hash_pair` owner; each worker union-finds
    its own partition and returns ``{node: local_min}`` for touched nodes; the
    driver merges worker-local components that share a node (into min-position
    ids).  The result is bit-for-bit identical to the single-process closure.

    ``executor`` is optional; when ``None`` a :class:`ProcessPoolExecutor` is
    used.  With ``n_workers == 1`` this degenerates to the single union-find.
    """
    if n_workers == 1:
        return distributed_closure(edges, n, records=records)

    owned: dict[int, list[ScoredPair]] = {w: [] for w in range(n_workers)}
    for e in edges:
        owner = hash_pair(e.left_position, e.right_position, n_workers)
        owned[owner].append(e)

    from functools import partial

    worker = partial(_local_component_map, n=n)
    if executor is not None:
        maps = list(executor.map(worker, [owned[w] for w in range(n_workers)]))
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            maps = list(ex.map(worker, [owned[w] for w in range(n_workers)]))
    return _merge_local_components(maps, n, records, len(edges))


def _merge_local_components(worker_maps, n: int,
                            records: Optional[Sequence[Any]], n_matched: int) -> ClusterAssignment:
    """Merge per-worker ``{node: local_min}`` maps into global min-position ids."""
    # Union (worker, local_label) keys that share a node.
    id_of: dict[tuple[int, int], int] = {}
    node_keys: dict[int, list[tuple[int, int]]] = {}
    for w, comp_map in enumerate(worker_maps):
        for node, label in comp_map.items():
            key = (w, label)
            if key not in id_of:
                id_of[key] = len(id_of)
            node_keys.setdefault(node, []).append(key)

    # Untouched nodes are singletons; no keys needed.
    touched = set(node_keys)
    uf = _DisjointSet(len(id_of) or 1)
    for keys in node_keys.values():
        first = id_of[keys[0]]
        for key in keys[1:]:
            uf.union(first, id_of[key])

    # Per merged key-component: the minimum node-label (a node id).  The key is
    # (worker, local_label), so the node label is key[1].
    comp_min: dict[int, int] = {}
    for key, idx in id_of.items():
        root = uf.find(idx)
        comp_min[root] = min(comp_min.get(root, 10 ** 18), key[1])

    node_cluster: dict[int, int] = {}
    for node, keys in node_keys.items():
        root = uf.find(id_of[keys[0]])
        node_cluster[node] = comp_min[root]
    return _dense_assignment(node_cluster, n, records, n_matched)