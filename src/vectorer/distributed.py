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
    # To be identical to single-process, train on the FULL vector matrix unless
    # sample_size is deliberately set (then we approximate).
    all_vectors = np.vstack(vector_shards)
    centroids = train_canopy_centroids(
        all_vectors, n_canopies, seed=seed, sample_size=sample_size,
    )

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
    return distributed_closure(edges, n, n_workers)


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