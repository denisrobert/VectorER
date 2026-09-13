"""Top-level distributed batch ER orchestrators.

:func:`distributed_batch_er` dispatches to the records form
(:func:`_distributed_batch_er_from_records` -- parse/embed on the driver, then
distribute) or the store-fed form (:func:`_distributed_batch_er_from_store` --
bounded pulls from a pre-populated ``VectorDatabase``).  Both wire the same
stage map from :mod:`._canopy`, :mod:`._scoring` and :mod:`._closure`.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Callable, Optional, Sequence

import numpy as np

from ..blocking import train_canopy_centroids
from ..clustering import ClusterAssignment
from ..embeddings import CharacterHashingEmbedding, EmbeddingModel
from ..records import EMBED_DEFAULT
from ..scoring import FellegiSunterScorer
from ..vectorstores import VectorDatabase
from ._canopy import _assign_shard, _embed_shard, _embedder_state_of, gather_canopy_sample
from ._closure import distributed_closure_reduce
from ._core import hash_pair
from ._scoring import _score_owned_pairs, _scorer_state_of


def distributed_batch_er(
    records: Optional[Sequence[Any]] = None,
    vector_database: Optional[VectorDatabase] = None,
    *,
    scorer: FellegiSunterScorer,
    n_canopies: int,
    overlap_m: int = 2,
    tau: float = 0.85,
    seed: int = 42,
    n_workers: int = 2,
    embed_dim: int = 384,
    embedder: Optional[EmbeddingModel] = None,
    embed_text: Optional[Callable[[dict], str]] = None,
    sample_size: Optional[int] = 200_000,
    use_threads: bool = False,
    executor: Optional[Any] = None,
    progress_callback: Optional[Callable[[int], None]] = None,
) -> ClusterAssignment:
    """Run the batch ER stages in parallel and return the cluster assignment.

    Supply **exactly one** data source:

    * ``records``: the dataset as a record list; parsed and embedded on the
      driver's shards, then distributed (the classic form).
    * ``vector_database``: a **pre-populated** :class:`VectorDatabase` (e.g. a
      checkpointed :class:`InMemoryVectorDatabase` or an external
      ``QdrantVectorDatabase``).  The stages then read the store in bounded
      pulls -- one shard's vectors at a time for canopy training/assignment,
      and only the record payloads each worker owns for FS scoring -- so a
      dataset that does **not fit on a single node** can be deduplicated as
      long as it fits in the cluster.  See
      :func:`_distributed_batch_er_from_store`.

    ``executor`` may be a ``concurrent.futures`` executor; otherwise a
    :class:`ProcessPoolExecutor` (or :class:`ThreadPoolExecutor` when
    ``use_threads``) with ``n_workers`` is created.

    Notes
    -----
    * ``embedder`` selects the embedding model used by the records form.
      Default (and ``embed_dim``'s effective value) is the deterministic
      :class:`~vectorer.embeddings.CharacterHashingEmbedding`, matching the
      pipeline default.  The embedder must implement
      :meth:`EmbeddingModel.to_settings`; workers rebuild it from those
      settings, so a heavy model (sentence-transformer, GPU) is **loaded once
      per worker** -- the expected cost in a distributed run.
    * ``embed_text`` renders each record to the embedding text, exactly like
      ``BatchPipeline(embed_text=...)`` / ``build_batch_pipeline(embed_text=...)``.
      It must match the serializer used at ingest/query (default
      :data:`~vectorer.records.EMBED_DEFAULT`, the schema-agnostic
      ``field: value`` lines); pass ``positional_embed_text(...)`` etc. to
      reproduce a custom single-process text space.
    * ``scorer`` is serialized to each worker via its settings -- the same
      m/u, prior and threshold as single-process.
    * The closure over the above-tau edges is the exact distributed union-find,
      equivalent to the single-process transitive closure.
    """
    if (records is None) == (vector_database is None):
        raise ValueError("supply exactly one of records or vector_database")
    if vector_database is not None:
        return _distributed_batch_er_from_store(
            vector_database,
            scorer=scorer, n_canopies=n_canopies, overlap_m=overlap_m, tau=tau,
            seed=seed, n_workers=n_workers, sample_size=sample_size,
            use_threads=use_threads, executor=executor,
            progress_callback=progress_callback,
        )
    embedder = embedder if embedder is not None else CharacterHashingEmbedding(dimension=embed_dim)
    return _distributed_batch_er_from_records(
        records,  # type: ignore[arg-type]
        embedder_state=_embedder_state_of(embedder),
        embed_text=embed_text if embed_text is not None else EMBED_DEFAULT,
        scorer=scorer, n_canopies=n_canopies, overlap_m=overlap_m, tau=tau,
        seed=seed, n_workers=n_workers,
        sample_size=sample_size, use_threads=use_threads, executor=executor,
        progress_callback=progress_callback,
    )


def _distributed_batch_er_from_records(
    records: Sequence[Any],
    *,
    embedder_state: dict,
    embed_text: Callable[[dict], str] = EMBED_DEFAULT,
    scorer: FellegiSunterScorer,
    n_canopies: int,
    overlap_m: int = 2,
    tau: float = 0.85,
    seed: int = 42,
    n_workers: int = 2,
    sample_size: Optional[int] = 200_000,
    use_threads: bool = False,
    executor: Optional[Any] = None,
    progress_callback: Optional[Callable[[int], None]] = None,
) -> ClusterAssignment:
    """Run the batch ER stages in parallel and return the cluster assignment.

    Parameters match :class:`~vectorer.batch.BatchPipeline` plus distribution
    knobs.  ``executor`` may be a ``concurrent.futures`` executor; otherwise a
    :class:`ProcessPoolExecutor` (or :class:`ThreadPoolExecutor` when
    ``use_threads``) with ``n_workers`` is created.

    Notes
    -----
    * ``embedder_state`` is the serialized embedding model (see
      :meth:`EmbeddingModel.to_settings`); each worker rebuilds the model from
      these settings, so results match ``build_batch_pipeline(embedder=...)``
      embedding with the same model.
    * ``embed_text`` is the record serializer (default
      :data:`~vectorer.records.EMBED_DEFAULT`); the workers call it with each
      parsed record before embedding, exactly as ``build_batch_pipeline(
      embed_text=...)`` would.
    * ``scorer`` is serialized to each worker via its settings -- the same
      m/u, prior and threshold as single-process.
    * The closure over the above-tau edges is the exact distributed union-find,
      equivalent to the single-process transitive closure.
    """
    n = len(records)
    if n == 0:
        return ClusterAssignment(node_cluster={}, clusters={}, n_pairs_evaluated=0, n_pairs_matched=0)

    # Contiguous shards so that the flattened ``records_by_pos`` order matches the
    # original record order -- required for global ids to equal the positions
    # the single-process pipeline produces (and for identical results).
    boundaries = [n * w // n_workers for w in range(n_workers + 1)]
    shards = [records[boundaries[w]:boundaries[w + 1]] for w in range(n_workers)]
    bases = boundaries[:-1]

    # --- stage 1: parse + embed (map) -------------------------------------
    # Embedding is pure numpy / a GPU model in its own worker; the thread pool
    # is safe because CharacterHashingEmbedding releases the GIL and heavier
    # models embed independently per worker.
    def _run_embed():
        if executor is not None:
            return list(executor.map(
                lambda shard: _embed_shard(shard, embedder_state, embed_text), shards))
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            return list(ex.map(
                lambda shard: _embed_shard(shard, embedder_state, embed_text), shards))

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
    records_by_pos = [r for shard in parsed_shards for r in shard]
    edges = _score_owned_pairs(
        pair_buckets,
        lambda positions: [records_by_pos[p] for p in positions],
        scorer_state, tau,
        n_workers=n_workers, use_threads=use_threads, executor=executor,
        progress_callback=progress_callback,
    )

    # --- stage 5: distributed closure over the above-tau edges -------------
    # Use the weighted reduce (Milestones B-C): the driver no longer holds all
    # edges in one union-find; each worker union-finds its partition and the
    # merge is exact/multi-machine.  With n_workers == 1 this degenerates to
    # the single union-find.
    return distributed_closure_reduce(edges, n, n_workers=n_workers, executor=executor)


def _distributed_batch_er_from_store(
    vector_database: VectorDatabase,
    *,
    scorer: FellegiSunterScorer,
    n_canopies: int,
    overlap_m: int = 2,
    tau: float = 0.85,
    seed: int = 42,
    n_workers: int = 2,
    sample_size: Optional[int] = 200_000,
    use_threads: bool = False,
    executor: Optional[Any] = None,
    progress_callback: Optional[Callable[[int], None]] = None,
) -> ClusterAssignment:
    """Store-fed distributed batch ER (populate-then-cluster on huge data).

    Records and vectors live in ``vector_database`` (cluster-resident for an
    external store); only bounded pieces are ever pulled:

    * one shard's vectors at a time -- for canopy centroid training via a
      cross-shard SAMPLE (never the full matrix, so a dataset that does not
      fit on one node can be clustered) and for that shard's canopy
      assignment;
    * only the record payloads a worker needs to score its owned pairs
      (batched ``records_at`` fetches), not a copy of the dataset;
    * representatives for the output clusters, one ``record_at`` per cluster.

    The scoring map and the exact distributed closure are the same as the
    records path, and given the same store contents the cluster assignment is
    identical (the canopy geometry and the edge stream agree).
    """
    n = len(vector_database)
    if n == 0:
        return ClusterAssignment(
            node_cluster={}, clusters={}, n_pairs_evaluated=0, n_pairs_matched=0
        )

    boundaries = [n * w // n_workers for w in range(n_workers + 1)]
    bases = boundaries[:-1]

    def shard_vectors(w: int) -> np.ndarray:
        start, stop = boundaries[w], boundaries[w + 1]
        return np.asarray(vector_database.vectors_for(start, stop), dtype="float32")

    # --- stage 2: canopy centroid training on a cross-shard SAMPLE --------
    if sample_size is not None and n > int(sample_size):
        # Mirror gather_canopy_sample's deterministic per-shard sampling, done
        # one shard at a time so the driver never holds the full matrix.
        rng = np.random.default_rng(seed)
        per_shard = int(sample_size) // max(1, n_workers)
        slices = []
        for w in range(n_workers):
            v = shard_vectors(w)
            take = max(1, min(per_shard, len(v)))
            idx = rng.choice(len(v), take, replace=False)
            slices.append(np.asarray(v)[idx])
        train_vectors = np.vstack(slices)
    else:
        # Small data (or an explicit full-materialization request): train on
        # the whole matrix, bit-identical to the records path.
        shards = [shard_vectors(w) for w in range(n_workers)]
        train_vectors = np.vstack(shards) if shards else np.zeros((0, 0))
    centroids = train_canopy_centroids(train_vectors, n_canopies, seed=seed, sample_size=None)
    del train_vectors

    # --- stage 3: per-shard canopy assignment (bounded), emit + hash-own ----
    centroid_to_ids: dict[int, set[int]] = {}
    for w in range(n_workers):
        v = shard_vectors(w)
        canopy = _assign_shard(v, centroids, overlap_m)
        assignments = canopy.assignments  # (shard_local_n, overlap_m) of centroids
        offset = bases[w]
        for local_i, row in enumerate(assignments):
            gi = offset + local_i
            for centroid in row:
                if centroid >= 0:
                    centroid_to_ids.setdefault(int(centroid), set()).add(gi)
        del v, canopy

    pair_buckets: dict[int, list[tuple[int, int]]] = {w: [] for w in range(n_workers)}
    seen: set[tuple[int, int]] = set()
    for ids in centroid_to_ids.values():
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

    # --- stage 4: score owned pairs; fetch ONLY owned records from the store
    scorer_state = _scorer_state_of(scorer)
    edges = _score_owned_pairs(
        pair_buckets,
        lambda positions: vector_database.records_at(positions),
        scorer_state, tau,
        n_workers=n_workers, use_threads=use_threads, executor=executor,
        progress_callback=progress_callback,
    )

    # --- stage 5: distributed closure + representatives from the store -----
    assignment = distributed_closure_reduce(
        edges, n, n_workers=n_workers, executor=executor
    )
    for cluster in assignment.clusters.values():
        cluster.representative = vector_database.record_at(cluster.representative_position)
    return assignment