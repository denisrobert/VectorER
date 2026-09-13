"""Distributed Fellegi-Sunter scoring map.

The FS scoring stage spreads across workers: each pair is owned by a
deterministic balanced hash of its positions (:func:`distributed_score_pairs`
via contiguous slices), each worker scores only its owned pairs with the same
serialized scorer, and only the above-``tau`` rows cross the wire
(:func:`_score_owned_pairs` / :func:`_score_owned_pairs_worker`).
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Callable, Optional, Sequence

from ..clustering import ClusterAssignment, ScoredPair
from ..scoring import FellegiSunterScorer
from ._closure import distributed_closure_reduce


def _score_shard(left_records, right_records, scorer_state, tau):
    """Score one worker's owned pairs; return (mask, probs, weights) where
    ``mask[i]`` is True iff pair i is at/above ``tau``.  The caller aligns the
    mask with the original pair list (``_score_shard`` filters nothing)."""
    scorer = _scorer_from_state(scorer_state)
    probs = scorer.score_pairs(left_records, right_records)
    weights = scorer.match_weight_pairs(left_records, right_records)
    mask = [float(p) >= tau for p in probs]
    return mask, list(map(float, probs)), list(map(float, weights))


def _score_owned_pairs_worker(pairs, left_records, right_records, scorer_state, tau,
                              progress_queue=None, chunk_size: int = 4096):
    """Module-level worker: score ``pairs`` whose record payloads are given as
    position-aligned ``left_records``/``right_records`` lists.

    A worker receives only the records of the pairs it owns instead of a copy
    of the whole dataset.  Returns the above-``tau`` :class:`ScoredPair` edges
    with global positions.
    """
    out = []
    for start in range(0, len(pairs), chunk_size):
        end = start + chunk_size
        chunk = pairs[start:end]
        mask, probs, weights = _score_shard(
            left_records[start:end], right_records[start:end], scorer_state, tau
        )
        if progress_queue is not None:
            progress_queue.put(len(chunk))
        out.extend(
            ScoredPair(left_position=p[0], right_position=p[1],
                       probability=prob, match_weight=weight)
            for p, prob, weight, keep in zip(chunk, probs, weights, mask)
            if keep
        )
    return out


def _score_owned_pairs(
    pair_buckets,
    record_fetcher,
    scorer_state: dict,
    tau: float,
    *,
    n_workers: int,
    use_threads: bool = False,
    executor: Optional[Any] = None,
    progress_callback: Optional[Callable[[int], None]] = None,
) -> list[ScoredPair]:
    """Score each worker's owned pairs in parallel (the shared stage 4).

    ``pair_buckets[w]`` is worker ``w``'s owned ``(i, j)`` pairs, and
    ``record_fetcher(positions)`` returns the record payloads for those
    positions (position-aligned) -- the only driver-side difference between
    the records path (a look-up into the parsed list) and the store path (a
    batched ``VectorDatabase.records_at``).  Each worker therefore receives
    only the records of the pairs it owns, never a copy of the whole dataset.
    Returns the above-``tau`` ``ScoredPair`` edges.
    """
    batches: dict[int, tuple[list, list, list]] = {}
    for w in range(n_workers):
        pairs = pair_buckets[w]
        if not pairs:
            batches[w] = (pairs, [], [])
            continue
        positions = [p for pair in pairs for p in pair]
        recs = record_fetcher(positions)
        batches[w] = (pairs, recs[0::2], recs[1::2])

    progress_queue = None
    if progress_callback is not None:
        import threading

        try:
            import multiprocessing

            progress_queue = multiprocessing.Manager().Queue()
            stop = threading.Event()

            def _drain():
                while not stop.is_set():
                    try:
                        got = progress_queue.get(timeout=0.2)
                    except Exception:  # noqa: BLE001  (progress is best-effort)
                        continue
                    progress_callback(got)

            drain_thread = threading.Thread(target=_drain, daemon=True)
            drain_thread.start()
        except Exception:  # noqa: BLE001  (progress is best-effort)
            progress_queue = None

    from functools import partial

    def _thread_score(worker_id):
        pairs, left, right = batches[worker_id]
        return _score_owned_pairs_worker(
            pairs, left, right, scorer_state, tau, progress_queue=progress_queue
        )

    if executor is not None:
        use_cls = type(executor)
        if use_cls is ThreadPoolExecutor:
            scored_lists = list(executor.map(_thread_score, range(n_workers)))
        else:
            futures = [
                executor.submit(
                    partial(
                        _score_owned_pairs_worker,
                        pairs=batches[w][0], left_records=batches[w][1],
                        right_records=batches[w][2], scorer_state=scorer_state,
                        tau=tau, progress_queue=progress_queue,
                    )
                )
                for w in range(n_workers)
            ]
            scored_lists = [f.result() for f in futures]
    else:
        if use_threads:
            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                scored_lists = list(ex.map(_thread_score, range(n_workers)))
        else:
            with ProcessPoolExecutor(max_workers=n_workers) as ex:
                futures = [
                    ex.submit(
                        partial(
                            _score_owned_pairs_worker,
                            pairs=batches[w][0], left_records=batches[w][1],
                            right_records=batches[w][2], scorer_state=scorer_state,
                            tau=tau, progress_queue=progress_queue,
                        )
                    )
                    for w in range(n_workers)
                ]
                scored_lists = [f.result() for f in futures]
    if progress_queue is not None:
        stop.set()
        drain_thread.join(timeout=1.0)
    return [e for lst in scored_lists for e in lst]


def _scorer_from_state(state):
    from ..scoring import FellegiSunterScorer

    if "settings" in state:
        return FellegiSunterScorer.from_settings(state["settings"])
    if "comparisons" in state:
        return FellegiSunterScorer.from_comparisons(state["comparisons"])
    raise ValueError("scorer_state must contain 'settings' or 'comparisons'")


def _scorer_state_of(scorer: FellegiSunterScorer) -> dict:
    return {"settings": scorer.to_settings()}


def _score_pairs_worker(indices, left_records, right_records, scorer_state, tau,
                        progress_callback=None):
    """Module-level worker: score the ``indices`` slice of equal-length
    ``left``/``right`` lists, returning only ``(index, prob, weight)`` rows at
    or above ``tau``.  Used by :func:`distributed_score_pairs`.

    ``progress_callback``, when given, is called with ``(worker, done)`` every
    ``progress_every`` pairs so the controller can aggregate progress (works
    for process workers via a shared queue, and Ray via a remote callback)."""
    scorer = _scorer_from_state(scorer_state)
    left = [left_records[i] for i in indices]
    right = [right_records[i] for i in indices]
    probs = scorer.score_pairs(left, right)
    weights = scorer.match_weight_pairs(left, right)
    if progress_callback is not None:
        progress_callback(len(indices))
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


def _balanced_owned_slices(n_pairs: int, n_workers: int) -> dict[int, list[int]]:
    """Assign pair indices to ``n_workers`` **contiguous, equal-size** slices.

    Unlike :func:`_owner_of_index` (which round-robins by index), contiguous
    slices guarantee each worker gets an (almost) equal *number* of pairs, so
    the workload across cores is as equal as possible while keeping pairing
    logic simple and deterministic.
    """
    owned: dict[int, list[int]] = {w: [] for w in range(n_workers)}
    base = n_pairs // n_workers
    extra = n_pairs % n_workers
    start = 0
    for w in range(n_workers):
        take = base + (1 if w < extra else 0)
        owned[w] = list(range(start, start + take))
        start += take
    return owned


def distributed_score_pairs(
    scorer: FellegiSunterScorer,
    left_records: Sequence[dict],
    right_records: Sequence[dict],
    *,
    tau: float,
    n_workers: int = 2,
    executor: Optional[Any] = None,
    pair_positions: Optional[Sequence[tuple[int, int]]] = None,
    progress_callback: Optional[Callable[[int], None]] = None,
) -> list[Any]:
    """Score equal-length ``left``/``right`` pair lists in parallel (map).

    Pairs are partitioned into **contiguous, equal-size slices** (one per
    worker) so each worker scores an (almost) equal number of pairs -- the
    workload across cores/machines is as equal as possible while keeping
    pairing simple and deterministic.  Each worker scores its slice with the
    same serialized scorer and returns only the rows at/above ``tau``.

    ``progress_callback``, when given, is invoked as ``progress_callback(n)``
    each time a worker finishes scoring ``n`` pairs (its slice size); the
    controller accumulates these to drive an aggregate tqdm bar.  ``None``
    disables progress reporting (no change to prior behaviour).

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
    owned = _balanced_owned_slices(n_pairs, n_workers)

    from functools import partial

    # Progress: process/thread workers report through a multiprocessing Queue
    # (thread-safe and picklable); Ray is handled by the RayExecutor (the
    # worker fn runs remotely and the callback is a remote ref).  For plain
    # executors, progress_callback is fed from a Queue-watcher thread.
    progress_queue = None

    import threading

    if progress_callback is not None and executor is None:
        try:
            import multiprocessing

            progress_queue = multiprocessing.Manager().Queue()
            stop = threading.Event()

            def _drain():
                while not stop.is_set():
                    try:
                        n = progress_queue.get(timeout=0.2)
                    except Exception:
                        continue
                    progress_callback(n)

            drain_thread = threading.Thread(target=_drain, daemon=True)
            drain_thread.start()
        except Exception:  # noqa: BLE001  (progress is best-effort)
            progress_queue = None

    worker = partial(_score_slice_worker,
                     left_records=left_records, right_records=right_records,
                     scorer_state=scorer_state, tau=tau,
                     progress_queue=progress_queue)
    if executor is not None:
        scored = list(executor.map(worker, [owned[w] for w in range(n_workers)]))
    else:
        # Manager().Queue() is shared/picklable so partial args reach the
        # workers; the controller's drain thread feeds progress_callback safely.
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            scored = list(ex.map(worker, [owned[w] for w in range(n_workers)]))

    if progress_queue is not None:
        stop.set()
        drain_thread.join(timeout=1.0)
    rows = [row for chunk in scored for row in chunk]
    if pair_positions is None:
        return rows
    return [
        ScoredPair(left_position=pair_positions[idx][0],
                   right_position=pair_positions[idx][1],
                   probability=prob, match_weight=weight)
        for idx, prob, weight in rows
    ]


def _queue_put(q):
    def put(n):
        q.put(n)
    return put


def _score_slice_worker(indices, left_records, right_records, scorer_state, tau,
                        progress_queue=None):
    """Module-level worker: score a contiguous ``indices`` slice.

    ``progress_queue`` (a multiprocessing.Queue), when given, receives ``n =
    len(indices)`` after the slice is scored, so the controller can aggregate
    across workers.  Returns the above-``tau`` ``(idx, prob, weight)`` rows.
    """
    scorer = _scorer_from_state(scorer_state)
    left = [left_records[i] for i in indices]
    right = [right_records[i] for i in indices]
    probs = scorer.score_pairs(left, right)
    weights = scorer.match_weight_pairs(left, right)
    if progress_queue is not None:
        progress_queue.put(len(indices))
    return [
        (i, float(p), float(w))
        for i, p, w in zip(indices, probs, weights)
        if float(p) >= tau
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