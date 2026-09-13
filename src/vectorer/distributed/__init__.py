"""Distributed batch ER (subpackage).

An additive distributed executor around the framework's public stage hooks.
The single-process ``BatchPipeline.run`` is NOT touched; this package produces
the *same* cluster assignment (given the same data, geometry and scorer) by
executing the same stages across workers:

- parse/embed: shard map
- canopy train: driver (sampled) + worker local assignment
- candidate pairs: emit, dedupe, own-each-by-pair-hash
- Fellegi-Sunter: parallel map over the owned pairs (only above-tau edges
  cross the wire)
- Swoosh closure: distributed connected components (exact) over the above-tau
  edges

The package is split by responsibility so each piece stays legible:

- :mod:`._core` -- shared primitives (pair ownership by hash).
- :mod:`._canopy` -- distributed canopy blocking: shard embedding, local
  assignment, cross-shard centroid sampling.
- :mod:`._executors` -- backend abstraction (``process``, ``thread``, ``ray``).
- :mod:`._scoring` -- the distributed Fellegi-Sunter scoring map (owned pairs,
  only above-tau edges cross the wire).
- :mod:`._closure` -- the exact distributed connected-components closure
  (R-Swoosh / D-Swoosh), bit-for-bit identical to the single-process closure.
- :mod:`._tf` -- global term-frequency pre-reduction so TF weights are
  consistent across machines.
- :mod:`._batch` -- the top-level orchestrators: :func:`distributed_batch_er`
  (records form / store-fed form) wiring the stages together.

Executors
---------
* ``multiprocessing`` (default; extra-free) -- :func:`distributed_batch_er`.
* Ray is straightforward to add by swapping the executor used internally; the
  seams (worker functions, pair ownership, closure) are backend-agnostic
  (:mod:`._executors`).

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

The streaming / multi-machine building blocks (v0.4.0 distribution plan):

* :func:`distributed_score_pairs` -- the FS scoring *map* (:mod:`._scoring`).
* :func:`distributed_score_and_reduce` -- composes scoring map + closure
  reduce for a single streaming score-then-cluster call.
* :func:`streaming_distributed_closure` -- transitive closure over an iterator
  of edge chunks (streaming reduce, bounded memory).
* :func:`distributed_closure_reduce` -- multi-machine exact connected
  components (:mod:`._closure`).
* :func:`merge_tf_counters` / :func:`build_global_tf_tables` -- global
  term-frequency pre-reduction (:mod:`._tf`).
* :func:`create_executor` / :class:`RayExecutor` -- backend abstraction
  (:mod:`._executors`).

Caveats (per the plan): G-Swoosh (`gswoosh`/`cluster_with_merger`), per-query
FS scoring (incremental / link-directed), and single-machine canopy training
remain single-process by design; use the transitive-closure mode for
distributed runs and the external-vector-DB adapter for the incremental store.
"""

from ._batch import distributed_batch_er
from ._canopy import gather_canopy_sample
from ._closure import (
    distributed_closure,
    distributed_closure_reduce,
    streaming_distributed_closure,
)
from ._core import hash_pair
from ._executors import RayExecutor, create_executor
from ._scoring import distributed_score_and_reduce, distributed_score_pairs
from ._tf import build_global_tf_tables, merge_tf_counters

__all__ = [
    "RayExecutor",
    "build_global_tf_tables",
    "create_executor",
    "distributed_batch_er",
    "distributed_closure",
    "distributed_closure_reduce",
    "distributed_score_and_reduce",
    "distributed_score_pairs",
    "gather_canopy_sample",
    "hash_pair",
    "merge_tf_counters",
    "streaming_distributed_closure",
]