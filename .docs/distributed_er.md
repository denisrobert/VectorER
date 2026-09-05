# Distributed ER: multi-node operation

The framework can run beyond a single machine. This document is the
**operations guide**: how to shard, stream, and — where it does not fit — treat
as a caveat.

## What distributes, and how

| Stage | Multi-node strategy |
|---|---|
| Parse + embed | record shards handled per machine (map). |
| Canopy blocking | centroids trained once on a **cross-machine sample** (`gather_canopy_sample`); each machine assigns its own records against them. |
| Candidate pairs | emitted per shard; each pair owned by a deterministic balanced hash. |
| **Fellegi-Sunter scoring** | the expensive stage: pair-hash-owned, each worker rebuilds the scorer from serialized settings, returns only above-`tau` edges (a streaming map). |
| Swoosh closure | per-machine union-find + shared-node merge into min-position ids (`distributed_closure_reduce`, exact). |
| Term-frequency tables | pre-reduced globally (`merge_tf_counters` / `build_global_tf_tables`) so TF weights match on every machine. |

## What stays single-process **by design** (caveats, not forced)

- **G-Swoosh** (`gswoosh` / `cluster_with_merger`) — merge order is globally
  significant (a merge changes the representative later pairs match against).
  Use the transitive-closure mode for distributed runs; the closure shards
  exactly.
- **Per-query FS scoring** (incremental / link-directed) — `k` is small, so
  distributing it adds latency, not scale.  Distribute the *store* instead.
- **k-means canopy training** — one global sample gather; cheap.

## Running it

### Batch (multi-node via Ray)

```bash
# start a Ray cluster
ray start --head --port=6379                          # on the head node
ray start --address=<head-ip>:6379                    # on each worker node

# from any node
python examples/multi_node_distributed_er.py --n-base 5000 --n-workers 4 \
    --ray-address <head-ip>:6379 --verify
```

`--verify` asserts the distributed assignment equals the single-process
`BatchPipeline.run`.  `--ray-address auto` starts/joins a **local** Ray instance
(same code path, useful for a single-host demo).

A quick equivalence check lives in `examples/distributed_streaming_er.py`;
`benchmarks/benchmark_bulk_er_multinode.py` times single vs a simulated 2-node
cluster on one host.

### Incremental / link-directed (external distributed vector DB)

For huge reference stores in online mode, point the `VectorDatabase` interface
at an external distributed vector database (Qdrant shipped):

```python
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance
from vectorer.incremental import IncrementalPipeline
from vectorer.vectorstore_adapters import QdrantVectorDatabase

client = QdrantClient(host="localhost", port=6333)
db = QdrantVectorDatabase(embedder=embedder, client=client,
                          collection="people", vector_size=384,
                          distance=Distance.COSINE)
db.add(reference_records)                       # embed + upsert into the cluster
pipeline = IncrementalPipeline.from_store(db, scorer, k=20, tau=0.85)
result = pipeline.resolve(record)               # ANN search + local FS scoring
```

Only the index and the record payloads go remote; the embedding model and the
FS scorer stay local.  The adapter is a `VectorDatabase` (any `index.search`/
`record_at`/`add`/`__len__` back end works the same), so this is the
contribution-friendly seam for other vector DBs.

## Building blocks reference

- `distributed_batch_er` — orchestration of the whole batch pipeline across
  the register map/reduce stages (process/thread/Ray executors).
- `distributed_score_pairs` — the FS scoring map (with `pair_positions=` for
  `ScoredPair` output).
- `distributed_score_and_reduce` — score map + closure reduce in one call.
- `streaming_distributed_closure` — transitive closure over an edge *stream*.
- `distributed_closure_reduce` — exact connected components across machines.
- `create_executor` / `RayExecutor` — backend abstraction.
- `gather_canopy_sample`, `merge_tf_counters`, `build_global_tf_tables` —
  Milestone-C helpers for memory-bounded canopy training and global TF tables.

All results are deterministic and **identical** to the single-process pipelines
(candidate-pair set, scorer settings, and union-find are shared; only *which
machine* computes each pair differs).