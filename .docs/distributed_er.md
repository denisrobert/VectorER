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

client = QdrantClient(host="127.0.0.1", port=6333)
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

## Latency trade-off: in-memory vs external store

The benchmark numbers below come from the same machine, the same hashing
embedder (384-d), the same k=20, and the same 100 close-variant queries, so
they isolate the *store*:

| store | refs | mean /query | p50 | p95 | p99 |
|---|---|---|---|---|---|
| in-memory `FlatIndex` (`incremental_latency.json`) | 20 000 | 4.7 ms | 4.7 ms | 5.7 ms | 7.0 ms |
| in-memory `FlatIndex` (`incremental_latency_paperscale.json`) | 50 000 | 5.7 ms | 5.7 ms | 7.0 ms | 7.6 ms |
| Qdrant server (`incremental_qdrant_latency.json`) | 20 000 | 17.3 ms | 16.1 ms | 33.8 ms | 48.4 ms |

**In-memory scales in latency with dataset size** — the local `FlatIndex` adds
a few microseconds/ms per extra record (a linear scan of the embedding matrix),
so 20k → 50k costs about +1 ms/query.

**The external store adds a fixed per-query round-trip** but its search cost is
essentially flat in dataset size (ANN index on the server): the 20k-Qdrant
number is ~3.6× the in-memory 20k number, and the gap does not widen as the
collection grows.  The adapter already keeps the query to **one** server
round-trip (payloads fetched in the same search call, count cached), so the
Qdrant overhead is dominated by network latency — roughly 13 ms in the numbers
above, independent of N.

Choose the store on where the curve crosses:

- **Small / single-node reference stores** (≤ ~50k records, all scoring on one
  host): in-memory `FlatIndex` is strictly better — no network hop.
- **Huge / multi-node reference stores** (one machine can't hold the index or
  the memory budget, or ingestion is shared): the fixed round-trip cost is the
  *same per query regardless of N*, so Qdrant becomes the right choice as N
  grows — the +13 ms is amortized against an in-memory index that either does
  not fit, is shared across a fleet, or would need to page.

Two practical notes from the benchmark:

- Use `127.0.0.1`, not `localhost`, for local Qdrant: `localhost` can resolve
  to `::1` (IPv6) while Qdrant listens on IPv4 only, which adds ~5 s per call.
- The Qdrant benchmark is `benchmarks/benchmark_incremental_er_qdrant.py`
  (`--recreate` rebuilds the collection; `--breakdown` splits embed / Qdrant
  block / scorer).

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