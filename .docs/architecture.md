# vector-er architecture

This document describes the runtime operation modes of the framework and the
architecture of its two embedding-and-vector-based entity resolution (ER)
pipelines. It is a design/systems document, not an API reference: it explains
*what each stage is, why it exists, and how the stages connect*, and points at
the modules that implement each contract.

---

## 1. Overview

`vector-er` is a framework for entity resolution over dense embeddings. Its
core design principle: **the vector index is the blocking engine, and the
Fellegi-Sunter scorer is a vectorized NumPy computation rather than a SQL
pipeline.** The intended workloads fit on a single machine: one process, one
NUMA node, an in-memory index and in-memory record store. Nothing in the
framework assumes a distributed query engine, a SQL planner, or an external
linkage service — only NumPy for the scoring math and FAISS for approximate
nearest-neighbour blocking (see [`src/vectorer/sim.py`](../src/vectorer/sim.py),
[`src/vectorer/comparisons.py`](../src/vectorer/comparisons.py)).

Everything downstream of parsing consumes the same primitive: a *record* is a
plain `Mapping[str, Any]`. Records flow through shared building blocks in two
orchestrated orders (the two "modes"), described in §2.

The package is organised around contracts rather than a monolithic pipeline:

| Module | Contract / responsibility |
|---|---|
| `records.py` | Record model, schemas, parsers, embedding-text serialization |
| `embeddings.py` | `EmbeddingModel` interface + reference implementations |
| `vectorstores.py` | `IndexingStrategy` (ANN), `VectorDatabase` (records + vectors + index) |
| `blocking.py` | `VectorBlocker` (top-k search), `CanopyIndex` (k-means multi-assignment) |
| `comparisons.py` | Extensible Fellegi-Sunter comparison set (19 registered options) |
| `sim.py` | Vectorized similarity/distance primitives (Jaro, JW, edit dist, date, geo, list) |
| `scoring.py` | `WeightTable` + `FellegiSunterScorer`: level assignment, bayes factors, calibration, EM |
| `classification.py` | FS decision rule (match / possible-match / non-match) |
| `clustering.py` | Swoosh (G-Swoosh), cluster assignments, representatives |
| `incremental.py` | **Operational mode 1**: streaming/online resolution |
| `batch.py` | **Operational mode 2**: whole-dataset clustering |

### 1.1 Why not SQL?

The framework is intentionally free of SQL and SQL-engine dependencies. SQL is
a natural fit when candidate generation is a *join* over a database, but in a
vector-based design the candidates come from an ANN index / canopy partition,
not a query planner. Realizing the comparison levels as SQL would either force
every candidate generation to round-trip through a database, or re-introduce a
SQL engine the vector path does not use. Instead each comparison level is a
vectorized NumPy predicate evaluated over the whole batch of candidate pairs at
once, and the Fellegi-Sunter math is pure array algebra.

The trade-off is consciously scoped: this makes the framework **single-machine
by design** (all pair data lives in process memory) and places the burden of
scaling on better blocking rather than on distributing a SQL plan. That is
intentional for the current workload target; a distributed/SQL-based engine —
such as [Splink](https://github.com/moj-analytical-services/splink), whose
candidate generation, comparison-level evaluation and parameter-estimation are
realized as queries translatable to DuckDB, Spark, SQLite or Athena (see
[Linacre *et al.*, "Splink: Free software for probabilistic record linkage at
scale", *International Journal of Population Data Science* 7(3), 2022,
DOI [10.23889/ijpds.v7i3.1794](https://doi.org/10.23889/ijpds.v7i3.1794)) —
is a different architecture with different trade-offs, better suited to very
large or multi-node tabular workloads.

---

## 2. Operation modes

The framework supports two principal operation modes, plus the training
sub-mode used by both. They share the same comparison set, scorer, similarity
primitives and classifier, but differ in *when* records appear and *what* the
output is.

```
Mode A  Incremental (online)         Mode B  Bulk (offline / batch)
                                     ┌──────────────────────────────────────┐
 parse -> embed -> vector search     │ parse all records                    │
 blocking (top-k) -> FS score ->     │  -> embed all -> canopy blocking     │
 classify                            │  -> FS score every canopy pair        │
                                     │  -> Swoosh clustering                 │
                                     └──────────────────────────────────────┘

Training (either mode): calibrated-from-pairs or EM m/u + prior (=: W → W')
```

### 2.1 Incremental mode (`incremental.py`)

Used when records arrive one at a time and must be resolved **against an
existing reference population** (a resolved store, a deduplicated "golden"
index, or the previously ingested stream). The pipeline is *per-query* and the
stage chain is:

```
                 ┌─────────────────────────────────────────────────────────────┐
  payload        │  IncrementalPipeline.resolve()                              │
 ───────────────►│                                                             │
                 │  1. parse        -> record dict        (records.py)         │
                 │  2. embed        -> dense query vector (embeddings.py)      │
                 │  3. block        -> top-k candidates   (blocking.VectorBlocker)
                 │  4. score        -> FS posteriors      (scoring)            │
                 │  5. classify     -> Decision/Matches   (classification.py)  │
                 └──────────┬──────────────────────────────────────────────────┘
                            ▼
                     Resolution(input, retrieved, matches, decision)
```

Characteristics:

- **Reference store is a `VectorDatabase`** (default `InMemoryVectorDatabase`):
  it owns the embedding model, the ANN index (`FlatIndex` over L2-normalized
  vectors = cosine), and the record payloads the index positions map back to.
- **Blocking is top-k vector search** (`VectorBlocker.block`): the query is
  embedded, run through the FAISS index, and the top-k nearest reference
  positions are returned with their blocking scores.
- **Scoring is per-query batch**: all k candidates are scored in *one*
  evaluation pass against the query (see §4), with a precomputed query vector
  reused across stages so nothing is embedded twice.
- **Ingestion is supported**: `add`/`ingest` append resolved records back into
  the store, growing the index for future queries (the incremental corpus can
  bootstrap itself from an empty store).

### 2.2 Bulk mode (`batch.py`)

Used when a *complete dataset* must be deduplicated/clustered in one job. The
stage chain is:

```
                ┌──────────────────────────────────────────────────────────────┐
  dataset       │  BatchPipeline.run(records)                                  │
 ──────────────►│                                                              │
                │  1. parse all    -> record dicts       (records.py)          │
                │  2. embed all    -> dense matrix       (embeddings.py)       │
                │  3. block        -> overlapping       (blocking.canopy_      │
                │                     canopy pairs       blocking)             │
                │  4. score        -> FS posterior per  (scoring.score_pairs)  │
                │                     canopy pair                            │
                │  5. cluster      -> Swoosh clusters   (clustering.py)        │
                └──────────┬───────────────────────────────────────────────────┘
                           ▼
                    BatchResult(assignment, canopy, scored_pairs, timing)
```

Characteristics:

- **Blocking is canopy clustering**: the embedded matrix is passed to FAISS
  k-means; every record is multi-assigned to its top-`m` centroids; a *canopy*
  is all records assigned to one centroid; candidate pairs are the pairs
  co-occurring in any canopy. Canopies **overlap** so a true match that falls
  near a centroid boundary is still blocked together (blocking recall).
- **Scoring is two-sided batch**: the scorer evaluates all canopy candidate
  pairs as equal-length left/right sequences in vectorized chunks, producing a
  posterior and match weight per pair (see §4).
- **Clustering is Swoosh-compatible**: `SwooshClusterer` first takes the
  transitive closure of above-threshold pairs (the cheap, standard mode), and
  `gswoosh`/`cluster_with_merger` additionally re-match merged representatives
  against the candidate pair set (full G-Swoosh behaviour).
- The output is a `ClusterAssignment`: record pos -> cluster id, plus clusters
  with member positions and a *representative* record (by completeness).

### 2.3 Training sub-mode (`scoring.py`)

Both modes consume a calibrated `FellegiSunterScorer`. Two native estimators
produce the `m`/`u` (per-comparison-level match/non-match) probabilities and
the base prior:

- **Supervised**: `calibrate_from_pairs` from labelled match/non-match pair
  records (Laplace-smoothed level proportions), re-derived through the same
  level-assignment machinery used at inference.
- **Unsupervised**: `fit_em` — candidate pairs under blocking rules, `u`
  estimated from uniform random pair samples, `m` and the blocked-pair match
  proportion fit by expectation maximisation, base prior = recall-adjusted
  share of blocked matches over all possible pairs.

Trained weights are serialized (`save`/`load`) as resolved comparison dicts;
the scorer is immutable-with-respect-to-weights (rebuilding the `WeightTable`
after training picks the new m/u up).

---

## 3. Blocking architecture

Blocking exists to bound the number of pairs the expensive scorer sees. The
framework's invariant is:

> **Any pair that never co-occurs in a blocked candidate set is treated as a
> non-match.** Blocking therefore trades recall for speed, and both pipelines
> expose `k` / `overlap`/`n_canopies` knobs to widen the candidate set.

Two different geometric blocking strategies implement this invariant:

| | Incremental (`VectorBlocker`) | Bulk (`canopy_blocking`) |
|---|---|---|
| Geometry | query vector vs stored vectors | dataset vectors vs k-means centroids |
| Index | FAISS flat inner-product (exact cosine top-k) | FAISS k-means + flat inner-product for assignment |
| Overlap | `k` nearest neighbours | top-`m` centroids per record (overlap when `m>1`) |
| Candidate unit | one query vs its top-k references | all pairs inside each canopy |
| Cost | O(d) embed + O(d·N) exact scan (flat) | O(d·N·t) training + O(d·N·m) assignment |

Canopy overlap is the bulk analogue of `k`: with overlapped canopies (`m>1`),
a record appears in several cells and its pairs are evaluated in each cell it
shares — the framework's answer to the classic canopy "missing pair at the
boundary" failure.

`blocking.py` defines the shared data contract: `BlockedCandidate(record,
score, position)` for vector blocking and `CanopyIndex.assignments/canopies/
candidate_pairs()` for canopy blocking.

---

## 4. Scoring engine architecture

The scorer is where the no-SQL design is most consequential. Architecture:

```
ComparisonSpec
  ├─ output_column_name
  ├─ fields               (record columns the comparison reads)
  ├─ prescore             (opt): one vectorized pass computing shared score
  │                        arrays for the batch (e.g. a single Jaro-Winkler
  │                        array consumed by every threshold level)
  └─ levels (ordered)
       ├─ null level      (bayes factor = 1, i.e. no evidence)
       ├─ agreement levels (vectorized predicate over the batch,
       │                    e.g. jw >= 0.9, jw >= 0.7, ...)
       └─ ELSE level      (catch-all)
```

Pipeline:

```
PairValues(left[field]: obj ndarray, right[field]: obj ndarray)
  │
  └─ for each ComparisonSpec:
       │  1. prescore(pv) -> cache            (shared score arrays)
       │  2. for level in spec.levels:        (CASE priority order)
       │        mask = level.test(pv, cache)  (vectorized predicate)
       │        assign first matching level per pair       (assign_levels)
       │  3. log_bf  <- log(m/u) of the assigned level      (null -> 0)
       │  4. optional term-frequency multiplier             (exact-mu / max(tf_l,tf_r))
       ▼
   log total = log(prior odds) + Σ_comparisons log BF          (clipped)
   posterior = sigmoid(log total) ; match_weight = log total / ln 2
```

Key properties:

- **Batch vectorization over the pair set**: every predicate is evaluated over
  all pairs at once (per-comparison score arrays cached and shared across
  threshold levels, so `[0.9, 0.7]` costs one Jaro-Winkler pass).
- **Small-batch scalar fast path** in `sim.py`: for ≤64 non-trivial pairs the
  Jaro/JW/edit-distance primitives use scalar loops, which beat per-cell NumPy
  at tiny batch sizes (k≈20). The vectorized path is retained for the large
  canopy pair sets.
- **Level semantics**: levels are ordered by decreasing agreement with a
  leading null level; each non-null level carries its own `m` and `u`. When a
  level is not given explicit probabilities, defaults are assigned at build
  time from a standard scheme: the exact-match level holds a 0.95 match
  probability, intermediate levels split the remainder, and the u-probabilities
  correspond to match weights that step from strongly non-matching (± −5) to
  the exact-match weight (+10). This scheme gives well-behaved scores out of
  the box (e.g. a single exact email match under a 1e-4 prior yields posterior
  0.0929) and is what the training sub-mode replaces with data-driven m/u.
- **Weighted score = single evaluation**: `score_and_weight_batch` returns
  posterior and match weight from one model evaluation.

`WeightTable` = compiled specs + per-spec log bayes factors + term-frequency
tables (value -> relative frequency, built from an optional reference
population). Base priors default to `1e-4`; the classifier default threshold is
`0.85`.

### 4.1 Comparison registry (`comparisons.py`)

The *function set* is extensible and covers 19 options spanning the standard
attribute-comparison families used in record linkage:

- `exact_match` (with optional term-frequency adjustment)
- Jaro-Winkler / Jaro similarities at thresholds (`jaro_winkler_at_thresholds`,
  `jaro_at_thresholds`)
- edit distances at thresholds (`levenshtein_at_thresholds`,
  `damerau_levenshtein_at_thresholds`)
- set/list similarities (`jaccard_at_thresholds`,
  `cosine_similarity_at_thresholds`, `array_intersect_at_sizes`,
  `pairwise_string_distance_function_at_thresholds`)
- `date_of_birth_comparison`
- `email_comparison`, `name_comparison`, `forename_surname_comparison`
- `postcode_comparison`, `distance_in_km_at_thresholds`
- `distance_function_at_thresholds` (any registered primitive or callable)
- `absolute_date_difference_at_thresholds`, `absolute_time_difference_at_thresholds`
- `custom_comparison` (declarative conditions or a `test(PairValues, cache)`
  callable)

`Comparison` objects are declared (name + params), serializable, and buildable
into `ComparisonSpec`s via the registry; custom builders are added through
`register_comparison`. A custom level is simply a vectorized predicate — the
same machinery the built-ins use.

---

## 5. Clustering architecture (`clustering.py`)

Swoosh operates on scored pairs (bulk mode output). Design:

- **Pair-driven fit with the FS scorer**: `ScoredPair(left_position,
  right_position, probability, match_weight)` is the unit, so the expensive
  scoring happened once in §4's batch pass.
- **Transitive closure mode** (`SwooshClusterer.cluster`): union the
  above-`tau` pairs. Cheap; the standard "score then cluster" workflow. Cluster
  ids are deterministic (minimum position); representatives are the most
  *complete* records (most non-null fields).
- **G-Swoosh mode** (`gswoosh`, `cluster_with_merger`): when a merge changes a
  cluster's representative, later pairs are re-tested against the *new*
  representative (with caching of unchanged representative pairs). This is the
  Swoosh loop the name claims: match tests retried until a pass merges nothing.

`ClusterAssignment` is the portable result: `node_cluster` (pos -> cluster id),
`clusters` (id -> `Cluster` with members, representative), and counters
(`n_pairs_evaluated`, `n_pairs_matched`).

---

## 6. Data flow and lifecycle

```
declared Comparisons ──► ComparisonRegistry ──► ComparisonSpec (levels+m/u)
                                                     │
        reference records ──► WeightTable(tf tables) ├─► FellegiSunterScorer
                                                     │       (incremental + bulk)
        training pairs / population ──► calibrate / EM ─► trained scorer
                                                     │
   incremental: VectorDatabase(embedder, index)  ───┤─► IncrementalPipeline
   bulk:        embedder + scorer + canopy params ───┘─► BatchPipeline
```

Persistence boundaries: scorer `save/load` (trained comparisons + prior),
`InMemoryVectorDatabase.save/load` (FAISS index + records),
`Comparison.to_dict/from_dict`.

---

## 7. Extension points

| What you want to change | Hook |
|---|---|
| Embedding model | implement `EmbeddingModel` (or `SentenceTransformerEmbedding` / `CharacterHashingEmbedding`) and pass `embedder=` |
| ANN index | implement `IndexingStrategy` (`FlatIndex` is the cosine reference) |
| Blocking geometry | implement a `VectorBlocker`-style blocker or a canopy variant feeding `CanopyIndex` |
| Comparison set | `register_comparison` / `make_comparison`; custom levels are `test(PairValues, cache) -> mask` |
| Similarity primitive | register a callable with `distance_function_at_thresholds` or use it directly in a custom level |
| m/u / prior | `calibrate_from_pairs`, `fit_em`, or override per-level m/u in resolved dicts |
| Decision rule | subclass `Classifier`/`ThresholdClassifier` (`tau`, optional `possible_low` band) |
| Cluster merge rule | pass `merge=` to `gswoosh`/`SwooshClusterer` (default: completeness) |
| Stage behaviour | every stage of both pipelines is a public method you can override or swap |

---

## 8. Performance characteristics (as benchmarked)

See `benchmarks/benchmark_incremental_er.py` and `benchmarks/benchmark_bulk_er.py`
for methodology and artifacts under `results/`. High-level numbers from the
current native stack (single machine, 384-d embeds by default):

- **Incremental (cold per-query, 50k reference index, k=20, tau=0.85):**
  ~4–5 ms mean end-to-end with the deterministic 384-d hashing embedder;
  ~33 ms with the MiniLM sentence embedder (embedding dominates, parity with
  the embedding cost of the transformer itself).
- **Bulk (canopy -> FS -> Swoosh):** Fellegi-Sunter scoring throughput of
  ~6.4k–8.2k candidate pairs/s in pure NumPy; a 52k-record dataset deduplicates
  in ~254 s with precision 1.0 and recall 0.97–1.0.

These figures are a property of the architecture (vectorized pair evaluation,
shared score arrays, scalar fast paths) and of the block: embedding dominates
in both modes when a real transformer model is used.