# vector-er architecture

This document describes the runtime operation modes of the framework and the
architecture of its three embedding-and-vector-based entity resolution (ER)
pipelines. It is a design/systems document, not an API reference: it explains
*what each stage is, why it exists, and how the stages connect*, and points at
the modules that implement each contract.

---

## 1. Overview

`vector-er` is a framework for entity resolution over dense embeddings. Its
core design principle: **the vector index is the blocking engine, and the
Fellegi-Sunter scorer is a vectorized NumPy computation rather than a SQL
pipeline [1][2].** The intended workloads fit on a single machine: an in-memory
index and in-memory record store, optionally parallelized across the machine's
cores via the distributed batch executor (inter-machine/cluster scale-out is
out of scope). Nothing in the
framework assumes a distributed query engine, a SQL planner, or an external
linkage service — only NumPy for the scoring math and FAISS for approximate
nearest-neighbour blocking [11] (see
[`src/vectorer/sim.py`](../src/vectorer/sim.py),
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
| `link.py` | **Operational mode 3**: two-database record linkage |
| `distributed.py` | Additive distributed executor for batch ER (same result as `batch.py`) |

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
by design** (all pair data lives in this machine's memory, optionally spread
across its processes by the distributed executor) and places the burden of
scaling on better blocking rather than on distributing a SQL plan. That is
intentional for the current workload target; a distributed/SQL-based engine —
such as [Splink](https://github.com/moj-analytical-services/splink) [19], whose
candidate generation, comparison-level evaluation and parameter-estimation are
realized as queries translatable to DuckDB, Spark, SQLite or Athena — is a
different architecture with different trade-offs, better suited to very large
or multi-node tabular workloads.

---

## 2. Operation modes

The framework supports three principal operation modes, plus the training
sub-mode used by them all. They share the same comparison set, scorer,
similarity primitives and classifier, but differ in *when* records appear and
*what* the output is.

```
Mode A  Incremental (online)      Mode B  Bulk (offline / batch)   Mode C  Link (two DBs)
                                  ┌─────────────────────────────┐  project canonical fields
 parse -> embed -> vector search │ parse all records           │  A ─► block vs indexed B ─►
 blocking (top-k) -> FS score -> │ -> embed all -> canopy      │  FS score top-k ─► link edges
 classify                        │ -> FS score every canopy    │  (directed), or cross-DB
                                  │ -> Swoosh clustering        │  canopy pairs (symmetric)
                                  └─────────────────────────────┘
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
- **Ingestion is supported, including novelty-only ingestion**:
  `add`/`ingest` append resolved records back into the store, growing the index
  for future queries (the incremental corpus can bootstrap itself from an empty
  store). For streams that should contain only *new* entities, the
  novelty-only switch (`ingest_novel`, `ingest_novel_many`) first resolves the
  record against the store and appends it **only when no reference record
  scores at or above the novelty threshold** (`tau` by default, or a custom
  `novelty_threshold`): exact and near-duplicates are skipped, so the index
  grows only on genuinely novel records rather than re-ingesting the already-known
  population.

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

- **Blocking is canopy clustering [12]**: the embedded matrix is passed to
  FAISS k-means; every record is multi-assigned to its top-`m` centroids; a
  *canopy* is all records assigned to one centroid; candidate pairs are the
  pairs co-occurring in any canopy. Canopies **overlap** so a true match that
  falls near a centroid boundary is still blocked together (blocking recall).
- **Scoring is two-sided batch**: the scorer evaluates all canopy candidate
  pairs as equal-length left/right sequences in vectorized chunks, producing a
  posterior and match weight per pair (see §4).
- **Clustering is Swoosh-compatible [13]**: `SwooshClusterer` first takes the
  transitive closure of above-threshold pairs (the cheap, standard mode), and
  `gswoosh`/`cluster_with_merger` additionally re-match merged representatives
  against the candidate pair set (full G-Swoosh behaviour).
- The output is a `ClusterAssignment`: record pos -> cluster id, plus clusters
  with member positions and a *representative* record (by completeness).

### 2.3 Training sub-mode (`scoring.py`)

All three modes consume a calibrated `FellegiSunterScorer`.  Two native
estimators produce the `m`/`u` (per-comparison-level match/non-match)
probabilities and the base prior, following the Fellegi-Sunter estimation
literature [1]:

- **Supervised**: `calibrate_from_pairs` from labelled match/non-match pair
  records (Laplace-smoothed level proportions), re-derived through the same
  level-assignment machinery used at inference — the calibration-of-false-match
  rates route of Belin & Rubin [10].  Labelled pairs may come from one
  database's duplicates, cross-database labelled links (the Link mode), or a
  gold set.
- **Unsupervised**: `fit_em` — candidate pairs under blocking rules, `u`
  estimated from uniform random pair samples, `m` and the blocked-pair match
  proportion fit by expectation maximisation [1][19], base prior = recall-adjusted
  share of blocked matches over all possible pairs.  Works on a single
  duplicate-bearing population; for the Link mode, fit on a pooled or
  representative population then apply the same scorer to both sides.

Trained weights are serialized (`save`/`load`) as resolved comparison dicts;
the scorer is immutable-with-respect-to-weights (rebuilding the `WeightTable`
after training picks the new m/u up).  Whichever mode trains the scorer, the
same calibrated model is then used by incremental, bulk, and Link alike — each
pipeline receives it via its `scorer=` parameter.

### 2.4 Link mode (`link.py`)

Used when records come from **two separately-managed databases** — different
schemas, overlapping compared fields — and must be *linked*, not merged:
the output is a table of **link edges** `(a_id, b_id, posterior, weight,
decision)`, and neither database is mixed into the other.  This is the
merger / cross-enterprise-collaboration use case.

```
canonical projection (per-DB FieldMap)
  -> directed:  index B (canonicalized) -> top-k ANN of each A record -> FS
  -> symmetric: canopy-block canonicalized A+B -> FS score only cross-DB pairs
  -> classify (tau / possible_low bands) -> LinkTable of edges
```

Key design:

- **`FieldMap`** maps each DB's own columns onto the canonical compared fields,
  with optional per-field normalisers, so the two heterogeneous schemas align
  without renaming the caller's data.
- **Blocking runs on canonical embeddings** (both sides are projected before
  embed/compare), so ANN/canopy blocking compares aligned text.  A canonical
  field absent on one side is simply `None` on that side; FS turns it into a
  null level (no evidence), so overlap *within* the compared fields is handled
  for free.
- **`RecordLinker`** (`link_directed`, `link_symmetric`, `link`) reuses the
  framework's incremental pipeline, `canopy_blocking`, `score_pairs`,
  `calibrate_from_pairs`, and `ThresholdClassifier` as orchestrated stages —
  no new scoring/blocking/clustering internals.
- **1:1 vs 1:N** is a caller choice (`enforce_11` in directed mode; symmetric
  emits all above-tau cross-DB pairs).
- **`LinkTable`** exposes `matches`/`possible_matches` bands, `as_pairs()`,
  `by_a()`/`by_b()`, and `to_dict()` for export.

---

## 3. Blocking architecture

Blocking exists to bound the number of pairs the expensive scorer sees. The
framework's invariant is:

> **Any pair that never co-occurs in a blocked candidate set is treated as a
> non-match.** Blocking therefore trades recall for speed, and all blocking modes
> expose knobs to widen the candidate set — `k` for incremental, `overlap` /
> `n_canopies` for bulk and Link's symmetric path.

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
boundary" failure [12].

Dense embeddings for the blocking stage follow a well-established line of work:
distributed tuple representations for ER (DeepER) [15], pre-trained-embedding
blocking compared head-to-head [16], universal dense blocking (UniBlocker) [17],
and deep-learning blocking against exact-match blocking [18].

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
   log total   (clipped)
   posterior, match_weight
```

The per-pair math, in Jax Math:

$$ \mathrm{logBF} = \log\left(\frac{m_i}{u_i}\right) \ \text{for the assigned agreement level } i, \qquad \mathrm{logBF} = 0 \ \text{for the null level} $$

$$ \log\mathrm{total} = \log\left(\frac{p_0}{1-p_0}\right) + \sum_{\text{assigned}} \mathrm{logBF}, \qquad L = \mathrm{clip}\left(\log\mathrm{total},\ -\ln 10^{300},\ \ln 10^{300}\right) $$

$$ p(\mathrm{match}) = \sigma(L) = \frac{1}{1+e^{-L}}, \qquad \mathrm{match\_weight} = \frac{L}{\ln 2} $$

Key properties:

- **Batch vectorization over the pair set**: every predicate is evaluated over
  all pairs at once (per-comparison score arrays cached and shared across
  threshold levels, so `[0.9, 0.7]` costs one Jaro-Winkler pass [6][7]).
- **Small-batch scalar fast path** in `sim.py`: for ≤64 non-trivial pairs the
  Jaro/JW/edit-distance primitives use scalar loops, which beat per-cell NumPy
  at tiny batch sizes (k≈20). The vectorized path is retained for the large
  canopy pair sets.
- **Level semantics**: levels are ordered by decreasing agreement with a
  leading null level; each non-null level carries its own $m$ and $u$ [1].
  When a level is not given explicit probabilities, defaults are assigned at
  build time from a standard scheme mirroring the classic match-weight
  construction [3]: the exact-match level holds a $0.95$ match probability,
  intermediate levels split the remainder, and the $u$-probabilities correspond
  to match weights that step from strongly non-matching ($-5$) to the exact-match
  weight ($+10$). This scheme gives well-behaved scores out of the box (e.g. a
  single exact email match under a $10^{-4}$ prior yields posterior $0.0929$) and is
  what the training sub-mode replaces with data-driven $m/u$.
- **Weighted score = single evaluation**: `score_and_weight_batch` returns
  posterior and match weight from one model evaluation.
- **Union-Class existence lift**: a compared field whose value is a
  ``set``/``frozenset`` marks a *union of alternatives* (a synthetic master
  record from :func:`union_merge`).  Such pairs are expanded over their value
  combinations and the **maximum** posterior returned — the Union-Class match
  function ``M(r1,r2) = true iff some value pair matches`` (Swoosh Prop. 2.4).
  List/tuple values are untouched (they are comparison columns for the
  list-aware comparisons), so union records coexist with vector/tag fields.
  Combined with the reflexive default (`idempotent=True`) and a set-union
  merge, this is the ICAR-compatible Union-Class construction.
- **Reflexive by default** (`idempotent=True` on `FellegiSunterScorer`):
  content-identical pairs (equal on every compared field) are forced to
  posterior `1.0`.  This guarantees the *idempotence/reflexivity* property
  `r ≈ r` of the Swoosh Union-Class ICAR construction.  Without it, a "thin"
  record (few non-null comparison fields) would score against itself at the
  prior (all null levels carry no evidence, e.g. 0.0001), failing `r ≈ r`.
  The check is a cheap O(#comparisons) content-equality test over the compared
  columns only; it is applied to the posterior in `score`/`score_batch`/
  `score_pairs` (match weights keep the true self-weight) and can be disabled
  with `idempotent=False` to recover the raw posterior for identical pairs.

`WeightTable` = compiled specs + per-spec log bayes factors + term-frequency
tables (value -> relative frequency, built from an optional reference
population), implementing frequency-based matching [4]. Base priors default to
$10^{-4}$; the classifier default threshold is $0.85$ (operating-score cut-off,
cf. FS decision rules [3][5]).

### 4.1 Comparison registry (`comparisons.py`)

The *function set* is extensible and covers 19 options spanning the standard
attribute-comparison families used in record linkage [14]:

- `exact_match` (with optional term-frequency adjustment [4])
- Jaro-Winkler / Jaro similarities at thresholds (`jaro_winkler_at_thresholds`,
  `jaro_at_thresholds`) [6][7]
- edit distances at thresholds (`levenshtein_at_thresholds`,
  `damerau_levenshtein_at_thresholds`) [8][9]
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

Swoosh [13] operates on scored pairs (bulk mode output). Design:

- **Pair-driven fit with the FS scorer**: `ScoredPair(left_position,
  right_position, probability, match_weight)` is the unit, so the expensive
  scoring happened once in §4's batch pass.
- **Transitive closure mode** (`SwooshClusterer.cluster`): union the
  above-`tau` pairs. Cheap; the standard "score then cluster" workflow (a
  connected-component-style reduction of the scored graph [14]). Cluster
  ids are deterministic (minimum position); representatives are produced by the
  configured merge function.
- **G-Swoosh mode** (`gswoosh`, `cluster_with_merger`): when a merge changes a
  cluster's representative, later pairs are re-tested against the *new*
  representative (with caching of unchanged representative pairs). This is the
  G-Swoosh loop formalized in the Swoosh family of algorithms [13]: match
  tests retried until a pass merges nothing.

**Merge functions.** `merge(records, positions) -> (representative, position)`
produces the cluster's representative. Three are provided out of the box:

* :func:`select_representative` (default) — the most complete member record
  (most non-`None` fields). Representative is an *existing* record, so its
  position is a valid index into ``records``.
* :func:`union_merge` — a **synthetic master record** whose fields hold the
  union of every value seen across the matched records (set-valued fields, the
  Swoosh Union Class). Returns position ``-1``; the representative is not any
  member.
* :func:`latest_merge` — a **synthetic master record** whose fields hold the
  most recent value per attribute, keyed on a ``timestamp_field`` (newest
  non-`None` value per field). Position anchors to the newest member.

The Swoosh algorithms store the representative **record object** (not just an
index), so synthetic representatives (``position=-1``) flow through unchanged.
Because synthetic master records can hold set-valued fields, the scorer
implements the **Union-Class existence lift**: when a compared field holds a
``set``/``frozenset``, the pair is expanded over its value combinations and the
**maximum** posterior (the ``∃`` value pair of the Union Class) is returned
(see §4). List/tuple field values are *not* union-lifted — they are
comparison-column values (embedding vectors, tag lists) for the list-aware
comparisons. Pass a merge function through the batch pipeline via
``build_batch_pipeline(..., merge=union_merge)``.

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
| Embedding model | implement `EmbeddingModel` (or `SentenceTransformerEmbedding` / `CharacterHashingEmbedding`) and pass `embedder=` — or wrap an already-instantiated, GPU/quantized model via `SentenceTransformerEmbedding(model=...)`; see `user_guide.md` §0 |
| ANN index | implement `IndexingStrategy` (`FlatIndex` is the cosine reference) |
| Reference store / scaling | implement `VectorDatabase` against an external (distributed) vector DB — see §7.1 |
| Blocking geometry | implement a `VectorBlocker`-style blocker or a canopy variant feeding `CanopyIndex` |
| Comparison set | `register_comparison` / `make_comparison`; custom levels are `test(PairValues, cache) -> mask` |
| Similarity primitive | register a callable with `distance_function_at_thresholds` or use it directly in a custom level |
| m/u / prior | `calibrate_from_pairs`, `fit_em`, or override per-level m/u in resolved dicts |
| Decision rule | subclass `Classifier`/`ThresholdClassifier` (`tau`, optional `possible_low` band) |
| Cluster merge rule | pass `merge=` to `gswoosh`/`SwooshClusterer` (default: completeness) |
| Stage behaviour | every stage of the pipelines is a public method you can override or swap |

### 7.1 Scaling the reference store: external distributed vector DBs

The **incremental** pipeline's only contact with storage is the
`VectorDatabase` interface (`vectorstores.py`): it calls `index.search(query, k)`
→ `(indices, scores)` for blocking, `record_at(position)` to fetch candidate
records, `embedding` to embed queries, and `add`/`update`/`delete` for
ingestion.  That makes an **external distributed vector database a drop-in
replacement** — implement `VectorDatabase` (and optionally `IndexingStrategy`)
against Qdrant, Milvus, Pinecone, Weaviate, Elasticsearch, ChromaDB, etc., and
the incremental pipeline scales horizontally across nodes with **no pipeline
changes**.

```
IncrementalPipeline ──► VectorDatabase (interface) ──► external vector DB
                         ├─ index.search  ──────────► HNSW/sharded ANN search
                         ├─ record_at     ──────────► fetch payload by id
                         ├─ add/update/delete ──────► upsert (vector, id, payload)
                         └─ embedding     ──────────► stays LOCAL (model)
```

Only the **index and the record payloads move remote**; the embedding model and
the FS scorer stay local.  A natural mapping is to use the framework's record
position as the external document id and store the record dict as the payload,
so `index.search` returns positions and `record_at(position)` becomes an id
fetch.  External DBs typically provide HNSW/IVF indexes and sharding, which
replace the in-memory FAISS flat scan (the real ANN bottleneck as N grows).

**When to choose this vs the intra-machine distributed executor:** the batch
distributed executor (`distributed_batch_er`) parallelizes across one machine's
cores and is the answer for *whole-dataset* dedup/clustering.  The external-DB
route is the answer for *huge reference stores* in incremental/online service
mode — horizontal ANN scaling and persistence without touching the pipeline.
The two are orthogonal and can coexist.

**Caveats:** the adapter must preserve cosine semantics (L2-normalize or use
the DB's cosine metric to keep scores comparable with the local `FlatIndex`);
be mindful of payload-size limits and serialization cost when storing records as
payloads; and account for eventual-consistency / freshness if ingestion is
asynchronous (`ingest_novel` expects immediate visibility).

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

---

## 9. References

1. Fellegi, I. P., & Sunter, A. B. (1969). *A Theory for Record Linkage.*
   Journal of the American Statistical Association, 64(328), 1183–1210.
   [DOI 10.1080/01621459.1969.10501049](https://doi.org/10.1080/01621459.1969.10501049)
2. Winkler, W. E. (2006). *Overview of Record Linkage and Current Research
   Directions.* Statistical Research Division, U.S. Bureau of the Census,
   Research Report Series RRS2006/02.
   [Link](https://www.census.gov/library/working-papers/2006/adrm/rrs2006-02.html)
3. Winkler, W. E. (1993). *Improved Decision Rules in the Fellegi–Sunter Model
   of Record Linkage.* Proceedings of the Section on Survey Research Methods,
   American Statistical Association, 274–279.
4. Winkler, W. E. (2000). *Frequency-Based Matching in the Fellegi–Sunter Model
   of Record Linkage.* Statistical Research Division, U.S. Bureau of the
   Census, Research Report Series RR2000/06.
   [Link](https://www.census.gov/library/working-papers/2000/adrm/rr2000-06.html)
5. Sadinle, M., & Fienberg, S. E. (2013). *A Generalized Fellegi–Sunter
   Framework for Multiple Record Linkage with Application to Homicide Record
   Systems.* Journal of the American Statistical Association, 108(502), 651–660.
   [arXiv:1205.3217](https://arxiv.org/abs/1205.3217)
6. Jaro, M. A. (1989). *Advances in Record-Linkage Methodology as Applied to
   Matching the 1985 Census of Tampa, Florida.* Journal of the American
   Statistical Association, 84(406), 414–420.
   [DOI 10.1080/01621459.1989.10478785](https://doi.org/10.1080/01621459.1989.10478785)
7. Winkler, W. E. (1990). *String Comparator Metrics and Enhanced Decision
   Rules in the Fellegi–Sunter Model of Record Linkage.* Proceedings of the
   Section on Survey Research Methods, American Statistical Association, 354–359.
8. Levenshtein, V. I. (1966). *Binary Codes Capable of Correcting Deletions,
   Insertions, and Reversals.* Soviet Physics Doklady, 10(8), 707–710.
9. Damerau, F. J. (1964). *A Technique for Computer Detection and Correction
   of Spelling Errors.* Communications of the ACM, 7(3), 171–176.
   [DOI 10.1145/363958.363994](https://doi.org/10.1145/363958.363994)
10. Belin, T. R., & Rubin, D. B. (1995). *A Method for Calibrating False-Match
    Rates in Record Linkage.* Journal of the American Statistical Association,
    90(430), 694–707.
    [DOI 10.1080/01621459.1995.10476563](https://doi.org/10.1080/01621459.1995.10476563)
11. Johnson, J., Douze, M., & Jégou, H. (2021). *Billion-Scale Similarity Search
    with GPUs.* IEEE Transactions on Big Data, 7(3), 535–547.
    [DOI 10.1109/TBDATA.2019.2921572](https://doi.org/10.1109/TBDATA.2019.2921572)
    · Preprint: [arXiv:1702.08734](https://arxiv.org/abs/1702.08734)
12. McCallum, A., Nigam, K., & Ungar, L. H. (2000). *Efficient Clustering of
    High-Dimensional Data Sets with Application to Reference Matching.*
    In Proceedings of the Sixth ACM SIGKDD International Conference on
    Knowledge Discovery and Data Mining (KDD), 169–178.
    [DOI 10.1145/347090.347123](https://doi.org/10.1145/347090.347123)
13. Benjelloun, O., Garcia-Molina, H., Menestrina, D., Su, Q., Whang, S. E., &
    Widom, J. (2009). *Swoosh: A Generic Approach to Entity Resolution.*
    The VLDB Journal, 18(1), 255–276.
    [DOI 10.1007/s00778-008-0098-x](https://doi.org/10.1007/s00778-008-0098-x)
14. Christen, P. (2012). *Data Matching: Concepts and Techniques for Record
    Linkage, Entity Resolution, and Duplicate Detection.* Springer,
    Data-Centric Systems and Applications.
    [DOI 10.1007/978-3-642-31164-2](https://doi.org/10.1007/978-3-642-31164-2)
15. Ebraheem, M., Thirumuruganathan, S., Joty, S., Ouzzani, M., & Tang, N.
    (2018). *Distributed Representations of Tuples for Entity Resolution
    (DeepER).* Proceedings of the VLDB Endowment, 11(11), 1454–1467.
    [DOI 10.14778/3236187.3236198](https://doi.org/10.14778/3236187.3236198)
16. Zeakis, A., Papadakis, G., Skoutas, D., & Koubarakis, M. (2023).
    *Pre-Trained Embeddings for Entity Resolution: An Experimental Analysis.*
    Proceedings of the VLDB Endowment, 16(11), 3239–3251.
    [DOI 10.14778/3598581.3598594](https://doi.org/10.14778/3598581.3598594)
17. Wang, T., Lin, H., Han, X., Chen, X., Cao, B., & Sun, L. (2024). *Towards
    Universal Dense Blocking for Entity Resolution (UniBlocker).* arXiv preprint
    2404.14831. [arXiv:2404.14831](https://arxiv.org/abs/2404.14831)
18. Thirumuruganathan, S., Li, H., Tang, N., Ouzzani, M., Govind, Y., Paulsen,
    D., Fung, G., & Doan, A. (2021). *Deep Learning for Blocking in Entity
    Matching.* Proceedings of the VLDB Endowment, 14(11), 2459–2472.
    [DOI 10.14778/3476249.3476294](https://doi.org/10.14778/3476249.3476294)
19. Linacre, R., Lindsay, S., Manassis, T., Slade, Z., Hepworth, T., Kennedy,
    R., & Bond, A. (2022). *Splink: Free Software for Probabilistic Record
    Linkage at Scale.* International Journal of Population Data Science, 7(3).
    [DOI 10.23889/ijpds.v7i3.1794](https://doi.org/10.23889/ijpds.v7i3.1794)