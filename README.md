# vector-er

[![PyPI version](https://img.shields.io/pypi/v/vectorer.svg)](https://pypi.org/project/vectorer/)
[![Python versions](https://img.shields.io/pypi/pyversions/vectorer.svg)](https://pypi.org/project/vectorer/)

A framework for **embedding-and-vector-based entity resolution** with two
composable pipelines and an extensible **Fellegi-Sunter** (FS) comparison set
spanning 19 options across the standard attribute-comparison families of record
linkage — implemented **natively in NumPy, with no SQL engine and no external
linkage dependencies**. It runs on anything from a single laptop to a
**multi-node cluster**: the batch pipeline's expensive stages (FS scoring,
candidate blocking, Swoosh closure) shard and stream across machines via an
optional Ray backend, and the incremental reference store scales horizontally
through external distributed vector databases.

| Pipeline | Stage chain | Use case |
|---|---|---|
| **Incremental** | parsing -> embedding -> vector search blocking (top-k) -> FS scoring on the top-k -> classification | online / streaming resolution of one record against a reference store |
| **Batch** | parsing -> embedding -> canopy blocking on the embedded dataset -> FS scoring of every canopy pair -> Swoosh clustering | offline deduplication / clustering of a whole dataset |
| **Link** | canonical projection -> (directed) index B + resolve A, or (symmetric) cross-DB canopy pairs -> FS scoring -> link edges | linking **two separately-managed databases** (mergers / cross-enterprise collaborations) |

**Why no SQL?** Comparison levels are naturally a vectorized computation when
the candidates come from an ANN index or canopy partition rather than a
database join. Every level here is a NumPy predicate evaluated over the whole
batch of candidate pairs at once, and the Fellegi-Sunter math (level assignment
-> bayes factors -> posterior) is pure array algebra.

## Documentation

- **[`.docs/architecture.md`](.docs/architecture.md)** — operation modes,
  pipeline architecture for incremental and bulk ER, blocking/scoring/clustering
  design, extension points, and design rationale.
- **[`.docs/user_guide.md`](.docs/user_guide.md)** — hands-on guide covering
  both pipelines end to end: comparison selection, reference-store persistence,
  ingestion modes, batch clustering, calibration, threshold choice, and
  production notes built around the incremental person-resolution use case.
- **API reference (`docs/`)** — Sphinx/ReadTheDocs pages. Build locally with:

  ```bash
  pip install -e ".[docs]"
  sphinx-build -b html docs docs/_build/html
  ```

## Installation

`vectorer` is published on PyPI. Install it with `pip` (Python `>= 3.10`):

```bash
# Core: vectorized scoring (numpy) + vector ANN/blocking (faiss-cpu).
pip install vectorer

# Optional: real (sentence-transformers) embedding model
pip install "vectorer[embedding]"
```

For development / to run the test suite, install from a source checkout
(recommended in a virtualenv):

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows PowerShell
pip install -e ".[test]"
```

The only runtime dependencies are `numpy` (vectorized scoring) and `faiss-cpu`
(vector ANN/blocking).

## Quick start

### Incremental pipeline

```python
from vectorer.comparisons import make_comparison
from vectorer.incremental import build_incremental_pipeline

references = [{"first_name": "John", "last_name": "Smith", "date_of_birth": "1985-06-15", ...}, ...]

pipeline = build_incremental_pipeline(
    references,
    comparisons=[
        make_comparison("jaro_winkler_at_thresholds", col_name="first_name"),
        make_comparison("jaro_winkler_at_thresholds", col_name="last_name"),
        make_comparison("date_of_birth_comparison", col_name="date_of_birth"),
    ],
    k=20, tau=0.85,
)

result = pipeline.resolve({"first_name": "Jon", "last_name": "Smyth", "date_of_birth": "1985-06-15"})
result.decision            # Decision.MATCH / Decision.NON_MATCH
result.matches[0].match_probability
```

**Novelty-only ingestion.** Ingestion supports a switch that adds a record to
the reference store *only if it is novel* — i.e. no record in the index scores
at or above the novelty threshold:

```python
pipeline.ingest(record)                 # always append (returns new position)
pipeline.ingest_novel(record)           # append only if novel; None if a match exists
pipeline.ingest_novel(record, novelty_threshold=0.5)  # custom novelty bar (default = tau)
positions = pipeline.ingest_novel_many(deck)   # aligned positions; None = skipped duplicate
```

`ingest_novel` resolves the record against the store first and appends it only
when the best candidate posterior is strictly below `tau` (or
`novelty_threshold`), so near-duplicates never bloat the growing index.

`build_incremental_pipeline` defaults to a deterministic, dependency-free
hashing embedder; pass `embedder=` a `SentenceTransformerEmbedding` (or any
`EmbeddingModel`) for a real model. To reuse a model you already loaded — on a
GPU, quantized, with custom pooling — hand the framework the instance:
`SentenceTransformerEmbedding(model=loaded)` wraps any preconfigured
`encode`-capable model as-is (see `docs/user_guide.md` §0).

### Batch pipeline

```python
from vectorer.batch import build_batch_pipeline
from vectorer.comparisons import make_comparison

result = build_batch_pipeline(
    comparisons=[make_comparison("email_comparison", col_name="email"), ...],
    n_canopies=512, overlap_m=3, tau=0.85,
).run(records)

result.n_clusters                        # number of entities found
result.cluster_of_position(7)            # cluster id of record 7
result.assignment.clusters               # cluster_id -> Cluster (members + representative)
result.n_candidate_pairs                 # canopy candidate pairs scored by FS
result.timing                            # per-stage seconds (parse/embed/canopy/FS/swoosh)
```

The batch stage chain is exposed as methods (`embed_all`, `block`, `score`,
`cluster`) so any stage can be replaced.

### Record Linkage (two databases)

For the merger / cross-enterprise-collaboration case — two **differently-schemed**
databases linked, not merged — use the Link mode. Declare the canonical
overlap once, then link:

```python
from vectorer.link import RecordLinker, FieldMap

linker = RecordLinker(
    embedder=embedder,
    comparisons=[make_comparison("email_comparison", col_name="email"), ...],
    field_maps={                     # canonical field <- each DB's own column
        "A": FieldMap({"email": "email", "name": "name"}, id_column="cust_id"),
        "B": FieldMap({"email": "em", "name": "legal_name"}, id_column="partner_id"),
    },
    k=20, tau=0.7,
)
table = linker.link(records_a, records_b, mode="directed")   # or "symmetric"
table.matches                    # LinkEdge list: a_id, b_id, probability, decision
table.as_pairs()                 # [(a_id, b_id), ...]
```

The output is a **link table**, never a merged store — each database keeps its
own schema and identity (see `examples/link_two_databases.py`).

### Distributed batch ER

The batch pipeline can be parallelized across workers (thread or process pools)
and produces the **same cluster assignment** as `BatchPipeline.run`:

```python
from vectorer import distributed_batch_er

assign = distributed_batch_er(
    records,
    scorer=scorer,               # FellegiSunterScorer (shipped to workers)
    n_canopies=512, overlap_m=2, tau=0.85,
    n_workers=4, use_threads=False,
)
assign.node_cluster             # {record_index: cluster_id} == single-process
```

Cross-shard canopies and mask-aligned parallel scoring keep the result
identical; only the FS stage parallelizes the heavy part.

## The comparison set (19 options, native)

`vectorer.comparisons` registers every comparison option under a canonical
snake-case name:

`exact_match`, `jaro_winkler_at_thresholds`, `jaro_at_thresholds`,
`levenshtein_at_thresholds`, `damerau_levenshtein_at_thresholds`,
`jaccard_at_thresholds`, `cosine_similarity_at_thresholds`,
`date_of_birth_comparison`, `email_comparison`, `name_comparison`,
`forename_surname_comparison`, `postcode_comparison`,
`distance_in_km_at_thresholds`, `distance_function_at_thresholds`,
`pairwise_string_distance_function_at_thresholds`,
`absolute_date_difference_at_thresholds`,
`absolute_time_difference_at_thresholds`, `array_intersect_at_sizes`,
`custom_comparison`.

```python
from vectorer.comparisons import available_comparisons, make_comparison, comparison_set

available_comparisons()                       # name -> description
c = make_comparison(                          # declared (serializable) comparison
    "jaro_winkler_at_thresholds", col_name="first_name",
    score_threshold_or_thresholds=[0.9, 0.8, 0.7],
)
comparison_set([c])                           # -> [ComparisonSpec], scorer input
```

Every comparison is a list of ordered levels (`null -> exact -> fuzzy
thresholds -> else`) with a standard default m/u scheme: the exact-match level
holds a 0.95 match probability, intermediate levels split the remainder, and
the u-probabilities correspond to match weights stepping from ~−5 to +10 —
well-behaved out of the box (e.g. a single exact email match under a 1e-4 prior
yields posterior 0.0929) and replaceable by training.

The set is **extensible** (and the scorer is fully vectorized: each level's
predicate runs over the whole candidate batch via NumPy, and multi-threshold
comparisons precompute their score array once):

```python
from vectorer.comparisons import register_comparison

def my_exact(col_name, **kwargs):
    from vectorer.comparisons import exact_match_spec
    return exact_match_spec(col_name)

register_comparison("my_exact", my_exact, fields=("col_name",), description="...")
```

Custom levels may be supplied as vectorized `test(PairValues, cache) -> mask`
callables, or as the declarative conditions `"ELSE"`, `'"col_l" IS NULL OR
"col_r" IS NULL'`, and `'"col_l" = "col_r"'` (`custom_comparison` types).

## Fellegi-Sunter scoring and calibration

`vectorer.scoring.FellegiSunterScorer` assigns each pair its highest-priority
level per comparison, combines per-level $\log(m/u)$ with the prior odds, and
derives

$$ P(\mathrm{match}) = \sigma\left(\log\frac{p_0}{1-p_0} + \sum_{\text{assigned}} \log\frac{m}{u}\right) $$

(vectorized; $\sigma$ is the sigmoid). Training is equally native:

```python
from vectorer.scoring import FellegiSunterScorer

scorer = FellegiSunterScorer.from_comparisons(comparisons, prior=1e-4)

# Supervised calibration from labelled match/non-match pairs
scorer_cal = scorer.calibrate_from_pairs(pair_records, smoothing=0.1)

# Unsupervised EM on a duplicate-bearing population (u from random-pair
# sampling, m + prior by EM over blocking-rule candidate pairs)
scorer_em = scorer.fit_em(training_records,
                          training_block_on=[("first_name",), ("date_of_birth",)])

scorer_em.score(left_record, right_record)          # posterior
scorer_em.score_batch(left_record, candidate_rows)  # vectorised over candidates
scorer_em.save("model.json"); FellegiSunterScorer.load("model.json")
```

## Persistence

* `FellegiSunterScorer.save/load` round-trips the trained comparisons and
  prior (`{"type", "params", "levels"}` per comparison — levels carry the
  current `m/u`).
* `InMemoryVectorDatabase.save/load` persists the embedded reference
  population (FAISS index + records) for the incremental pipeline.
* `Comparison.to_dict/from_dict` serializes individual comparison specs.

## Project structure

```text
vector-er/
├── pyproject.toml          # core deps: numpy + faiss-cpu; [embedding]/[test] extras
├── src/vectorer/
│   ├── records.py          # parsing stage: schemas, parsers, text serialization
│   ├── embeddings.py       # EmbeddingModel, sentence-transformers, hashing (tests/demos)
│   ├── vectorstores.py     # IndexingStrategy, FlatIndex (FAISS), VectorDatabase, in-memory store
│   ├── blocking.py         # VectorBlocker (top-k) + canopy blocking (k-means multi-assignment)
│   ├── sim.py              # vectorized similarity/distance primitives (no SQL, no fuzzy deps)
│   ├── comparisons.py      # extensible comparison registry (19 options, native)
│   ├── scoring.py          # WeightTable + FellegiSunterScorer (calibration + EM + inference)
│   ├── classification.py   # FS decision rule (match / possible / non-match)
│   ├── clustering.py       # Swoosh (G-Swoosh), cluster assignment helpers
│   ├── incremental.py      # IncrementalPipeline
│   ├── batch.py            # BatchPipeline
│   ├── link.py             # RecordLinker (two-database record linkage)
│   ├── distributed.py      # distributed_batch_er (parallel batch ER, same result)
│   └── pins.py             # pinned embedding model id/revision
├── .docs/architecture.md   # operation modes, pipeline architecture, design rationale
├── .docs/user_guide.md     # hands-on usage: incremental + batch ER, calibration
├── tests/                  # pytest suite (~75 tests, offline)
├── benchmarks/             # incremental + bulk (batch) latency/throughput benchmarks
├── examples/               # incremental_er.py, batch_er.py
└── results/                # benchmark result artifacts (gitignored)
```

## Examples

```bash
python examples/incremental_er.py --n-references 500 --tau 0.85
python examples/batch_er.py --n-base 500 --dup-rate 0.04 --n-canopies 64 --overlap 2
python examples/distributed_batch_er.py --n-base 500 --n-workers 4 --verify
python examples/link_two_databases.py
```

Both `batch_er.py` and `distributed_batch_er.py` accept `--embedder sentence`
to switch from the deterministic hashing embedder to
`sentence-transformers/all-MiniLM-L6-v2` (revision-pinned; requires the
`[embedding]` extra). The distributed example's `--verify` flag runs the
single-process pipeline and asserts the same cluster assignment.

## Benchmarks

```bash
python benchmarks/benchmark_incremental_er.py --n-references 50000 --query-count 30 --breakdown
python benchmarks/benchmark_bulk_er.py --n-records 50000 --dup-rate 0.04 --overlap 1
# multi-node (simulated 2-node Ray cluster on one host, or a real cluster via --ray-address)
python benchmarks/benchmark_bulk_er_multinode.py --n-records 8000 --n-workers 2 --verify
```

The incremental benchmark reproduces the original project's cold per-query
latency methodology (50k reference index, k=20, close-variant queries,
percentiles + phase breakdown); the bulk benchmark measures whole-dataset
throughput across canopy -> FS -> Swoosh with duplicate-recovery metrics.  The
multi-node bulk benchmark times the same workload with a Ray executor
(`n-workers` actors simulate the cluster nodes; pass `--ray-address ip:port`
for actual machines) and asserts the distributed assignment equals
single-process.  See `results/*.json`.

## Tests

```bash
python -m pytest
```

The suite covers parsing, the comparison registry (every option constructible),
canonical Jaro-Winkler reference values, vectorized FS scoring (including a
default-prior posterior parity check), calibration and native EM, vector +
canopy blocking, Swoosh clustering, persistence round-trips, and both pipelines
end to end — all offline with the deterministic hashing embedder.

## Notes and caveats

* The framework is **multi-node capable** via the optional Ray backend
  (`vectorer.distributed`) and external distributed vector databases.  What
  shards/streams across machines: parsing, embedding, canopy assignment,
  Fellegi-Sunter scoring (pair-hash owned, only above-`tau` edges cross the
  wire), and the Swoosh closure (per-machine union-find + a shared-node
  merge).  What stays single-process **by design** (surfaced as caveats, not
  forced): G-Swoosh (`gswoosh`/`cluster_with_merger` — the sequential merge
  order is globally significant; use the transitive-closure mode), per-query
  FS scoring (k is tiny — distribute the store instead), and the k-means
  canopy-train step (one global sample gather, cheap).
* Array-based comparisons (`cosine`, `jaccard`, `array_intersect`,
  `pairwise_string_distance`) operate on list-valued columns natively — no
  special column typing or database casts are needed.
* `custom_comparison` accepts declarative conditions (null/exact/else) or
  user-supplied vectorized test callables; free-form SQL is intentionally not
  supported.
* Swoosh here is pair-driven: pairs that never share a canopy are treated as
  non-matches (blocking recall). `cluster` does the transitive merge over the
  scored pairs; `cluster_with_merger` / `gswoosh` re-match merged
  representatives against the candidate pair set for full G-Swoosh behaviour.
* Scoring is vectorized over the candidate batch (single Jaro-Winkler / edit-
  distance / date-parse pass per comparison, threshold levels read cached
  score arrays), so the incremental path is sub-millisecond per query and the
  batch path scores tens of thousands of canopy pairs in pure Python + NumPy.