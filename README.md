# vector-er

A framework for **embedding-and-vector-based entity resolution** with two
composable pipelines and an extensible **Fellegi-Sunter** (FS) comparison set
covering the same 19 comparison options currently available in
[Splink](https://moj-analytical-services.github.io/splink/) — implemented
**natively in NumPy, with no Splink, no DuckDB, and no SQL engine**.

| Pipeline | Stage chain | Use case |
|---|---|---|
| **Incremental** | parsing -> embedding -> vector search blocking (top-k) -> FS scoring on the top-k -> classification | online / streaming resolution of one record against a reference store |
| **Batch** | parsing -> embedding -> canopy blocking on the embedded dataset -> FS scoring of every canopy pair -> Swoosh clustering | offline deduplication / clustering of a whole dataset |

**Why no SQL?** Splink evaluates comparison levels by lowering them to SQL
over a DuckDB table. That is awkward when the *blocking engine is a vector
database*: the candidates are produced by ANN search / canopy clustering, not
by a SQL join. Here every comparison level is a vectorized NumPy predicate
evaluated over the whole batch of candidate pairs at once, and the Fellegi-
Sunter math (level assignment -> bayes factors -> posterior) is pure array
algebra. The comparison *semantics* mirror Splink's, so a model declared with
these names behaves like the equivalent Splink model.

## Installation

```bash
# Python >= 3.10. Core deps: numpy + faiss-cpu only.
python -m venv .venv
.venv\Scripts\activate        # Windows PowerShell
pip install -e ".[test]"

# Optional: real (sentence-transformers) embedding model
pip install -e ".[embedding]"
```

There is no `splink`, `duckdb`, or `pandas` dependency anywhere in the package.

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

`build_incremental_pipeline` defaults to a deterministic, dependency-free
hashing embedder; pass `embedder=` a `SentenceTransformerEmbedding` (or any
`EmbeddingModel`) for a real model.

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

## The comparison set (the 19 Splink options, native)

`vectorer.comparisons` registers every comparison option of
`splink.comparison_library` under the canonical name:

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

Every built-in name replicates the **level structure and default m/u** of its
Splink counterpart (verified against Splink by tests): e.g. an exact email
match under the default prior yields posterior `0.0929`, the `m/u` values come
from Splink's `0.95`/`weights [-5..3, +10]` scheme, and each comparison is a
list of ordered levels (`null -> exact -> fuzzy thresholds -> else`).

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

`vectorer.scoring.FellegiSunterScorer` computes the same match-weight algebra
Splink uses, without SQL: each pair is assigned its highest-priority level per
comparison, the log-bayes-factor sum is combined with the prior odds, and
`match_probability = sigmoid(log(prior odds) + sum log(m/u))`. Training is
equally native:

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
│   ├── comparisons.py      # extensible comparison registry (19 Splink-compatible options, native)
│   ├── scoring.py          # WeightTable + FellegiSunterScorer (calibration + EM + inference)
│   ├── classification.py   # FS decision rule (match / possible / non-match)
│   ├── clustering.py       # Swoosh (G-Swoosh), cluster assignment helpers
│   ├── incremental.py      # IncrementalPipeline
│   ├── batch.py            # BatchPipeline
│   └── pins.py             # pinned embedding model id/revision
├── tests/                  # pytest suite (~75 tests, offline)
└── examples/
    ├── incremental_er.py   # online resolution + EM training demo
    └── batch_er.py         # canopy -> FS -> Swoosh deduplication demo
```

## Examples

```bash
python examples/incremental_er.py --n-references 500 --tau 0.85
python examples/batch_er.py --n-base 500 --dup-rate 0.04 --n-canopies 64 --overlap 2
```

Both accept `--embedder sentence` to switch from the deterministic hashing
embedder to `sentence-transformers/all-MiniLM-L6-v2` (revision-pinned; requires
the `[embedding]` extra).

## Tests

```bash
python -m pytest
```

The suite covers parsing, the comparison registry (every Splink option
constructible), canonical Jaro-Winkler reference values, vectorized FS scoring
(including a Splink-default posterior parity check), calibration and native EM,
vector + canopy blocking, Swoosh clustering, persistence round-trips, and both
pipelines end to end — all offline with the deterministic hashing embedder.

## Notes and caveats

* Comparison semantics replicate Splink's recent `splink.comparison_library`
  level structures and default m/u (verified by tests); they are not an exact
  upstream contract and will diverge if Splink changes its defaults.
* Array-based comparisons (`cosine`, `jaccard`, `array_intersect`,
  `pairwise_string_distance`) operate on list-valued columns natively — no
  special column typing or DuckDB casts are needed.
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
  batch path scores tens of thousands of canopy pairs per second in pure
  Python + NumPy.