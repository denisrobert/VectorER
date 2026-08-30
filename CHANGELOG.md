# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Link-mode benchmark tooling**: `benchmarks/benchmark_incremental_roc.py`
  sweeps the match threshold `tau` on the incremental pipeline and reports the
  ROC curve, precision-recall curve, and ROC AUC.  The methodology calibrates
  Fellegi-Sunter m/u on an 80/20 train/validation split and evaluates the sweep
  on the held-out validation set, making the numbers statistically meaningful.
- **Prepared-dataset support in the benchmarks**: a shared
  `benchmarks/benchmark_data.py` loader (`load_records`,
  `require_compared_fields`) lets all three benchmark scripts take
  `--data-file` (JSONL or JSON, optionally `--data-key`); the bulk benchmark
  additionally accepts `--gt-file` for evaluating recall/precision on real
  datasets.  This enables re-running the benchmarks at a later time with
  realistic populations (synthetic or real), not just the built-in generators.
- **`--embedder` option** for the incremental ROC benchmark (`hashing`,
  `minilm`, `mdbr`) so model comparisons (e.g.
  `sentence-transformers/all-MiniLM-L6-v2` vs `MongoDB/mdbr-leaf-mt`) can be
  generated against the same dataset.
- **`.docs/recipes.md`**: copy-pasteable recipes for specialised comparators,
  including a recipe for a *time-sliced comparator* (time-decayed address
  matching).

### Changed

- **`time_decay_wrapper`** (`vectorer.comparisons`): a new framework utility
  that wraps any comparison into its time-sliced, time-decayed version
  (crosses inner levels with time bands, decays each level's `m`, adds a
  missing-timestamp null level, and preserves the inner comparison's
  prescore).  Also registered as the `time_decayed_comparison` comparison
  option.

## [0.2.0] - 2026-08-30

### Added

- **Record Linkage mode** (`vectorer.link`): a third pipeline that links
  records across **two separately-managed databases** (different schemas,
  overlapping compared fields) instead of deduplicating one.  `RecordLinker`
  supports directed linking (index one database, resolve the other) and
  symmetric linking (cross-database canopy pairs), emits `LinkTable`/`LinkEdge`
  results, and never merges the two stores.  `FieldMap` provides per-database
  canonical field projection with optional normalisers and id columns.
- **`embed_text=` hook** on `InMemoryVectorDatabase` so stored records keep
  their native schema while embedding/comparison text uses a canonical
  projection (used by the Link mode).
- **`examples/link_two_databases.py`**: a runnable merger / cross-enterprise
  collaboration example.
- **User guide + architecture docs** for the Link mode.

## [0.1.1] - 2026-08-30

### Changed

- **README installation instructions** updated for the PyPI release:
  `pip install vectorer` and `pip install "vectorer[embedding]"` (with
  `pip install -e ".[test]"` for development), plus PyPI badges.

## [0.1.0] - 2026-08-30

Initial public release of `vectorer` on PyPI.

### Added

- **Incremental ER** (`vectorer.incremental`): stream one record at a time
  against a reference store — parse, embed, top-k vector-search blocking,
  Fellegi-Sunter scoring, classification.  Supports novelty-only ingestion
  (`ingest_novel` / `ingest_novel_many`).
- **Batch ER** (`vectorer.batch`): whole-dataset deduplication — parse, embed,
  overlapping canopy blocking, Fellegi-Sunter scoring of every canopy pair,
  Swoosh clustering (transitive closure, plus full G-Swoosh via `gswoosh` /
  `SwooshClusterer.cluster_with_merger`).
- **Fellegi-Sunter comparison set** (`vectorer.comparisons`): 19 registered
  comparison options covering the standard attribute-comparison families
  (exact, jaro/jaro-winkler, Levenshtein/Damerau-Levenshtein, jaccard, cosine,
  array intersect, date-of-birth, email, name, forename/surname, postcode,
  distance-in-km, absolute date/time difference) implemented natively and
  vectorized in NumPy with no SQL engine.
- **Scoring and calibration** (`vectorer.scoring`): `FellegiSunterScorer` with
  reflexive (Union-Class-compatible) matching, term-frequency adjustment,
  supervised calibration from labelled pairs, and unsupervised EM fitting.
- **Custom merge functions** (`vectorer.clustering`): `select_representative`
  (default), `union_merge` (Swoosh Union Class), `latest_merge`, and the
  documented contract for user-supplied merges.
- **Preconfigured embedding models**: embedders are always passed as
  instances; `SentenceTransformerEmbedding(model=...)` wraps an
  already-instantiated model (GPU, quantized, custom pooling) without the
  framework re-initializing it.
- **Persistence**: scorer `save`/`load`, `InMemoryVectorDatabase.save`/`load`
  (FAISS index + records), JSON-serializable declared comparisons.
- **Documentation**: `README.md`, `.docs/architecture.md`, `.docs/user_guide.md`,
  `.source-papers/`.

[Unreleased]: https://github.com/denisrobert/VectorER
[0.2.0]: https://github.com/denisrobert/VectorER
[0.1.1]: https://github.com/denisrobert/VectorER
[0.1.0]: https://github.com/denisrobert/VectorER