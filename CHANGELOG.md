# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Initial release of `vectorer`: a framework for embedding-and-vector-based
  entity resolution with two composable pipelines.
- **Incremental ER** (`vectorer.incremental`): stream one record at a time
  against a reference store — parse, embed, top-k vector-search blocking,
  Fellegi-Sunter scoring, classification. Supports novelty-only ingestion
  (`ingest_novel` / `ingest_novel_many`).
- **Batch ER** (`vectorer.batch`): whole-dataset deduplication — parse,
  embed, overlapping canopy blocking, Fellegi-Sunter scoring of every canopy
  pair, Swoosh clustering (transitive closure, plus full G-Swoosh via
  `gswoosh` / `SwooshClusterer.cluster_with_merger`).
- **Fellegi-Sunter comparison set** (`vectorer.comparisons`): 19 registered
  comparison options covering the standard attribute-comparison families
  (exact, jaro/jaro-winkler, Levenshtein/Damerau-Levenshtein, jaccard,
  cosine, array intersect, date-of-birth, email, name, forename/surname,
  postcode, distance-in-km, absolute date/time difference, custom), implemented
  natively and vectorized in NumPy with no SQL engine.
- **Scoring and calibration** (`vectorer.scoring`): `FellegiSunterScorer`
  with reflexive (Union-Class-compatible) matching, term-frequency adjustment,
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

## [0.1.0] - TBD

To be filled in at release time (rename from *Unreleased* and date it).

[Unreleased]: https://github.com/denisrobert/VectorER
[0.1.0]: https://github.com/denisrobert/VectorER