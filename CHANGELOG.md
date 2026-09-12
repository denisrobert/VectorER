# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **`vectorer.comparisons` module split into a package** — ``comparisons.py``
  (~1500 lines mixing ~700 lines of factories with the registry/serialization
  plumbing) is now ``comparisons/`` with focused modules: `_core`
  (``PairValues``, ``Level``/``ComparisonSpec``, the default ``m/u`` scheme and
  the shared level/prescore toolkit), `_distance_families` (the thresholded
  distance group: exact, jaro, jaro_winkler, edit distance, jaccard, cosine,
  array-intersect, distance-function, pairwise), `_temporal_families`
  (date-of-birth, absolute date/time difference), `_structural_families`
  (email, name, forename/surname, postcode, distance-in-km), `_custom`
  (custom comparison + the time-decay wrapper), `_registry` (the name-keyed
  registry + the single built-in manifest), and `_comparison` (the
  ``Comparison`` object + ``make_comparison``/``make_comparisons`` /
  ``comparison_set`` and serialization helpers).  The grouping axis is the
  shared machinery each family rides on, so no single file exceeds ~340 lines
  and family modules import only ``_core`` (never one another).  The public
  API surface is unchanged and ``from vectorer.comparisons import ...`` keeps
  working; the package re-exports every previously-imported name.  Pure
  reorganization -- no behaviour change (full test suite green).
- **`vectorer.scoring` module split into a package** — ``scoring.py`` (~1500
  lines, one ~1100-line class) is now ``scoring/`` with focused modules:
  `_constants` (DEFAULT_PRIOR / DEFAULT_THRESHOLD / log clip),
  `_levels` (level assignment + m/u helpers), `_weights` (WeightTable),
  `_construction` (constructor + serialization + input normalization),
  `_inference` (scalar evaluation + public scoring API), `_union`
  (Union-Class existential lift, split into its own file),
  `_training` (``calibrate_from_pairs`` / ``fit_em`` / prior recovery),
  `_candidates` (blocking-rule candidate generation, fuzzy keys, pair
  sampling), `_splink` (Splink importer), and `_scorer` (the assembled
  ``FellegiSunterScorer``).  ``FellegiSunterScorer`` is composed from five
  focused mixins; the public API surface is unchanged and the existing
  ``from vectorer.scoring import ...`` imports keep working.  Pure
  reorganization -- no behaviour change (full test suite green).
- **`FellegiSunterScorer.from_comparisons` accepts resolved comparison dicts**
  — ``_as_comparisons`` now also handles raw ``{"type", "params", "levels"}``
  entries (matching ``_as_specs``), so a resolved settings-dict entry can be
  passed directly as a comparison source instead of silently being dropped
  from ``scorer.comparisons``.  Behavioural bug-fix surfaced by the post-
  refactor coverage tests.
- **`scoring._math` module** — the numeric primitives `_sigmoid` and
  `_values_equal` moved out of `scoring._levels` into a small shared
  `scoring/_math.py`, since both are consumed by the inference and training
  modules but are unrelated to level assignment.  Pure relocation -- no
  behaviour change.
- **Test-suite coverage of the scoring package** — added coverage-driven tests
  for the split-out modules (multi-column blocking rules, full-enumeration
  pair sampling, `_values_equal` array/list branches, log-bayes saturation,
  empty-candidate public paths, `fit_em` error paths + `prior=` override,
  Union-Class no-set-field expansion and max-lift, dict-form ``from_cls``,
  TF-table helpers).  Package coverage rose from 87% to 92%.

- **Documented and enforced the comparison level-ordering contract** — the
  requirement that the **final level of every ``ComparisonSpec`` be the ELSE
  fallback** (``test=None``, catching every pair that matches no earlier
  level) is now documented in ``ComparisonSpec.build_spec``, the comparisons
  module docstring and ``_assign_levels``.  It is a hard invariant, not a
  convention: a test-bearing final level would silently label unmatched pairs
  as matches of that level.  A parametrized test now verifies all 20 built-in
  comparisons end with the fallback.

### Added

- **`QdrantVectorDatabase` round-trip optimisations + live-server benchmark** —
  the adapter now fetches hit `payloads` in the same search call as the index
  query and serves the subsequent per-candidate `record_at` calls from that
  cache (k+1 network round-trips per `resolve` -> 1), and caches the point
  count so `block`'s `len()` cap no longer issues a per-query `count`.  New
  `benchmarks/benchmark_incremental_er_qdrant.py` measures the incremental
  (online) pipeline against a live Qdrant store: ingest timing split into
  embedding vs upsert, cold per-query latency with optional phase breakdown,
  and ground-truth blocking quality.  Use `127.0.0.1` (not `localhost`) for
  local Qdrant -- `localhost` can resolve to `::1` while Qdrant listens on
  IPv4 only, adding ~5 s per call.  The `.docs/distributed_er.md` now
  documents the in-memory vs external-store latency/dataset-size trade-off.
- **`OpenAIEmbedding` keyless local-server support** — ``api_key=None`` (with
  no ``OPENAI_API_KEY`` set) is now allowed when ``base_url`` points at a
  local OpenAI-compatible endpoint (Ollama, LM Studio, vLLM, etc.); the
  ``urllib`` path omits the ``Authorization`` header entirely and the SDK
  path passes a no-op placeholder key the server ignores.  The default
  ``api.openai.com`` endpoint still requires a key.
- **`benchmarks/benchmark_yancey_enrichment.py`** — benchmark that tests
  Yancey's match-enrichment procedure for improving EM (Yancey 2004, RRS
  #2004-01).  On the generated duplicate-bearing population it compares
  plain EM, Yancey-enriched EM with prior recalibration, enriched m/u with a
  fixed/swept prior, an oracle-prior arm, and default m/u on the same labelled
  eval pairs (precision/recall/F1 across a tau grid).  Verifies that
  enrichment rescues EM's sparse-match-class failure (recall 0.60 → 0.76), and
  that freezing the prior at an operating value is what takes recall further
  (F1 0.999 at prior 1e-3), confirming the prior-recalibration caveat.
- **Benchmark eval-set validity — genuinely disjoint negatives + full
  confusion matrix** — both the yancey enrichment and incremental benchmarks
  previously drew their non-match eval pairs as *random index-vs-index pairs*,
  which are invalid in this synthetic population: the tiny name grid makes any
  two population records near-duplicates, so a working pipeline would
  (correctly) match them.  They now use
  ``benchmark_data.build_unrelated_negatives`` — freshly generated census-
  distributed people whose ``(first, last, dob)`` identity is disjoint from the
  population (colliding DOBs shifted +5 days, colliding emails nulled) — so a
  correct resolver must *not* match them, and the precision/F1 numbers are
  meaningful again.  The enrichment benchmark's ``prf`` now also emits the
  full 2x2 confusion matrix (``tp``/``fp``/``fn``/``tn``) per tau.

### Changed

- **`fit_em` candidate pool construction (`_blocked_pairs`)** — the training
  pool now uses *fuzzy* blocking for single string columns (case/typo-flipped
  twin values land in the same trigram bucket as their base) and *per-rule
  budgets ordered by specificity*: high-cardinality, twin-discriminating rules
  (e.g. `date_of_birth`) get enough of the cap to actually produce matches,
  instead of being starved after a common-name rule saturates the budget.
  Previously the pool contained ~0 true twins (perturbed twins never shared an
  exact first-name string and the DOB rule never ran), so the learned m/u
  collapsed onto 1e-8 floors and true twins scored ~0.
- **`fit_em` base-prior estimate** — the reported prior is now the model's own
  self-consistent match rate over a **uniform pair sample** (a fixed-point
  solve of ``pi = mean(posterior(pi))`` on precomputed per-pair evidence; the
  convergence test runs in logit space, which contracts faster in the slow
  near-boundary regime).  A **per-iterate trajectory boundary guard** falls
  back to the evidence-positive share ``mean(ev > 0)`` the moment the iterate
  is pushed past ``0.9`` and keeps climbing — so a pool whose m/u make ``mean
  (log-odds) > 0`` cannot run the prior to the spurious ``1.0`` fixed point,
  regardless of where it would have converged.  Replaces both the blocked-pool
  ``pi`` extrapolation to ``C(n,2)`` and the flat-``0.5``-prior posterior mean
  (both of which systematically overstate the base rate).  The ``pi`` used
  inside the EM M-step is also now estimated from **evidence-only**
  responsibilities (``sigmoid(sum log m/u)`` without the ``logit(pi)`` term),
  removing the self-reinforcing ``pi -> 1`` runaway of the classic formulation.
  The blocked pool is deliberately match-biased, so the old extrapolation
  overstated the base rate by orders of magnitude; ``scorer._em`` records
  ``pi`` (the blocked share, clipped to ``(1e-6, 1-1e-6)``) and
  ``prior_empirical`` (the honest uniform-posterior base rate) for inspection
  and for ``recalibrate_prior``.
- **`recalibrate_prior(method=...)`**: the prior recovery after match-enrichment
  EM now takes a ``method`` switch.  ``method="yancey"`` (default) implements
  the paper's count-ratio correction (Yancey 2004 §2.4): the full-set prior is
  ``pi * |S0| / |S|`` — the enriched EM mixing share ``pi`` times the enriched
  blocked-pair count ``|S0|`` (recorded by ``fit_em``), scaled to the full pair
  domain ``|S| = C(n, 2)``.  This is deterministic and free (no scoring pass).
  ``method="empirical"`` preserves the previous behaviour: score a uniform
  full-set pair sample with the trained ``m/u`` and set the prior to the model's
  own expected match rate.  ``fit_em`` now records the mixing share and blocked
  pair count on the returned scorer (carried through ``to_settings``/``from_settings``);
  ``method="yancey"`` raises ``ValueError`` when that metadata is absent.

- **Dataset-relative default `n_canopies` in the batch pipeline**: a fixed
  `512` default is replaced by a default resolved from the dataset size —
  `n_canopies = max(1, len(records) // 39)` (the FAISS k-means minimum-points
  heuristic) when the caller does not supply `n_canopies=`.  `BatchPipeline.__init__`
  and `build_batch_pipeline` now default to `n_canopies=Optional[int] = None`;
  the value is resolved lazily in `run()` (and in `block()` for standalone
  use).  The canopy grid therefore scales with the number of records instead of
  over/under-partitioning small or large datasets.  Explicit `n_canopies=` is
  still honoured.

## [0.5.2] - 2026-09-07

### Added

- **`FellegiSunterScorer.recalibrate_prior(records, sample_size, seed, recall)`**:
  after match-enrichment EM (Yancey 2004), the EM match-prior reflects the
  *enriched subset*, and Yancey recalibrates only the non-match classes (C2/C3).
  For a posterior/threshold system the match prior must be corrected to the full
  population: this method scores a uniform full-set pair sample with the trained
  `m/u`, sets the prior to the model's own expected match rate (optionally
  recall-adjusted), and returns a scorer with the corrected prior.
- **User guide §6.3**: added the "Recalibrate the prior after enrichment"
  note documenting the Yancey gap and the fixed-prior remedy.

## [0.5.1] - 2026-09-06

### Added (fixed-prior EM — the calibration-paradox remedy)

- **`fit_em(fixed_prior=...)`**: holds the base prior **frozen across every EM
  iteration** — `pi` is used in the E-step's responsibilities and never
  re-estimated in the M-step, so only `m/u` are learned (the Splink-style
  fixed-prior EM).  This enables sweeping prior × threshold to find an optimal
  operating point rather than trusting EM's own often-miscalibrated base rate.
  `prior=` remains the previous behaviour (EM still learns `pi`; the reported
  base rate is overridden).  Passing both with different values raises.
- **`benchmark_bulk_er_em.py --prior-sweep-priors/--prior-sweep-taus`**: trains
  a scorer per fixed prior over a labelled eval sample (`--gt-file`) and reports
  precision/recall (and F1) across the tau grid — the operating-point surface
  proposed to resolve the calibration paradox (verified: recall 0.70→0.755 at
  precision 1.0 by sweeping prior).
- **`generate_census_population_with_duplicates.py --gt-output`**: writes the
  exact `{duplicate_index: base_index}` ground-truth map so recovery metrics are
  measurable (also `.gitignore`d).

## [0.5.0] - 2026-09-06

### Added

- **Balanced distributed FS scoring**: `distributed_score_pairs` and
  `distributed_batch_er` now partition the candidate-pair workload into
  **contiguous, equal-size slices** (`_balanced_owned_slices`) instead of
  hash-round-robin ownership, so each core/machine scores an (almost) equal
  number of pairs. (Per-pair *cost* still varies with the comparison set; the
  pair *count* is balanced.)
- **Worker → controller progress reporting**: the distributed scoring stages
  accept an optional `progress_callback`; process workers report their scored
  pair count through a shared `Manager` queue, and the controller aggregates it
  (wired to an aggregate `tqdm` bar in the EM bulk benchmark).  `None` disables
  reporting and preserves prior behaviour.
- **`benchmark_bulk_er_em.py`**: multi-process bulk dedup via `--n-procs N`
  (uses `distributed_batch_er`'s process pool) with an aggregate `fs scoring
  (distributed)` progress bar.

**Interface change (semver: this is a MINOR bump — 0.4.1 → 0.5.0):**
`distributed_score_pairs`, `distributed_batch_er`, and
`bucket_override`-adjacent helpers gained **new optional keyword arguments**
(`progress_callback`, and in `distributed_score_pairs` no positional change),
and `cluster_quality_distributed` gained an optional `progress=` flag. None of
the existing required parameters or the return types changed; the additions
are purely additive, but because the distributed API surface grew (new kwargs
and a new `_score_slice_worker`/`_score_worker` progress path improve
behaviour), a **minor version bump** is appropriate per semver.

## [0.4.1] - 2026-09-05

### Added

**Completes the v0.4.0 distribution plan (Milestones C, D, E).**

- **Milestone C — canopy-sample gather + global TF wired in**:
  - `gather_canopy_sample` — memory-bounded, deterministic cross-machine
    sample for k-means centroid training, wired into `distributed_batch_er`
    (trains on the sample; `sample_size=None` = full matrix for bit-identical
    small-data results).
  - `distributed_batch_er` now routes the Swoosh closure through
    `distributed_closure_reduce` (multi-machine, exact) instead of a single
    union-find.
- **Milestone D — external distributed vector DB**:
  - `QdrantVectorDatabase` (`vectorstore_adapters.py`) — a `VectorDatabase`
    backed by a distributed Qdrant collection: `add`/`update`/`delete`/
    `record_at`/`len`/`index.search` (HNSW cosine), embedding model stays
    local.  Handles both Qdrant `query_points` (≥1.15) and legacy `search`.
  - New optional `[qdrant]` extra; exported as `vectorer.QdrantVectorDatabase`.
  - Usable directly with `IncrementalPipeline.from_store` for the multi-node
    online serving path.
- **Milestone E — operations doc**:
  - `.docs/distributed_er.md` — multi-node operation guide (what shards /
    streams, what stays single-process by design with caveats, how to run a Ray
    cluster and a Qdrant-backed incremental store, building-block reference).
  - `.docs/architecture.md` §7.2 "Multi-node operation" added; single-node-only
    claims removed across README + architecture docs.
- **Tests**: gather/TF/sample (2), Qdrant adapter (3).

## [0.4.0] - 2026-09-05

### Added

- **Multi-node distribution (v0.4.0 plan, Milestones A-B)** in
  `vectorer.distributed`, all additive on top of the single-machine pipelines
  (which stay byte-for-byte unchanged):
  - **`Executor` backend abstraction**: `create_executor("process"|"thread"|
    "ray")` and a minimal optional `RayExecutor`; the orchestration is
    backend-agnostic.
  - **`distributed_score_pairs`** — the FS scoring *map* across workers: pairs
    owned by a deterministic balanced hash, each worker rebuilds the scorer
    from its serialized settings and returns only the above-`tau` rows (only
    those cross the wire).  Optional `pair_positions=` returns proper
    `ScoredPair`s.
  - **`distributed_score_and_reduce`** — composes the scoring map with the
    closure reduce for a single, streaming score-then-cluster call.
  - **`streaming_distributed_closure`** — transitive closure over an
    *iterator of edge chunks* (streaming reduce, bounded memory).
  - **`distributed_closure_reduce`** — multi-machine exact connected
    components: per-worker local union-find + a shared-node merge into
    min-position ids, bit-for-bit identical to the single-process closure
    (verified at `n_workers` 1/2/3 and with a thread executor).
  - **TF pre-reduce** (`merge_tf_counters`, `build_global_tf_tables`) so
    per-value term frequencies stay globally consistent across machines.
  - `examples/distributed_streaming_er.py` with `--verify`.
  - `examples/multi_node_distributed_er.py` — runs the same streaming
    score-and-cluster flow across a **Ray cluster**: pass the head node
    `ip:port`, or `--ray-address auto` to start/join a local instance (same
    code path as a real multi-node setup).  `--verify` asserts the cluster
    assignment is identical to single-process.  `RayExecutor` starts a local
    Ray instance when no `RAY_ADDRESS`/address is configured.
  - `benchmarks/benchmark_bulk_er_multinode.py` — bulk ER on a **simulated
    2-node Ray cluster** (single host, `n_workers` actors), timing single vs
    distributed and asserting identical assignments (`--verify`).
- The caveat map from the plan is honoured: G-Swoosh, per-query FS (incremental/
  link-directed) and single-machine-canopy training remain single-process by
  design; the transitive-closure mode and the store-backed incremental path
  are the distributed surfaces.

## [0.3.1] - 2026-09-05

### Added

- **`build_incremental_pipeline(vector_database=...)`** — the convenience
  constructor now also accepts a **pre-built vector store** (with its own
  embedder) for the production *serving* modality: the reference population was
  embedded separately (previously) and loaded (e.g. from a persisted or
  distributed vector DB), so it is not re-embedded.  Providing both/or neither
  of `records=` and `vector_database=` raises a clear error.
- **`IncrementalPipeline.from_store`** — a discoverable alias for the
  production *serving* modality: build a pipeline from an already-embedded,
  loaded vector store without re-embedding the reference records.
- **`import_splink_scorer`** (`vectorer.scoring`): import `m/u` (and
  term-frequency weights) trained by Splink onto a matching native comparison
  set, for use in batch, Link, or incremental modes.  Matches Splink
  comparisons to native ones by output column name, transfers per-level
  `m_probability` / `u_probability` and TF metadata, preserves level order, and
  validates level counts.  Documented in the user guide (§6.3) and recipes with
  the caveats (level alignment, native tests only, TF tables rebuilt from
  `base_records=`, `idempotent` reflexivity).
- **`OpenAIEmbedding`** (`vectorer.embeddings`): OpenAI API-backed embedder
  (`text-embedding-3-*`, `text-embedding-ada-002`), reading `OPENAI_API_KEY`
  or `api_key=`, with `dimensions=` truncation, batched `embed_many`, and
  `base_url=` override.  **Prefers the official `openai` SDK when installed**
  (optional `[openai]` extra), falling back to a dependency-free `urllib`
  client.
- **`non_standardized_address` perturbation** (`benchmarks/perturbations.py`):
  address rewritten in realistic ways (street-type abbreviations/expansions
  incl. wayfare kinds, unit/civic forms, field-order variants, PO-box/RR
  substitutes).
- **`address_change` perturbation**: a genuine move within the same city and
  postal-code prefix — the most frequent move — to test resilience to stale
  addresses.
- **Per-perturbation confusion matrices** in
  `benchmarks/benchmark_incremental_roc.py` (`--perturbation`,
  `--tau-report`), with the six typed clerical/transmission perturbation
  categories, benchmarked against a 300k 2021-census-distributed population.
- **Sphinx API reference** (`docs/`, `.readthedocs.yaml`) published via
  ReadTheDocs, including a vector-ER logo (the glyph ER with a vector macron).

## [0.3.0] - 2026-08-31

### Added

- **Distributed batch ER** (`vectorer.distributed`): `distributed_batch_er`
  executes the whole batch pipeline across workers (thread or process pools)
  and produces the **same cluster assignment** as the single-process
  `BatchPipeline.run`.  Seams added to `blocking.py`:
  `train_canopy_centroids` / `assign_canopies` (`canopy_blocking` now wraps
  them, behaviour unchanged); **global-canopy cross-shard pair emission** (so a
  true match split across shards is not missed); **mask-aligned** parallel FS
  scoring (`_score_shard` returns mask/probs/weights so filtered below-τ rows
  never misalign positions); `distributed_closure` (exact connected components
  over the above-τ edges).  Scorer transport via `to_settings`/`from_settings`;
  `distributed_batch_er`, `distributed_closure`, `hash_pair` are exported from
  `vectorer`.
- **`examples/distributed_batch_er.py`**: a runnable illustration of the
  distributed batch pipeline with `--n-workers` / `--use-threads` and
  `--verify` (asserts the distributed cluster assignment equals the
  single-process `BatchPipeline.run`).
- **Tests** (`tests/test_distributed.py`): distributed == single-process for
  thread/process pools at `n_workers` 1/2/3, `hash_pair` determinism,
  train/assign == `canopy_blocking`, closure equivalence.

### Changed

- The single-process `BatchPipeline.run` is **untouched**; the distributed
  executor reproduces its result exactly (verified by tests and the
  example's `--verify`).

## [0.2.2] - 2026-08-30

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

## [0.2.1] - 2026-08-30

### Changed

- General cleanup / maintenance release.

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

[0.5.2]: https://github.com/denisrobert/VectorER
[0.5.1]: https://github.com/denisrobert/VectorER
[0.5.0]: https://github.com/denisrobert/VectorER
[0.4.1]: https://github.com/denisrobert/VectorER
[0.4.0]: https://github.com/denisrobert/VectorER
[0.3.1]: https://github.com/denisrobert/VectorER
[0.3.0]: https://github.com/denisrobert/VectorER
[0.2.2]: https://github.com/denisrobert/VectorER
[0.2.1]: https://github.com/denisrobert/VectorER
[0.2.0]: https://github.com/denisrobert/VectorER
[0.1.1]: https://github.com/denisrobert/VectorER
[0.1.0]: https://github.com/denisrobert/VectorER