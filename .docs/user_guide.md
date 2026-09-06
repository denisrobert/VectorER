# User's Guide

This guide explains how to use the framework for **incremental** and **batch**
entity resolution (ER), from a first end-to-end pipeline through to calibrating
the model, tuning parameters, and production-serving. It builds on
[`architecture.md`](architecture.md) and keeps every step runnable against the
package in this repository.

The running illustration throughout is the canonical use case inherited from
the original project: **incremental ER of Canadian-style person records** —
each record has `first_name`, `last_name`, `date_of_birth`, `email` (present
~70% of the time) and `address` (present ~70%), missing values are `None`, and
queries arrive one at a time against a pre-built reference index of 50,000
records.

---

## 0. Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows PowerShell
pip install -e ".[test]"      # core: numpy + faiss-cpu
pip install -e ".[embedding]" # optional: sentence-transformers
```

Import surface:

```python
from vectorer.comparisons import make_comparison, available_comparisons, comparison_set
from vectorer.scoring import FellegiSunterScorer
from vectorer.incremental import build_incremental_pipeline, IncrementalPipeline
from vectorer.batch import build_batch_pipeline, BatchPipeline
from vectorer.embeddings import SentenceTransformerEmbedding, CharacterHashingEmbedding
from vectorer.records import RecordSchema, DictParser, JsonLinesParser
from vectorer.vectorstores import InMemoryVectorDatabase, FlatIndex
from vectorer import Decision
```

The framework treats a *record* as any `Mapping[str, Any]`. Dataclasses and
objects exposing `to_dict()` are coerced automatically
(`vectorer.records.to_record_dict`).

### Embedders

- `SentenceTransformerEmbedding(model_id, revision, device)` — real transformer
  embeddings; use `vectorer.pins.EMBEDDING_MODEL_ID` (all-MiniLM-L6-v2) with its
  pinned `EMBEDDING_MODEL_REVISION` for reproducible runs.
- `OpenAIEmbedding(api_key=..., model="text-embedding-3-small", dimensions=...)` —
  calls the OpenAI Embeddings API (`text-embedding-3-*`,
  `text-embedding-ada-002`, or a compatible endpoint).  Reads the
  `OPENAI_API_KEY` env var by default, batches via `embed_many`, and supports
  the `dimensions` truncation.  Prefers the official `openai` SDK when
  installed (`pip install -e ".[openai]"`), otherwise falls back to a
  dependency-free `urllib` client.  Requires network + an API key (or a
  `base_url=` pointing at a compatible self-hosted endpoint).
- `CharacterHashingEmbedding(dimension=384)` — deterministic, dependency-free
  hashed character n-gram embedder, and the default. Same dimensionality as
  MiniLM, instant, no download. Because it keys on character n-grams, a typo,
  transposition, or OCR slip corrupts only the handful of n-grams it touches —
  so for the character-level noise of clerical error this default is usually
  the *best* choice, not just a cheap one. Switch to the sentence-transformer
  embedder when you need semantic blocking.

Everywhere the pipelines take an embedder, they take an *instance*, not a
model name — so "the model" is whatever you hand over, already configured:

```python
from sentence_transformers import SentenceTransformer
from vectorer.embeddings import SentenceTransformerEmbedding
from vectorer.incremental import build_incremental_pipeline
from vectorer.pins import EMBEDDING_MODEL_ID, EMBEDDING_MODEL_REVISION

# Framework loads it for you from an id...
embedder = SentenceTransformerEmbedding(EMBEDDING_MODEL_ID, EMBEDDING_MODEL_REVISION)

# ...or you wrap a *preconfigured* model object (GPU, device_map,
# quantization, custom pooling, any SentenceTransformer-compatible encode).
loaded = SentenceTransformer(EMBEDDING_MODEL_ID, device="cuda:0",
                             model_kwargs={"torch_dtype": "float16"})
embedder = SentenceTransformerEmbedding(model=loaded)
pipeline = build_incremental_pipeline(references, embedder=embedder, ...)
```

`model=` wraps any object with an `encode` method as-is — the framework never
re-initializes it, so everything you configured on it (device, batch size,
prompt/instruction, ONNX/quantized backend, revision) is what gets used. Pass
preconfigured `EmbeddingModel` implementations anywhere an `embedder=` is
accepted; `SentenceTransformerEmbedding(model=...)` just adapts an existing
`sentence_transformers` model into that interface. Only `model_id`-constructed
wrappers need the pinned `revision=`.

---

## 1. The comparison set (choose your model)

Both pipelines consume a *declared* comparison set built through
`make_comparison(name, **params)`; every option in the registry is available
via `available_comparisons()`.

```python
from vectorer.comparisons import make_comparison, available_comparisons

print(available_comparisons())   # 19 options, name -> description

comparisons = [
    make_comparison("jaro_winkler_at_thresholds", col_name="first_name",
                    score_threshold_or_thresholds=[0.9, 0.8, 0.7]),
    make_comparison("jaro_winkler_at_thresholds", col_name="last_name",
                    score_threshold_or_thresholds=[0.9, 0.8, 0.7]),
    make_comparison("date_of_birth_comparison", col_name="date_of_birth"),
    make_comparison("email_comparison", col_name="email"),
    make_comparison("jaro_winkler_at_thresholds", col_name="address",
                    score_threshold_or_thresholds=[0.85, 0.75, 0.65]),
]
```

A comparison object is an ordered list of levels
(`null -> exact -> fuzzy thresholds -> else`) evaluated over the column or
columns it names. Each non-null level carries an `m`/`u` pair (default m/u are
assigned at build time; see `architecture.md` §4, and §6 of this guide for
calibrating them). The mapping between attributes and comparison objects is
not one-to-one: a comparison may read several columns (`forename_surname_comparison`,
`distance_in_km_at_thresholds`), and the same attribute may back several
comparison objects. The thresholded built-ins are a compact form of that —
`jaro_winkler_at_thresholds` runs a single Jaro-Winkler pass but defines one
level per threshold, each with its own `m`/`u`; you could write the same
decomposition by hand, e.g. as a custom comparison per non-overlapping level.

Custom comparisons are registered with `register_comparison`; for the built-in
families that need extra columns, e.g.:

```python
make_comparison("forename_surname_comparison",
                forename_col_name="forename", surname_col_name="surname")
make_comparison("distance_in_km_at_thresholds",
                lat_col="lat", long_col="long", km_thresholds=[1, 10, 100])
make_comparison("absolute_date_difference_at_thresholds",
                col_name="hired", input_is_string=True,
                metrics=["day", "day", "day"], thresholds=[1, 7, 30])
```

---

## 2. The algorithms in practice: Fellegi–Sunter and the Swoosh family

Before wiring your data in, it is worth having a precise mental model of the two
algorithms you are composing and the three choices they expose. The framework
lets you pick a **comparison set** (the FS match function), a **merge function**
(how a matched group is combined into one record), and thereby an implicit
**domination order** (which records can be treated as "covered" and discarded).
Getting these right is what separates a config that behaves sensibly on your
data from one that quietly under- or over-merges.

### 2.1 Fellegi–Sunter in one paragraph

FS treats each attribute comparison as *evidence*, not a verdict. For a pair of
records and one attribute, the pair lands in exactly one **level** (e.g. `null`,
`exact`, `jaro-winkler ≥ 0.9`, `... ≥ 0.7`, `else`). Each non-null level carries
two probabilities you can reason about like this:

- **m** = *"if these two records really are the same entity, how likely is this
  level?"* — usually highest for `exact`.
- **u** = *"if they are in fact different entities, how likely is this level?"* —
  tiny for `exact` on a rare value, larger for `else`.

The score accumulates the log of the ratio $m/u$ (the *match weight*) per
comparison plus a base prior (the probability two *random* records match), and
the posterior (probability of a match given the data) is the sigmoid of that
total:

$$ P = \sigma\left( \log\frac{p_0}{1-p_0} + \sum_{\text{assigned}} \log\frac{m}{u} \right), \qquad \sigma(x) = \frac{1}{1+e^{-x}} $$

A comparison whose level is `null` contributes **no evidence** (weight 0) — this
is why a record with most fields missing has a low self-score, and why the
scorer is reflexive by default (see the architecture doc §4). The **match
threshold $\tau$** on that posterior is the operating point you choose in §7.

The key practitioner takeaway: **you are not defining hard rules; you are
choosing which *attributes* you trust and how strongly.** An `exact` match on a
high-cardinality field (email) is worth far more than an `exact` match on a
low-cardinality field (gender). That ordering — and the relative value of each
level — is exactly what `m/u` encodes.

### 2.2 Choosing comparators

Three decisions per attribute:

1. **Which attributes are discriminative?** Prefer fields that vary across
   entities (full name, date of birth, email, postcode) over fields that are
   often identical by chance (gender, province, "0000" zip). A comparison on a
   non-discriminative field mostly dilutes the score — leave it out or give it
   coarse thresholds.
2. **What kind of fuzziness does the attribute need?**
   - *Spelling / OCR* → `jaro_winkler_at_thresholds` (names), or `levenshtein`
     / `damerau_levenshtein` for harder edit noise.
   - *Set-valued* (tags, phone-list, embedding column) → `jaccard`,
     `cosine_similarity`, `array_intersect`, `pairwise_string_distance`.
   - *Numeric / temporal* → `absolute_date_difference`,
     `absolute_time_difference`, `distance_in_km` (lat/long).
   - *Structured identifiers* → `email_comparison`, `postcode_comparison`,
     `date_of_birth_comparison`, `name_comparison`.
   - When nothing built-in fits, register a custom comparison (a level whose
     test is a vectorized `test(PairValues, cache) -> mask` callable) or wrap an
     arbitrary distance function via `distance_function_at_thresholds`.
3. **How many thresholds / how coarse?** Thresholds should mirror how much
   noise you tolerate in that field. Two or three thresholds is usually right:
   a near-exact band, a fuzzy band, and everything else. Too many thresholds
   over-fit the m/u and slow calibration.

**Level-ordering matters.** In `vector-er` levels are ordered most-to-least
agreement (`exact` before fuzzy), and a pair takes its *first matching* level.
Keep that order: an attribute with thresholds `[0.9, 0.8, 0.7]` means "exact →
very close → reasonably close → else", which is what the default m/u scheme
assumes.

A sensible starting point for people (from the original project) is the set in
§1: fuzzy first/last name, exact date-of-birth, exact+fuzzy email, fuzzy
address. For organizations consider name, postcode (with km threshold as a
fallback when postcodes are stale), and date-of-birth.

### 2.3 The Swoosh family: match, merge, domination

Swoosh turns the pairwise scorer into *clusters* by repeatedly **merging**
records that match. It is defined by a **match function** `M` (already chosen —
the FS posterior ≥ `tau`) and a **merge function** `μ` (how a matched group
becomes one record). From those it inherits a **domination order** used to
discard "covered" records.

The Swoosh paper — Benjelloun, O., Garcia-Molina, H., Menestrina, D., Su, Q.,
Whang, S. E., & Widom, J. (2009). *Swoosh: A Generic Approach to Entity
Resolution.* The VLDB Journal, 18(1), 255–276.
[doi:10.1007/s00778-008-0098-x](https://doi.org/10.1007/s00778-008-0098-x) —
identifies four *ICAR* properties of `M` and `μ` that make the computation
order-independent and let you discard dominated records safely:
**I**dempotence, **C**ommutativity, **A**ssociativity, and **R**epresentativity
(re-matching). If `M` and `μ` are ICAR, record pairs can match in *any* order
and a dominated record can be thrown away anytime — which is what enables the
fast R-Swoosh/F-Swoosh variants. If they are not ICAR, the only *correct*
algorithm is **G-Swoosh**, which re-tests merged representatives and cannot
discard records.

**Which of our algorithms is which:**

| Entry point | Swoosh algorithm | When to use |
|---|---|---|
| `SwooshClusterer.cluster` (default batch path; `BatchPipeline.run`) | Transitive closure over scored pairs (an R-Swoosh-style approximation — assumes matches are effectively transitive, i.e. a strong domination) | Production, large data; fast; results near G-Swoosh (§5.5 of the paper) |
| `cluster_with_merger` / `gswoosh` | **G-Swoosh** (correct for non-ICAR `M`/`μ`; re-tests representatives) | Exact clustering on modest data, audits, or custom merge functions whose re-match behaviour you care about |

Since the FS matcher is only partially ICAR (commutative, merge often
associative, but not always representative), most production use is the fast
transitive-closure path; G-Swoosh is there when you need the exact answer.
That exactness comes at a significant performance cost: it re-scans the pair
set until a full pass merges nothing, and every merge that yields a new
representative invalidates the scorer cache for that representative's edges,
re-invoking the expensive scorer and re-testing pairs — an entire pass's worth
of work for each merge round that the one-shot closure path never pays.

### 2.4 Choosing the merge function

The merge function decides what a *resolved entity* looks like. The framework
ships three; pick by what your users need to consume:

1. **`select_representative`** (default) — the cluster is represented by the
   most *complete* existing member (most non-`None` fields). Concrete, one row,
   but you "lose" the other records' differing values (no golden value; a
   completely-populated older record may shadow a sparser newer one).
   *Choose when* each cluster must map to a single real record (e.g. a golden
   reference index, dedupe of an identity table where a member row is the
   canonical one).
2. **`union_merge`** — a synthetic master record whose fields hold **all**
   values seen (set-valued fields; the Swoosh Union Class). Nothing is chosen
   or lost — the caller reviews the value sets. Best *recall* of variants;
   pairs re-match by *existence* over value sets, so it is the cleanest
   ICAR-compatible merge.
   *Choose when* you want to preserve every alias/variant (people with multiple
   spellings, products with multiple titles) and don't need a single
   "winning" value per field.
3. **`latest_merge(timestamp_field=...)`** — a synthetic master whose fields
   hold the **newest** non-`None` value per attribute (per-field recency), i.e.
   the master reflects the latest information as of the newest record.
   *Choose when* records are temporal (census waves, address history,
   evolving profiles) and the newest observation is authoritative.

The merge function also fixes your **domination order** (see §2.5). If your
custom merge does not preserve all values (e.g. averages a price, or keeps only
the longest string), the combination is **not likely to be ICAR** (special
cases aside, such as a monotonic timestamp that justifies discarding older
values) — so you should cluster with G-Swoosh, not rely on transitive-closure
approximations blindly.

**Writing your own.** Any callable with the same signature as the built-ins
works wherever a merge is accepted:

```python
def my_merge(records, positions):
    """Return (representative_record, representative_position).

    representative_record  -- the object that represents the fused cluster;
                              either one of the members or a synthetic master.
    representative_position -- the index into `records` of that representative
                              member, or -1 when the representative is
                              synthetic (not identical to any single member).
    """
    members = [records[p] for p in positions]
    ...
    return representative, representative_position
```

The contract:

- `records` is the full record list and `positions` the positions of the
  members being fused (an already-merged cluster can appear as a single group,
  so the members may span several original positions).
- The returned `representative_record` is the record object that will
  represent the fused cluster in future matches. `representative_position` is
  the **index into the original `records` list** of the member that
  `representative_record` corresponds to — i.e. the position the cluster is
  "anchored" at for reporting, and the value stored in
  `Cluster.representative_position`. It must be one of the `positions` you
  were given: if `representative_record` *is* one of the members, return its
  index into `records` (this is what `select_representative` does; a
  timestamp-based merge returns the newest member's index). If instead you
  build a **synthetic master record** from the members' values — a record that
  is not identical to any single member — there is no such index, and you
  return `-1` (this is what `union_merge` does). Synthetic masters holding
  `set`/`frozenset` fields are re-matched through the scorer's Union-Class
  existence lift.
- Pass it anywhere the built-ins go: `build_batch_pipeline(..., merge=my_merge)`
  for the default transitive-closure path, or
  `SwooshClusterer(tau=..., merge=my_merge)` /
  `gswoosh(records, pairs, match_probability, tau=..., merge=my_merge)` for
  full G-Swoosh re-matching.

Keep the caveats above in mind: if the merge **discards** values (chooses one
winner per field, picks the longest string), it is **not likely to be ICAR** —
although it *can* be in special cases (e.g. a monotonically increasing
timestamp field decides which values to keep — `latest_merge`).  When you are
not sure and want exactness, use G-Swoosh rather than trusting the closure path, and expect
re-match behaviour to be merge-specific.

### 2.5 Domination in this framework

**Intuitive definition.**  Once a cluster of records has been merged into one
representative, that representative becomes the cluster's *stand-in* for all
future comparisons.  We say a record $r_1$ is **dominated by** $r_2$ when $r_2$
can safely take $r_1$'s place in **every** comparison that $r_1$ could still
participate in — that is, for any new record $r'$ that would have matched
$r_1$, the representative $r_2$ matches $r'$ too:

$$ \text{for every possible } r':\quad r_1 \approx r' \implies r_2 \approx r' $$

If domination holds, comparing the *representative* against a new record tells
you everything comparing any *member* against it would — so the dominated
members can be dropped, and only the representative needs to keep being matched.
The formal Swoosh definition is *merge domination*:

$$ r_1 \preceq r_2 \iff r_1 \approx r_2 \ \text{ and } \ \mu(r_1, r_2) = r_2 $$

i.e. $r_1$ matches $r_2$ (so they belong together) and merging them leaves $r_2$
as the representative.  This is what lets Swoosh discard dominated records and
shrink the candidate set while still producing the same final clusters.

The framework does **not** take a separate domination oracle; the domination
order is an *emergent consequence of the merge function you pick*:

- **`select_representative`**: after a merge, the non-representative members
  are dominated by the representative (the representative *is* $\mu(r_1, r_2)$).
- **`union_merge`**: every member is dominated by the union master (the master
  contains all their values), so **all** values are preserved — the strongest,
  cleanest domination; it is exactly the Swoosh "Union Class" where domination
  is set inclusion.
- **`latest_merge`**: a member is dominated once every field the master keeps
  is at least as new as its own (recency domination); older records are covered
  by the newest master.

When can you *act* on domination (discard dominated records early)? Only when
`M` and `μ` are ICAR. In practice:

- With **`union_merge` + the existence-lifted match**, ICAR holds (the
  reflexive fix + commutativity + Union-Class), so dominated records can be
  discarded and R/F-Swoosh reasoning applies.
- With **`select_representative` or any non-union custom merge**, ICAR is not
  guaranteed, so the framework does *not* discard on domination — it uses
  G-Swoosh (re-testing) when you ask for it, or the transitive-closure
  approximation (which is effectively *assuming* strong transitivity) in the
  default batch path.

Practitioner rule of thumb: if your merge **preserves all values** (union) or
recency-dominates cleanly (timestamped latest-wins), you can lean on the cheap
closure path and trust it; if your merge **chooses/discards** values or fields
(the `select_representative` "keep one row" case), treat the results as an
approximation and use G-Swoosh when correctness is critical. The domination
order is therefore *your choice of merge function*, not a separate box to fill
in.

---

## 3. Incremental ER (one record at a time)

Use incremental ER when records arrive as a stream and each must be checked
against an existing reference population — the online / API / streaming-insert
setting of the original project's resolver.

### 3.1 Build the reference store and pipeline

```python
from vectorer.incremental import build_incremental_pipeline
from vectorer.pins import EMBEDDING_MODEL_ID, EMBEDDING_MODEL_REVISION

references = load_reference_people()          # list[dict]: 50k Canadian persons

pipeline = build_incremental_pipeline(
    references,
    embedder=SentenceTransformerEmbedding(EMBEDDING_MODEL_ID, EMBEDDING_MODEL_REVISION),
    comparisons=comparisons,                  # from §1
    k=20,                                     # top-k vector search blocking
    tau=0.85,                                 # match threshold on the posterior
)
```

`build_incremental_pipeline` builds an `InMemoryVectorDatabase` (embedder +
FAISS flat index over L2-normalized vectors = cosine) over `references`, then a
`FellegiSunterScorer` from the `comparisons`. If you omit the embedder you get
the deterministic hashing embedder; if you already have a calibrated scorer,
pass `scorer=` instead of `comparisons=`.

### 3.2 Resolve a query

```python
query = {"first_name": "Jon", "last_name": "Smyth",
         "date_of_birth": "1985-06-15", "email": None, "address": "1 Main St Toronto"}

r = pipeline.resolve(query)
r.decision                 # Decision.MATCH / Decision.NON_MATCH
r.matches                  # sorted MatchResult list (posterior desc)
r.matches[0].match_probability, r.matches[0].match_weight, r.matches[0].blocking_score
r.retrieved                # all top-k ScoredCandidates (even non-matches)
r.embedding                # the query's embedding vector
```

`resolve` runs the whole chain: parse -> embed -> vector search blocking (top-k)
-> FS scoring of the top-k -> classify. A query is a match if any candidate's
posterior `>= tau`.

### 3.3 Persisting the reference store (the serving modality)

**Two modalities, one entry point.** `build_incremental_pipeline` accepts
**either** the raw ``records`` (embeds them into a new store — good for
illustration and small/one-off setups) **or** a pre-built ``vector_database=``
(the production *serving* case, where the population was embedded separately,
previously):

```python
from vectorer.incremental import build_incremental_pipeline
from vectorer.vectorstores import InMemoryVectorDatabase

# Modality 1 — embed raw records (illustration / small):
pipeline = build_incremental_pipeline(records, comparisons=comparisons, ...)

# Modality 2 — serve against an already-embedded store (production):
# the population was embedded earlier, e.g. after a Splink pre-dedupe, then
# persisted to / loaded from a (possibly distributed) vector DB.
pipeline = build_incremental_pipeline(
    vector_database=loaded_store,        # carries its own embedder already
    comparisons=comparisons, k=20, tau=0.85,
)
```

In modality 2 the reference records are **not** re-embedded — only queries are
embedded at ``resolve()`` time.  ``IncrementalPipeline.from_store(store,
scorer, ...)`` is exactly the same serving path, written as a named, explicit
classmethod on the pipeline.

Embedding 50k records is the expensive, one-off step; persist it so a new
process resolves without re-embedding:

```python
from vectorer.vectorstores import InMemoryVectorDatabase, FlatIndex
from vectorer.scoring import FellegiSunterScorer

# build once...
db = InMemoryVectorDatabase(embedder, FlatIndex(normalize=True))
db.add(references)
db.save("data/person_index")          # writes index.faiss + records.pkl + metadata.json

# ...reload later (embedder is required to deserialize the store)
db = InMemoryVectorDatabase.load("data/person_index", embedding=embedder)
pipeline = IncrementalPipeline(
    vector_database=db,
    scorer=FellegiSunterScorer.from_comparisons(comparisons, threshold=0.85),
    k=20, tau=0.85,
)
```

### 3.4 Scaling the reference store with an external vector DB

The incremental pipeline talks to its reference store exclusively through the
`VectorDatabase` interface (`add`, `index.search`, `record_at`, `__len__`,
`embedding`).  The bundled `InMemoryVectorDatabase` is a FAISS flat index in
memory — great up to a few million records, but a flat scan becomes the
bottleneck as N grows.  To scale horizontally, implement `VectorDatabase` (and
optionally `IndexingStrategy`) against an **external distributed vector DB**
such as Qdrant, Milvus, Pinecone, Weaviate, or Elasticsearch *(contributions
welcome)*:

```python
from vectorer.incremental import IncrementalPipeline
from vectorer.scoring import FellegiSunterScorer
from my_adapters import QdrantVectorDatabase     # your VectorDatabase impl

db = QdrantVectorDatabase(embedder=embedder, collection="people")
db.add(references)                               # upsert (vector, position, payload=record)

pipeline = IncrementalPipeline(
    vector_database=db,
    scorer=FellegiSunterScorer.from_comparisons(comparisons, threshold=0.85),
    k=20, tau=0.85,
)
result = pipeline.resolve(incoming_record)        # unchanged pipeline code
```

The adapter maps the framework's interface onto the DB client: use the record
*position* as the external document id and store the record dict as the
payload, so `index.search` returns positions and `record_at(position)` is an id
fetch.  Only the index and records move remote — the embedding model and the FS
scorer stay local.

When to use it:

- **Incremental / online service at large scale**: HNSW/IVF sharded ANN replaces
  the flat FAISS scan and gives horizontal scaling + persistence.
- **Complementary to the multi-node batch executor**: `distributed_batch_er` /
  `distributed_score_and_reduce` shards whole-dataset dedup across machines
  (Ray) — see `examples/multi_node_distributed_er.py`; the external-DB
  route is for huge *reference stores* in online mode.  They can coexist.

> **Multi-node operation.**  For the full how-to — what shards/streams across
> machines, what stays single-process by design, and how to run a Ray cluster
> or a Qdrant-backed incremental store — see
> [`.docs/distributed_er.md`](distributed_er.md) and architecture doc §7.3.

Caveats: preserve cosine semantics (L2-normalize or use the DB's cosine metric
so scores stay comparable with the local `FlatIndex`); watch payload-size
limits / serialization cost when storing records as payloads; and account for
eventual consistency if the DB's upserts are async (`ingest_novel` expects
immediate visibility).  See the architecture doc §7.1 for the detailed design.

### 3.5 (Re)building from a scratch or empty store

The incremental store can bootstrap itself: start with an empty database and
`add`/`ingest` records as they resolve. Combined with the embedding model:

```python
db = InMemoryVectorDatabase(embedder, FlatIndex(normalize=True))
pipeline = IncrementalPipeline(db, scorer, k=20, tau=0.85)

for record in stream:
    pipeline.ingest(record)          # parse + embed + append; or
    # pipeline.ingest_novel(record)  # append only if no existing match (see §3.6)
```

### 3.6 Ingestion modes

- `pipeline.add(records)` — append already-parsed records.
- `pipeline.ingest(payload)` — parse + embed + append, returns new position.
- `pipeline.ingest_novel(payload, novelty_threshold=None)` — **novelty-only
  ingestion**: resolves first, appends only if *no* candidate scores at or
  above the novelty threshold (`tau` by default). Returns the new position, or
  `None` for a duplicate — useful for de-duplicated ingest into the index.
- `pipeline.ingest_novel_many(deck)` — batch version, returns a position list
  aligned with the input (`None` = skipped duplicate).

### 3.7 Using stage hooks

Every stage is a public method you can override or call directly:

```python
record = pipeline.parse(payload)          # stage 1
vector = pipeline.embed(record)           # stage 2 (cached for the next stages)
cands   = pipeline.block(record, k=20)    # stage 3
scored  = pipeline.score(record, cands)   # stage 4
matches = pipeline.classify(record, scored)  # stage 5
```

Subclass `IncrementalPipeline` and override a stage to customise it (e.g. swap
the blocker, add a pre-filter, cache features).

---

## 4. Batch ER (whole dataset, deduplicate + cluster)

Use batch ER when you have a complete dataset and want to find all duplicate
entities at once — the bulk deduplication analogue of the incremental use case.
The chain is: parse all -> embed all -> canopy blocking on the embedded dataset
-> FS scoring of every canopy candidate pair -> Swoosh clustering.

### 4.1 Build and run

```python
from vectorer.batch import build_batch_pipeline, BatchPipeline
from vectorer.embeddings import SentenceTransformerEmbedding

dataset = load_all_records()                # list[dict], e.g. 10k+ people

pipeline = build_batch_pipeline(
    embedder=SentenceTransformerEmbedding(EMBEDDING_MODEL_ID, EMBEDDING_MODEL_REVISION),
    comparisons=comparisons,                # same comparison set as §1
    n_canopies=256,                         # FAISS k-means canopy count
    overlap_m=2,                            # top-2 canopy assignment (overlap)
    canopy_seed=42,
    tau=0.85,
)
result = pipeline.run(dataset)
```

### 4.2 Reading the result

```python
result.n_clusters                      # number of distinct entities found
result.n_singletons                    # entities with exactly one record
result.n_non_singletons                # entities with >= 2 records (duplicates found)
result.cluster_of_position(7)          # cluster id of record 7
result.assignment.clusters             # cluster_id -> Cluster(members, representative)
result.scored_pairs                    # every scored canopy candidate pair
result.timing                          # per-stage seconds
```

`Cluster` exposes `member_positions` (record indices) and a `representative`
record (the most complete member, by non-null field count).

### 4.3 Saving outputs

```python
# Persist the Swoosh assignment (cluster_id per record) as JSON
assign = result.assignment
json.dump(
    {pos: cluster_id for pos, cluster_id in assign.node_cluster.items()},
    open("clusters.json", "w"),
)
# Or map to user-facing ids via a schema
schema = RecordSchema(("first_name", "last_name", "date_of_birth", "email"),
                      id_column="record_id")
id_to_cluster = result.cluster_ids_of(schema)   # {"r000": 5, ...}
```

### 4.4 Tuning candidate generation

- `n_canopies` — more canopies = smaller cells = fewer candidate pairs but
  higher risk of splitting near-boundary true matches.
- `overlap_m` — degree of multi-assignment. `1` = hard partition (cheapest);
  `2–3` is the common sweet spot that recovers most boundary matches.
- The canopy index is deterministic for a given `canopy_seed`.

**Swoosh re-matching (optional).** By default `run` does the transitive closure
over above-`tau` pairs (standard *score then cluster*). If you want full
G-Swoosh behaviour — re-matching merged representatives against the candidate
pair set — use the pair-driven entry points below.

> **⚠ Expensive.** G-Swoosh re-runs the Fellegi-Sunter matcher on merged
> representatives against the candidate pair set until convergence — dozens of
> times slower than the default closure (the bulk benchmarks show the swoosh
> stage jumping from ~0.05 s to ~22–400 s). Use it only when **absolute
> correctness is required** (audits, final reconciliation, or non-ICAR custom
> merges); prefer the default closure for production.

`scorer_match` receives the two
**representative records** (not positions), which may be synthetic master
records; `merge` defaults to `select_representative` but can be `union_merge`
or `latest_merge`:

```python
from vectorer.blocking import canopy_blocking
from vectorer.clustering import SwooshClusterer, union_merge

parsed = result.records                            # the record dicts (canopy uses them)
vectors = pipeline.embed_all(parsed)
canopy = pipeline.block(vectors)
pairs = list(canopy.candidate_pairs())
scored = pipeline.score(parsed, pairs)
assignment = SwooshClusterer(
    tau=0.85,
    merge=union_merge,          # or latest_merge(..., timestamp_field="ts")
).cluster_with_merger(
    parsed, pairs,
    scorer_match=lambda l, r: pipeline.scorer.score(l, r),   # records, not positions
)
# union_merge representatives are synthetic (representative_position == -1)
# set-valued master records; the scorer's Union-Class lift scores them.
```

### Choosing a merge function

- `select_representative` (default) — pick the most complete member as the
  cluster's record (which is a real member; `representative_position` is a
  valid index). Good default when you want a concrete record per entity.
- `union_merge` — a master record whose fields hold **all** values seen across
  the matched records (set-valued fields; the Swoosh Union Class). No "golden
  value" is chosen; caller reviews the value sets. The scorer's Union-Class
  lift re-matches these set-valued records by existence over value pairs.
- `latest_merge(timestamp_field="timestamp")` — a master record whose fields
  hold the **most recent** value per attribute (newest non-`None` member per
  field), so the merged record reflects the latest information.

For the batch pipeline, pass the merge function directly:
`build_batch_pipeline(..., merge=union_merge)`.

To provide your own, use the same `merge(records, positions) ->
(representative, position)` signature as the built-ins (see §2.4) — for
example a custom `my_merge` above becomes
`build_batch_pipeline(..., merge=my_merge)`; pass it to
`SwooshClusterer(tau=..., merge=my_merge)` the same way for the pair-driven
G-Swoosh path.

---

## 5. Record Linkage: linking two databases (mergers / collaboration)

The two modes so far take *one* database and resolve or deduplicate it. For the
**merger / cross-enterprise collaboration** use case you have *two*
independently-managed databases with **different schemas** and want to say which
records refer to the same entity — without ever merging the two stores. That is
what the **Link** mode (`vectorer.link.RecordLinker`) does: it emits a table of
**link edges** `(a_id, b_id, posterior, weight, decision)`.

### 5.1 The core idea: canonical field projection

The two schemas differ, but the *compared fields overlap*. You declare that
overlap once:

```python
from vectorer.link import RecordLinker, FieldMap

linker = RecordLinker(
    embedder=embedder,
    comparisons=[                       # declared on the CANONICAL names
        make_comparison("jaro_winkler_at_thresholds", col_name="name"),
        make_comparison("date_of_birth_comparison", col_name="dob"),
        make_comparison("email_comparison", col_name="email"),
    ],
    field_maps={
        "A": FieldMap({"name": "name", "dob": "birth_date", "email": "email_c"},
                      id_column="cust_id"),
        "B": FieldMap({"name": "legal_name", "dob": "dob", "email": "email_p"},
                      id_column="partner_id"),
    },
    k=20, tau=0.7,
)
```

`FieldMap({"canonical": "source_col", ...})` projects each side into the
canonical compared fields; `id_column` names the side's own primary key so the
edges carry your ids.  A canonical field absent on one side projects to `None`,
which FS treats as a null level (no evidence) — so overlap *within* the
compared fields is handled automatically.

### 5.2 Directed link (recommended)

Index database B once, resolve every A record against it:

```python
table = linker.link_directed(records_a, records_b)
# or, with a strict one-to-one mapping (each B used at most once)
table = linker.link_directed(records_a, records_b, enforce_11=True)
```

### 5.3 Symmetric link

Block A and B together with overlapping canopies, score only the **cross-DB**
pairs, emit edges — no merge happens:

```python
table = linker.link_symmetric(records_a, records_b,
                              n_canopies=256, overlap_m=2)
```

### 5.4 Reading the result

```python
table.matches                 # LinkEdge list (decision == "match")
table.possible_matches        # review-tier band (if possible_low was set)
table.as_pairs()              # [(a_id, b_id), ...] over the matches
table.by_a() / table.by_b()   # edges grouped per record
table.to_dict()               # exportable {a_count, b_count, n_links, links}

for edge in table.matches:
    print(edge.a_id, edge.b_id, edge.probability, edge.match_weight)
```

Whether the output is 1:1 or 1:N is your call: directed + `enforce_11` gives
strict one-to-one; the default (both modes) keeps every cross-DB pair above
`tau`, i.e. 1:N links.

### 5.5 Practical notes

- **Blocking rides on the overlap**: embeddings are computed on the canonical
  text of both sides, so semantic tolerance (MiniLM) helps when names differ
  across DBs; bump `k` for thin overlaps.
- **Calibrate on cross-DB labels**: `calibrate_from_pairs` already takes
  `(a_field_l, a_field_r)` pairs with `is_match` — feed it hand-labelled
  cross-DB pairs (use the canonical field names) instead of relying on default
  m/u.  The example in §1 used `prior=1e-2` to make the scores usable on small
  test sets; for production, calibrate the prior too.
- **Three-tier review**: pass `possible_low=` (e.g. `0.5`) to split automatic
  links from the "possible match" band you review before the merger decision.
- See `examples/link_two_databases.py` for a full runnable two-database merge
  scenario.

---

## 6. Calibrating the Fellegi-Sunter parameters

The default `m/u` values produce reasonable scores out of the box, but for
production-grade probabilities you should calibrate them — either supervised
(we have labels) or unsupervised (we have a *duplicate-bearing* population;
see §6.2 for why that qualifier matters).

### 6.0 Term-frequency adjustments (weighting rare exact matches more)

The base `u` for an exact match assumes that *any* two matching values are
equally unlikely — but that is wrong for skewed fields. On a "surname" column
where most records are "smith", an exact match on "smith" is weak evidence
(these records probably differ elsewhere), while an exact match on "soetoro"
is strong evidence the records are the same person. **Term-frequency (TF)
adjustment** rescales the exact-match evidence by how *rare* the matched value
is in the reference population: the effective $u$ of the exact level becomes

$$ u_{\mathrm{eff}} = \frac{u_{\mathrm{exact}}}{\max\left(\mathrm{tf}_{\mathrm{left}},\ \mathrm{tf}_{\mathrm{right}}\right)^{w}} $$

so a rare exact match posts a much larger posterior than a common one ($w$ is
the TF weight, with a $\mathrm{tf}_{\min}$ floor).

How to use it — it is **opt-in and needs a reference population**, and without
`base_records=` the adjustment silently does nothing:

```python
comparisons = [
    *base_comparisons,  # fuzzy first/last name, DOB, email...
    make_comparison("exact_match", col_name="surname", term_frequency_adjustments=True),
]

scorer = FellegiSunterScorer.from_comparisons(
    comparisons, base_records=reference_population,  # REQUIRED for TF
)
```

Guidance:

- Enable it on fields with **highly skewed value distributions** (surnames,
  street names, employer, "state"), and leave it off for fields that are
  effectively unique (email, full DOB, national IDs) where every value is
  already rare.
- Only the `exact_match` comparison carries the flag; fuzzy comparisons are
  unaffected.
- The reference population should be **representative of the data you score**
  (ideally the same store/population). Pass it to `from_comparisons`/`from_settings`
  as `base_records=`.
- The TF metadata (column, weight, `tf_minimum_u_value` floor) survives
  calibration and persistence; calibrated `u` of the exact level is what gets
  scaled.

### 6.1 Supervised: calibrate from labelled pairs

Construct a list of records with an `is_match` flag (1 = match, 0 = non-match)
and, for each compared field, both the left and right values:

```python
labelled_pairs = [
    {"is_match": 1,
     "first_name_l": "john",   "first_name_r": "john",
     "last_name_l": "smith",   "last_name_r": "smith",
     "date_of_birth_l": "1985-06-15", "date_of_birth_r": "1985-06-15",
     "email_l": None, "email_r": None},
    {"is_match": 0,
     "first_name_l": "robert", "first_name_r": "anna",
     "last_name_l": "green",   "last_name_r": "walker",
     "date_of_birth_l": "1978-02-28", "date_of_birth_r": "1991-07-20",
     "email_l": None, "email_r": None},
    # ... at least a few hundred pairs, mixing matches and non-matches
]

scorer_calibrated = FellegiSunterScorer.from_comparisons(comparisons).calibrate_from_pairs(
    labelled_pairs, smoothing=0.5,
)
# Use it in any mode via scorer=scorer_calibrated (incremental, batch, or Link)
pipeline = build_incremental_pipeline(refs, embedder=embedder,
                                      scorer=scorer_calibrated, k=20, tau=0.85)
# Calibrated posteriors are only as good as the labelled sample: with a handful
# of pairs they stay conservative (~0.3–0.5 for exact matches). Calibrate on a
# few hundred labelled pairs to get sharp, production-useful probabilities.
```

### 6.2 Unsupervised: expectation maximisation

EM needs **duplicate pairs in the training population** — this is not
optional. The estimator generates candidate pairs under blocking rules,
estimates `u` from uniform random pairs, and fits `m` plus the match prior by
EM:

```python
scorer_em = FellegiSunterScorer.from_comparisons(comparisons).fit_em(
    training_records,                         # duplicate-bearing people
    training_block_on=[("first_name",), ("date_of_birth",)],
    recall=0.7,                               # approximate blocking recall
    seed=42,
)
pipeline = build_batch_pipeline(embedder=embedder, scorer=scorer_em,
                                n_canopies=256, overlap_m=2, tau=0.85)
```

The EM loop infers which blocked pairs are true matches and fits `m` (plus the
match prior) from them. If the population contains no duplicates, there are no
true matches among the blocked candidates, and EM has nothing to fit `m`
against — it silently mis-fits instead of failing.

The common failure is feeding a **pre-deduplicated** population (one row per
entity, e.g. an already-cleaned reference store). Such a set still generates
blocked candidate pairs — records that share a blocking key (same first name,
same date of birth) but are different entities — so the run does **not** raise
an error. The EM loop treats those chance co-blocked non-matches as evidence
for the agreement levels, over-estimating the `m` probabilities and distorting
the base prior; the trained scorer then over-confidently matches records that
merely look alike. Fit on the *raw* input instead:

- the batch input itself, **when it contains genuine near-duplicates** (e.g. a
  raw export with duplicate rows) — the natural choice for §4's batch path;
- or a sample of the reference population into which you have **planted
  duplicates** (copy records, perturb a field or two) for the incremental use
  case.

Avoid fully deduplicated / clean reference sets. If that is all you have,
either plant duplicates (carefully — the planted duplicates define exactly
what the fit sees) or use the supervised route (§6.1).

Sanity-check the fit before trusting it: each comparison's fitted `m` should
concentrate on the agreement levels (exact, near-exact), and the base prior
(`probability_two_random_records_match`) should be small but not implausibly
tiny. A near-zero prior, or an `m` spread evenly across all levels, says no
true matches were seen — the input was effectively duplicate-free, and the
result should be discarded. (The one case that *does* raise a clear error is a
population that generates zero blocked candidate pairs at all — for example a
tiny set where no two records share a blocking key.)

### 6.3 Calibrating EM in practice (the operating-point sweep)

**The problem.** EM estimates the match proportion — the FS base prior — jointly
with the comparison-vector probabilities, and that proportion is the weak
dimension. When the true match share is small, EM cannot reliably detect the
match class and can converge to a prior (and therefore a threshold behaviour)
that is not relevant to linkage. Yancey documents the phenomenon precisely:

> "Failure to converge to the desired class parameters happens when the
> proportion of one of the classes M, U′, U″ is too small to be detected by the
> algorithm. In practice, it is the class M of matches that is generally
> smallest, and when the proportion of this class drops below 0.05 or so, the
> EM algorithm can converge to parameter values that are not relevant to record
> linkage calculation." — Yancey (2004), *Improving EM Algorithm Estimates for
> Record Linkage Parameters* [20].

**Two remedies — use both.**

1. **Match-enrich the training set (Yancey's fix).** Feed EM a population in
   which the duplicate share is noticeably above the "invisible ≈5%" floor —
   e.g. the `population_with_duplicates.json` generator (5% + 1% + 0.1%
   perturbed duplicates), or a blocked-pair sample that is match-enriched.
   Enriched training lets EM see the match class and fit sensible `m/u`.

2. **Freeze the prior and sweep the operating point.** Even with enrichment,
   EM's own prior estimate can drift. `fit_em(fixed_prior=...)` holds the base
   prior **frozen** across every EM iteration (only `m/u` are learned — the
   Splink-style fixed-prior EM), so you can treat the prior as a knob.  Choose
   the operating point by sweeping the prior×threshold grid on *labelled* data:

   ```bash
   python benchmarks/benchmark_bulk_er_em.py \
     --data-file benchmarks/population_with_duplicates.json \
     --gt-file benchmarks/population_gt.json \
     --prior-sweep-priors "1e-5,1e-4,1e-3" \
     --prior-sweep-taus "0.5,0.7,0.85,0.95" \
     --n-training-subsample 20000 --em-max-pairs 30000
   ```

   This reports precision/recall/F1 at each (prior, tau) on the labelled pairs.
   **Pick the operating point by its precision/recall, not by a nominal
   threshold** — on the duplicate population this recovered ~5 points of recall
   (0.70 → 0.755) at precision 1.0 just by moving the prior from EM's estimate
   to `1e-3`.  Use `fixed_prior` in production with the chosen value, and set
   `tau` from the grid.

**Why this matters for the benchmark flows.** A scorer trained by default
`fit_em` (learned prior) may be *correct* yet still under-merge at `tau=0.85`
because the learned prior is too small — the framework faithfully applies the
weights; the calibration, not the machinery, is the issue. Calibrating by the
sweep removes that failure mode.

### 6.4 Importing trained parameters from Splink

If the **base population was already deduplicated / linked with Splink**, or a
huge distributed population was first deduped by Splink before being loaded
into a distributed vector DB for the incremental pipeline, you can **reuse
Splink's trained `m/u` (and term-frequency weights) directly** — the batch, Link
and incremental modes all accept a scorer built this way.

Because this framework evaluates the comparison family natively (levels are
vectorized NumPy predicates, not Splink's SQL), Splink's settings JSON cannot be
loaded verbatim.  `import_splink_scorer` bridges the gap: it matches Splink's
trained comparisons to your native comparison set by **output column name** and
transfers the per-level `m_probability` / `u_probability` (and TF metadata) onto
the corresponding native levels, preserving the standard level order
(`null -> exact -> fuzzy -> else`).

```python
from vectorer.scoring import import_splink_scorer
from vectorer.comparisons import make_comparison

# 1. Splink's trained model (e.g. Linker.misc.save_model_to_json()).
splink_json = {/* 'comparisons': [...], 'probability_two_random_records_match': 1e-5 */}

# 2. A native comparison set over the SAME columns and thresholds as Splink.
native = [
    make_comparison("jaro_winkler_at_thresholds", col_name="first_name",
                    score_threshold_or_thresholds=[0.9, 0.7]),
    make_comparison("email_comparison", col_name="email"),
]

scorer = import_splink_scorer(
    splink_json, native,
    threshold=0.85,
    base_records=base_population,   # rebuilds TF value tables natively
)

# 3. Use it in any mode.
pipeline = build_batch_pipeline(scorer=scorer, ...)     # batch dedupe
linker = RecordLinker(scorer=scorer, ...)                # link mode
inc = IncrementalPipeline(vector_database=db, scorer=scorer, ...)   # incremental
```

**Use cases**

- A base database that has **already been used to train Splink for batch dedupe
  or linkage** — import that model instead of re-calibrating here.
- A **huge distributed base population** used by the incremental pipeline:
  dedupe it once with Splink, then load the cleansed population into a
  distributed vector DB and resolve against it with this framework — the
  Splink-trained scorer carries the same decision boundary into the online
  path.

**Caveats (read before relying on this)**

- **Levels must match exactly.** The native comparison set must have the same
  output column names, the same thresholds, and therefore the same number of
  levels in the same order as the Splink model; the helper validates the level
  count and raises if they differ.  `jaro_winkler_at_thresholds(...)` with the
  same `score_threshold_or_thresholds` and `date_of_birth_comparison` /
  `email_comparison` / `exact_match` etc. mirror Splink's level structure.
- **Splink's `m/u` are only as good as the Splink training data.** If the
  populations differ (e.g. Splink trained on a different corpus), the weights
  will be miscalibrated for this dataset.  For the distributed-base use case,
  the Splink prior and m/u reflect *that* base — appropriate since you score
  against the same population.
- **Term-frequency tables are rebuilt natively** from `base_records=`.  The
  trained `tf_adjustment_weight`/`tf_minimum_u_value` transfer; the value→
  frequency tables come from your population (pass `base_records=`), not
  Splink's internal tables.
- **No SQL is inherited.** Only the trained numbers transfer; level *tests* are
  the native predicates.  If your Splink comparison used a custom comparison
  level this framework doesn't implement (`custom_comparison`), map it manually
  or avoid importing that comparison.
- **`idempotent=True`** (reflexivity) is applied by default; single-pair
  posteriors may differ slightly from Splink's on exact-identical pairs (this
  framework forces `P=1.0`).  Disable with `idempotent=False` to compare
  raw.

### 6.5 Persisting the trained model

```python
scorer_em.save("model.json")                  # comparisons (with m/u) + prior
scorer = FellegiSunterScorer.load("model.json")
pipeline = IncrementalPipeline(db, scorer, k=20, tau=0.85)
```

`to_settings()`/`to_dict()` return the same structure if you prefer to store it
inline; `save`/`load` wrap them as JSON.

---

## 7. Choosing the match threshold (`tau`)

`tau` is the posterior at which a pair is declared a match. It is the
*operating point* of the Fellegi-Sunter decision rule:

- **High tau (0.9–0.99)** — high precision, lower recall. Fewer false matches;
  good when a wrong match is costly (identity linking).
- **Low tau (0.5–0.7)** — higher recall, more false matches. Good for
  candidate decks / clerical review, or when recall is the priority.

For binary classification and no labelled data, the natural default is `0.5`
(the posterior where evidence is neutral). The original project used `0.85`
for a precision-leaning operating point. For three-band decisions (match /
possible match / non-match) outside the two-band pipeline, use the classifier
directly:

```python
from vectorer.classification import ThresholdClassifier

classifier = ThresholdClassifier(tau=0.9, possible_low=0.5)
classifier.decide(0.93)   # Decision.MATCH
classifier.decide(0.60)   # Decision.POSSIBLE_MATCH
classifier.decide(0.10)   # Decision.NON_MATCH
```

For principled threshold selection, calibrate the scorer first (posteriors are
then meaningful probabilities) and, if you have labels, sweep `tau` over
precision/recall as the previous project's experiments did.

---

## 8. Practical notes from the original project's use case

### 8.1 Data preparation

- Lowercase and normalize values before embedding and comparison (suffixes,
  abbreviations, whitespace). The original project serializes each person with
  a stable template (`first_name`, `last_name`, `date_of_birth`, address,
  email) so identical entities embed near-identically.
- Missing fields should be `None`, not empty strings: the comparison levels'
  null handling and the term-frequency logic both rely on `None`.
- Because names are shared across records (the synthetic population has a
  small name vocabulary), name fields by themselves are high-frequency: an
  exact email or a full date-of-birth match adds far more evidence per bit. If
  a field is very common in your data, enable term-frequency adjustment:
  `make_comparison("exact_match", col_name="surname", term_frequency_adjustments=True)`
  with the reference population passed as `base_records=` to the scorer
  (`FellegiSunterScorer.from_comparisons(comparisons, base_records=refs)`).
  See §6.0 for how TF adjustment works and when to enable it.

### 8.2 Blocking recall

- Incremental top-k: the original project's `k=20` achieved **99.6%** top-k
  blocking recall on the 50k person index (missing-rate 0.3). Raise `k` to
  recover more recall, at a linear cost in scoring time.
- Bulk canopies: overlap `m ≥ 2` recovers near-boundary matches; the 52k-record
  benchmark deduplicated with recall ≥ 0.97 at `overlap_m=1–2`.

### 8.3 What the numbers look like (sanity anchors)

On the original 50k-reference incremental use case (MiniLM embeddings,
k=20):

- cold per-query `resolve`: ~33 ms end-to-end with the MiniLM embedder
  (embedding-dominated; ~4–5 ms with the hashing embedder);
- exact duplicates match at posterior 1.0; a 2-char-typo `smith→smithq`
  duplicate still matches at > 0.99 (jaro-winkler + dob + email evidence).

On the bulk 52k use case (hashing embedder): 254 s total, of which ~230 s is
vectorized FS scoring of 1.7 M canopy pairs (~7.3k pairs/s), precision 1.0 and
recall 0.97–1.0.

Treat every number as a sanity check, not a promise: your embedding, data
quality and hardware will move them.

### 8.4 Errors to expect

- `calibrate_from_pairs` requires both 1s and 0s in `is_match`.
- `fit_em` raises if no blocking-rule candidate pairs are generated — add
  duplicates or adjust `training_block_on`.
- `distance_function_at_thresholds` raises for an unknown function name;
  pass a callable for anything bespoke.
- Building a comparator over columns absent from your records is only detected
  at scoring time (the score statuses degrade gracefully to null levels). Pass
  the `base_records`/schema you actually serialize with.

---

## 9. End-to-end script skeleton

Putting the pieces together for the incremental use case:

```python
import json
from vectorer.comparisons import make_comparison
from vectorer.embeddings import SentenceTransformerEmbedding
from vectorer.incremental import build_incremental_pipeline
from vectorer.pins import EMBEDDING_MODEL_ID, EMBEDDING_MODEL_REVISION
from vectorer.scoring import FellegiSunterScorer

comparisons = [
    make_comparison("jaro_winkler_at_thresholds", col_name="first_name"),
    make_comparison("jaro_winkler_at_thresholds", col_name="last_name"),
    make_comparison("date_of_birth_comparison", col_name="date_of_birth"),
    make_comparison("email_comparison", col_name="email"),
]

# 1. Index the reference population once (with term-frequency base, if desired)
embedder = SentenceTransformerEmbedding(EMBEDDING_MODEL_ID, EMBEDDING_MODEL_REVISION)
references = json.load(open("reference_people.json"))
pipeline = build_incremental_pipeline(references, embedder=embedder,
                                      comparisons=comparisons, k=20, tau=0.85)

# 2. Optionally calibrate (labels) or fit EM on a duplicate-bearing sample
#    scorer = FellegiSunterScorer.from_comparisons(comparisons)
#    scorer = scorer.calibrate_from_pairs(labelled_pairs)   # or .fit_em(...)
#    pipeline.scorer = scorer

# 3. Persist the store + model
pipeline.vector_database.save("data/person_index")
pipeline.scorer.save("data/scorer.json")

# 4. Resolve incoming queries
for raw in incoming:
    result = pipeline.resolve(json.loads(raw))
    yield {"decision": result.decision.value,
           "best_probability": result.matches[0].match_probability if result.matches else None,
           "best_candidate": result.matches[0].candidate_position if result.matches else None}
```

For the batch analogue, see §4. All modes — incremental, batch, and record
linkage (§5) — share the comparison set, scorer and calibration, so a model
trained once serves them all.