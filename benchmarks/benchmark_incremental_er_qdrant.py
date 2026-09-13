"""Measure per-query latency of the incremental ER pipeline against a live **Qdrant** store.

This is the external-vector-DB sibling of ``benchmark_incremental_er.py``: the
reference population is embedded locally and served from a Qdrant collection,
so the cold query path is embedding -> remote Qdrant top-k ANN blocking ->
Fellegi-Sunter scoring on the top-k -> classification, with the index and the
record payloads living in the cluster (local dev: ``localhost:6333``; pass a
``--url`` for a distributed instance).

It reports the same statistics as the in-memory benchmark (mean / median /
percentiles / min / max / stdev), an optional embedding/Qdrant-blocking/scorer
phase breakdown, and ground-truth blocking quality (top-k recall + match rate
at tau) against the same close-variant query set.  Additionally it times the
two ingest stages separately -- embedding the population (CPU/embedder cost)
vs the ``upsert`` round-trips (network cost) -- since for a remote store those
are the two things that scale.

Requires a reachable Qdrant server and ``pip install qdrant-client``.

Example::

    # Qdrant on 127.0.0.1:6333, no auth, hashing embedder, 20k references
    python benchmarks/benchmark_incremental_er_qdrant.py \\
        --url http://127.0.0.1:6333 --n-references 20000 --query-count 100 \\
        --breakdown --output results/incremental_qdrant_latency.json

    # real sentence-transformers embedder (needs `pip install -e ".[embedding]"`)
    python benchmarks/benchmark_incremental_er_qdrant.py --embedder sentence

The collection is created (or reused) by default; ``--recreate`` drops and
rebuilds it, and ``--collection`` names it.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Optional, Sequence

from tqdm import tqdm

from vectorer.comparisons import make_comparison
from vectorer.embeddings import CharacterHashingEmbedding
from vectorer.incremental import IncrementalPipeline
from vectorer.scoring import FellegiSunterScorer
from vectorer.vectorstore_adapters import QdrantVectorDatabase

DEFAULT_QUERY_COUNT = 100
DEFAULT_THRESHOLD = 0.85
DEFAULT_BLOCKING_K = 20
DEFAULT_MISSING_RATE = 0.3
DEFAULT_CLOSE_VARIATION_RATE = 0.15
DEFAULT_REFERENCE_COUNT = 20000
DEFAULT_QDRANT_URL = "http://127.0.0.1:6333"
DEFAULT_COLLECTION = "vectorer_bench"


def percentile(values: Sequence[float], p: float) -> float:
    """Nearest-rank percentile over ascending ``values`` (0..1)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(p * len(ordered))))
    return ordered[idx]


# ---------------------------------------------------------------------------
# Population + queries (identical to benchmark_incremental_er.py)
# ---------------------------------------------------------------------------


FIRST_NAMES = [
    "john", "mary", "robert", "susan", "james", "linda", "michael", "patricia",
    "david", "jennifer", "william", "elizabeth", "richard", "barbara", "joseph",
    "thomas", "sarah", "charles", "karen", "daniel", "nancy", "paul", "betty",
    "mark", "helen", "steven", "sandra", "george", "ashley", "ken",
]
LAST_NAMES = [
    "smith", "jones", "martinez", "brown", "wilson", "davis", "garcia", "miller",
    "lee", "taylor", "anderson", "thomas", "moore", "jackson", "martin", "thompson",
    "white", "lopez", "hill", "scott", "green", "adams", "baker", "gonzalez",
    "nelson", "carter", "mitchell", "perez", "roberts", "turner",
]
STREET_KINDS = ["St", "Ave", "Rd", "Blvd", "St.", "Ave.", "Rd.", "Blvd."]
CITIES = ["Toronto", "Vancouver", "Montreal", "Calgary", "Ottawa", "Edmonton", "Winnipeg", "Halifax"]


def make_comparisons() -> list:
    return [
        make_comparison(
            "jaro_winkler_at_thresholds", col_name="first_name",
            score_threshold_or_thresholds=[0.9, 0.8, 0.7],
        ),
        make_comparison(
            "jaro_winkler_at_thresholds", col_name="last_name",
            score_threshold_or_thresholds=[0.9, 0.8, 0.7],
        ),
        make_comparison("date_of_birth_comparison", col_name="date_of_birth"),
        make_comparison("email_comparison", col_name="email"),
        make_comparison(
            "jaro_winkler_at_thresholds", col_name="address",
            score_threshold_or_thresholds=[0.85, 0.75, 0.65],
        ),
    ]


def generate_people(n: int, missing_rate: float = 0.3, seed: int = 42) -> list[dict]:
    """Synthetic Canadian-style person population (shared with the in-memory
    benchmark, so the same references/queries benchmark both backends)."""
    rng = random.Random(seed)
    people = []
    for _ in range(n):
        first = rng.choice(FIRST_NAMES)
        last = rng.choice(LAST_NAMES)
        year = rng.randint(1945, 2005)
        month = rng.randint(1, 12)
        day = rng.randint(1, 28)
        dob = f"{year}-{month:02d}-{day:02d}"
        email = (
            f"{first}.{last}{rng.randint(1, 9999)}@example.com"
            if rng.random() >= missing_rate else None
        )
        address = (
            f"{rng.randint(1, 9999)} {rng.choice(STREET_KINDS)} "
            f"{rng.choice(['Main', 'Oak', 'Maple', 'Queen', 'King', 'Elm'])} "
            f"{rng.choice(CITIES)}, ON"
            if rng.random() >= missing_rate else None
        )
        people.append({
            "first_name": first,
            "last_name": last,
            "date_of_birth": dob,
            "email": email,
            "address": address,
        })
    return people


def introduce_variations(person: dict, variation_rate: float = 0.1) -> dict:
    """Create a slightly modified version of a person (same logic as the
    in-memory benchmark)."""
    new_person = dict(person)
    if random.random() < variation_rate and len(new_person["first_name"]) > 2:
        name = list(new_person["first_name"])
        idx = random.randint(0, len(name) - 1)
        name[idx] = random.choice("abcdefghijklmnopqrstuvwxyz")
        new_person["first_name"] = "".join(name)
    if random.random() < variation_rate and len(new_person["last_name"]) > 2:
        name = list(new_person["last_name"])
        idx = random.randint(0, len(name) - 1)
        name[idx] = random.choice("abcdefghijklmnopqrstuvwxyz")
        new_person["last_name"] = "".join(name)
    if random.random() < variation_rate and new_person["address"]:
        addr = new_person["address"]
        replacements = [
            ("St ", "Street "), ("St.", "Street"), ("Ave ", "Avenue "),
            ("Ave.", "Avenue"), ("Rd ", "Road "), ("Rd.", "Road"),
            ("Blvd ", "Boulevard "), ("Blvd.", "Boulevard"),
        ]
        for old, new in replacements:
            if old in addr and random.random() < 0.5:
                addr = addr.replace(old, new)
        new_person["address"] = addr
    if random.random() < variation_rate and new_person["email"]:
        email = new_person["email"]
        if random.random() < 0.5:
            local, domain = email.split("@")
            email = f"{local}{random.randint(1, 99)}@{domain}"
        new_person["email"] = email
    return new_person


# ---------------------------------------------------------------------------
# Qdrant-backed ingest + benchmark
# ---------------------------------------------------------------------------


def build_qdrant_database(
    records: Sequence[dict],
    embedder,
    collection: str,
    url: str,
    recreate: bool,
    batch_size: int = 256,
) -> tuple[QdrantVectorDatabase, dict[str, Any]]:
    """Create (or recreate/reuse) the collection, ingest the population, and
    time the two ingest stages separately (embedding vs upsert round-trips)."""
    from qdrant_client import QdrantClient
    from qdrant_client.http import models

    client = QdrantClient(url=url)
    try:
        client.get_collection(collection_name=collection)
        exists = True
    except Exception:
        exists = False

    if exists and recreate:
        print(f"Dropping existing collection {collection!r} ...")
        client.delete_collection(collection_name=collection)
        exists = False

    timing: dict[str, Any] = {}

    t0 = time.perf_counter()
    db = QdrantVectorDatabase(
        embedder=embedder,
        client=client,
        collection=collection,
        vector_size=int(embedder.dimension or 0),
    )
    timing["collection_new"] = not exists
    timing["setup_seconds"] = time.perf_counter() - t0

    if not exists or len(db) < len(records):
        # Embed first (one batched pass), then upsert in batches: the two costs
        # scale very differently for a remote store.
        t0 = time.perf_counter()
        from vectorer.records import to_record_dict

        vectors: list[list[float]] = []
        for i in range(0, len(records) + 1, batch_size):
            chunk = records[i:i + batch_size]
            if not chunk:
                continue
            vectors.extend(embedder.embed_many([
                "\n".join(f"{k}: {v}" for k, v in to_record_dict(r).items() if v is not None)
                for r in chunk
            ]))
        timing["embed_seconds"] = time.perf_counter() - t0
        timing["embed_per_record_ms"] = (
            (timing["embed_seconds"] / max(len(records), 1)) * 1000
        )

        start = len(db)
        t0 = time.perf_counter()
        points = []
        for i, (record, vec) in enumerate(zip(records, vectors)):
            rec = to_record_dict(record)
            points.append(models.PointStruct(
                id=start + i, vector=vec, payload={"record": rec},
            ))
            if len(points) >= batch_size:
                client.upsert(collection_name=collection, points=points)
                points = []
        if points:
            client.upsert(collection_name=collection, points=points)
        timing["upsert_seconds"] = time.perf_counter() - t0
        timing["upsert_per_record_ms"] = (
            (timing["upsert_seconds"] / max(len(records), 1)) * 1000
        )
        # The manual upsert bypassed db.add, so keep the adapter's count cache
        # in sync (else blocking's min(k, len(db)) would see 0 and short-circuit).
        db._len_cache = start + len(records)
    else:
        timing["reused"] = True

    return db, timing


def measure(pipeline: IncrementalPipeline, queries: Sequence[dict], breakdown: bool) -> dict[str, Any]:
    """Cold per-query latency stats; optional phase breakdown (embed / Qdrant
    block / scorer)."""
    totals: list[float] = []
    block_times: list[float] = []
    scorer_times: list[float] = []
    embed_times: list[float] = []
    db = pipeline.vector_database

    for person in tqdm(queries, desc="resolving queries", unit="query"):
        if breakdown:
            te = time.perf_counter()
            vector = db.embedding.embed(pipeline.serialize(person))
            embed_times.append((time.perf_counter() - te) * 1000)

            tb = time.perf_counter()
            candidates = pipeline.block(person, k=pipeline.k)  # embed + Qdrant ANN
            block_times.append((time.perf_counter() - tb) * 1000)

            ts = time.perf_counter()
            pipeline.scorer.score_batch(person, [c.record for c in candidates])
            scorer_times.append((time.perf_counter() - ts) * 1000)

        t0 = time.perf_counter()
        pipeline.resolve(person)
        totals.append((time.perf_counter() - t0) * 1000)

    stats = {
        "count": len(totals),
        "mean_ms": statistics.mean(totals) if totals else 0.0,
        "median_ms": percentile(totals, 0.50),
        "p50_ms": percentile(totals, 0.50),
        "p75_ms": percentile(totals, 0.75),
        "p90_ms": percentile(totals, 0.90),
        "p95_ms": percentile(totals, 0.95),
        "p99_ms": percentile(totals, 0.99),
        "min_ms": min(totals) if totals else 0.0,
        "max_ms": max(totals) if totals else 0.0,
        "stdev_ms": statistics.stdev(totals) if len(totals) > 1 else 0.0,
        "scope": (
            "cold per-query IncrementalPipeline.resolve over a live Qdrant store: "
            "embedding + remote top-k ANN blocking + native vectorized "
            "Fellegi-Sunter scoring (no SQL engine)"
        ),
    }
    if breakdown:
        stats["embedding_mean_ms"] = statistics.mean(embed_times) if embed_times else 0.0
        stats["blocking_mean_ms"] = statistics.mean(block_times) if block_times else 0.0
        stats["scorer_mean_ms"] = statistics.mean(scorer_times) if scorer_times else 0.0
        stats["breakdown_scope"] = (
            "independently timed: embedding, block (embed + Qdrant search), "
            "scorer.score_batch (vectorized Fellegi-Sunter scoring)"
        )
    return stats


def blocking_quality(pipeline: IncrementalPipeline, base: Sequence[dict], queries: Sequence[dict]) -> dict[str, Any]:
    """Ground-truth quality of the same cold path: top-k recall + match rate."""
    top_k_recall = 0
    matches = 0
    for position, (person, query) in enumerate(tqdm(zip(base, queries), desc="quality check", unit="query",
                                                    total=len(queries))):
        candidates = pipeline.block(query, k=pipeline.k)
        positions = [c.position for c in candidates]
        if position in positions:
            top_k_recall += 1
        result = pipeline.resolve(query)
        if result.matches:
            matches += 1
    total = len(queries)
    return {
        "top_k_blocking_recall": round(top_k_recall / total, 4) if total else 0.0,
        "match_rate_at_tau": round(matches / total, 4) if total else 0.0,
    }


def environment_block() -> dict[str, Any]:
    """Record the interpreter and key library versions plus CPU/thread hints."""
    libs = ["numpy", "faiss", "sentence_transformers", "qdrant_client", "grpc"]
    versions: dict[str, str] = {}
    for name in libs:
        try:
            mod = importlib.import_module(name)
            versions[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            pass
    try:
        import importlib.metadata as md

        versions["qdrant_client"] = md.version("qdrant-client")
    except Exception:
        pass
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "threads": os.cpu_count(),
        "libs": versions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure per-query latency of the incremental ER pipeline against a live Qdrant store"
    )
    parser.add_argument("--url", default=DEFAULT_QDRANT_URL,
                        help=f"Qdrant server URL (default {DEFAULT_QDRANT_URL})")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION,
                        help=f"Qdrant collection name (default {DEFAULT_COLLECTION})")
    parser.add_argument("--recreate", action="store_true",
                        help="drop the collection and rebuild it (default: reuse when fully populated)")
    parser.add_argument("--n-references", type=int, default=DEFAULT_REFERENCE_COUNT,
                        help="reference population size")
    parser.add_argument("--query-count", type=int, default=DEFAULT_QUERY_COUNT)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--k", dest="blocking_k", type=int, default=DEFAULT_BLOCKING_K)
    parser.add_argument("--missing-rate", type=float, default=DEFAULT_MISSING_RATE)
    parser.add_argument("--close-variation-rate", type=float, default=DEFAULT_CLOSE_VARIATION_RATE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embedder", choices=["hashing", "sentence"], default="hashing",
                        help="'sentence' uses sentence-transformers + MiniLM; 'hashing' is deterministic")
    parser.add_argument("--breakdown", action="store_true",
                        help="also record embedding / Qdrant blocking / scorer phase times")
    parser.add_argument("--output", default="results/incremental_qdrant_latency.json")
    args = parser.parse_args()

    random.seed(args.seed)
    if args.embedder == "sentence":
        from vectorer.pins import EMBEDDING_MODEL_ID, EMBEDDING_MODEL_REVISION
        from vectorer.embeddings import SentenceTransformerEmbedding

        embedder = SentenceTransformerEmbedding(EMBEDDING_MODEL_ID, EMBEDDING_MODEL_REVISION)
    else:
        embedder = CharacterHashingEmbedding(dimension=384)

    print(f"Generating {args.n_references:,} reference records (missing-rate {args.missing_rate})...")
    records = generate_people(args.n_references, missing_rate=args.missing_rate, seed=args.seed)

    print(f"Ingesting {len(records):,} records into Qdrant @ {args.url} "
          f"(collection {args.collection!r}, {args.embedder} embedder)...")
    database, timing = build_qdrant_database(
        records, embedder, collection=args.collection, url=args.url, recreate=args.recreate,
    )
    print(f"Ingest: {json.dumps(timing, indent=2)}")

    scorer = FellegiSunterScorer.from_comparisons(make_comparisons(), threshold=args.threshold)
    pipeline = IncrementalPipeline(vector_database=database, scorer=scorer,
                                   k=args.blocking_k, tau=args.threshold)

    base = records
    n = min(args.query_count, len(base))
    queries = [
        introduce_variations(base[i % len(base)], variation_rate=args.close_variation_rate)
        for i in range(n)
    ]
    print(f"Resolving {len(queries):,} close-variant queries (k={args.blocking_k}, "
          f"tau={args.threshold}, embedder={args.embedder})...")

    stats = measure(pipeline, queries, args.breakdown)
    quality = blocking_quality(pipeline, base, queries)

    results = {
        "parameters": {
            "reference_records": len(records),
            "qdrant_url": args.url,
            "collection": args.collection,
            "recreate": args.recreate,
            "query_count": len(queries),
            "match_threshold": args.threshold,
            "blocking_k": args.blocking_k,
            "missing_rate": args.missing_rate,
            "seed": args.seed,
            "close_variation_rate": args.close_variation_rate,
            "embedder": args.embedder,
        },
        "ingest": timing,
        "latency": stats,
        "quality": quality,
        "environment": environment_block(),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print(f"Quality: top-k blocking recall={quality['top_k_blocking_recall']}, "
          f"match rate at tau={quality['match_rate_at_tau']}")
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()