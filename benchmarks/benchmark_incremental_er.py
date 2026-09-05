"""Measure per-query latency of the incremental (online) ER pipeline.

Emulates ``experiments/whitepaper/experiment_online_latency.py`` from the
original entity-resolution project on top of the new native stack
(``vectorer``): identical methodology, identical reporting.

The original project's headline latency figures were amortized-batch averages.
Its ``experiment_online_latency.py`` instead measured the *cold, per-query*
cost of the production online resolver path (embed query -> FAISS top-k
blocking -> Fellegi-Sunter scoring on the top-k -> classify) by calling
``resolve`` once per query and recording the actual per-query wall times.  It
reported mean / median / p50 / p75 / p90 / p95 / p99 / min / max / stdev, an
optional phase breakdown (embedding / FAISS blocking / scorer), and an
environment block recording interpreter and library versions.

This script reproduces that benchmark against ``vectorer.incremental``:

* a reference population of synthetic Canadian-style person records (missing
  address/email at ``--missing-rate``) is embedded and indexed once
  (optionally persisted to and reloaded from ``--index-dir``, mirroring the
  original's persisted FAISS store);
* queries are *close variants* of reference records (typos in forename /
  surname, address normalisation, email perturbations) produced by the same
  perturbation logic the original used;
* the cold ``IncrementalPipeline.resolve`` path is timed per query.

Example::

    python benchmarks/benchmark_incremental_er.py --n-references 20000 --query-count 100 \\
        --breakdown --output results/incremental_latency.json

    # optional: real sentence-transformers embedder (needs `pip install -e ".[embedding]"`)
    python benchmarks/benchmark_incremental_er.py --embedder sentence --query-count 50
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
from vectorer.vectorstores import FlatIndex, InMemoryVectorDatabase

DEFAULT_QUERY_COUNT = 100
DEFAULT_THRESHOLD = 0.85
DEFAULT_BLOCKING_K = 20
DEFAULT_MISSING_RATE = 0.3
DEFAULT_CLOSE_VARIATION_RATE = 0.15
DEFAULT_REFERENCE_COUNT = 20000

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


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    rank = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction))))
    return ordered[rank]


# ---------------------------------------------------------------------------
# Dataset + perturbation (mirrors the original's generate_data)
# ---------------------------------------------------------------------------


def generate_people(count: int, missing_rate: float = 0.3, seed: int = 42) -> list[dict]:
    """Deterministic synthetic Canadian person records (dicts)."""
    rng = random.Random(seed)
    people = []
    for i in range(count):
        first = rng.choice(FIRST_NAMES)
        last = rng.choice(LAST_NAMES)
        dob = f"{rng.randint(1930, 2010)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
        email = (
            f"{first}.{last}{i}@example.com"
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
    """Create a slightly modified version of a person (same logic as the original).

    Per-field with probability ``variation_rate``: a random character typo in the
    forename/surname, US/Canadian address normalisation (``St`` -> ``Street``
    etc.), and a ``local<digits>@domain`` email mutation.
    """
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
# Comparison set (mirrors the original's priority order)
# ---------------------------------------------------------------------------


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


def build_pipeline(
    records: Sequence[dict],
    embedder,
    k: int,
    threshold: float,
    index_dir: Optional[str],
) -> tuple[IncrementalPipeline, dict[str, Any]]:
    """Build (or reload) the reference store and return a cold incremental pipeline."""
    timing: dict[str, Any] = {}
    if index_dir is not None:
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)

    store_path = Path(index_dir) / "index.faiss" if index_dir else None
    if store_path is not None and store_path.exists():
        t0 = time.perf_counter()
        database = InMemoryVectorDatabase.load(index_dir, embedding=embedder)
        timing["index_load_seconds"] = time.perf_counter() - t0
    else:
        t0 = time.perf_counter()
        database = InMemoryVectorDatabase(embedder, FlatIndex(normalize=True))
        database.add(records)
        timing["index_build_seconds"] = time.perf_counter() - t0
        if index_dir is not None:
            database.save(index_dir)
            index_size = sum(
                f.stat().st_size for f in Path(index_dir).iterdir() if f.is_file()
            )
            timing["index_bytes"] = index_size

    scorer = FellegiSunterScorer.from_comparisons(make_comparisons(), threshold=threshold)
    pipeline = IncrementalPipeline(vector_database=database, scorer=scorer, k=k, tau=threshold)
    return pipeline, timing


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


def measure(pipeline: IncrementalPipeline, queries: Sequence[dict], breakdown: bool) -> dict[str, Any]:
    """Run ``resolve`` per query and return cold per-query latency statistics.

    Every query is a close variant of a reference record, so ``resolve`` runs
    its full cold path: embedding -> vector (FAISS) top-k blocking -> vectorized
    Fellegi-Sunter scoring on the top-k -> classification.  When ``breakdown``
    is set the three phases are additionally timed independently per query.
    """
    totals: list[float] = []
    block_times: list[float] = []
    scorer_times: list[float] = []
    embed_times: list[float] = []
    db = pipeline.vector_database

    for person in tqdm(queries, desc="resolving queries", unit="query"):
        if breakdown:
            te = time.perf_counter()
            vector = db.embedding.embed(pipeline._embed_text(person))
            embed_times.append((time.perf_counter() - te) * 1000)

            tb = time.perf_counter()
            candidates = pipeline.block(person, k=pipeline.k)  # embed + FAISS search
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
            "cold per-query end-to-end IncrementalPipeline.resolve: embedding + "
            "FAISS top-k blocking + native vectorized Fellegi-Sunter scoring "
            "(no SQL engine)"
        ),
    }
    if breakdown:
        stats["embedding_mean_ms"] = statistics.mean(embed_times) if embed_times else 0.0
        stats["blocking_mean_ms"] = statistics.mean(block_times) if block_times else 0.0
        stats["scorer_mean_ms"] = statistics.mean(scorer_times) if scorer_times else 0.0
        stats["breakdown_scope"] = (
            "independently timed: embedding, block (embed + FAISS search), "
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
    libs = ["numpy", "faiss", "sentence_transformers"]
    versions: dict[str, str] = {}
    for name in libs:
        try:
            mod = importlib.import_module(name)
            versions[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            pass
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "threads": os.cpu_count(),
        "libs": versions,
    }


def format_comparison(new_stats: dict[str, Any], old_stats: dict[str, Any]) -> str:
    """Side-by-side latency table (new stack vs a reference artifact).

    ``speedup = original / new`` so a value of ``2.3x`` reads "2.3x faster".
    """
    keys = [
        "mean_ms", "median_ms", "p50_ms", "p75_ms", "p90_ms", "p95_ms", "p99_ms",
        "min_ms", "max_ms", "stdev_ms",
    ]
    rows = []
    for key in keys:
        new = new_stats.get(key, 0.0)
        old = old_stats.get(key, 0.0)
        speedup = old / new if new else float("nan")
        rows.append(f"  {key:12s} {old:10.3f} {new:10.3f} {speedup:8.2f}x")
    header = (
        "  key           original    new-stack   speedup\n"
        "  ---------------------------------------------------"
    )
    return header + "\n" + "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure per-query latency of the incremental ER pipeline (native vectorer stack)"
    )
    parser.add_argument("--n-references", type=int, default=DEFAULT_REFERENCE_COUNT,
                        help="reference population size (original paper scale: 50000)")
    parser.add_argument("--query-count", type=int, default=DEFAULT_QUERY_COUNT)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--k", dest="blocking_k", type=int, default=DEFAULT_BLOCKING_K)
    parser.add_argument("--missing-rate", type=float, default=DEFAULT_MISSING_RATE)
    parser.add_argument("--close-variation-rate", type=float, default=DEFAULT_CLOSE_VARIATION_RATE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embedder", choices=["hashing", "sentence"], default="hashing",
                        help="'sentence' uses sentence-transformers + MiniLM; 'hashing' is deterministic")
    parser.add_argument("--index-dir", default=None,
                        help="persist the reference store here and reload it if present")
    parser.add_argument("--data-file", default=None,
                        help="optional real/prepared dataset (JSONL or JSON) to use as the "
                             "reference population instead of the synthetic generator; "
                             "expects the compared fields (first_name, last_name, "
                             "date_of_birth, email, address), None allowed")
    parser.add_argument("--data-key", default=None,
                        help="when --data-file is a single JSON object, the key holding "
                             "the records list")
    parser.add_argument("--breakdown", action="store_true",
                        help="also record embedding / FAISS blocking / scorer phase times")
    parser.add_argument("--compare", default=None,
                        help="path to an original-project latency artifact JSON (e.g. "
                             "results/erwhitepaper/online_resolver_latency.json) to tabulate "
                             "the new stack's latency against")
    parser.add_argument("--output", default="results/incremental_latency.json")
    args = parser.parse_args()

    random.seed(args.seed)
    if args.embedder == "sentence":
        from vectorer.pins import EMBEDDING_MODEL_ID, EMBEDDING_MODEL_REVISION
        from vectorer.embeddings import SentenceTransformerEmbedding

        embedder = SentenceTransformerEmbedding(EMBEDDING_MODEL_ID, EMBEDDING_MODEL_REVISION)
    else:
        # 384-d to mirror the vector dimensionality of all-MiniLM-L6-v2,
        # keeping the FAISS blocking stage dimensionally comparable.
        embedder = CharacterHashingEmbedding(dimension=384)

    from benchmark_data import load_records, require_compared_fields

    if args.data_file:
        print(f"Loading reference population from {args.data_file} ...")
        records = load_records(args.data_file, key=args.data_key)
        require_compared_fields(records, ["first_name", "last_name", "date_of_birth", "email", "address"])
        print(f"  {len(records):,} records loaded")
    else:
        print(f"Generating {args.n_references:,} reference records (missing-rate {args.missing_rate})...")
        records = generate_people(args.n_references, missing_rate=args.missing_rate, seed=args.seed)

    pipeline, timing = build_pipeline(
        records, embedder, k=args.blocking_k, threshold=args.threshold, index_dir=args.index_dir,
    )
    print(f"Index ready: {timing}")

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
            "data_file": args.data_file,
            "data_key": args.data_key,
            "index_dir": args.index_dir,
            "query_count": len(queries),
            "match_threshold": args.threshold,
            "blocking_k": args.blocking_k,
            "missing_rate": args.missing_rate,
            "seed": args.seed,
            "close_variation_rate": args.close_variation_rate,
            "embedder": args.embedder,
        },
        "index": timing,
        "latency": stats,
        "quality": quality,
        "environment": environment_block(),
    }
    if args.compare:
        reference = json.loads(Path(args.compare).read_text(encoding="utf-8"))
        old_stats = reference.get("latency", reference)
        comparison = format_comparison(stats, old_stats)
        results["comparison"] = {
            "reference_artifact": args.compare,
            "reference_parameters": reference.get("parameters", {}),
            "speedup_mean": round(old_stats["mean_ms"] / stats["mean_ms"], 3)
            if stats.get("mean_ms") else None,
            "speedup_median": round(old_stats["median_ms"] / stats["median_ms"], 3)
            if stats.get("median_ms") else None,
        }
        print(comparison)
        print(f"  Speedup: mean {results['comparison']['speedup_mean']}x, "
              f"median {results['comparison']['speedup_median']}x faster")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print(f"Quality: top-k blocking recall={quality['top_k_blocking_recall']}, "
          f"match rate at tau={quality['match_rate_at_tau']}")
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()