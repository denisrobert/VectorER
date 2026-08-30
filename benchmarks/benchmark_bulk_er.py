"""Measure the bulk (batch) entity-resolution pipeline.

Benchmarks the whole-dataset path::

    parse -> embed -> canopy blocking on the embedded dataset ->
    Fellegi-Sunter scoring of every canopy candidate pair -> Swoosh clustering

This emulates the original project's bulk experiments (whitepaper section 7:
bulk embedding + blocking + batch Fellegi-Sunter scoring) on the new native
stack.  Unlike the original's per-query FAISS + per-query candidate deck, the
batch pipeline clusters the whole dataset into overlapping canopies (FAISS
k-means, multi-assignment) and scores every canopy candidate pair in one
vectorized pass -- so the reported numbers are whole-dataset throughput and
stage costs, plus ground-truth duplicate recovery.

Metrics reported (JSON artifact):

* per-stage wall seconds (parse, embed, canopy, fellegi_sunter, swoosh) and the
  proportionate total;
* throughput: records/second, canopy candidate pairs/second, and Fellegi-Sunter
  pair-scoring throughput;
* quality: Swoosh cluster counts and duplicate-recovery precision/recall
  against the planted duplicate twins;
* (with ``--compare``) a table of the new scoring throughput against the
  original project's batch-scoring throughput.

Example::

    python benchmarks/benchmark_bulk_er.py --n-records 10000 --dup-rate 0.04 \\
        --n-canopies 256 --overlap 2 --output results/bulk_latency.json
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
from typing import Any, Sequence

from vectorer.batch import BatchPipeline
from vectorer.comparisons import make_comparison
from vectorer.embeddings import CharacterHashingEmbedding
from vectorer.scoring import FellegiSunterScorer

DEFAULT_N_RECORDS = 10000
DEFAULT_DUP_RATE = 0.04
DEFAULT_N_CANOPIES = 256
DEFAULT_OVERLAP = 2
DEFAULT_TAU = 0.85
DEFAULT_MISSING_RATE = 0.3

FIRST_NAMES = [
    "john", "mary", "robert", "susan", "james", "linda", "michael", "patricia",
    "david", "jennifer", "william", "elizabeth", "richard", "barbara", "joseph",
    "thomas", "sarah", "charles", "karen", "daniel", "nancy", "paul", "betty",
]
LAST_NAMES = [
    "smith", "jones", "martinez", "brown", "wilson", "davis", "garcia", "miller",
    "lee", "taylor", "anderson", "thomas", "moore", "jackson", "martin", "thompson",
    "white", "lopez", "hill", "scott", "green", "adams", "baker", "gonzalez",
]
STREET_KINDS = ["St", "Ave", "Rd", "Blvd", "St.", "Ave.", "Rd.", "Blvd."]
CITIES = ["Toronto", "Vancouver", "Montreal", "Calgary", "Ottawa", "Edmonton"]


def generate_dataset(n_base: int, dup_rate: float, missing_rate: float, seed: int = 42):
    """Deterministic synthetic people; returns records and duplicate-twin masks."""
    rng = random.Random(seed)
    base = []
    for i in range(n_base):
        first = rng.choice(FIRST_NAMES)
        last = rng.choice(LAST_NAMES)
        dob = f"{rng.randint(1930, 2010)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
        email = f"{first}.{last}{i}@example.com" if rng.random() >= missing_rate else None
        address = (
            f"{rng.randint(1, 9999)} {rng.choice(STREET_KINDS)} "
            f"{rng.choice(['Main', 'Oak', 'Maple', 'Queen', 'King', 'Elm'])} "
            f"{rng.choice(CITIES)}, ON"
            if rng.random() >= missing_rate else None
        )
        base.append({
            "first_name": first,
            "last_name": last,
            "date_of_birth": dob,
            "email": email,
            "address": address,
        })

    records = list(base)
    dup_positions = rng.sample(range(n_base), int(round(n_base * dup_rate)))
    twin_entities: dict[int, int] = {}
    for position in dup_positions:
        twin = dict(base[position])
        # ~40% noisy twins (single-char surname perturbation), rest exact.
        if rng.random() < 0.4 and len(twin["last_name"]) > 2:
            name = list(twin["last_name"])
            idx = rng.randint(0, len(name) - 1)
            name[idx] = rng.choice("abcdefghijklmnopqrstuvwxyz")
            twin["last_name"] = "".join(name)
        twin_position = len(records)
        records.append(twin)
        twin_entities[twin_position] = position
    return records, twin_entities


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


def build_embedder(kind: str):
    if kind == "sentence":
        from vectorer.pins import EMBEDDING_MODEL_ID, EMBEDDING_MODEL_REVISION
        from vectorer.embeddings import SentenceTransformerEmbedding

        return SentenceTransformerEmbedding(EMBEDDING_MODEL_ID, EMBEDDING_MODEL_REVISION)
    # 384-d mirrors the all-MiniLM-L6-v2 embedding dimensionality.
    return CharacterHashingEmbedding(dimension=384)


def cluster_quality(
    pipeline: BatchPipeline,
    records: Sequence[dict],
    twin_entities: dict[int, int],
    seed: int,
) -> dict[str, Any]:
    """Run the batch pipeline and evaluate duplicate recovery against ground truth."""
    result = pipeline.run(records)
    timing = result.timing

    tp = fp = fn = 0
    for twin_position, base_position in twin_entities.items():
        same = (
            result.cluster_of_position(twin_position)
            == result.cluster_of_position(base_position)
        )
        if same:
            tp += 1
        else:
            fn += 1
    n_twins = len(twin_entities)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    # MergeRate-like reduction: fraction of records in non-singleton clusters.
    non_singleton_members = sum(
        len(c.member_positions)
        for c in result.assignment.clusters.values()
        if len(c.member_positions) > 1
    )
    merge_rate = non_singleton_members / len(records)

    largest = max(
        result.assignment.clusters.values(),
        key=lambda c: len(c.member_positions),
        default=None,
    )
    return {
        "total_records": len(records),
        "duplicate_pairs_planted": n_twins,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "merge_rate": round(merge_rate, 4),
        "n_clusters": result.n_clusters,
        "n_non_singletons": result.n_non_singletons,
        "largest_cluster": (len(largest.member_positions) if largest else 0),
        "n_candidate_pairs": result.n_candidate_pairs,
        "timing_seconds": {k: round(v, 4) for k, v in timing.items()},
        "total_seconds": round(sum(timing.values()), 4),
        "records_per_second": round(len(records) / max(sum(timing.values()), 1e-9), 1),
        "candidate_pairs_per_second": round(
            result.n_candidate_pairs / max(timing.get("fellegi_sunter", 0), 1e-9), 1
        ),
        "fellegi_sunter_seconds_per_million_pairs": round(
            timing.get("fellegi_sunter", 0) * 1e6 / max(result.n_candidate_pairs, 1),
            3,
        ),
        "seed": seed,
    }


def environment_block() -> dict[str, Any]:
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the bulk (batch) entity-resolution pipeline"
    )
    parser.add_argument("--n-records", type=int, default=DEFAULT_N_RECORDS)
    parser.add_argument("--dup-rate", type=float, default=DEFAULT_DUP_RATE)
    parser.add_argument("--missing-rate", type=float, default=DEFAULT_MISSING_RATE)
    parser.add_argument("--n-canopies", type=int, default=DEFAULT_N_CANOPIES)
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP,
                        help="top-m canopy assignments per record (1 = hard partition)")
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embedder", choices=["hashing", "sentence"], default="hashing")
    parser.add_argument("--merge", choices=["rep", "union"], default="rep",
                        help="Swoosh merge function: 'rep' = select_representative "
                             "(default, a real member record); 'union' = union_merge "
                             "(synthetic master record with set-valued fields)")
    parser.add_argument("--compare", default=None,
                        help="path to original section7_results.json to tabulate scoring "
                             "throughput against")
    parser.add_argument("--output", default="results/bulk_latency.json")
    args = parser.parse_args()

    # FAISS k-means requires ~39x records >= clusters; clamp the canopy grid.
    n_canopies = min(args.n_canopies, max(1, args.n_records // 39))

    print(f"Generating {args.n_records:,} base records (dup-rate {args.dup_rate}, "
          f"missing-rate {args.missing_rate}, seed {args.seed})...")
    records, twin_entities = generate_dataset(
        args.n_records, args.dup_rate, args.missing_rate, args.seed
    )
    print(f"Dataset: {len(records):,} records, {len(twin_entities):,} duplicate twins")

    embedder = build_embedder(args.embedder)
    scorer = FellegiSunterScorer.from_comparisons(make_comparisons(), threshold=args.tau)
    from vectorer.clustering import union_merge
    merge_fn = union_merge if args.merge == "union" else None
    pipeline = BatchPipeline(
        embedder=embedder,
        scorer=scorer,
        n_canopies=n_canopies,
        overlap_m=args.overlap,
        canopy_seed=args.seed,
        tau=args.tau,
        merge=merge_fn,
    )
    print(f"Running batch pipeline (canopies={n_canopies}, overlap={args.overlap}, "
          f"tau={args.tau}, embedder={args.embedder}, merge={args.merge})...")

    t0 = time.perf_counter()
    quality = cluster_quality(pipeline, records, twin_entities, args.seed)
    quality["wall_seconds"] = round(time.perf_counter() - t0, 4)

    results = {
        "parameters": {
            "total_records": len(records),
            "duplicate_pairs_planted": len(twin_entities),
            "n_canopies": n_canopies,
            "overlap": args.overlap,
            "tau": args.tau,
            "missing_rate": args.missing_rate,
            "seed": args.seed,
            "embedder": args.embedder,
            "merge": args.merge,
        },
        "quality": quality,
        "environment": environment_block(),
    }

    print(json.dumps(quality, indent=2))

    if args.compare:
        reference = json.loads(Path(args.compare).read_text(encoding="utf-8"))
        # Original bulk numbers (previous project's artifact): per-query FAISS
        # latency + batch scoring totals.
        ref = reference.get("strategies", {})
        strategy = ref.get("default", {})
        ref_score_seconds = next(
            (v for k, v in strategy.items() if k.endswith("_batch_seconds")), None
        )
        ref_queries = len(strategy.get("threshold_metrics", {})) and strategy.get("latency_ms", {}).get("samples", 0)
        ref_pairs = ref_queries * strategy.get("scoring_k", 0)
        ref_pairs_per_second = ref_pairs / ref_score_seconds if ref_score_seconds and ref_pairs else None
        new_pairs_per_second = quality["candidate_pairs_per_second"]

        rows = [
            ("fellegi_sunter_seconds(original)",
             round(ref_score_seconds, 1) if ref_score_seconds else None),
            ("scored_pairs(original)", ref_pairs if ref_pairs else None),
            ("pairs_per_second(original)", round(ref_pairs_per_second, 1) if ref_pairs_per_second else None),
            ("pairs_per_second(new)", new_pairs_per_second),
        ]
        rows.append((
            "scoring_throughput_speedup",
            round(new_pairs_per_second / ref_pairs_per_second, 2)
            if ref_pairs_per_second and new_pairs_per_second else None,
        ))
        table = "\n".join(f"  {name:42s} {value}" for name, value in rows)
        print("----- scoring throughput vs original (whitepaper section 7) -----")
        print(table)
        results["comparison"] = {
            "reference_artifact": args.compare,
            "original_fellegi_sunter_seconds": rows[0][1],
            "original_scored_pairs": rows[1][1],
            "original_pairs_per_second": rows[2][1],
            "new_pairs_per_second": rows[3][1],
            "scoring_throughput_speedup": rows[4][1],
        }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()