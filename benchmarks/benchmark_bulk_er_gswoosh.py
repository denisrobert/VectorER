"""Benchmark the bulk entity-resolution pipeline using **G-Swoosh** clustering.

The R-Swoosh-style path (``BatchPipeline.run``, transitive closure over pre-
scored canopy pairs) is the default bulk clustering: it scores every canopy
candidate pair once with Fellegi-Sunter and unions the above-``tau`` pairs,
never re-scoring after a merge.  G-Swoosh (``gswoosh`` / ``SwooshClusterer.
cluster_with_merger``) additionally **re-tests merged representatives** against
the candidate pair set until a full pass produces no merges — the only correct
algorithm when the match/merge functions are not ICAR, but much more expensive.

This script measures that extra cost and the resulting cluster quality on the
*same* input as ``benchmark_bulk_er.py`` (same synthetic dataset, canopy
parameters, and comparison set), so the two algorithms can be compared:

* the canopy candidate pairs are scored once with FS (identical to the R path);
* ``gswoosh`` serves already-scored pairs from a cache and **only re-scores**
  pairs whose representatives changed after a merge (the genuinely new work);
* ``n_pairs_evaluated`` and ``n_re_scored`` quantify how much match-testing
  G-Swoosh performs over and above the one-shot scoring;
* ``--swoosh-only`` limits the timing to the G-Swoosh loop alone.

Compare with the R-Swoosh artifacts:

    python benchmarks/benchmark_bulk_er_gswoosh.py --n-records 10000 \\
        --compare results/bulk_latency.json --output results/bulk_latency_gswoosh.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any, Optional, Sequence

from vectorer.comparisons import make_comparison
from vectorer.scoring import FellegiSunterScorer
from vectorer.embeddings import CharacterHashingEmbedding
from vectorer.clustering import gswoosh, select_representative, union_merge
from vectorer.batch import BatchPipeline

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
    """Identical to benchmark_bulk_er.generate_dataset (same data as the R runs)."""
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
        make_comparison("jaro_winkler_at_thresholds", col_name="first_name",
                        score_threshold_or_thresholds=[0.9, 0.8, 0.7]),
        make_comparison("jaro_winkler_at_thresholds", col_name="last_name",
                        score_threshold_or_thresholds=[0.9, 0.8, 0.7]),
        make_comparison("date_of_birth_comparison", col_name="date_of_birth"),
        make_comparison("email_comparison", col_name="email"),
        make_comparison("jaro_winkler_at_thresholds", col_name="address",
                        score_threshold_or_thresholds=[0.85, 0.75, 0.65]),
    ]


def environment_block() -> dict[str, Any]:
    versions = {}
    for name in ["numpy", "faiss"]:
        try:
            mod = __import__(name)
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


def cluster_quality(
    assignment, n_records: int, twin_entities: dict[int, int],
    n_candidate_pairs: int,
) -> dict:
    """Same quality metrics as benchmark_bulk_er.cluster_quality."""
    tp = fp = fn = 0
    for twin_position, base_position in twin_entities.items():
        same = assignment.node_cluster[twin_position] == assignment.node_cluster[base_position]
        if same:
            tp += 1
        else:
            fn += 1
    n_twins = len(twin_entities)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    non_singleton_members = sum(
        len(c.member_positions)
        for c in assignment.clusters.values()
        if len(c.member_positions) > 1
    )
    largest = max(assignment.clusters.values(), key=lambda c: len(c.member_positions), default=None)
    return {
        "total_records": n_records,
        "duplicate_pairs_planted": n_twins,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "merge_rate": round(non_singleton_members / n_records, 4),
        "n_clusters": len(assignment.clusters),
        "n_non_singletons": sum(1 for c in assignment.clusters.values() if len(c.member_positions) > 1),
        "largest_cluster": (len(largest.member_positions) if largest else 0),
        "n_candidate_pairs": n_candidate_pairs,
        "n_pairs_evaluated": assignment.n_pairs_evaluated,
        "n_pairs_matched": assignment.n_pairs_matched,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bulk ER benchmark with G-Swoosh (re-matching) clustering"
    )
    parser.add_argument("--n-records", type=int, default=DEFAULT_N_RECORDS)
    parser.add_argument("--dup-rate", type=float, default=DEFAULT_DUP_RATE)
    parser.add_argument("--missing-rate", type=float, default=DEFAULT_MISSING_RATE)
    parser.add_argument("--n-canopies", type=int, default=DEFAULT_N_CANOPIES)
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--merge", choices=["rep", "union"], default="rep",
                        help="Swoosh merge function: 'rep' = select_representative "
                             "(default, a real member record); 'union' = union_merge "
                             "(synthetic master record with set-valued fields, scored "
                             "through the scorer's Union-Class existence lift)")
    parser.add_argument("--swoosh-only", action="store_true",
                        help="timing covers only the G-Swoosh loop; the canopy+FS setup is run untimed")
    parser.add_argument("--compare", default=None,
                        help="path to an R-Swoosh bulk artifact (benchmark_bulk_er output) to tabulate against")
    parser.add_argument("--output", default="results/bulk_latency_gswoosh.json")
    args = parser.parse_args()

    n_canopies = min(args.n_canopies, max(1, args.n_records // 39))
    embedder = CharacterHashingEmbedding(dimension=384)
    scorer = FellegiSunterScorer.from_comparisons(make_comparisons(), threshold=args.tau)

    records, twin_entities = generate_dataset(
        args.n_records, args.dup_rate, args.missing_rate, args.seed
    )
    pipeline = BatchPipeline(
        embedder=embedder,
        scorer=scorer,
        n_canopies=n_canopies,
        overlap_m=args.overlap,
        canopy_seed=args.seed,
        tau=args.tau,
    )

    timing: dict[str, float] = {}
    if args.swoosh_only:
        t0 = time.perf_counter()
        parsed = [dict(r) for r in records]
        vectors = pipeline.embed_all(parsed)
        canopy = pipeline.block(vectors)
        pairs = list(canopy.candidate_pairs())
        scored = pipeline.score(parsed, pairs)
        timing["setup_seconds"] = round(time.perf_counter() - t0, 4)
    else:
        t0 = time.perf_counter()
        parsed = [dict(r) for r in records]
        timing["parse"] = round(time.perf_counter() - t0, 4)
        t0 = time.perf_counter()
        vectors = pipeline.embed_all(parsed)
        timing["embed"] = round(time.perf_counter() - t0, 4)
        t0 = time.perf_counter()
        canopy = pipeline.block(vectors)
        timing["canopy"] = round(time.perf_counter() - t0, 4)
        pairs = list(canopy.candidate_pairs())
        t0 = time.perf_counter()
        scored = pipeline.score(parsed, pairs)
        timing["fellegi_sunter"] = round(time.perf_counter() - t0, 4)

    n_candidate_pairs = len(pairs)

    # Cache of one-shot FS scores so G-Swoosh only re-scores merged reps.
    pre_score: dict[tuple[int, int], float] = {}
    for p in scored:
        i, j = p.left_position, p.right_position
        pre_score[(i, j) if i <= j else (j, i)] = float(p.probability)

    # Map representative record objects -> original positions (select_rep keeps
    # real member records, so identity works).
    id_to_pos = {id(parsed[i]): i for i in range(len(parsed))}

    stats = {"n_cached": 0, "n_re_scored": 0, "n_distinct_records_tested": 0}

    def match_probability(left_rep, right_rep) -> float:
        li = id_to_pos.get(id(left_rep))
        ri = id_to_pos.get(id(right_rep))
        cached = None
        if li is not None and ri is not None:
            key = (li, ri) if li <= ri else (ri, li)
            cached = pre_score.get(key)
        if cached is not None:
            stats["n_cached"] += 1
            return cached
        # Synthetic union representatives are new dict objects (not in the
        # one-shot position cache), so they are always re-scored through the
        # scorer's Union-Class existence lift.
        stats["n_re_scored"] += 1
        stats["n_distinct_records_tested"] += 1
        return scorer.score(left_rep, right_rep)

    merge_fn = select_representative if args.merge == "rep" else union_merge

    t0 = time.perf_counter()
    assignment = gswoosh(
        records=parsed,
        pairs=pairs,
        match_probability=match_probability,
        tau=args.tau,
        merge=merge_fn,
    )
    timing["gswoosh_swoosh"] = round(time.perf_counter() - t0, 4)
    timing.setdefault("parse", 0.0)
    timing.setdefault("embed", 0.0)
    timing.setdefault("canopy", 0.0)
    timing.setdefault("fellegi_sunter", 0.0)
    total = timing.get("setup_seconds", 0) + sum(
        v for k, v in timing.items() if k in ("parse", "embed", "canopy", "fellegi_sunter", "gswoosh_swoosh")
    )
    timing["total_seconds"] = round(total, 4)

    quality = cluster_quality(assignment, len(parsed), twin_entities, n_candidate_pairs)
    quality["timing_seconds"] = {k: round(v, 4) for k, v in timing.items()}
    quality["total_seconds"] = timing["total_seconds"]
    quality["records_per_second"] = round(len(parsed) / max(total, 1e-9), 1)
    quality["fellegi_sunter_call_stats"] = stats
    quality["merge_function"] = args.merge
    quality["algorithm"] = "G-Swoosh"

    results = {
        "parameters": {
            "total_records": len(parsed),
            "duplicate_pairs_planted": len(twin_entities),
            "n_canopies": n_canopies,
            "overlap": args.overlap,
            "tau": args.tau,
            "missing_rate": args.missing_rate,
            "seed": args.seed,
            "embedder": "hashing-384",
            "algorithm": "G-Swoosh",
            "merge": args.merge,
            "swoosh_only": args.swoosh_only,
        },
        "quality": quality,
        "environment": environment_block(),
    }

    print(json.dumps(quality, indent=2))

    if args.compare:
        ref = json.load(open(args.compare, encoding="utf-8"))
        rq = ref["quality"]
        r_sw = rq["timing_seconds"].get("swoosh", rq.get("timing_seconds", {}).get("gswoosh_swoosh", 0))
        g_sw = quality["timing_seconds"].get("gswoosh_swoosh", 0)
        ref_merge = rq.get("merge_function") or ref.get("parameters", {}).get("merge", "?")
        rows = [
            ("reference_merge", ref_merge),
            ("this_merge", args.merge),
            ("swoosh_stage_seconds_reference", round(r_sw, 4)),
            ("swoosh_stage_seconds_g_swoosh", round(g_sw, 4)),
        ]
        match_same = (ref_merge == args.merge) or (ref.get("parameters", {}).get("merge") == args.merge)
        if match_same and r_sw:
            rows.append(("swoosh_slowdown_x", round(g_sw / max(r_sw, 1e-9), 2)))
        rows += [
            ("n_pairs_evaluated_reference", rq.get("n_pairs_evaluated", rq["n_candidate_pairs"])),
            ("n_pairs_evaluated_g_swoosh", assignment.n_pairs_evaluated),
            ("n_fs_rescores_g_swoosh", stats["n_re_scored"]),
            ("total_seconds_reference", rq.get("total_seconds")),
            ("total_seconds_g_swoosh", quality["total_seconds"]),
            ("n_clusters_reference", rq.get("n_clusters")),
            ("n_clusters_g_swoosh", quality["n_clusters"]),
            ("recall_reference", rq.get("recall")),
            ("recall_g_swoosh", quality["recall"]),
            ("precision_reference", rq.get("precision")),
            ("precision_g_swoosh", quality["precision"]),
        ]
        table = "\n".join(f"  {name:36s} {value}" for name, value in rows)
        print("----- G-Swoosh vs reference bulk artifact -----")
        print(table)
        results["comparison"] = {
            "reference_artifact": args.compare,
            **{name: value for name, value in rows},
        }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()