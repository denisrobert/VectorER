"""Benchmark the bulk ER pipeline on a **simulated 2-node Ray cluster**.

Single host, two Ray "nodes" for benchmarking the multi-node path without
infrastructure.  Ray's local mode runs all actors on one machine but the
execution path (pair-hash-owned streaming score + multi-node closure reduce) is
exactly the one a real cluster would use; pass a real ``--ray-address`` to
target actual machines instead.

The benchmark compares, on the same dataset / canopies / scorer:

* **single-process**: ``BatchPipeline.run`` (the classic, in-process path);
* **2-node (simulated)**: ``distributed_score_and_reduce`` with a Ray executor
  whose pair-scoring map and component reduce run as two actors (the "nodes").

Metrics:

* per-stage wall seconds and total for both runs,
* candidate-pairs/sec for the expensive FS scoring stage,
* cluster / duplicate-recovery quality (identical for both, shown once),
* ``--verify`` asserts the distributed cluster assignment equals the
  single-process one.

Example::

    python benchmarks/benchmark_bulk_er_multinode.py --n-records 8000 \\
        --dup-rate 0.04 --n-workers 2 --ray-address auto \\
        --output results/bulk_latency_multinode.json
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
from typing import Any, Sequence

from vectorer.batch import BatchPipeline
from vectorer.comparisons import make_comparison
from vectorer.distributed import create_executor, distributed_score_and_reduce
from vectorer.embeddings import CharacterHashingEmbedding
from vectorer.scoring import FellegiSunterScorer

FIRST_NAMES = [
    "john", "mary", "robert", "susan", "james", "linda", "michael", "patricia",
    "david", "jennifer", "william", "elizabeth", "richard", "barbara", "joseph",
]
LAST_NAMES = [
    "smith", "jones", "martinez", "brown", "wilson", "davis", "garcia", "miller",
    "lee", "taylor", "anderson", "thomas", "moore", "jackson",
]


def generate_dataset(n_base: int, dup_rate: float, seed: int = 42):
    rng = random.Random(seed)
    records = []
    dup_positions = []
    for i in range(n_base):
        first = rng.choice(FIRST_NAMES)
        last = rng.choice(LAST_NAMES)
        records.append({
            "first_name": first,
            "last_name": last,
            "date_of_birth": f"{rng.randint(1930, 2010)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
            "email": f"{first}.{last}{i}@mail.com",
        })
    for position in rng.sample(range(n_base), int(round(n_base * dup_rate))):
        twin = dict(records[position])
        if rng.random() < 0.4:
            twin["last_name"] += "e"
        dup_positions.append((len(records), position))
        records.append(twin)
    return records, dup_positions


def make_comparisons():
    return [
        make_comparison("jaro_winkler_at_thresholds", col_name="first_name",
                        score_threshold_or_thresholds=[0.9, 0.8, 0.7]),
        make_comparison("jaro_winkler_at_thresholds", col_name="last_name",
                        score_threshold_or_thresholds=[0.9, 0.8, 0.7]),
        make_comparison("date_of_birth_comparison", col_name="date_of_birth"),
        make_comparison("email_comparison", col_name="email"),
    ]


def build_pairs(records, n_canopies: int, overlap_m: int, seed: int):
    import numpy as np

    from vectorer.blocking import assign_canopies, train_canopy_centroids
    from vectorer.records import to_record_dict

    embedder = CharacterHashingEmbedding(384)
    vecs = np.asarray(embedder.embed_many([
        "\n".join(f"{k}: {v}" for k, v in to_record_dict(r).items() if v is not None)
        for r in records
    ]), dtype="float32")
    centroids = train_canopy_centroids(vecs, n_canopies, seed=seed, sample_size=None)
    canopy = assign_canopies(vecs, centroids, overlap_m)
    seen = set()
    pairs = []
    for (i, j) in canopy.candidate_pairs():
        key = (i, j) if i <= j else (j, i)
        if key not in seen:
            seen.add(key)
            pairs.append(key)
    return pairs


def _quality(assignment, records, dup_positions):
    n_clusters = len(assignment.clusters)
    singletons = sum(1 for c in assignment.clusters.values() if len(c.member_positions) == 1)
    twins_merged = sum(
        1 for twin_idx, base_idx in dup_positions
        if assignment.node_cluster[twin_idx] == assignment.node_cluster[base_idx]
    )
    return {
        "records": len(records),
        "clusters": n_clusters,
        "singletons": singletons,
        "merge_rate": round((len(records) - singletons) / len(records), 4),
        "twins_merged": twins_merged,
        "n_duplicate_twins": len(dup_positions),
    }


def environment_block() -> dict:
    versions = {}
    for name in ["numpy", "faiss", "ray"]:
        try:
            versions[name] = __import__(name).__version__
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
        description="Bulk ER on a simulated 2-node Ray cluster"
    )
    parser.add_argument("--n-records", type=int, default=8000)
    parser.add_argument("--dup-rate", type=float, default=0.04)
    parser.add_argument("--n-canopies", type=int, default=32)
    parser.add_argument("--overlap", type=int, default=2)
    parser.add_argument("--tau", type=float, default=0.85)
    parser.add_argument("--n-workers", type=int, default=2,
                        help="number of Ray actors simulating the nodes")
    parser.add_argument("--ray-address", default="auto",
                        help="'auto' = simulate on this host; ip:port = real cluster")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verify", action="store_true",
                        help="assert distributed == single-process assignment")
    parser.add_argument("--output", default="results/bulk_latency_multinode.json")
    args = parser.parse_args()

    records, dup_positions = generate_dataset(args.n_records, args.dup_rate, args.seed)
    scorer = FellegiSunterScorer.from_comparisons(make_comparisons(), threshold=args.tau)
    print(f"dataset: {len(records):,} records, {len(dup_positions)} duplicate twins")

    pairs = build_pairs(records, args.n_canopies, args.overlap, args.seed)
    print(f"candidate pairs: {len(pairs):,}")
    left = [records[i] for i, j in pairs]
    right = [records[j] for i, j in pairs]

    # ---- single-process run ------------------------------------------------
    pipeline = BatchPipeline(
        embedder=CharacterHashingEmbedding(384), scorer=scorer,
        n_canopies=args.n_canopies, overlap_m=args.overlap, canopy_seed=args.seed,
        tau=args.tau,
    )
    t0 = time.perf_counter()
    single = pipeline.run(records)
    t_single = time.perf_counter() - t0
    print(f"\nsingle-process: {t_single:.2f}s total")

    # ---- simulated 2-node run ---------------------------------------------
    print(f"running distributed with {args.n_workers} Ray worker(s) "
          f"(address={args.ray_address!r}, single host simulation)...")
    executor = create_executor("ray", n_workers=args.n_workers, address=args.ray_address)
    t0 = time.perf_counter()
    assignment = distributed_score_and_reduce(
        scorer, left, right,
        pair_positions=pairs,
        tau=args.tau, n=len(records),
        n_workers=args.n_workers, executor=executor,
    )
    t_dist = time.perf_counter() - t0
    print(f"distributed ({args.n_workers} node-sim): {t_dist:.2f}s total")

    if args.verify:
        identical = single.assignment.node_cluster == assignment.node_cluster
        print(f"distributed == single-process: {identical}")
        if not identical:
            raise SystemExit("ERROR: distributed differs from single-process!")
        print("OK: identical cluster assignment")

    quality = _quality(assignment, records, dup_positions)
    print(f"clusters: {quality['clusters']}, "
          f"twins merged: {quality['twins_merged']}/{quality['n_duplicate_twins']}")

    results = {
        "parameters": {
            "total_records": len(records),
            "duplicate_pairs_planted": len(dup_positions),
            "n_canopies": args.n_canopies,
            "overlap": args.overlap,
            "tau": args.tau,
            "n_workers": args.n_workers,
            "ray_address": args.ray_address,
            "seed": args.seed,
            "mode": "single-host simulation of 2-node Ray cluster",
        },
        "candidate_pairs": len(pairs),
        "single_process_seconds": round(t_single, 4),
        "distributed_seconds": round(t_dist, 4),
        "speedup": round(t_single / t_dist, 3),
        "candidate_pairs_per_second":
            round(len(pairs) / max(t_dist, 1e-9), 1),
        "quality": quality,
        "distributed_equals_single": bool(args.verify and
                                           single.assignment.node_cluster == assignment.node_cluster),
        "environment": environment_block(),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()