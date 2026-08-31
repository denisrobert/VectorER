"""Example: distributed batch entity resolution.

The same whole-dataset dedup pipeline as examples/batch_er.py, but executed
across multiple workers via vectorer.distributed.distributed_batch_er.  The
cluster assignment is identical to the single-process BatchPipeline.run; the
FS scoring stage is the one parallelized.

Run from the project root:

    python examples/distributed_batch_er.py [--n-base 500] [--dup-rate 0.04]
        [--tau 0.85] [--n-canopies 64] [--overlap 2] [--n-workers 4]
        [--use-threads] [--verify]

With ``--verify`` the example also runs the single-process pipeline on the same
data and asserts that the distributed cluster assignment is byte-for-byte
identical (proving the executor reproduces the local result).
"""

from __future__ import annotations

import argparse
import random

from vectorer.batch import build_batch_pipeline
from vectorer.comparisons import make_comparison
from vectorer.distributed import distributed_batch_er
from vectorer.embeddings import CharacterHashingEmbedding
from vectorer.scoring import FellegiSunterScorer

FIRST_NAMES = [
    "john", "mary", "robert", "susan", "james", "linda", "michael", "patricia",
    "david", "jennifer", "william", "elizabeth", "richard", "barbara", "joseph",
]
LAST_NAMES = [
    "smith", "jones", "martinez", "brown", "wilson", "davis", "garcia",
    "miller", "lee", "taylor", "anderson", "thomas", "moore", "jackson",
]


def generate_dataset(n_base: int, dup_rate: float, seed: int = 42) -> tuple[list[dict], list[int]]:
    rng = random.Random(seed)
    base = []
    for i in range(n_base):
        first = rng.choice(FIRST_NAMES)
        last = rng.choice(LAST_NAMES)
        dob = f"{rng.randint(1930, 2010)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
        base.append({
            "first_name": first,
            "last_name": last,
            "date_of_birth": dob,
            "email": f"{first}.{last}{i}@mail.com",
            "address": None,
        })
    records = list(base)
    dup_positions = rng.sample(range(n_base), int(round(n_base * dup_rate)))
    for position in dup_positions:
        twin = dict(base[position])
        if rng.random() < 0.4:
            twin["last_name"] = twin["last_name"] + "e"
        records.append(twin)
    return records, dup_positions


def make_comparisons() -> list:
    return [
        make_comparison("jaro_winkler_at_thresholds", col_name="first_name",
                        score_threshold_or_thresholds=[0.9, 0.8, 0.7]),
        make_comparison("jaro_winkler_at_thresholds", col_name="last_name",
                        score_threshold_or_thresholds=[0.9, 0.8, 0.7]),
        make_comparison("date_of_birth_comparison", col_name="date_of_birth"),
        make_comparison("email_comparison", col_name="email"),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Distributed batch ER example")
    parser.add_argument("--n-base", type=int, default=500)
    parser.add_argument("--dup-rate", type=float, default=0.04)
    parser.add_argument("--tau", type=float, default=0.85)
    parser.add_argument("--n-canopies", type=int, default=64)
    parser.add_argument("--overlap", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-workers", type=int, default=4,
                        help="number of worker processes/threads")
    parser.add_argument("--use-threads", action="store_true",
                        help="use a thread pool instead of a process pool")
    parser.add_argument("--verify", action="store_true",
                        help="also run single-process and assert identical clusters")
    args = parser.parse_args()

    records, dup_positions = generate_dataset(args.n_base, args.dup_rate, args.seed)
    n_dup = len(dup_positions)
    print(f"Dataset: {len(records):,} records ({n_dup} duplicate twins)")

    scorer = FellegiSunterScorer.from_comparisons(make_comparisons(), threshold=args.tau)

    print(f"Running distributed batch pipeline with {args.n_workers} "
          f"{'threads' if args.use_threads else 'processes'} "
          f"(canopies={args.n_canopies}, overlap={args.overlap}, tau={args.tau})...")
    assignment = distributed_batch_er(
        records,
        scorer=scorer,
        n_canopies=args.n_canopies,
        overlap_m=args.overlap,
        tau=args.tau,
        seed=args.seed,
        n_workers=args.n_workers,
        use_threads=args.use_threads,
    )

    from collections import Counter

    cluster_sizes = Counter(assignment.node_cluster.values())
    n_clusters = len(cluster_sizes)
    n_singletons = sum(1 for size in cluster_sizes.values() if size == 1)
    n_merged = n_clusters - n_singletons
    print(f"Distributed Swoosh clusters: {n_clusters} "
          f"({n_merged} merged, {n_singletons} singletons)")

    # Ground-truth duplicate-recovery checks.
    twins_merged = 0
    not_in_cluster = 0
    for twin_position in range(args.n_base, args.n_base + n_dup):
        twin_base = dup_positions[twin_position - args.n_base]
        c_twin = assignment.node_cluster[twin_position]
        c_base = assignment.node_cluster[twin_base]
        if c_twin == c_base:
            twins_merged += 1
        else:
            not_in_cluster += 1
    print(f"Duplicate recovery: {twins_merged}/{n_dup} twins merged "
          f"({twins_merged / n_dup:.1%})")

    if args.verify:
        print("Verifying against the single-process pipeline...")
        pipeline = build_batch_pipeline(
            embedder=CharacterHashingEmbedding(dimension=384),
            scorer=scorer,
            n_canopies=args.n_canopies,
            overlap_m=args.overlap,
            canopy_seed=args.seed,
            tau=args.tau,
        )
        single = pipeline.run(records).assignment.node_cluster
        identical = single == assignment.node_cluster
        print(f"  single-process clusters: {len(set(single.values()))}")
        print(f"  distributed == single-process: {identical}")
        if not identical:
            raise SystemExit("ERROR: distributed result differs from single-process!")
        else:
            print("  OK: cluster assignment identical")


if __name__ == "__main__":
    main()