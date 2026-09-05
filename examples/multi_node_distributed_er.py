"""Example: multi-node distributed batch ER.

Demonstrates running the batch pipeline's expensive stages across **multiple
machines** via Ray.  The framework's distributed layer
(``vectorer.distributed``) is backend-agnostic: the same orchestration that
spawns process-pool workers on one host drives Ray actors on a cluster.

Setup
-----
1. Install the optional Ray backend once on every node:

       pip install -e ".[benchmark]"      # tqdm (optional)
       pip install ray>=2.0

2. Start a Ray cluster.  On the head node:

       ray start --head --port=6379

   On each worker node:

       ray start --address=<head-ip>:6379

   (Ray ships its own coordination; the framework needs no changes.)

3. Run the example from any node, pointing at the head:

       python examples/multi_node_distributed_er.py --ray-address auto --n-base 2000 \
           --n-workers 4 --verify

   - ``--ray-address auto`` (default) reuses an *already-running* local Ray
     cluster; pass ``auto`` to start one on this host, or ``ip:port`` to join.
   - ``--n-workers`` = number of Ray actors (each may run on a different node).
   - ``--verify`` additionally runs the single-process ``BatchPipeline.run``
     locally and asserts the cluster assignment is **identical** -- proving the
     distributed path reproduces the single-machine result.

   To exercise true multi-node placement, place ~half of ``--n-workers`` objects
   on each node via the ``ray`` CLI's node labels; without that, Ray will pack
   workers wherever it sees capacity (still a valid multi-process demo).

Notes
-----
* The deterministic hashing embedder is used (identical to the single-process
  default) so results match ``BatchPipeline.run``.
* Candidate-pair ownership by hash means each pair is scored on exactly one
  worker; only above-``tau`` edges cross the wire.
* This example uses the transitive-closure mode; G-Swoosh stays single-machine
  by design (see ``.distributed_er_plan.md`` caveats).
"""

from __future__ import annotations

import argparse
import random

from vectorer.batch import build_batch_pipeline
from vectorer.comparisons import make_comparison
from vectorer.distributed import create_executor
from vectorer.distributed import distributed_score_and_reduce
from vectorer.scoring import FellegiSunterScorer

FIRST_NAMES = ["john", "mary", "robert", "susan", "james", "linda", "michael", "patricia",
               "david", "jennifer", "william", "elizabeth"]
LAST_NAMES = ["smith", "jones", "martinez", "brown", "wilson", "davis", "garcia", "miller",
              "lee", "taylor", "anderson", "thomas"]


def generate(n_base: int, dup_rate: float, seed: int = 42):
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
    from vectorer.embeddings import CharacterHashingEmbedding
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-node distributed batch ER (Ray)")
    parser.add_argument("--n-base", type=int, default=2000)
    parser.add_argument("--dup-rate", type=float, default=0.04)
    parser.add_argument("--tau", type=float, default=0.85)
    parser.add_argument("--n-canopies", type=int, default=32)
    parser.add_argument("--overlap", type=int, default=2)
    parser.add_argument("--n-workers", type=int, default=4,
                        help="number of Ray actors (spread across the cluster)")
    parser.add_argument("--ray-address", default="auto",
                        help="head node address ('auto' reuses an existing actor "
                             "cluster on this host, or pass ip:port to join one)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verify", action="store_true",
                        help="run single-process and assert identical clusters")
    args = parser.parse_args()

    records, dup_positions = generate(args.n_base, args.dup_rate, args.seed)
    scorer = FellegiSunterScorer.from_comparisons(make_comparisons(), threshold=args.tau)
    print(f"dataset: {len(records):,} records, {len(dup_positions)} duplicate twins")

    print(f"building candidate pairs (canopies={args.n_canopies}, overlap={args.overlap})...")
    pairs = build_pairs(records, args.n_canopies, args.overlap, args.seed)
    print(f"candidate pairs: {len(pairs):,}")

    print(f"scoring + clustering across {args.n_workers} Ray workers "
          f"(address={args.ray_address!r})...")
    executor = create_executor("ray", n_workers=args.n_workers, address=args.ray_address)
    assignment = distributed_score_and_reduce(
        scorer,
        [records[i] for i, j in pairs],
        [records[j] for i, j in pairs],
        pair_positions=pairs,
        tau=args.tau,
        n=len(records),
        n_workers=args.n_workers,
        executor=executor,
    )

    twins_merged = sum(
        1 for twin_idx, base_idx in dup_positions
        if assignment.node_cluster[twin_idx] == assignment.node_cluster[base_idx]
    )
    print(f"Swoosh clusters: {len(assignment.clusters)}; "
          f"duplicate twins merged: {twins_merged}/{len(dup_positions)}")

    if args.verify:
        print("verifying against single-process...")
        pipeline = build_batch_pipeline(
            comparisons=make_comparisons(), scorer=scorer,
            n_canopies=args.n_canopies, overlap_m=args.overlap, tau=args.tau,
        )
        single = pipeline.run(records).assignment
        identical = single.node_cluster == assignment.node_cluster
        print(f"distributed (multi-node) == single-process: {identical}")
        if not identical:
            raise SystemExit("ERROR: distributed differs from single-process!")
        print("OK: identical cluster assignment")


if __name__ == "__main__":
    main()