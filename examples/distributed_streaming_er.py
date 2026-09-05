"""Example: distributed score + closure (streaming reduce) for batch ER.

Illustrates the v0.4.0 distributed building blocks added to
``vectorer.distributed`` for scaling the expensive stages across machines:

* ``distributed_score_pairs`` -- the FS scoring *map*: pairs owned by worker via
  a deterministic, balanced hash; each worker scores its chunk with the same
  serialized scorer and returns only the above-``tau`` edges (only those cross
  the wire).
* ``distributed_score_and_reduce`` -- the full streaming flow: score the pairs,
  then ``distributed_closure_reduce`` computes exact connected components
  across workers (per-worker union-find + a shared-node merge into min-position
  ids), bit-for-bit identical to the single-process transitive closure.

Run from the project root:

    python examples/distributed_streaming_er.py [--n-base 500] [--n-workers 4]
        [--tau 0.85] [--verify]

``--verify`` also runs the single-process ``BatchPipeline.run`` and asserts the
distributed cluster assignment is identical.
"""

from __future__ import annotations

import argparse
import random

from vectorer.batch import build_batch_pipeline
from vectorer.clustering import ScoredPair
from vectorer.comparisons import make_comparison
from vectorer.distributed import (
    distributed_score_and_reduce,
    distributed_score_pairs,
)
from vectorer.scoring import FellegiSunterScorer

FIRST_NAMES = ["john", "mary", "robert", "susan", "james", "linda", "michael", "patricia"]
LAST_NAMES = ["smith", "jones", "martinez", "brown", "wilson", "davis", "garcia", "miller"]


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Distributed score + closure example")
    parser.add_argument("--n-base", type=int, default=500)
    parser.add_argument("--dup-rate", type=float, default=0.04)
    parser.add_argument("--tau", type=float, default=0.85)
    parser.add_argument("--n-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verify", action="store_true",
                        help="compare against the single-process pipeline")
    args = parser.parse_args()

    records, dup_positions = generate(args.n_base, args.dup_rate, args.seed)
    scorer = FellegiSunterScorer.from_comparisons(make_comparisons(), threshold=args.tau)

    # Build the candidate pairs using the same canopy paths as BatchPipeline.
    import numpy as np

    from vectorer.blocking import assign_canopies, train_canopy_centroids
    from vectorer.embeddings import CharacterHashingEmbedding
    from vectorer.records import to_record_dict

    embedder = CharacterHashingEmbedding(384)
    vecs = np.asarray(embedder.embed_many([
        "\n".join(f"{k}: {v}" for k, v in to_record_dict(r).items() if v is not None)
        for r in records
    ]), dtype="float32")
    centroids = train_canopy_centroids(vecs, n_clusters=32, seed=args.seed, sample_size=None)
    canopy = assign_canopies(vecs, centroids, overlap_m=2)

    # cross-check canopy pairs -> unique candidate pairs
    seen = set()
    pairs: list[tuple[int, int]] = []
    for (i, j) in canopy.candidate_pairs():
        key = (i, j) if i <= j else (j, i)
        if key not in seen:
            seen.add(key)
            pairs.append(key)
    print(f"candidate pairs: {len(pairs):,}")

    # Streaming score map (only above-tau edges cross the wire), then reduce.
    left = [records[i] for i, j in pairs]
    right = [records[j] for i, j in pairs]
    assignment = distributed_score_and_reduce(
        scorer, left, right,
        pair_positions=pairs,
        tau=args.tau, n=len(records),
        n_workers=args.n_workers,
    )

    twins_merged = sum(
        1 for twin_idx, base_idx in dup_positions
        if assignment.node_cluster[twin_idx] == assignment.node_cluster[base_idx]
    )
    print(f"Swoosh clusters: {len(assignment.clusters)}; "
          f"duplicate twins merged: {twins_merged}/{len(dup_positions)}")

    if args.verify:
        pipeline = build_batch_pipeline(
            comparisons=make_comparisons(), scorer=scorer,
            n_canopies=32, overlap_m=2, tau=args.tau,
        )
        single = pipeline.run(records).assignment
        identical = single.node_cluster == assignment.node_cluster
        print(f"distributed == single-process: {identical}")
        if not identical:
            raise SystemExit("ERROR: distributed differs from single-process!")
        print("OK: identical cluster assignment")


if __name__ == "__main__":
    main()