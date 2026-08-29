"""Example: batch entity resolution (deduplication).

Pipeline: parsing -> embedding -> canopy blocking on the embedded dataset ->
Fellegi-Sunter scoring of every canopy candidate pair -> Swoosh clustering on
the results.

Run from the project root:

    python examples/batch_er.py [--n-base 500] [--dup-rate 0.04] [--tau 0.85]
                                [--n-canopies 64] [--overlap 2]
                                [--embedder hashing]

The example builds a synthetic population with a fraction of exact/near
duplicates, deduplicates it with the batch pipeline, and reports cluster
statistics plus ground-truth precision/recall.
"""

from __future__ import annotations

import argparse
import random

from vectorer.batch import build_batch_pipeline
from vectorer.comparisons import make_comparison
from vectorer.embeddings import CharacterHashingEmbedding, SentenceTransformerEmbedding

FIRST_NAMES = [
    "john", "mary", "robert", "susan", "james", "linda", "michael", "patricia",
    "david", "jennifer", "william", "elizabeth", "richard", "barbara", "joseph",
]
LAST_NAMES = [
    "smith", "jones", "martinez", "brown", "wilson", "davis", "garcia",
    "miller", "lee", "taylor", "anderson", "thomas", "moore", "jackson",
]


def generate_dataset(n_base: int, dup_rate: float, seed: int = 42) -> list[dict]:
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
        make_comparison(
            "jaro_winkler_at_thresholds",
            col_name="first_name",
            score_threshold_or_thresholds=[0.9, 0.8, 0.7],
        ),
        make_comparison(
            "jaro_winkler_at_thresholds",
            col_name="last_name",
            score_threshold_or_thresholds=[0.9, 0.8, 0.7],
        ),
        make_comparison("date_of_birth_comparison", col_name="date_of_birth"),
        make_comparison("email_comparison", col_name="email"),
    ]


def build_embedder(kind: str):
    if kind == "sentence":
        from vectorer.pins import EMBEDDING_MODEL_ID, EMBEDDING_MODEL_REVISION
        return SentenceTransformerEmbedding(EMBEDDING_MODEL_ID, EMBEDDING_MODEL_REVISION)
    return CharacterHashingEmbedding(dimension=384)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch ER example")
    parser.add_argument("--n-base", type=int, default=500)
    parser.add_argument("--dup-rate", type=float, default=0.04)
    parser.add_argument("--tau", type=float, default=0.85)
    parser.add_argument("--n-canopies", type=int, default=64)
    parser.add_argument("--overlap", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embedder", choices=["hashing", "sentence"], default="hashing")
    args = parser.parse_args()

    records, dup_positions = generate_dataset(args.n_base, args.dup_rate, seed=args.seed)
    n_dup = len(dup_positions)
    print(f"Dataset: {len(records):,} records ({n_dup} duplicate twins of base rows)")

    # Keep the canopy grid proportionate to the dataset (FAISS k-means
    # requires n >= ~39 * clusters training points).
    n_canopies = min(args.n_canopies, max(1, len(records) // 39))

    pipeline = build_batch_pipeline(
        embedder=build_embedder(args.embedder),
        comparisons=make_comparisons(),
        n_canopies=n_canopies,
        overlap_m=args.overlap,
        canopy_seed=args.seed,
        tau=args.tau,
    )
    result = pipeline.run(records)

    # Ground-truth evaluation: duplicates share an entity iff their base rows do.
    duplicate_entity = {
        args.n_base + i: dup_positions[i] for i in range(n_dup)
    }
    tp = fp = fn = 0
    for twin_position, base_position in duplicate_entity.items():
        same = result.cluster_of_position(twin_position) == result.cluster_of_position(base_position)
        if same:
            tp += 1
        else:
            fn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    print(f"Canopies: {result.canopy.n_clusters}, candidate pairs: {result.n_candidate_pairs}")
    print(f"Timing (s): { {k: round(v, 3) for k, v in result.timing.items()} }")
    print(
        f"Swoosh clusters: {result.n_clusters} "
        f"({result.n_non_singletons} merged, {result.n_singletons} singletons) "
        f"over {result.n_candidate_pairs} scored pairs"
    )
    print(f"Duplicate recovery: precision={precision:.3f} recall={recall:.3f} ({tp}/{n_dup})")
    if result.n_non_singletons:
        biggest = max(result.assignment.clusters.values(), key=lambda c: len(c.member_positions))
        print(
            f"Largest cluster: {len(biggest.member_positions)} records, "
            f"representative #{biggest.representative_position}"
        )


if __name__ == "__main__":
    main()