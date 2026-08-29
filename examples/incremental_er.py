"""Example: incremental entity resolution.

Pipeline: parsing -> embedding -> vector search blocking (top-k) -> Fellegi-
Sunter scoring on the top-k -> classification.

Run from the project root:

    python examples/incremental_er.py [--n-references 500] [--tau 0.85]
                                      [--k 20] [--embedder hashing]

The example builds a synthetic reference population, resolves a noisy duplicate
of a known record (should match) and an unrelated record (should be rejected),
then trains the Fellegi-Sunter parameters on the population via Splink
expectation maximisation and repeats the resolutions.
"""

from __future__ import annotations

import argparse
import random
from typing import Optional

from vectorer.comparisons import make_comparison
from vectorer.embeddings import CharacterHashingEmbedding, SentenceTransformerEmbedding
from vectorer.incremental import build_incremental_pipeline
from vectorer.records import DictParser
from vectorer.scoring import FellegiSunterScorer

FIRST_NAMES = [
    "john", "mary", "robert", "susan", "james", "linda", "michael", "patricia",
    "david", "jennifer", "william", "elizabeth", "richard", "barbara", "joseph",
]
LAST_NAMES = [
    "smith", "jones", "martinez", "brown", "wilson", "davis", "garcia",
    "miller", "lee", "taylor", "anderson", "thomas", "moore", "jackson",
]


def generate_people(n: int, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    people = []
    for i in range(n):
        first = rng.choice(FIRST_NAMES)
        last = rng.choice(LAST_NAMES)
        dob = f"{rng.randint(1930, 2010)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
        email = f"{first}.{last}{i}@mail.com" if rng.random() < 0.7 else None
        address = f"{rng.randint(1, 9999)} main street" if rng.random() < 0.7 else None
        people.append({
            "first_name": first,
            "last_name": last,
            "date_of_birth": dob,
            "email": email,
            "address": address,
        })
    return people


def noisy_duplicate(person: dict, seed: int = 1) -> dict:
    rng = random.Random(seed)
    copy = dict(person)
    if rng.random() < 0.6:
        copy["last_name"] = person["last_name"] + "e"
    if rng.random() < 0.3:
        copy["date_of_birth"] = person["date_of_birth"][:-1] + "0"
    return copy


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
    parser = argparse.ArgumentParser(description="Incremental ER example")
    parser.add_argument("--n-references", type=int, default=500)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--tau", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embedder", choices=["hashing", "sentence"], default="hashing")
    args = parser.parse_args()

    references = generate_people(args.n_references, seed=args.seed)
    base = references[0]
    duplicate = noisy_duplicate(base, seed=args.seed + 1)
    seed_offset = args.seed + 2
    unrelated = generate_people(1, seed=seed_offset)[0]
    while (
        unrelated["first_name"] == base["first_name"]
        and unrelated["last_name"] == base["last_name"]
        and unrelated["date_of_birth"] == base["date_of_birth"]
    ):
        seed_offset += 1
        unrelated = generate_people(1, seed=seed_offset)[0]

    comparisons = make_comparisons()
    embedder = build_embedder(args.embedder)

    print(f"Building incremental pipeline over {len(references):,} references...")
    pipeline = build_incremental_pipeline(
        references,
        embedder=embedder,
        comparisons=comparisons,
        k=args.k,
        tau=args.tau,
    )

    def report(label: str, resolution) -> None:
        if resolution.decision.value == "match":
            best = resolution.matches[0]
            print(
                f"  {label}: MATCH  p={best.match_probability:.4f} "
                f"(block score {best.blocking_score:.3f}, candidate #{best.candidate_position})"
            )
        else:
            print(f"  {label}: NO MATCH (retrieved {len(resolution.retrieved)} candidates)")

    print("Resolving a noisy duplicate of reference #0:")
    report("duplicate", pipeline.resolve(duplicate))
    print("Resolving a genuinely unrelated record:")
    report("unrelated", pipeline.resolve(unrelated))

    print("Training Fellegi-Sunter m/u via expectation maximisation...")
    trained = FellegiSunterScorer.from_comparisons(comparisons).fit_em(
        references,
        training_block_on=[("first_name",), ("date_of_birth",)],
        seed=args.seed,
    )
    pipeline.scorer = trained
    print("After training:")
    report("duplicate", pipeline.resolve(duplicate))
    report("unrelated", pipeline.resolve(unrelated))


if __name__ == "__main__":
    main()