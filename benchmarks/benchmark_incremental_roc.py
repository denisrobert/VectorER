"""Benchmark: τ-sweep ROC / Precision-Recall for the incremental pipeline.

Methodology (calibrated): a **large** synthetic population (default 100k
records) is split **80/20 into training and validation sets**.  A labelled
Fellegi-Sunter training pair set is built from the training half (variants =
match pairs, distinct records = non-match pairs), the scorer's ``m/u`` are
**calibrated** on it, and then the τ sweep is run on the **held-out validation
half**: every validation query is resolved against an index of the validation
records, its best-candidate posterior is taken as the score, and sweeping
``tau`` yields the ROC / PR curves and ROC AUC.  This makes the numbers
meaningful (the posteriors are data-fitted, not raw defaults).

Usage::

    python benchmarks/benchmark_incremental_roc.py --n-records 100000 \\
        --train-fraction 0.8 --train-pairs 2000 \\
        --n-positives 300 --n-negatives 300 --tau-count 80 \\
        --output results/incremental_roc.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
from pathlib import Path
from typing import Any, Sequence

from vectorer.comparisons import make_comparison
from vectorer.embeddings import CharacterHashingEmbedding
from vectorer.incremental import build_incremental_pipeline
from vectorer.scoring import FellegiSunterScorer
from vectorer.vectorstores import FlatIndex, InMemoryVectorDatabase

FIRST_NAMES = [
    "john", "mary", "robert", "susan", "james", "linda", "michael", "patricia",
    "david", "jennifer", "william", "elizabeth", "richard", "barbara", "joseph",
    "thomas", "sarah", "charles", "karen", "daniel", "nancy", "paul", "betty",
    "mark", "helen", "steven", "sandra", "george", "ashley", "ken", "donna",
    "brian", "michelle", "kevin", "laura", "jason", "kathleen", "edward",
    "deborah", "ronald", "samantha", "frank", "carol", "raymond", "maryann",
    "gregory", "nancyann", "scott", "diane", "eric", "kathryn", "stephen",
    "dorothy", "lawrence", "gloria", "nicholas", "phyllis", "andrew", "jane",
]
LAST_NAMES = [
    "smith", "jones", "martinez", "brown", "wilson", "davis", "garcia", "miller",
    "lee", "taylor", "anderson", "thomas", "moore", "jackson", "martin", "thompson",
    "white", "lopez", "hill", "scott", "green", "adams", "baker", "gonzalez",
    "nelson", "carter", "mitchell", "perez", "roberts", "turner", "phillips",
    "campbell", "parker", "evans", "edwards", "collins", "stewart", "sanchez",
    "morris", "rogers", "reed", "cook", "morgan", "bell", "murphy", "bailey",
    "rivera", "cooper", "richardson", "cox", "howard", "ward", "torres",
]
STREET_KINDS = ["St", "Ave", "Rd", "Blvd", "St.", "Ave.", "Rd.", "Blvd."]
CITIES = ["Toronto", "Vancouver", "Montreal", "Calgary", "Ottawa", "Edmonton"]


def generate_people(count: int, missing_rate: float = 0.3, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    people = []
    for i in range(count):
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
        people.append({
            "first_name": first,
            "last_name": last,
            "date_of_birth": dob,
            "email": email,
            "address": address,
        })
    return people


def introduce_variations(person: dict, variation_rate: float = 0.15) -> dict:
    """True-match variant: perturb 1-3 fields, keep the person's identity."""
    new_person = dict(person)
    if random.random() < variation_rate and len(new_person["first_name"]) > 2:
        name = list(new_person["first_name"])
        name[random.randint(0, len(name) - 1)] = random.choice("abcdefghijklmnopqrstuvwxyz")
        new_person["first_name"] = "".join(name)
    if random.random() < variation_rate and len(new_person["last_name"]) > 2:
        name = list(new_person["last_name"])
        name[random.randint(0, len(name) - 1)] = random.choice("abcdefghijklmnopqrstuvwxyz")
        new_person["last_name"] = "".join(name)
    if random.random() < variation_rate and new_person["address"]:
        addr = new_person["address"]
        for old, new in [
            ("St ", "Street "), ("St.", "Street"), ("Ave ", "Avenue "),
            ("Ave.", "Avenue"), ("Rd ", "Road "), ("Rd.", "Road"),
            ("Blvd ", "Boulevard "), ("Blvd.", "Boulevard"),
        ]:
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


_LABELLED_FIELDS = ("first_name", "last_name", "date_of_birth", "email", "address")


def build_labelled_training_pairs(
    train_records: Sequence[dict],
    n_pairs: int,
    variation_rate: float = 0.15,
    seed: int = 11,
) -> list[dict]:
    """Build an ``is_match``-labelled pair set from the training population.

    Half are true matches (a record + its perturbed variant), half true
    non-matches (two distinct records).  Each row has ``is_match`` plus
    ``<field>_l`` / ``<field>_r`` for every compared field, as required by
    ``FellegiSunterScorer.calibrate_from_pairs``.
    """
    rng = random.Random(seed)
    half = n_pairs // 2
    pairs: list[dict] = []
    # True matches: same identity, perturbed.
    for _ in range(half):
        rec = train_records[rng.randrange(len(train_records))]
        variant = introduce_variations(rec, variation_rate=variation_rate)
        row = {"is_match": 1}
        for f in _LABELLED_FIELDS:
            row[f"{f}_l"] = rec[f]
            row[f"{f}_r"] = variant[f]
        pairs.append(row)
    # True non-matches: distinct identities.
    for _ in range(half):
        i = rng.randrange(len(train_records))
        j = rng.randrange(len(train_records))
        while j == i:
            j = rng.randrange(len(train_records))
        row = {"is_match": 0}
        for f in _LABELLED_FIELDS:
            row[f"{f}_l"] = train_records[i][f]
            row[f"{f}_r"] = train_records[j][f]
        pairs.append(row)
    return pairs


# ---------------------------------------------------------------------------
# Curves
# ---------------------------------------------------------------------------


def roc_curve(scores, labels, thresholds):
    """Return (fpr, tpr) arrays over ``thresholds`` (tau >= t means positive)."""
    scores = list(scores)
    labels = list(labels)
    positives = sum(labels)
    negatives = len(labels) - positives
    fpr = []
    tpr = []
    for t in thresholds:
        tp = fp = 0
        for s, y in zip(scores, labels):
            if s >= t:
                if y:
                    tp += 1
                else:
                    fp += 1
        fpr.append(fp / negatives if negatives else 0.0)
        tpr.append(tp / positives if positives else 0.0)
    return fpr, tpr


def precision_recall_curve(scores, labels, thresholds):
    """Return (recall, precision) arrays over ``thresholds`` (>= t is positive)."""
    scores = list(scores)
    labels = list(labels)
    positives = sum(labels)
    rec = []
    prec = []
    for t in thresholds:
        tp = fp = 0
        for s, y in zip(scores, labels):
            if s >= t:
                if y:
                    tp += 1
                else:
                    fp += 1
        rec.append(tp / positives if positives else 0.0)
        prec.append(tp / (tp + fp) if (tp + fp) else 0.0)
    return rec, prec


def roc_auc(fpr, tpr):
    """Trapezoidal AUC over the ROC curve points."""
    auc = 0.0
    for i in range(1, len(fpr)):
        auc += 0.5 * (fpr[i] - fpr[i - 1]) * (tpr[i] + tpr[i - 1])
    return auc


def environment_block() -> dict:
    versions = {}
    for name in ["numpy", "faiss"]:
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
        description="Incremental-pipeline ROC / PR / AUC benchmark by sweeping tau"
    )
    parser.add_argument("--n-records", type=int, default=100000,
                        help="total synthetic population size (split into train/validation)")
    parser.add_argument("--train-fraction", type=float, default=0.8,
                        help="fraction of the population used for FS calibration; "
                             "the rest is the held-out validation set for the sweep")
    parser.add_argument("--train-pairs", type=int, default=2000,
                        help="number of labelled pairs built from the training half")
    parser.add_argument("--n-positives", type=int, default=300,
                        help="validation true-match query count (variants of validation records)")
    parser.add_argument("--n-negatives", type=int, default=300,
                        help="validation true-non-match query count (distinct records)")
    parser.add_argument("--missing-rate", type=float, default=0.3)
    parser.add_argument("--close-variation-rate", type=float, default=0.15)
    parser.add_argument("--smoothing", type=float, default=0.5,
                        help="Laplace smoothing for calibrate_from_pairs")
    parser.add_argument("--k", dest="blocking_k", type=int, default=20)
    parser.add_argument("--tau-count", type=int, default=80,
                        help="number of tau thresholds to sweep")
    parser.add_argument("--tau-max", type=float, default=1.0)
    parser.add_argument("--tau-min", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embedder", choices=["hashing", "minilm", "mdbr"], default="hashing",
                        help="embedding model: 'hashing' (deterministic 384-d), "
                             "'minilm' (sentence-transformers/all-MiniLM-L6-v2), "
                             "'mdbr' (MongoDB/mdbr-leaf-mt)")
    parser.add_argument("--data-file", default=None,
                        help="optional prepared/real dataset (JSONL or JSON) to use as the "
                             "population instead of the synthetic generator; expects the "
                             "compared fields, None allowed")
    parser.add_argument("--data-key", default=None,
                        help="when --data-file is a single JSON object, the key holding the records list")
    parser.add_argument("--output", default="results/incremental_roc.json")
    args = parser.parse_args()

    random.seed(args.seed)

    # ---- 1. Population + split -------------------------------------------
    from benchmark_data import load_records, require_compared_fields

    if args.data_file:
        print(f"Loading population from {args.data_file} ...")
        population = load_records(args.data_file, key=args.data_key)
        require_compared_fields(population, ["first_name", "last_name", "date_of_birth", "email", "address"])
        print(f"  {len(population):,} records loaded")
    else:
        print(f"Generating {args.n_records:,} synthetic people ...")
        population = generate_people(args.n_records, missing_rate=args.missing_rate, seed=args.seed)
    n_train = int(round(len(population) * args.train_fraction))
    train_records = population[:n_train]
    val_records = population[n_train:]
    print(f"  train={len(train_records):,} validation={len(val_records):,}")

    # ---- 2. Train the FS scorer on the TRAINING half ----------------------
    print(f"Building {args.train_pairs:,} labelled training pairs ...")
    labelled_pairs = build_labelled_training_pairs(
        train_records, args.train_pairs,
        variation_rate=args.close_variation_rate, seed=args.seed + 5,
    )
    print("Calibrating Fellegi-Sunter m/u on the training set ...")
    scorer = FellegiSunterScorer.from_comparisons(make_comparisons())
    scorer = scorer.calibrate_from_pairs(labelled_pairs, smoothing=args.smoothing)

    # ---- 3. Validation pipeline: index VALIDATION records only ------------
    def build_embedder():
        if args.embedder == "minilm":
            from vectorer.embeddings import SentenceTransformerEmbedding
            from vectorer.pins import EMBEDDING_MODEL_ID, EMBEDDING_MODEL_REVISION

            return SentenceTransformerEmbedding(EMBEDDING_MODEL_ID, EMBEDDING_MODEL_REVISION)
        if args.embedder == "mdbr":
            from vectorer.embeddings import SentenceTransformerEmbedding

            return SentenceTransformerEmbedding("MongoDB/mdbr-leaf-mt")
        return CharacterHashingEmbedding(dimension=384)

    embedder = build_embedder()
    pipeline = build_incremental_pipeline(
        val_records,
        embedder=embedder,
        scorer=scorer,
        k=args.blocking_k,
        tau=0.5,  # nominal; the sweep decides, the posteriors are the data
    )

    # ---- 4. Validation queries --------------------------------------------
    # Positives: perturbed copies of validation records (best-posterior match).
    rng_perturb = random.Random(args.seed + 7)
    base_indices = list(range(len(val_records)))
    rng_perturb.shuffle(base_indices)
    positives = [
        introduce_variations(val_records[base_indices[i % len(base_indices)]],
                             variation_rate=args.close_variation_rate)
        for i in range(args.n_positives)
    ]
    # Negatives: distinct people NOT in the validation set (full-identity-disjoint).
    neg_base = generate_people(max(args.n_negatives * 10, 500),
                               missing_rate=args.missing_rate, seed=args.seed + 1)
    ref_names = {(r["first_name"], r["last_name"], r["date_of_birth"]) for r in val_records}
    ref_dobs = {r["date_of_birth"] for r in val_records}
    ref_emails = {r["email"] for r in val_records if r["email"]}
    negatives: list[dict] = []
    for candidate in neg_base:
        if len(negatives) >= args.n_negatives:
            break
        nc = dict(candidate)
        if (nc["first_name"], nc["last_name"], nc["date_of_birth"]) in ref_names:
            continue
        if nc["date_of_birth"] in ref_dobs:
            import datetime as _dt

            d = _dt.date.fromisoformat(nc["date_of_birth"])
            nc["date_of_birth"] = (d + _dt.timedelta(days=5)).isoformat()
        if nc.get("email") in ref_emails:
            nc["email"] = None
        negatives.append(nc)

    print(f"Resolving {len(positives):,} positives + {len(negatives):,} negatives "
          f"on the validation set ...")
    scores: list[float] = []
    labels: list[int] = []
    for q in positives:
        r = pipeline.resolve(q)
        best = max((c.probability for c in r.retrieved), default=0.0)
        scores.append(best)
        labels.append(1)
    for q in negatives:
        r = pipeline.resolve(q)
        best = max((c.probability for c in r.retrieved), default=0.0)
        scores.append(best)
        labels.append(0)

    # ---- 5. Sweep tau on the validation scores ---------------------------
    thresholds = [args.tau_max - (args.tau_max - args.tau_min) * i / (args.tau_count - 1)
                  for i in range(args.tau_count)]
    thresholds = sorted(set(round(t, 6) for t in thresholds), reverse=True)

    fpr, tpr = roc_curve(scores, labels, thresholds)
    rec, prec = precision_recall_curve(scores, labels, thresholds)
    auc = roc_auc(fpr, tpr)

    # Operating points at a few canonical taus (incl. 0.85).
    def operating_point(tau: float) -> dict:
        tp = fp = tn = fn = 0
        for s, y in zip(scores, labels):
            pred = s >= tau
            tp += bool(pred and y)
            fp += bool(pred and not y)
            tn += bool(not pred and not y)
            fn += bool(not pred and y)
        return {
            "tau": tau,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": round(tp / (tp + fp), 4) if tp + fp else None,
            "recall": round(tp / (tp + fn), 4) if tp + fn else None,
            "fpr": round(fp / (fp + tn), 4) if (fp + tn) else None,
        }

    best_youden = max(range(len(thresholds)),
                      key=lambda i: tpr[i] - fpr[i])
    results = {
        "methodology": (
            "calibrated: 80/20 train/validation split; FS m/u fit on the training "
            "pair set; sweep evaluated on the held-out validation records"
        ),
        "parameters": {
            "n_records": args.n_records,
            "n_train": len(train_records),
            "n_validation": len(val_records),
            "train_fraction": args.train_fraction,
            "train_pairs": len(labelled_pairs),
            "data_file": args.data_file,
            "data_key": args.data_key,
            "n_positives": args.n_positives,
            "n_negatives": args.n_negatives,
            "missing_rate": args.missing_rate,
            "close_variation_rate": args.close_variation_rate,
            "smoothing": args.smoothing,
            "blocking_k": args.blocking_k,
            "tau_count": args.tau_count,
            "tau_range": [args.tau_min, args.tau_max],
            "seed": args.seed,
            "embedder": args.embedder,
            "score_definition": "best candidate posterior per resolved query",
        },
        "training": {
            "n_pairs": len(labelled_pairs),
            "n_match_pairs": sum(1 for p in labelled_pairs if p["is_match"]),
            "n_non_match_pairs": sum(1 for p in labelled_pairs if not p["is_match"]),
        },
        "n_positives": len(positives),
        "n_negatives": len(negatives),
        "roc_auc": round(auc, 4),
        "operating_points": {
            "default_tau_0_85": operating_point(0.85),
            "best_youden": {**operating_point(thresholds[best_youden]),
                            "youden_j": round(tpr[best_youden] - fpr[best_youden], 4)},
        },
        "thresholds": thresholds,
        "fpr": [round(x, 6) for x in fpr],
        "tpr": [round(x, 6) for x in tpr],
        "recall": [round(x, 6) for x in rec],
        "precision": [round(x, 6) for x in prec],
        "environment": environment_block(),
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"\nROC AUC (validation) = {auc:.4f}")
    print(f"tau=0.85: {results['operating_points']['default_tau_0_85']}")
    print(f"best J:  {results['operating_points']['best_youden']}")
    print(f"Saved ROC/PR curves to {args.output}")

    # Compact table preview
    print("\n  tau      TPR      FPR      recall   precision")
    for i in range(0, len(thresholds), max(1, len(thresholds) // 12)):
        print(f"  {thresholds[i]:.3f}   {tpr[i]:.3f}    {fpr[i]:.3f}    {rec[i]:.3f}   {prec[i]:.3f}")


if __name__ == "__main__":
    main()