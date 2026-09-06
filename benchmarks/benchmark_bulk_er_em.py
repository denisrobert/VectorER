"""Benchmark the bulk (batch) entity-resolution pipeline **with EM training**.

Identical methodology to ``benchmark_bulk_er.py`` (whole-dataset canopy ->
Fellegi-Sunter -> Swoosh, duplicate-recovery metrics), except the Fellegi-Sunter
``m/u`` and prior are **fit by expectation maximisation** instead of using
default m/u.  The training population is the deduplicatable,
duplicate-bearing dataset -- by default the generated
``benchmarks/population_with_duplicates.json`` (300k census-distributed people
plus ~22k perturbed duplicates), so EM sees realistic near-duplicates and fits
data-driven match weights.

Usage::

    python benchmarks/benchmark_bulk_er_em.py \\
        --data-file benchmarks/population_with_duplicates.json \\
        --n-training-subsample 120000 \\   # EM fits m/u + prior on this sample/limit
        --bench-limit 40000 \\             # bulk dedupe is run on this sample (tractable)
        --output results/bulk_latency_em.json

Notes
-----
* ``--data-file`` defaults to the gitignored duplicate-bearing population;
  pass a different JSON/JSONL to train + bench on your own data.
* ``--em-block-on`` chooses the blocking rules used to generate candidate pairs
  for EM; ``--em-max-pairs`` caps them, ``--em-seed`` seeds the random-pair
  sampling, and ``--em-recall`` adjusts the base prior for blocking recall.
* Because the EM loop and the full-file bulk run are both expensive, the bulk
  dedup stage is evaluated on ``--bench-limit`` records (sampled deterministically
  from the file) while EM trains on up to ``--n-training-subsample``.  ``None`` /
  0 means "use the whole file for that stage" (slow at 300k+).
* Without ground-truth twin labels (``--gt-file``) only throughput/cluster stats
  are reported; the EM-trained prior and m/u still drive the scores.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any, Optional, Sequence

from vectorer.batch import BatchPipeline
from vectorer.comparisons import make_comparison
from vectorer.embeddings import CharacterHashingEmbedding
from vectorer.scoring import FellegiSunterScorer

# Default training population (gitignored, regeneratable).
DEFAULT_DATA_FILE = "benchmarks/population_with_duplicates.json"

FIRST_NAMES = [
    "john", "mary", "robert", "susan", "james", "linda", "michael", "patricia",
    "david", "jennifer", "william", "elizabeth", "richard", "barbara", "joseph",
]
LAST_NAMES = [
    "smith", "jones", "martinez", "brown", "wilson", "davis", "garcia", "miller",
    "lee", "taylor", "anderson", "thomas", "moore", "jackson",
]


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
    return CharacterHashingEmbedding(dimension=384)


def em_train(
    records: Sequence[dict],
    comparisons: Optional[Sequence] = None,
    *,
    training_block_on: Sequence[Sequence[str]] = (("first_name",), ("date_of_birth",)),
    max_pairs: float = 1e6,
    recall: float = 0.7,
    em_seed: Optional[int] = None,
    tau: float = 0.85,
) -> FellegiSunterScorer:
    """Build a scorer whose ``m/u`` and prior are fit by expectation maximisation
    over ``records`` (a duplicate-bearing population)."""
    comparisons = comparisons if comparisons is not None else make_comparisons()
    scorer = FellegiSunterScorer.from_comparisons(comparisons, threshold=tau)
    return scorer.fit_em(
        records,
        training_block_on=training_block_on,
        max_pairs=max_pairs,
        recall=recall,
        seed=em_seed,
    )


def cluster_quality(
    pipeline: BatchPipeline,
    records: Sequence[dict],
    twin_entities: dict[int, int],
    seed: int,
) -> dict[str, Any]:
    result = pipeline.run(records)
    timing = result.timing

    tp = fp = fn = 0
    n_rec = len(records)
    for twin_position, base_position in twin_entities.items():
        if not (0 <= twin_position < n_rec and 0 <= base_position < n_rec):
            continue
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
        "n_procs": 1,
        "seed": seed,
    }


def cluster_quality_distributed(
    records: Sequence[dict],
    twin_entities: dict[int, int],
    seed: int,
    scorer: FellegiSunterScorer,
    n_canopies: int,
    overlap_m: int,
    tau: float,
    n_procs: int,
    progress: bool = True,
) -> dict[str, Any]:
    """Bulk dedup via ``distributed_batch_er``: the FS scoring stage runs
    across ``n_procs`` process workers (single machine, multiple cores).

    With ``progress=True``, the controller shows an aggregate tqdm bar over
    scored pair chunks (each worker reports its chunk count through a shared
    queue)."""
    import time as _t

    from vectorer.distributed import distributed_batch_er

    progress_callback = None
    if progress:
        try:
            from tqdm import tqdm

            bar = tqdm(desc="fs scoring (distributed)", unit="pairs",
                       leave=False, ascii=True)

            def progress_callback(n):
                bar.update(n)

        except Exception:  # noqa: BLE001  (tqdm optional)
            bar = None

    t_start = _t.perf_counter()
    t0 = _t.perf_counter()
    assignment = distributed_batch_er(
        records,
        scorer=scorer,
        n_canopies=n_canopies,
        overlap_m=overlap_m,
        tau=tau,
        seed=seed,
        n_workers=n_procs,
        use_threads=False,   # process pool -> parallel FS scoring
        embed_dim=384,
        sample_size=200_000,
        progress_callback=progress_callback,
    )
    if progress and bar is not None:
        bar.close()
    t_dist = _t.perf_counter() - t0
    n_candidate_pairs = getattr(assignment, "n_pairs_evaluated", 0)

    tp = fp = fn = 0
    n_rec = len(records)
    for twin_position, base_position in twin_entities.items():
        if not (0 <= twin_position < n_rec and 0 <= base_position < n_rec):
            continue
        same = (
            assignment.node_cluster[twin_position]
            == assignment.node_cluster[base_position]
        )
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
    merge_rate = non_singleton_members / len(records)
    largest = max(
        assignment.clusters.values(),
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
        "n_clusters": len(assignment.clusters),
        "n_non_singletons": sum(1 for c in assignment.clusters.values() if len(c.member_positions) > 1),
        "largest_cluster": (len(largest.member_positions) if largest else 0),
        "n_candidate_pairs": n_candidate_pairs,
        "timing_seconds": {
            "distributed_total": round(t_dist, 4),
            "wall_seconds": round(_t.perf_counter() - t_start, 4),
        },
        "total_seconds": round(t_dist, 4),
        "records_per_second": round(len(records) / max(t_dist, 1e-9), 1),
        "candidate_pairs_per_second": round(
            n_candidate_pairs / max(t_dist, 1e-9), 1
        ),
        "fellegi_sunter_seconds_per_million_pairs": None,
        "n_procs": n_procs,
        "seed": seed,
    }


def prior_sweep(
    records: Sequence[dict],
    gt_pairs: Sequence[tuple[int, int, int]],   # (idx_a, idx_b, is_match)
    priors: Sequence[float],
    taus: Sequence[float],
    *,
    seed: int = 42,
    em_max_pairs: float = 100000,
) -> dict[str, Any]:
    """Sweep ``fit_em(fixed_prior=...)`` over ``priors`` x scoring ``taus``.

    Each prior trains a scorer with the prior FROZEN (only m/u are learned --
    the calibration-paradox remedy), and precision/recall are computed on the
    labelled eval pairs at each ``tau`` on that scorer.  This produces an
    operating-point surface so you can pick the best prior+threshold instead of
    trusting EM's own (often miscalibrated) base rate.
    """
    import numpy as np

    from vectorer.scoring import FellegiSunterScorer

    comps = make_comparisons()
    lefts = [records[a] for a, b, _ in gt_pairs]
    rights = [records[b] for a, b, _ in gt_pairs]
    y = np.asarray([m for _, _, m in gt_pairs], dtype=int)

    rows = []
    for prior in priors:
        scorer = FellegiSunterScorer.from_comparisons(comps).fit_em(
            records, training_block_on=[("first_name",), ("date_of_birth",)],
            max_pairs=em_max_pairs, recall=0.7, seed=seed,
            fixed_prior=prior,
        )
        probs = scorer.score_pairs(lefts, rights)
        for tau in taus:
            pred = probs >= tau
            tp = int((pred & (y == 1)).sum())
            fp = int((pred & (y == 0)).sum())
            fn = int((~pred & (y == 1)).sum())
            rows.append({
                "fixed_prior": prior,
                "tau": tau,
                "precision": round(tp / (tp + fp), 4) if (tp + fp) else 0.0,
                "recall": round(tp / (tp + fn), 4) if (tp + fn) else 0.0,
            })
    return {"prior_sweep_rows": rows}


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
        description="Bulk ER benchmark with EM-trained Fellegi-Sunter m/u"
    )
    parser.add_argument("--data-file", default=DEFAULT_DATA_FILE,
                        help=f"dataset to train EM on AND benchmark (default {DEFAULT_DATA_FILE})")
    parser.add_argument("--data-key", default=None,
                        help="when --data-file is a single JSON object, the key holding the records list")
    parser.add_argument("--gt-file", default=None,
                        help="optional JSON object mapping record index -> true duplicate-twin index")
    parser.add_argument("--n-training-subsample", type=int, default=None,
                        help="optional: fit EM on this many rows (subsample) for "
                             "faster tuning; None = use the whole file")
    parser.add_argument("--bench-limit", type=int, default=None,
                        help="optional: run the bulk dedup benchmark on this many "
                             "rows (deterministic sample of the file); None = whole file")
    parser.add_argument("--n-canopies", type=int, default=128)
    parser.add_argument("--overlap", type=int, default=2,
                        help="top-m canopy assignments per record (1 = hard partition)")
    parser.add_argument("--tau", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embedder", choices=["hashing", "sentence"], default="hashing")
    parser.add_argument("--merge", choices=["rep", "union"], default="rep")
    parser.add_argument("--em-block-on", nargs="*", default=None,
                        help="blocking rules for EM candidate generation; default "
                             "[first_name, date_of_birth]")
    parser.add_argument("--em-max-pairs", type=float, default=1e6)
    parser.add_argument("--em-recall", type=float, default=0.7,
                        help="blocking-recall adjustment for the EM prior")
    parser.add_argument("--em-seed", type=int, default=None,
                        help="seed for EM random-pair sampling (defaults to --seed+313)")
    parser.add_argument("--n-procs", type=int, default=None,
                        help="run the bulk dedup FS-scoring stage across N process "
                             "workers (multi-core) instead of single-process. "
                             "Use e.g. --n-procs 8 on an 8-core machine.")
    parser.add_argument("--prior-sweep-priors", default=None,
                        help="comma-separated fixed priors to sweep with fit_em("
                             "fixed_prior=...) against a gt-file, e.g. "
                             "'1e-6,1e-5,1e-4,1e-3,0.01').  When set, training runs "
                             "once per prior and precision/recall vs tau are "
                             "reported on the labeled eval sample.")
    parser.add_argument("--prior-sweep-taus", default="0.5,0.7,0.85,0.9,0.99",
                        help="comma-separated tau grid for the prior sweep")
    parser.add_argument("--output", default="results/bulk_latency_em.json")
    args = parser.parse_args()

    from benchmark_data import load_records, require_compared_fields

    print(f"Loading dataset from {args.data_file} ...")
    records = load_records(args.data_file, key=args.data_key)
    require_compared_fields(records, ["first_name", "last_name", "date_of_birth", "email", "address"])
    print(f"Dataset: {len(records):,} records")

    twin_entities: dict[int, int] = {}
    if args.gt_file:
        raw_gt = json.loads(Path(args.gt_file).read_text(encoding="utf-8"))
        for k, v in raw_gt.items():
            try:
                twin_entities[int(k)] = int(v)
            except (TypeError, ValueError):
                twin_entities[k] = v

    # --- EM training on the duplicate-bearing population -------------------
    if args.n_training_subsample is not None and args.n_training_subsample < len(records):
        rng = random.Random(args.seed)
        em_data = rng.sample(records, min(args.n_training_subsample, len(records)))
        print(f"EM training on a subsample of {len(em_data):,} records ...")
    else:
        em_data = records
        print(f"EM training on all {len(records):,} records ...")

    block_on = args.em_block_on if args.em_block_on else [("first_name",), ("date_of_birth",)]
    em_seed = args.em_seed if args.em_seed is not None else args.seed + 313
    n_procs = args.n_procs if args.n_procs else None
    t0 = time.perf_counter()
    scorer = em_train(
        em_data,
        make_comparisons(),
        training_block_on=block_on,
        max_pairs=args.em_max_pairs,
        recall=args.em_recall,
        em_seed=em_seed,
        tau=args.tau,
    )
    em_seconds = time.perf_counter() - t0
    em_diagnostics = scorer.to_settings()
    print(f"EM training done in {em_seconds:.1f}s; learned prior = "
          f"{em_diagnostics['probability_two_random_records_match']:.3g}")

    # --- prior sweep mode (calibration-paradox remedy) ---------------------
    if args.prior_sweep_priors:
        if not args.gt_file:
            raise SystemExit("--prior-sweep-priors requires --gt-file "
                             "(record index -> true twin base index)")
        priors = [float(x) for x in args.prior_sweep_priors.split(",")]
        taus = [float(x) for x in args.prior_sweep_taus.split(",")]
        # Build labelled pairs: true (dup, base) pairs and an equal set of
        # non-match pairs sampled from the population.
        gt = {}
        raw = json.loads(Path(args.gt_file).read_text(encoding="utf-8"))
        for k, v in raw.items():
            try:
                gt[int(k)] = int(v)
            except (TypeError, ValueError):
                gt[k] = v
        rng = random.Random(args.seed)
        gt_pairs = []
        for a, b in gt.items():
            gt_pairs.append((a, b, 1))
        # Non-matches: random index pairs not in gt.
        n_pos = len(gt_pairs)
        neg = 0
        index_all = set(range(len(records)))
        while neg < n_pos and neg < 200_000:
            a = rng.randrange(len(records))
            b = rng.randrange(len(records))
            if a == b or gt.get(a) == b or gt.get(b) == a:
                continue
            gt_pairs.append((a, b, 0))
            neg += 1
        print(f"Prior sweep: {len(priors)} priors x {len(taus)} taus, "
              f"{n_pos} positive + {neg} negative eval pairs")
        sweep = prior_sweep(
            records, gt_pairs, priors, taus, seed=args.seed,
            em_max_pairs=args.em_max_pairs,
        )
        results = {
            "parameters": {
                "mode": "prior_sweep",
                "data_file": args.data_file,
                "gt_file": args.gt_file,
                "training_subsample": len(em_data),
                "em_max_pairs": args.em_max_pairs,
                "em_seed": em_seed,
                "priors": priors,
                "taus": taus,
                "seed": args.seed,
            },
            **sweep,
            "environment": environment_block(),
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print("\nprior_sweep rows (best by F1):")
        best = max(sweep["prior_sweep_rows"],
                   key=lambda r: (r["precision"] + r["recall"]))  # noqa: max(...)
        for r in sweep["prior_sweep_rows"]:
            f1 = 2 * r["precision"] * r["recall"] / (r["precision"] + r["recall"]) if (r["precision"] + r["recall"]) else 0
            tag = " <--" if r["tau"] == best["tau"] and r["fixed_prior"] == best["fixed_prior"] else ""
            print(f"  prior={r['fixed_prior']:g} tau={r['tau']} "
                  f"precision={r['precision']} recall={r['recall']} f1={f1:.3f}{tag}")
        print(f"\nSaved prior-sweep results to {args.output}")
        return

    # --- bulk dedup with the EM-trained scorer ------------------------------
    # Keep the heavy dedup tractable: option to sample the evaluation set.
    bench_records = records
    if args.bench_limit is not None and args.bench_limit < len(records):
        rng = random.Random(args.seed + 1)
        bench_records = rng.sample(records, min(args.bench_limit, len(records)))
        # Re-map ground-truth twins (indices are into `records`, so only valid
        # when eval == full file; otherwise report counts without GT mapping).
        twin_entities = {}
        print(f"Bulk benchmark on a deterministic subsample of {len(bench_records):,} "
              f"records (ground-truth mapping disabled for a subsample)")

    n_canopies = min(args.n_canopies, max(1, len(bench_records) // 39))
    from vectorer.clustering import union_merge

    merge_fn = union_merge if args.merge == "union" else None

    import time as _t

    t0 = _t.perf_counter()
    if n_procs and n_procs > 1:
        print(f"Running distributed batch pipeline (canopies={n_canopies}, "
              f"overlap={args.overlap}, tau={args.tau}, EM-scorer, "
              f"n_procs={n_procs})...")
        quality = cluster_quality_distributed(
            bench_records, twin_entities, args.seed, scorer,
            n_canopies, args.overlap, args.tau, n_procs,
        )
    else:
        print(f"Running batch pipeline (canopies={n_canopies}, overlap={args.overlap}, "
              f"tau={args.tau}, embedder={args.embedder}, merge={args.merge}, EM-scorer)...")
        pipeline = BatchPipeline(
            embedder=build_embedder(args.embedder),
            scorer=scorer,
            n_canopies=n_canopies,
            overlap_m=args.overlap,
            canopy_seed=args.seed,
            tau=args.tau,
            merge=merge_fn,
        )
        quality = cluster_quality(pipeline, bench_records, twin_entities, args.seed)
    quality["wall_seconds"] = round(_t.perf_counter() - t0, 4)

    results = {
        "parameters": {
            "total_records": len(records),
            "duplicate_pairs_planted": len(twin_entities),
            "data_file": args.data_file,
            "data_key": args.data_key,
            "gt_file": args.gt_file,
            "training_subsample": len(em_data),
            "bench_limit": len(bench_records),
            "em_max_pairs": args.em_max_pairs,
            "em_recall": args.em_recall,
            "em_seed": em_seed,
            "em_block_on": block_on,
            "n_canopies": n_canopies,
            "overlap": args.overlap,
            "tau": args.tau,
            "seed": args.seed,
            "embedder": args.embedder,
            "merge": args.merge,
            "n_procs": n_procs if n_procs else 1,
        },
        "em_training_seconds": round(em_seconds, 4),
        "em_learned_prior": em_diagnostics["probability_two_random_records_match"],
        "em_learned_comparisons": [
            {
                "type": c["type"],
                "m_sum": round(sum(lv.get("m_probability", 0) for lv in c["levels"] if not lv.get("is_null_level")), 4),
            }
            for c in em_diagnostics["comparisons"]
        ],
        "quality": quality,
        "environment": environment_block(),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(quality, indent=2))
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()