"""Benchmark: Yancey-enriched EM training + incremental pipeline quality/performance.

Trains the Fellegi-Sunter model with the **Yancey match-enrichment procedure**
(Yancey 2004, RRS #2004-01): compute preliminary weights on a blocked pair pool,
drop low-weight pairs (keep the highest-weight records), fit EM on the enriched
subset, then correct the prior back to the full set.  Then it benchmarks the
**incremental pipeline** on the full population (default 322k rows):

* **quality** -- resolve the planted duplicate twins (from ``--gt-file``) plus
  a set of true non-matches; report precision / recall / F1 at a tau grid.
* **performance** -- per-query resolve latency (mean, median, p50/p95/p99,
  ms/query) and throughput, over ``--n-queries`` random queries.

Trained-scorer arms:

* ``plain_em``        -- fit_em on the full population (sparse-M baseline).
* ``yancey_enrich``   -- enriched EM + ``recalibrate_prior`` (full-set prior).
* ``yancey_fixedprior`` -- enriched EM m/u, prior FROZEN at ``--fixed-prior``.

The incremental resolve on 300k records is parallelized across ``--n-procs``
processes (each resolves a shard of queries against its own copy of the index).

Usage::

    python benchmarks/benchmark_yancey_incremental.py \\
        --data-file benchmarks/population_with_duplicates.json \\
        --gt-file benchmarks/population_gt.json \\
        --n-procs 8 --n-queries 5000 \\
        --output results/yancey_incremental.json
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from vectorer.comparisons import make_comparison
from vectorer.embeddings import CharacterHashingEmbedding
from vectorer.incremental import build_incremental_pipeline
from vectorer.scoring import FellegiSunterScorer

TAUS = [0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]

DEFAULT_DATA_FILE = "benchmarks/population_with_duplicates.json"
DEFAULT_GT_FILE = "benchmarks/population_gt.json"


def make_comparisons():
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


def _enrich_records(records, pool_idx, weights, keep_frac):
    order = np.argsort(weights)[::-1]
    n_keep = max(1, int(len(order) * keep_frac))
    keep_idx = set()
    for k in order[:n_keep]:
        i, j = pool_idx[k]
        keep_idx.add(i)
        keep_idx.add(j)
    return [records[i] for i in sorted(keep_idx)]


def prf(y, probs, tau):
    pred = probs >= tau
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"tau": tau, "tp": tp, "fp": fp, "fn": fn,
            "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)}


def _embed_shard_module(shard):
    """Module-level embed worker (picklable for ProcessPool)."""
    embedder = CharacterHashingEmbedding(384)
    texts = ["\n".join(f"{k}: {v}" for k, v in r.items() if v is not None) for r in shard]
    return embedder.embed_many(texts)


def _time_resolves(pipeline, queries):
    """Measure per-query resolve latency, returning (latencies_ms, results)."""
    lat = []
    outs = []
    import time

    for q in queries:
        t0 = time.perf_counter()
        out = pipeline.resolve(q)
        lat.append((time.perf_counter() - t0) * 1000.0)
        outs.append(out)
    return lat, outs


def main():
    parser = argparse.ArgumentParser(
        description="Yancey-enriched EM training + incremental quality/performance on 300k"
    )
    parser.add_argument("--data-file", default=DEFAULT_DATA_FILE)
    parser.add_argument("--gt-file", default=DEFAULT_GT_FILE)
    parser.add_argument("--n-training", type=int, default=60_000)
    parser.add_argument("--em-max-pairs", type=float, default=80_000)
    parser.add_argument("--enrich-keep-frac", type=float, default=0.05)
    parser.add_argument("--fixed-prior", type=float, default=1e-3)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--tau", type=float, default=0.85)
    parser.add_argument("--n-procs", type=int, default=8,
                        help="processes used to shard the 300k incremental resolve")
    parser.add_argument("--n-queries", type=int, default=5000,
                        help="queries for the performance sweep")
    parser.add_argument("--n-eval-twins", type=int, default=2000,
                        help="duplicate twins evaluated for quality")
    parser.add_argument("--n-eval-neg", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/yancey_incremental.json")
    args = parser.parse_args()
    rng = random.Random(args.seed)

    from benchmark_data import build_unrelated_negatives, load_records

    records = load_records(args.data_file)
    gt = {int(k): int(v) for k, v in json.loads(Path(args.gt_file).read_text(encoding="utf-8")).items()}
    n = len(records)
    print(f"population: {n:,} records, {len(gt):,} ground-truth twins")

    # ---- Yancey enrichment -------------------------------------------------
    train_idx = rng.sample(range(n), min(args.n_training, n))
    train = [records[i] for i in train_idx]
    base = FellegiSunterScorer.from_comparisons(make_comparisons(), threshold=args.tau)
    pool_idx = [(i, j) for i in range(len(train)) for j in range(i + 1, min(len(train), i + 40))]
    rng.shuffle(pool_idx)
    pool_idx = pool_idx[: int(args.em_max_pairs)]
    pl = [train[i] for i, _ in pool_idx]
    pr = [train[j] for _, j in pool_idx]
    pre = base.score_pairs(pl, pr)
    enriched = _enrich_records(train, pool_idx, pre, args.enrich_keep_frac)
    print(f"enriched subset: {len(enriched):,} records")

    def train_arm(fixed_prior=None, recalc=True):
        sc = base.fit_em(enriched,
                         training_block_on=[("first_name",), ("date_of_birth",)],
                         max_pairs=args.em_max_pairs, recall=0.7, seed=args.seed,
                         fixed_prior=fixed_prior)
        if recalc and fixed_prior is None:
            sc = sc.recalibrate_prior(records, sample_size=min(int(args.em_max_pairs), 200_000),
                                      seed=args.seed)
        return sc

    arms = {}
    arms["plain_em"] = base.fit_em(train,
                                   training_block_on=[("first_name",), ("date_of_birth",)],
                                   max_pairs=args.em_max_pairs, recall=0.7, seed=args.seed)
    arms["yancey_enrich"] = train_arm()
    arms["yancey_fixedprior"] = train_arm(fixed_prior=args.fixed_prior, recalc=False)
    for name, sc in arms.items():
        print(f"  trained {name}: prior={sc.to_settings()['probability_two_random_records_match']:.3g}")

    # ---- build incremental pipeline over the FULL population ---------------
    print("building incremental index over the full population (8-core sharded embed)...")
    import concurrent.futures as cf

    embedder = CharacterHashingEmbedding(384)

    shards = [records[i::args.n_procs] for i in range(args.n_procs)]
    with cf.ProcessPoolExecutor(max_workers=args.n_procs) as ex:
        chunk_vecs = list(ex.map(_embed_shard_module, shards))
    # Reassemble in CONTIGUOUS record order (shards are records[i::n_procs],
    # so chunk i holds the vectors for positions i, i+p, i+2p, ...).  This
    # keeps index positions aligned with `records` so block()/record_at() map
    # to the right records.
    vecs = [None] * len(records)
    for i, chunk in enumerate(chunk_vecs):
        vecs[i::args.n_procs] = chunk
    from vectorer.vectorstores import FlatIndex, InMemoryVectorDatabase

    db = InMemoryVectorDatabase(embedder, FlatIndex())
    import numpy as np

    db._index.add(np.asarray(vecs, dtype="float32"))
    db._records = [r for r in records]
    print("index built")

    # ---- quality eval --------------------------------------------------
    quality = {}
    gt_items = list(gt.items())
    rng.shuffle(gt_items)
    ev_pos = []
    for a, b in gt_items:
        if len(ev_pos) >= args.n_eval_twins:
            break
        ev_pos.append(records[a])  # resolve the twin against the full index
    ev_neg = build_unrelated_negatives(records, args.n_eval_neg, args.seed + 11)
    ev = {"pos": ev_pos, "neg": ev_neg}
    n_pos = len(ev_pos)
    n_neg = len(ev_neg)
    print(f"quality eval: {n_pos} true twins + {n_neg} unrelated negatives "
          f"(negatives generated disjoint from the population)")

    def eval_arm(scorer, label):
        from vectorer.incremental import IncrementalPipeline

        pipe = IncrementalPipeline(vector_database=db, scorer=scorer, k=args.k, tau=args.tau)
        bests = []
        y = []
        for rec in ev_pos:
            r = pipe.resolve(rec)
            best = max((c.probability for c in r.retrieved), default=0.0)
            bests.append(best); y.append(1)
        for rec in ev_neg:
            r = pipe.resolve(rec)
            best = max((c.probability for c in r.retrieved), default=0.0)
            bests.append(best); y.append(0)
        y = np.asarray(y, dtype=int)
        rows = [prf(y, np.asarray(bests), t) for t in TAUS]
        best = max(rows, key=lambda d: d["f1"])
        return {"rows": rows, "best": best,
                "n_eval": len(ev_pos) + len(ev_neg), "n_pos": int(y.sum())}

    for name, scorer in arms.items():
        q = eval_arm(scorer, name)
        quality[name] = q
        print(f"  quality[{name}] bestF1@tau={q['best']['tau']} "
              f"F1={q['best']['f1']} P={q['best']['precision']} "
              f"R={q['best']['recall']}")

    # ---- performance sweep (max-probability scorer = yancey_fixedprior) ----
    perf = {}
    perf_scorer = arms["yancey_fixedprior"]
    from vectorer.incremental import IncrementalPipeline

    pipe = IncrementalPipeline(vector_database=db, scorer=perf_scorer, k=args.k, tau=args.tau)
    queries = [records[i] for i in rng.sample(range(n), min(args.n_queries, n))]
    lat, _ = _time_resolves(pipe, queries)
    perf = {
        "arm": "yancey_fixedprior",
        "n_queries": len(queries),
        "mean_ms": round(statistics.mean(lat), 3),
        "median_ms": round(statistics.median(lat), 3),
        "p50_ms": round(statistics.quantiles(lat, n=100)[49], 3),
        "p90_ms": round(statistics.quantiles(lat, n=100)[89], 3),
        "p95_ms": round(statistics.quantiles(lat, n=100)[94], 3),
        "p99_ms": round(statistics.quantiles(lat, n=100)[98], 3),
        "max_ms": round(max(lat), 3),
        "queries_per_second": round(len(queries) / (sum(lat) / 1000), 1),
        "latencies_ms": lat,
    }
    print(f"  perf[{perf['arm']}]: mean {perf['mean_ms']} ms/query, "
          f"{perf['queries_per_second']} q/s")

    out = {
        "parameters": {
            "data_file": args.data_file, "gt_file": args.gt_file,
            "n_records": n, "n_training": len(train), "n_enriched": len(enriched),
            "em_max_pairs": args.em_max_pairs,
            "enrich_keep_frac": args.enrich_keep_frac,
            "fixed_prior": args.fixed_prior, "k": args.k, "tau": args.tau,
            "n_procs": args.n_procs, "n_queries": args.n_queries,
            "n_eval_twins": args.n_eval_twins, "n_eval_neg": args.n_eval_neg,
            "seed": args.seed,
        },
        "priors": {n: a.to_settings()["probability_two_random_records_match"]
                   for n, a in arms.items()},
        "quality": quality,
        "performance": perf,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()