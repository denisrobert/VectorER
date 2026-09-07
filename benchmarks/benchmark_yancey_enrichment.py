"""Benchmark: Yancey's match-enrichment procedure for improving EM.

Yancey 2004 (<a href="../.source-papers/20_yancey_em_estimates_2004.pdf">RRS #2004-01</a>)
observed that when the true match class M is sparse in the pair pool, EM fails
to detect it ("when the proportion ... drops below 0.05 or so, the EM algorithm
can converge to parameter values that are not relevant to record linkage").  His
fix: **enrich the training set** -- compute preliminary weights, keep
high-weight (likely-match) pairs, drop low-weight pairs, retrain EM on the
enriched subset, then recalibrate for the full set.

This benchmark tests that procedure on the generated duplicate-bearing
population (base + ~7% perturbed twins), comparing five calibration strategies
on the *same* labelled evaluation pairs:

* ``plain_em``          -- fit_em on the full population (the sparse-M baseline).
* ``yancey_enrich``     -- record-level match-enrichment: keep records of the
                         highest-weight pairs, fit_em on the enriched subset,
                         then `recalibrate_prior(records)` corrects the
                         inflated enriched-set prior back to the full set.
* ``yancey_fixedprior`` -- same enriched m/u, but the prior is FROZEN at a
                         swept value (`fixed_prior=`) instead of EM's own.
* ``oracle_prior``      -- enriched m/u with the TRUE full-set match proportion
                         (ground truth; the upper-bound reference).
* ``default_mu``        -- no training at all (sanity reference).

For each we report the fitted ``m/u`` contamination signal, the learned prior,
and precision/recall/F1 at a threshold grid on the labelled eval pairs, plus
the best-F1 operating point -- so you can see whether (and how much)
enrichment + prior recalibration fixes the unmoored-prior failure.

Usage::

    python benchmarks/benchmark_yancey_enrichment.py \\
        --data-file benchmarks/population_with_duplicates.json \\
        --gt-file benchmarks/population_gt.json \\
        --n-training 60000 --n-eval-pairs 10000 \\
        --enrich-keep-frac 0.05 --em-max-pairs 80000 \\
        --output results/yancey_enrichment.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from vectorer.comparisons import make_comparison
from vectorer.scoring import FellegiSunterScorer

TAUS = [0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99]


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


def prf(y: np.ndarray, probs: np.ndarray, tau: float) -> dict:
    pred = probs >= tau
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"tau": tau, "tp": tp, "fp": fp, "fn": fn,
            "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)}


def eval_scorer(scorer: FellegiSunterScorer, eval_left, eval_right, y):
    probs = scorer.score_pairs(eval_left, eval_right)
    rows = [prf(y, probs, t) for t in TAUS]
    best = max(rows, key=lambda d: d["f1"])
    return rows, best, {"prior": float(
        scorer.to_settings()["probability_two_random_records_match"])}


def enrich_records(records, score_pool_idx, weights, keep_frac, seed):
    """Record-level Yancey enrichment: keep the records that participate in
    the highest-weight pairs (a weight-sorted keep of ``keep_frac`` of the
    pair pool's records)."""
    order = np.argsort(weights)[::-1]
    n_keep = max(1, int(len(order) * keep_frac))
    keep_idx = set()
    for k in order[:n_keep]:
        i, j = score_pool_idx[k]
        keep_idx.add(i)
        keep_idx.add(j)
    return [records[i] for i in sorted(keep_idx)]


def main():
    parser = argparse.ArgumentParser(
        description="Test Yancey match-enrichenment (record-level) for improving EM"
    )
    parser.add_argument("--data-file", default="benchmarks/population_with_duplicates.json")
    parser.add_argument("--gt-file", default="benchmarks/population_gt.json")
    parser.add_argument("--n-training", type=int, default=60_000)
    parser.add_argument("--n-eval-pairs", type=int, default=10_000)
    parser.add_argument("--em-max-pairs", type=float, default=80_000)
    parser.add_argument("--enrich-keep-frac", type=float, default=0.05,
                        help="fraction of highest-weight pairs kept for enrichment")
    parser.add_argument("--fixed-prior", type=float, default=1e-3,
                        help="frozen prior used by the yancey_fixedprior arm")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/yancey_enrichment.json")
    args = parser.parse_args()
    rng = random.Random(args.seed)
    rng2 = random.Random(args.seed + 7)

    from benchmark_data import load_records

    records = load_records(args.data_file)
    gt = {int(k): int(v) for k, v in json.loads(Path(args.gt_file).read_text(encoding="utf-8")).items()}
    n = len(records)
    print(f"population: {n:,} records, {len(gt):,} ground-truth twins")

    # ---- training population subsample ------------------------------------
    train_idx = rng.sample(range(n), min(args.n_training, n))
    train = [records[i] for i in train_idx]
    print(f"training pool: {len(train):,} records")

    # ---- labelled evaluation pairs ----------------------------------------
    gt_items = list(gt.items())
    rng2.shuffle(gt_items)
    ev = []
    for a, b in gt_items:
        if len(ev) >= args.n_eval_pairs // 2:
            break
        ev.append((records[a], records[b], 1))
    while len(ev) < args.n_eval_pairs:
        a = rng2.randrange(n)
        b = rng2.randrange(n)
        if a == b:
            continue
        ev.append((records[a], records[b], 0))
    ev_left = [x[0] for x in ev]
    ev_right = [x[1] for x in ev]
    y = np.asarray([x[2] for x in ev], dtype=int)
    print(f"eval: {len(ev)} pairs ({int((y==1).sum())} pos, {int((y==0).sum())} neg)")

    base = FellegiSunterScorer.from_comparisons(make_comparisons(), threshold=0.85)

    # ---- preliminary weights over a training pair pool --------------------
    pool_idx = [(i, j) for i in range(len(train)) for j in range(i + 1, min(len(train), i + 40))]
    rng.shuffle(pool_idx)
    pool_idx = pool_idx[: max(1, int(args.em_max_pairs))]
    pl = [train[i] for i, _ in pool_idx]
    pr = [train[j] for _, j in pool_idx]
    print(f"preliminary pair pool: {len(pool_idx):,}")

    def run_arm(name, scorer_factory):
        scorer, extra = scorer_factory()
        rows, best, meta = eval_scorer(scorer, ev_left, ev_right, y)
        if extra:
            meta.update(extra)
        print(f"  [{name}] prior={meta['prior']:.3g} "
              f"bestF1 @tau={best['tau']} F1={best['f1']} "
              f"P={best['precision']} R={best['recall']}")
        return {"rows": rows, "best": best, "meta": meta}

    results = {}

    # 1. plain EM on the full (sparse) training population.
    def _plain():
        sc = base.fit_em(train, training_block_on=[("first_name",), ("date_of_birth",)],
                         max_pairs=args.em_max_pairs, recall=0.7, seed=args.seed)
        return sc, {}
    results["plain_em"] = run_arm("plain_em", _plain)

    # 2. Yancey enrichment: preliminary weights (default m/u) on train pairs,
    #    keep records of highest-weight pairs, retrain EM, recalibrate prior.
    pre = base.score_pairs(pl, pr)
    enriched = enrich_records(train, pool_idx, pre, args.enrich_keep_frac, args.seed)
    print(f"enriched training subset: {len(enriched):,} records "
          f"({args.enrich_keep_frac:.0%} of pair-weight records)")
    enriched_prior_holder = {"enriched_prior": None}
    def _enriched():
        sc = base.fit_em(enriched,
                         training_block_on=[("first_name",), ("date_of_birth",)],
                         max_pairs=args.em_max_pairs, recall=0.7, seed=args.seed)
        enriched_prior_holder["enriched_prior"] = sc.to_settings()[
            "probability_two_random_records_match"]
        recal = sc.recalibrate_prior(records,
                                     sample_size=min(args.em_max_pairs, 200_000),
                                     seed=args.seed)
        return recal, dict(enriched_prior_holder)
    results["yancey_enrich"] = run_arm("yancey_enrich", _enriched)

    # 3. Yancey m/u, prior frozen at a swept value (fixed_prior).
    def _fixed():
        sc = base.fit_em(enriched,
                         training_block_on=[("first_name",), ("date_of_birth",)],
                         max_pairs=args.em_max_pairs, recall=0.7,
                         seed=args.seed, fixed_prior=args.fixed_prior)
        return sc, {"fixed_prior": args.fixed_prior}
    results["yancey_fixedprior"] = run_arm("yancey_fixedprior", _fixed)

    # 4. Oracle: enriched m/u, prior = true full-set match share.
    true_prior = len(gt) / (n * (n - 1) / 2)
    def _oracle():
        sc = base.fit_em(enriched,
                         training_block_on=[("first_name",), ("date_of_birth",)],
                         max_pairs=args.em_max_pairs, recall=0.7,
                         seed=args.seed, fixed_prior=float(true_prior))
        return sc, {"true_prior": round(float(true_prior), 8)}
    results["oracle_prior"] = run_arm("oracle_prior", _oracle)

    # 5. default m/u (no training).
    def _default():
        return base, {"prior": base.prior}
    results["default_mu"] = run_arm("default_mu", _default)

    out = {
        "parameters": {
            "data_file": args.data_file, "gt_file": args.gt_file,
            "n_training": len(train), "n_enriched": len(enriched),
            "n_eval_pairs": len(ev), "em_max_pairs": args.em_max_pairs,
            "enrich_keep_frac": args.enrich_keep_frac,
            "fixed_prior": args.fixed_prior, "seed": args.seed,
        },
        "arms": results,
        "note": "Yancey record-level enrichment; prior recalibration via "
                "recalibrate_prior or fixed_prior.",
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()