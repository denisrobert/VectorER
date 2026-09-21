"""Bound the Fellegi-Sunter prior sweep with capture-recapture (Lincoln-Petersen).

Two **independent** linkage passes (different field subsets) score a pair
arena; their match tallies ``n1``/``n2`` and the recapture overlap ``m``
feed :func:`~vectorer.scoring.estimate_prior_capture_recapture`, which returns
a distribution-free estimate of the number of matches IN THE ARENA and a
confidence interval.  The interval bounds the ``fit_em(fixed_prior=...)``
sweep (the "calibration-paradox" remedy): the benchmark sweeps the band
x the tau grid and reports the best F1 within the band, alongside the
EM-learned-prior and default-prior baselines at the same taus.

Usage::

    python benchmarks/benchmark_lp_prior_sweep.py \\
        --data-file benchmarks/population_with_duplicates.json \\
        --gt-file benchmarks/population_gt.json \\
        --capture-pairs 300000 --capture-tau 0.9 \\
        --output results/lp_prior_sweep.json

The arena is the pair domain the two runs enumerate:

* ``blocked`` (default) -- the **candidate pair space** the framework's own
  blocking produces (:func:`~vectorer.scoring._candidates.build_blocked_pairs`,
  the same generator EM trains on).  True matches are dense there (the whole
  point of blocking), so a modest arena yields a healthy overlap: on the
  default 300k population, blocking on ``date_of_birth`` concentrates the
  ~22k duplicates into a few-million-pair candidate space.  The match count
  the estimator returns is the count WITHIN the blocked space, so the
  full-file prior is ``N_arena / block_recall / C(n, 2)`` -- blocking recall
  is the same knob ``fit_em(recall=...)`` / ``recalibrate_prior(recall=...)``
  already carry (default 0.7).
* ``uniform`` -- a uniform sample of the full pair domain (only sound when
  true matches are not a microscopic fraction of pairs, e.g. small files or
  high-overlap two-file linkage); needs ``--capture-pairs`` around
  ``m_needed * C(n, 2) / true_matches`` -- on the default 300k population
  that is ~10^7+ pairs and the captures are false-positive-dominated, so
  prefer ``blocked`` there.

The two passes are intentionally built on **orthogonal** evidence, and neither
run uses the arena's block key (so its evidence varies inside the arena):

* run A: first_name + last_name,
* run B: first_name + address.

Independence is what makes the overlap informative; with a single model or
nested thresholds the tallies are correlated and the prior is biased upward.
A small overlap (``m < 7``) triggers the estimator's warning -- the band is
too wide to trust.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from vectorer.comparisons import make_comparison
from vectorer.scoring import (
    DEFAULT_PRIOR,
    FellegiSunterScorer,
    estimate_prior_capture_recapture,
)

DEFAULT_DATA_FILE = "benchmarks/population_with_duplicates.json"


def _clip_prior(x: float) -> float:
    return float(min(max(x, 1e-8), 0.5))


def comparisons_for(names: Sequence[str]) -> list:
    builders = {
        "first_name": lambda: make_comparison(
            "jaro_winkler_at_thresholds", col_name="first_name",
            score_threshold_or_thresholds=[0.9, 0.8, 0.7],
        ),
        "last_name": lambda: make_comparison(
            "jaro_winkler_at_thresholds", col_name="last_name",
            score_threshold_or_thresholds=[0.9, 0.8, 0.7],
        ),
        "date_of_birth": lambda: make_comparison(
            "date_of_birth_comparison", col_name="date_of_birth",
        ),
        "email": lambda: make_comparison("email_comparison", col_name="email"),
        "address": lambda: make_comparison(
            "jaro_winkler_at_thresholds", col_name="address",
            score_threshold_or_thresholds=[0.85, 0.75, 0.65],
        ),
    }
    return [builders[name]() for name in names]


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def _score_rows(scorer, lefts, rights, y, taus) -> list[dict]:
    probs = scorer.score_pairs(lefts, rights)
    rows = []
    for tau in taus:
        pred = probs >= tau
        tp = int((pred & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum())
        fn = int((~pred & (y == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        rows.append({
            "tau": tau,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(_f1(precision, recall), 4),
        })
    return rows


def _best_row(rows) -> Optional[dict]:
    return max(rows, key=lambda r: r.get("f1", 0.0)) if rows else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lincoln-Petersen prior bounds -> fixed_prior sweep -> best F1"
    )
    parser.add_argument("--data-file", default=DEFAULT_DATA_FILE,
                        help=f"dataset to train and evaluate on (default {DEFAULT_DATA_FILE})")
    parser.add_argument("--data-key", default=None,
                        help="when --data-file is a single JSON object, the key holding the records list")
    parser.add_argument("--gt-file", default=None, required=True,
                        help="JSON object mapping record index -> true duplicate-twin base index")
    parser.add_argument("--output", default="results/lp_prior_sweep.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--arena-mode", choices=["blocked", "uniform"], default="blocked",
                        help="pair domain the two capture runs enumerate: 'blocked' "
                             "(default; the framework's candidate pair space -- dense "
                             "matches, cheap arena) or 'uniform' (full pair domain -- "
                             "only for small files / high-overlap two-file linkage)")
    parser.add_argument("--block-on", nargs="*", default=["date_of_birth"],
                        help="blocking fields for the blocked arena (default "
                             "date_of_birth: concentrates duplicates ~4 orders of "
                             "magnitude over the full domain on the demo population)")
    parser.add_argument("--block-recall", type=float, default=0.7,
                        help="blocking recall used to convert the within-arena match "
                             "count to a full-file prior (matches the fit_em recall "
                             "knob)")
    parser.add_argument("--capture-pairs", type=int, default=300_000,
                        help="arena size scored by both capture runs (candidate pairs "
                             "in blocked mode, uniformly-sampled pairs in uniform "
                             "mode)")
    parser.add_argument("--capture-tau", type=float, default=0.9,
                        help="posterior threshold each capture run uses to declare a match")
    parser.add_argument("--confidence", type=float, default=0.95,
                        help="two-sided coverage of the capture-recapture interval")
    parser.add_argument("--sweep-points", type=int, default=3,
                        help="prior grid points spread log-linearly inside the LP interval "
                             "(in addition to the point estimate)")
    parser.add_argument("--prior-sweep-taus", default="0.5,0.7,0.85,0.9,0.99",
                        help="comma-separated tau grid for the prior sweep")
    parser.add_argument("--em-max-pairs", type=float, default=1e6)
    parser.add_argument("--em-recall", type=float, default=0.7)
    parser.add_argument("--em-seed", type=int, default=None)
    parser.add_argument("--enrich-keep-frac", type=float, default=0.05,
                        help="Yancey match-enrichment fraction for every EM fit")
    parser.add_argument("--recalibration-method", choices=["yancey", "empirical"],
                        default="yancey")
    parser.add_argument("--training-limit", type=int, default=None,
                        help="optional: train the two capture runs on this many rows "
                             "(deterministic sample); eval stays on the full file")
    args = parser.parse_args()

    from benchmark_data import (
        build_unrelated_negatives,
        load_records,
        require_compared_fields,
    )
    from benchmark_bulk_er_em import em_train, prior_sweep

    print(f"Loading dataset from {args.data_file} ...")
    records = load_records(args.data_file, key=args.data_key)
    require_compared_fields(
        records, ["first_name", "last_name", "date_of_birth", "email", "address"]
    )
    print(f"Dataset: {len(records):,} records")

    gt: dict = {}
    raw_gt = json.loads(Path(args.gt_file).read_text(encoding="utf-8"))
    for k, v in raw_gt.items():
        try:
            gt[int(k)] = int(v)
        except (TypeError, ValueError):
            gt[k] = v

    em_seed = args.em_seed if args.em_seed is not None else args.seed + 313
    train_data = records
    if args.training_limit is not None and args.training_limit < len(records):
        train_data = random.Random(args.seed + 7).sample(records, args.training_limit)
        print(f"Training the capture runs on a subsample of {len(train_data):,} records")

    # --- two independent capture runs (orthogonal field evidence; neither
    # run uses the arena's block key, so its evidence varies inside the arena)
    run_a = comparisons_for(["first_name", "last_name"])
    run_b = comparisons_for(["first_name", "address"])
    print("Training capture run A (first_name+last_name) ...")
    t0 = time.perf_counter()
    scorer_a = em_train(
        train_data, run_a,
        training_block_on=[("first_name",), ("date_of_birth",)],
        max_pairs=args.em_max_pairs, recall=args.em_recall, em_seed=em_seed,
        tau=args.capture_tau, enrich_keep_frac=args.enrich_keep_frac,
        recalibration_method=args.recalibration_method,
    )
    print(f"  done in {time.perf_counter() - t0:.1f}s")
    print("Training capture run B (first_name+address) ...")
    t0 = time.perf_counter()
    scorer_b = em_train(
        train_data, run_b,
        training_block_on=[("first_name",), ("date_of_birth",)],
        max_pairs=args.em_max_pairs, recall=args.em_recall, em_seed=em_seed,
        tau=args.capture_tau, enrich_keep_frac=args.enrich_keep_frac,
        recalibration_method=args.recalibration_method,
    )
    print(f"  done in {time.perf_counter() - t0:.1f}s")

    # --- capture-recapture over the pair arena ----------------------------
    from vectorer.scoring._candidates import _sample_all_pairs, build_blocked_pairs

    n = len(records)
    total_full_pairs = n * (n - 1) // 2
    if args.arena_mode == "blocked":
        arena = build_blocked_pairs(
            records, [tuple(args.block_on)], int(args.capture_pairs),
            np.random.default_rng(args.seed),
        )
        print(f"blocked arena ({args.block_on}) from the framework's candidate "
              f"space: {len(arena):,} pairs")
    else:
        arena = _sample_all_pairs(n, int(args.capture_pairs),
                                  np.random.default_rng(args.seed))
        print(f"uniform arena: {len(arena):,} pairs of "
              f"{total_full_pairs:,} full-domain pairs")
    if not arena:
        raise RuntimeError("empty capture arena; records < 2")
    left = [records[i] for i, _ in arena]
    right = [records[j] for _, j in arena]
    pa = np.asarray(scorer_a.score_pairs(left, right), dtype=float) >= args.capture_tau
    pb = np.asarray(scorer_b.score_pairs(left, right), dtype=float) >= args.capture_tau
    n1, n2, m = int(pa.sum()), int(pb.sum()), int((pa & pb).sum())
    print(f"captures: n1={n1} n2={n2} m={m} over {len(arena):,} arena pairs")
    if m == 0:
        raise RuntimeError(
            "no overlap between the two capture runs over this arena: widen the "
            "capture evidence (different fields / thresholds / more independent "
            "runs) or lower --capture-tau; in uniform mode increase "
            "--capture-pairs (true matches are a tiny fraction of the full "
            "pair domain)"
        )
    est = estimate_prior_capture_recapture(
        n1, n2, m, total_pairs=len(arena), confidence=args.confidence,
    )
    # Convert to the full-file prior.
    if args.arena_mode == "blocked":
        factor = 1.0 / max(args.block_recall, 1e-3) / total_full_pairs
        prior = _clip_prior(est.matches_estimate * factor)
        prior_ci = (
            _clip_prior(float(est.matches_ci[0]) * factor),
            _clip_prior(float(est.matches_ci[1]) * factor),
        )
        print(f"Lincoln-Petersen (within blocked arena {len(arena):,} pairs):")
        print(f"  matches ~{est.matches_estimate:,.0f} "
              f"[{est.matches_ci[0]:,.0f}, {est.matches_ci[1]:,.0f}]")
        print(f"  full-file prior (recall={args.block_recall}) ~ {prior:.3g} "
              f"[{prior_ci[0]:.3g}, {prior_ci[1]:.3g}]")
    else:
        prior, prior_ci = est.prior, est.prior_ci
        print(f"Lincoln-Petersen: {est}")

    # --- sweep grid from the LP interval ----------------------------------
    lo, hi = prior_ci
    grid = sorted(
        float(x)
        for x in np.logspace(np.log10(max(lo, 1e-9)), np.log10(hi), args.sweep_points)
    )
    grid = list(dict.fromkeys(grid + [float(prior)]))  # dedupe, keep order
    if len(grid) == 1:
        grid = [lo, prior, hi]
    print(f"prior sweep grid from the LP band: {[f'{g:.3g}' for g in grid]}")

    # --- labelled eval pairs (dup/base positives + unrelated negatives) ----
    gt_pairs = [(a, b, 1) for a, b in gt.items()]
    n_pos = len(gt_pairs)
    neg_pairs = build_unrelated_negatives(
        records, n_pos, args.seed + 11, pool_factor=4,
        pair_left_record=True, rng=random.Random(args.seed),
    )
    print(f"eval pairs: {n_pos} positive + {len(neg_pairs)} negative")
    taus = [float(x) for x in args.prior_sweep_taus.split(",")]

    lefts = [records[a] for a, b, _ in gt_pairs] + [a for a, _ in neg_pairs]
    rights = [records[b] for a, b, _ in gt_pairs] + [b for _, b in neg_pairs]
    y = np.asarray([m for _, _, m in gt_pairs] + [0] * len(neg_pairs), dtype=int)

    # --- sweep fixed_prior across the band --------------------------------
    print(f"Sweeping {len(grid)} priors x {len(taus)} taus inside the LP band ...")
    sweep = prior_sweep(
        records, gt_pairs, grid, taus, seed=args.seed,
        em_max_pairs=args.em_max_pairs, neg_pairs=neg_pairs,
        enrich_keep_frac=args.enrich_keep_frac,
    )
    for r in sweep["prior_sweep_rows"]:
        r["f1"] = round(_f1(r["precision"], r["recall"]), 4)

    # --- baselines ---------------------------------------------------------
    em_scorer = em_train(
        records, comparisons_for(
            ["first_name", "last_name", "date_of_birth", "email", "address"]
        ),
        training_block_on=[("first_name",), ("date_of_birth",)],
        max_pairs=args.em_max_pairs, recall=args.em_recall, em_seed=em_seed,
        tau=0.85, enrich_keep_frac=args.enrich_keep_frac,
        recalibration_method=args.recalibration_method,
    )
    default_scorer = FellegiSunterScorer.from_comparisons(
        comparisons_for(["first_name", "last_name", "date_of_birth", "email", "address"]),
        prior=DEFAULT_PRIOR,
    )
    em_rows = _score_rows(em_scorer, lefts, rights, y, taus)
    default_rows = _score_rows(default_scorer, lefts, rights, y, taus)

    best_lp = _best_row(sweep["prior_sweep_rows"])
    best_em = _best_row(em_rows)
    best_default = _best_row(default_rows)

    print("\nLP-band sweep rows (best by F1):")
    for r in sorted(sweep["prior_sweep_rows"], key=lambda r: -r["f1"]):
        print(f"  prior={r['fixed_prior']:g} tau={r['tau']} "
              f"precision={r['precision']} recall={r['recall']} f1={r['f1']}")
    print(f"\nbest F1 in LP band:      prior={best_lp['fixed_prior']:g} "
          f"tau={best_lp['tau']} f1={best_lp['f1']}")
    print(f"best F1 EM-learned prior: tau={best_em['tau']} f1={best_em['f1']} "
          f"(learned prior={em_scorer.prior:.3g})")
    print(f"best F1 default prior:    tau={best_default['tau']} f1={best_default['f1']} "
          f"(prior={default_scorer.prior:.3g})")

    results = {
        "parameters": {
            "mode": "lp_prior_sweep",
            "data_file": args.data_file,
            "gt_file": args.gt_file,
            "arena_mode": args.arena_mode,
            "block_on": list(args.block_on),
            "block_recall": args.block_recall,
            "total_full_pairs": total_full_pairs,
            "capture_pairs": len(arena),
            "capture_tau": args.capture_tau,
            "confidence": args.confidence,
            "sweep_points": args.sweep_points,
            "grid": grid,
            "taus": taus,
            "em_max_pairs": args.em_max_pairs,
            "em_recall": args.em_recall,
            "enrich_keep_frac": args.enrich_keep_frac,
            "recalibration_method": args.recalibration_method,
            "training_limit": len(train_data),
            "seed": args.seed,
        },
        "capture_recapture": {
            "n1": n1,
            "n2": n2,
            "m": m,
            "arena_pairs": len(arena),
            "matches_estimate": round(est.matches_estimate, 2),
            "matches_ci": [round(est.matches_ci[0], 2), round(est.matches_ci[1], 2)],
            "prior_full": prior,
            "prior_full_ci": [prior_ci[0], prior_ci[1]],
            "standard_error": round(est.standard_error, 2),
        },
        "prior_sweep_rows": sweep["prior_sweep_rows"],
        "best": {
            "lp_band": best_lp,
            "em_learned_prior": best_em,
            "default_prior": best_default,
        },
        "environment": {
            "python": sys.version.split()[0],
            "libs": {"numpy": np.__version__},
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved LP prior-sweep results to {args.output}")


if __name__ == "__main__":
    main()