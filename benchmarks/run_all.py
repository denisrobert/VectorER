"""Re-run every benchmark in the repo and refresh the results artifacts.

Usage::

    python benchmarks/run_all.py                # full refresh (long)
    python benchmarks/run_all.py --skip-em      # skip the 2h EM bulk job
    python benchmarks/run_all.py --skip-mdbr     # skip the hosted mdbr embedder
    python benchmarks/run_all.py --skip-compare  # skip vs-original comparison
    python benchmarks/run_all.py --n-procs 8    # multiprocessing for the EM bulk job

Order / rationale
-----------------
* The EM bulk benchmark (``benchmark_bulk_er_em.py``) is the long pole (~2 h at
  the recorded scale) and is scheduled *first*, so a later interrupt still has
  its artifact.
* The remaining runs are grouped by benchmark script to reuse loaded data and
  keep the output artifacts adjacent in age.

Every command mirrors the parameters of the committed ``results/*.json``
artifacts (see the ``parameters`` block of each), so re-running refreshes the
same numbers after the code changes.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = Path(ROOT) / ".venv" / "Scripts" / "python.exe"
if not PY.exists():
    PY = sys.executable

_DRY_RUN = False

RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

EM_BULK_DATASET = "benchmarks/population_with_duplicates.json"
EM_BULK_GT = "benchmarks/population_gt.json"
ROC_DATASET = "benchmarks/population.json"


def run(cmd: list[str], *, tag: str, show_output: bool = False) -> None:
    """Run one benchmark command, printing the tag before and after."""
    print(f"\n=== [{tag}] {' '.join(cmd)}\n", flush=True)
    if _DRY_RUN:
        return
    t0 = time.perf_counter()
    if show_output:
        result = subprocess.run(cmd, cwd=ROOT)
    else:
        result = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.DEVNULL,
                                stderr=subprocess.STDOUT)
    dt = time.perf_counter() - t0
    status = "OK" if result.returncode == 0 else f"FAILED ({result.returncode})"
    print(f"=== [{tag}] {status} in {dt / 60:.1f} min\n", flush=True)
    if result.returncode != 0:
        sys.exit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-em", action="store_true",
                        help="skip the long EM bulk benchmark (benchmark_bulk_er_em.py)")
    parser.add_argument("--skip-mdbr", action="store_true",
                        help="skip the mdbr (hosted API) incremental ROC run")
    parser.add_argument("--skip-compare", action="store_true",
                        help="skip the vs-original incremental latency comparison")
    parser.add_argument("--n-procs", type=int, default=8,
                        help="worker count for the EM bulk benchmark (default 8)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the planned commands without executing them")
    args = parser.parse_args()

    global _DRY_RUN
    _DRY_RUN = args.dry_run

    print(f"Using python: {PY}", flush=True)
    print(f"n_procs: {args.n_procs}  skip_em: {args.skip_em}  "
          f"skip_mdbr: {args.skip_mdbr}  skip_compare: {args.skip_compare}",
          flush=True)

    # ---- 1. The long pole first: EM bulk dedup ---------------------------
    # EM training is Yancey-enriched by default (enrich_keep_frac=0.05 +
    # recalibrate_prior), which is what makes the learned m/u/prior sane.
    if not args.skip_em:
        run([
            str(PY), "benchmarks/benchmark_bulk_er_em.py",
            "--data-file", EM_BULK_DATASET,
            "--gt-file", EM_BULK_GT,
            "--n-canopies", "4096", "--overlap", "2", "--merge", "rep",
            "--n-procs", str(args.n_procs),
            "--output", "results/bulk_latency_em_4096_mp.json",
        ], tag="EM bulk (rep, 4096 canopies, mp)", show_output=True)
    else:
        print("\n=== [skip] EM bulk (per --skip-em) ===\n", flush=True)

    # ---- 2. Bulk R-Swoosh latency variants ------------------------------
    run([
        str(PY), "benchmarks/benchmark_bulk_er.py",
        "--n-records", "10000", "--dup-rate", "0.04",
        "--n-canopies", "256", "--overlap", "2",
        "--output", "results/bulk_latency.json",
    ], tag="bulk R-Swoosh (10k, 256 canopies)")
    run([
        str(PY), "benchmarks/benchmark_bulk_er.py",
        "--n-records", "52000", "--dup-rate", "0.04",
        "--n-canopies", "1024", "--overlap", "1",
        "--output", "results/bulk_latency_paperscale.json",
    ], tag="bulk R-Swoosh (paperscale 52k, 1024 canopies)")
    run([
        str(PY), "benchmarks/benchmark_bulk_er.py",
        "--n-records", "10000", "--dup-rate", "0.04",
        "--n-canopies", "256", "--overlap", "2", "--embedder", "sentence",
        "--output", "results/bulk_latency_sentence.json",
    ], tag="bulk R-Swoosh (10000, sentence embedder)")
    run([
        str(PY), "benchmarks/benchmark_bulk_er.py",
        "--n-records", "10000", "--dup-rate", "0.04",
        "--n-canopies", "256", "--overlap", "2", "--merge", "union",
        "--output", "results/bulk_latency_union.json",
    ], tag="bulk union (10k, 256 canopies)")
    run([
        str(PY), "benchmarks/benchmark_bulk_er.py",
        "--n-records", "4000", "--dup-rate", "0.04",
        "--n-canopies", "102", "--overlap", "2", "--merge", "union",
        "--output", "results/bulk_latency_union_4k.json",
    ], tag="bulk union (4k, 102 canopies)")

    # ---- 3. G-Swoosh variants -------------------------------------------
    run([
        str(PY), "benchmarks/benchmark_bulk_er_gswoosh.py",
        "--n-records", "10000", "--dup-rate", "0.04",
        "--n-canopies", "256", "--overlap", "2",
        "--output", "results/bulk_latency_gswoosh.json",
    ], tag="G-Swoosh (10k, 256 canopies)")
    run([
        str(PY), "benchmarks/benchmark_bulk_er_gswoosh.py",
        "--n-records", "4000", "--dup-rate", "0.04",
        "--n-canopies", "102", "--overlap", "2", "--merge", "rep",
        "--output", "results/bulk_latency_gswoosh_rep_4k.json",
    ], tag="G-Swoosh rep (4k, 102 canopies)")
    run([
        str(PY), "benchmarks/benchmark_bulk_er_gswoosh.py",
        "--n-records", "4000", "--dup-rate", "0.04",
        "--n-canopies", "102", "--overlap", "2", "--merge", "union",
        "--output", "results/bulk_latency_gswoosh_union_4k.json",
    ], tag="G-Swoosh union (4k, 102 canopies)")

    # ---- 4. Multinode (Ray) ----------------------------------------------
    run([
        str(PY), "benchmarks/benchmark_bulk_er_multinode.py",
        "--n-records", "8000", "--dup-rate", "0.04",
        "--n-workers", "2", "--ray-address", "auto",
        "--output", "results/bulk_latency_multinode.json",
    ], tag="bulk multinode (2-node sim, 8k)")

    # ---- 5. Incremental latency ------------------------------------------
    run([
        str(PY), "benchmarks/benchmark_incremental_er.py",
        "--n-references", "20000", "--query-count", "100", "--breakdown",
        "--output", "results/incremental_latency.json",
    ], tag="incremental latency (20k refs, hashing)")
    run([
        str(PY), "benchmarks/benchmark_incremental_er.py",
        "--n-references", "50000", "--query-count", "100",
        "--output", "results/incremental_latency_paperscale.json",
    ], tag="incremental latency (50k refs, hashing)")
    run([
        str(PY), "benchmarks/benchmark_incremental_er.py",
        "--n-references", "50000", "--query-count", "50",
        "--embedder", "sentence",
        "--output", "results/incremental_latency_sentence.json",
    ], tag="incremental latency (50k refs, sentence)")
    if not args.skip_compare:
        run([
            str(PY), "benchmarks/benchmark_incremental_er.py",
            "--n-references", "50000", "--query-count", "100",
            "--compare", r"C:\src\experiments\entity\results\erwhitepaper\online_resolver_latency.json",
            "--output", "results/incremental_latency_vs_original.json",
        ], tag="incremental latency vs original", show_output=True)
    else:
        print("\n=== [skip] vs-original comparison (per --skip-compare) ===\n", flush=True)

    # ---- 6. Incremental ROC + confusion matrices -------------------------
    for embedder, out in [
        ("hashing", "results/incremental_roc.json"),
        ("minilm", "results/incremental_roc_minilm.json"),
        ("mdbr", "results/incremental_roc_mdbr.json"),
    ]:
        if embedder == "mdbr" and args.skip_mdbr:
            print(f"\n=== [skip] incremental ROC {embedder} (per --skip-mdbr) ===\n",
                  flush=True)
            continue
        run([
            str(PY), "benchmarks/benchmark_incremental_roc.py",
            "--data-file", ROC_DATASET,
            "--n-records", "100000", "--train-fraction", "0.8",
            "--train-pairs", "2000", "--n-positives", "300", "--n-negatives", "300",
            "--tau-count", "80", "--k", "20",
            "--embedder", embedder,
            "--output", out,
        ], tag=f"incremental ROC ({embedder}, 100k)")

    for embedder in ["hashing", "minilm"]:
        for perturb in ["all", "address_change", "nsa"]:
            parts = ["confusion_matrix_census"]
            if perturb != "all":
                parts.append(perturb)
            if embedder == "minilm":
                parts.append("minilm")
            else:
                # keep the historical "hashing" suffix on the perturbed hashing
                # artifacts (confusion_matrix_census_address_change_hashing.json)
                if perturb != "all":
                    parts.append("hashing")
            out = f"results/{'_'.join(parts)}.json"
            run([
                str(PY), "benchmarks/benchmark_incremental_roc.py",
                "--data-file", ROC_DATASET,
                "--n-records", "100000", "--train-fraction", "0.8",
                "--train-pairs", "2000", "--n-positives", "120", "--n-negatives", "120",
                "--tau-count", "40", "--k", "20",
                "--perturbation", perturb,
                "--embedder", embedder,
                "--output", out,
            ], tag=f"confusion matrix ({embedder}, {perturb})")

    # ---- 7. Yancey (EM enrichment) benchmarks ----------------------------
    run([
        str(PY), "benchmarks/benchmark_yancey_enrichment.py",
        "--data-file", EM_BULK_DATASET, "--gt-file", EM_BULK_GT,
        "--recalibration-method", "yancey", "--n-eval-pairs", "10000",
        "--output", "results/yancey_enrichment.json",
    ], tag="yancey enrichment (yancey recalibration)")
    run([
        str(PY), "benchmarks/benchmark_yancey_enrichment.py",
        "--data-file", EM_BULK_DATASET, "--gt-file", EM_BULK_GT,
        "--recalibration-method", "empirical", "--n-eval-pairs", "10000",
        "--output", "results/yancey_enrichment_empirical.json",
    ], tag="yancey enrichment (empirical recalibration)")
    run([
        str(PY), "benchmarks/benchmark_yancey_incremental.py",
        "--data-file", EM_BULK_DATASET, "--gt-file", EM_BULK_GT,
        "--output", "results/yancey_incremental.json",
    ], tag="yancey incremental")

    print("\nAll benchmarks finished.", flush=True)


if __name__ == "__main__":
    main()