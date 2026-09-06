"""Generate a synthetic Canadian population (2021-census-distributed) **with
duplicates**, for EM training of the Fellegi-Sunter model in benchmark tests.

Builds the same base population as ``generate_census_population.py`` (seed,
2021 census sex/province distribution, correct postal-code prefixes,
name-independent safe emails), then adds duplicates of base records **with
perturbations** so the dataset contains genuine near-duplicate groups:

* **5%** of the base population is duplicated (perturbed) -- one extra twin;
* **1%** of the base population is duplicated **twice** (perturbed) -- two
  extra twins (multiplicity 3);
* **0.1%** of the base population is duplicated more than twice but **fewer
  than 6 times** (i.e. 3, 4 or 5 extra twins; multiplicity 4-6).

Duplicate copies are perturbed with the framework's clerical/transmission
perturber (``vectorer.benchmarks.perturbations``) so they are *near-duplicates*,
not exact copies -- exactly the kind of population EM needs (duplicate-bearing
rows with realistic clerical error) to fit ``m/u`` and the match prior.

The output is a single JSON list of records (base + appended duplicates, in
order).  The default output file is gitignored because it is large and
regeneratable.

Run from the project root:

    python benchmarks/generate_census_population_with_duplicates.py \\
        --n 300000 --seed 42 \\
        --output benchmarks/population_with_duplicates.json
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from tqdm import tqdm

# Reuse the base census generator unchanged.
from generate_census_population import (
    PROVINCES,
    TOTAL_POP,
    POSTAL_PREFIX,
    PopulationConfig,
    generate,
    build_province_sampler,
    postal_code,
)

# Clerical/transmission perturbation set (near-duplicate generator).
# Deterministic variant-only perturbations keep the record recognisable;
# ``serialization`` can compound several errors (harder EM cases).
from perturbations import (
    PERTURBATION_TYPES,
    apply_perturbation,
)

#: Default output; gitignored (large, regeneratable).
DEFAULT_OUTPUT = "benchmarks/population_with_duplicates.json"

# Duplication scheme (fraction of the base population, multiplicities of copies).
DUP_SINGLE = 0.05    # 5%  -> 1 perturbed twin
DUP_DOUBLE = 0.01    # 1%  -> 2 perturbed twins (multiplicity 3)
DUP_MANY = 0.001     # 0.1% -> 3-5 perturbed twins (multiplicity 4-6)
DUP_MANY_MIN, DUP_MANY_MAX = 3, 5  # extra twins in the "many" bucket


def add_duplicates(base: list[dict], seed: int = 42) -> list[dict]:
    """Append perturbed duplicates of ``base`` per the duplication scheme.

    Returns ``records`` = ``base`` + appended duplicates (original order
    preserved), with per-base-record perturbation choices made deterministically
    from ``seed``.
    """
    rng = random.Random(seed + 101)  # different stream from base generation
    n_base = len(base)
    n_single = int(round(n_base * DUP_SINGLE))
    n_double = int(round(n_base * DUP_DOUBLE))
    n_many = int(round(n_base * DUP_MANY))

    # Which base indices get duplicated, in the given buckets.
    indices = rng.sample(range(n_base), n_single + n_double + n_many)
    single_idx = set(indices[:n_single])
    double_idx = set(indices[n_single:n_single + n_double])
    many_idx = set(indices[n_single + n_double:])

    records = list(base)
    added = 0

    def perturbed(base_record: dict) -> dict:
        # A seeded random stream per copy so perturbation choice is stable.
        return apply_perturbation(rng.choice(list(PERTURBATION_TYPES)), base_record, rng)

    # Deterministic, reproducible order over the base indices.
    for i in tqdm(range(n_base), desc="adding duplicates", unit="base"):
        if i in single_idx:
            records.append(perturbed(base[i]))
            added += 1
        if i in double_idx:
            records.append(perturbed(base[i]))
            records.append(perturbed(base[i]))
            added += 2
        if i in many_idx:
            n_copies = rng.randint(DUP_MANY_MIN, DUP_MANY_MAX)
            for _ in range(n_copies):
                records.append(perturbed(base[i]))
            added += n_copies

    return records, added


def summarize(records: list[dict], n_base: int, added: int) -> None:
    n = len(records)
    prov = Counter(r["province_code"] for r in records)
    sex = Counter(r["sex"] for r in records)
    print(f"total: {n:,} (base {n_base:,} + {added:,} duplicates)")
    print("sex:")
    for k in ("M", "F"):
        print(f"  {k}: {sex[k]:,} ({sex[k] / n:.2%})")
    print("province (sample vs census):")
    census = {abbr: p / TOTAL_POP for _, abbr, p in PROVINCES}
    for abbr in census:
        got = prov.get(abbr, 0) / n
        print(f"  {abbr}: {got:.2%}  (census {census[abbr]:.2%})")
    bad = sum(1 for r in records if not r["postal_code"][0] in POSTAL_PREFIX[r["province_code"]])
    print(f"postal-code/province mismatches: {bad}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    print(f"Generating {args.n:,} census-distributed base people (seed={args.seed})...")
    base = generate(PopulationConfig(n=args.n, seed=args.seed))

    print("Adding perturbed duplicates "
          f"({DUP_SINGLE:.1%} single, {DUP_DOUBLE:.2%} doubled, {DUP_MANY:.2%} many)...")
    records, added = add_duplicates(base, seed=args.seed)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(records, fh)
    print(f"Wrote {len(records):,} records to {out}")

    summarize(records, len(base), added)


if __name__ == "__main__":
    main()