"""Generate a synthetic Canadian population of 300,000 people with a
distribution modelled on the 2021 Canadian census, for use as the base
population of every benchmark.

Distribution rules
------------------
* **Sex**: male/female split follows the 2021 census (`18,226,240` men+ vs
  `18,765,740` women+ -> ~49.27% / 50.73%).
* **Province**: each record's province is drawn in proportion to the 2021
  census population of the 13 provinces/territories (Ontario 38.45%, Quebec
  22.98%, BC 13.52%, Alberta 11.52%, ...).
* **Age/DOB**: birth year is sampled from the 2021 census broad age buckets
  (0-14 = 16.3%, 15-64 = 64.8%, 65+ = 19.0%), which yields a plausible
  birth-year distribution over 1921-2021.
* **Names**: given names are drawn sex-specific from common Canadian names
  (with a French-flavoured list for Quebec); surnames from common Canadian
  surnames (English + French origin).
* **Address / postal code**: street of the person's province/major city.  The
  postal code's first letter is guaranteed to be one of the acceptable forward
  sortation area prefix letters for that province (ON: K,L,M,N,P; QC: G,H,J;
  BC: V; AB: T; ...), so the initial always matches the province.
* **Email**: a safe, *name-independent* address (``<neutral-word><digits>
  @example.com``).  The local part is drawn from a small neutral vocabulary and
  random digits -- deliberately NOT derived from the person's name, so the
  name and email columns do not artificially correlate in the benchmarks.

Records keep the schema used by every benchmark script
(``first_name, last_name, date_of_birth, email, address``) plus ``sex``,
``province``, ``city`` and ``postal_code`` for richer downstream use.

Run from the project root:

    python benchmarks/generate_census_population.py [--n 300000] [--seed 42]
                                                    [--output benchmarks/population.json]
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

# ---------------------------------------------------------------------------
# 2021 Census of Population (StatCan) data
# ---------------------------------------------------------------------------

# (province, abbr, 2021 census population)
PROVINCES = [
    ("Ontario", "ON", 14_223_942),
    ("Quebec", "QC", 8_501_833),
    ("British Columbia", "BC", 5_000_879),
    ("Alberta", "AB", 4_262_635),
    ("Manitoba", "MB", 1_342_153),
    ("Saskatchewan", "SK", 1_132_505),
    ("Nova Scotia", "NS", 969_383),
    ("New Brunswick", "NB", 775_610),
    ("Newfoundland and Labrador", "NL", 510_550),
    ("Prince Edward Island", "PE", 154_331),
    ("Northwest Territories", "NT", 41_070),
    ("Yukon", "YT", 40_232),
    ("Nunavut", "NU", 36_858),
]
TOTAL_POP = sum(p for _, _, p in PROVINCES)

# Male / female counts from the 2021 census (Canada, 100% data).
MALE = 18_226_240
FEMALE = 18_765_740

# Broad age buckets (percent of population), used for birth-year sampling.
AGE_BUCKETS = [
    ("0-14", 16.3, (2007, 2021)),
    ("15-64", 64.8, (1957, 2006)),
    ("65+", 19.0, (1921, 1956)),
]

# Acceptable postal-code first letters (forward sortation area prefix) per
# province/territory (Canada Post).  ON and QC have multiple postal districts.
POSTAL_PREFIX = {
    "NL": ["A"], "NS": ["B"], "PE": ["C"], "NB": ["E"],
    "QC": ["G", "H", "J"], "ON": ["K", "L", "M", "N", "P"],
    "MB": ["R"], "SK": ["S"], "AB": ["T"], "BC": ["V"],
    "NT": ["X"], "NU": ["X"], "YT": ["Y"],
}

# Letters Canada Post uses; D,F,I,O,Q,U excluded everywhere, W/Z never first.
_LETTERS_ALL = [c for c in "ABCDEFGHIJKLMNOPRSTUVWXYZ" if c not in "DFIOQU"]
_CITY_PREFIX = {
    "ON": {"Toronto": "M", "Ottawa": "K", "Hamilton": "L", "London": "N", "Sudbury": "P"},
    "QC": {"Montreal": "H", "Quebec City": "G", "Gatineau": "J", "Saguenay": "G", "Sherbrooke": "J"},
    "BC": {"Vancouver": "V", "Victoria": "V", "Kelowna": "V", "Prince George": "V"},
    "AB": {"Calgary": "T", "Edmonton": "T", "Red Deer": "T"},
    "MB": {"Winnipeg": "R", "Brandon": "R"},
    "SK": {"Saskatoon": "S", "Regina": "S"},
    "NS": {"Halifax": "B", "Sydney": "B"},
    "NB": {"Moncton": "E", "Saint John": "E", "Fredericton": "E"},
    "NL": {"St. John's": "A", "Corner Brook": "A"},
    "PE": {"Charlottetown": "C"},
    "NT": {"Yellowknife": "X"},
    "YT": {"Whitehorse": "Y"},
    "NU": {"Iqaluit": "X"},
}

# Common Canadian names (sex-specific).  Quebec gets a French-flavoured list
# to reflect its linguistic makeup; everyone else draws from the English list.
MALE_EN = [
    "Noah", "Liam", "Theodore", "Leo", "William", "Oliver", "Lucas", "James",
    "Benjamin", "Thomas", "Ethan", "Jacob", "Logan", "Owen", "Jackson", "Henry",
    "Jack", "Ryan", "Daniel", "Matthew", "Samuel", "John", "David", "Michael",
    "Alexander", "Nathan", "Caleb", "Adam", "Evan", "Brayden", "Aiden", "Eric",
]
FEMALE_EN = [
    "Olivia", "Emma", "Charlotte", "Amelia", "Sophia", "Ava", "Mia", "Isabella",
    "Evelyn", "Abigail", "Emily", "Harper", "Grace", "Chloe", "Lily", "Ella",
    "Sofia", "Avery", "Zoe", "Nora", "Sarah", "Mary", "Jessica", "Ashley",
    "Hannah", "Madison", "Elizabeth", "Samantha", "Lauren", "Megan", "Rachel",
]
MALE_FR = [
    "Noah", "Leo", "Liam", "William", "Thomas", "Louis", "Arthur", "Edouard",
    "Emile", "Theo", "Felix", "Gabriel", "Antoine", "Alexandre", "Charles",
    "Mathis", "Nathan", "Jeremy", "Olivier", "Vincent",
]
FEMALE_FR = [
    "Olivia", "Emma", "Charlotte", "Lea", "Alice", "Florence", "Alicia",
    "Eve", "Juliette", "Laurence", "Camille", "Marion", "Sarah", "Rosalie",
    "Victoria", "Gabrielle", "Elisabeth", "Anne", "Marie", "Marianne",
]
SURNAMES_EN = [
    "Smith", "Brown", "Martin", "Roy", "Wilson", "MacDonald", "Johnson",
    "Taylor", "Campbell", "Anderson", "Jones", "Williams", "Lee", "Miller",
    "Thompson", "Clark", "Lewis", "Young", "Walker", "Hall", "Allen",
    "King", "Wright", "Scott", "Baker", "Adams", "Mitchell", "Carter",
    "Hill", "Moore", "White", "Green", "Roberts", "Turner", "Phillips",
    "Parker", "Evans", "Edwards", "Collins", "Stewart", "Morris", "Reed",
]
SURNAMES_FR = [
    "Tremblay", "Gagnon", "Roy", "Cote", "Bouchard", "Gauthier", "Morin",
    "Lavoie", "Fortin", "Gagné", "Ouellet", "Pelletier", "Belanger",
    "Lefebvre", "Girard", "Boucher", "Caron", "Beaulieu", "Cloutier",
    "Dube", "Poirier", "Fournier", "Lapointe", "Leclerc", "Dion",
]

# Neutral email local-part vocabulary (name-independent on purpose).
_EMAIL_WORDS = [
    "user", "contact", "person", "resident", "customer", "member", "citizen",
    "individual", "persona", "client",
]

# Street vocabulary.
STREETS = ["Main", "Queen", "King", "Elm", "Maple", "Oak", "Cedar", "Birch",
           "Park", "Church", "Victoria", "Wellington", "Bay", "Lake", "Hill",
           "River", "Douglas", "Albert"]
STREET_KINDS = ["St", "Ave", "Rd", "Blvd", "Cres", "Dr", "Way", "Ct", "Pl"]

# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


@dataclass
class PopulationConfig:
    n: int = 300_000
    seed: int = 42


def build_province_sampler(rng: random.Random):
    """Return a weighted (province_name, abbr) sampler per the 2021 census."""
    pop = [(abbr, p) for _, abbr, p in PROVINCES]
    abbrs = [a for a, _ in pop]
    weights = [p for _, p in pop]
    table = []
    cumulative = 0.0
    total = float(sum(weights))
    for abbr, w in zip(abbrs, weights):
        cumulative += w / total
        table.append((abbr, cumulative))
    return table


def sample_province(rng: random.Random, table) -> str:
    r = rng.random()
    for abbr, cum in table:
        if r <= cum:
            return abbr
    return table[-1][0]


def sample_birth_date(rng: random.Random) -> str:
    """Birth date sampled from the 2021 census broad age buckets."""
    r = rng.random() * 100.0
    acc = 0.0
    lo, hi = 1957, 2006  # default: 15-64
    for _, pct, (a, b) in AGE_BUCKETS:
        acc += pct
        if r <= acc:
            lo, hi = a, b
            break
    year = rng.randint(lo, hi)
    month = rng.randint(1, 12)
    day = rng.randint(1, 28)
    return f"{year:04d}-{month:02d}-{day:02d}"


def postal_code(rng: random.Random, province_abbr: str, city: str) -> str:
    """A plausible Canadian postal code whose first letter matches the province."""
    prefix_letters = POSTAL_PREFIX[province_abbr]
    city_prefix = _CITY_PREFIX.get(province_abbr, {}).get(city)
    first = city_prefix if city_prefix and city_prefix in prefix_letters else rng.choice(prefix_letters)
    # FSA: letter digit letter.  Urban areas us non-zero digit (more common).
    digit = rng.randint(1, 9) if rng.random() < 0.9 else 0
    second = rng.choice(_LETTERS_ALL)
    # LDU: digit letter digit
    d1 = rng.randint(1, 9) if rng.random() < 0.95 else 0
    l1 = rng.choice(_LETTERS_ALL)
    d2 = rng.randint(1, 9) if rng.random() < 0.95 else 0
    return f"{first}{digit}{second} {d1}{l1}{d2}"


def pick_name(rng: random.Random, sex: str, province_abbr: str):
    if province_abbr == "QC":
        male, female, surnames = MALE_FR, FEMALE_FR, SURNAMES_FR
    else:
        male, female, surnames = MALE_EN, FEMALE_EN, SURNAMES_EN
    first = rng.choice(male if sex == "M" else female)
    last = rng.choice(surnames)
    return first, last


def generate(config: PopulationConfig) -> list[dict]:
    rng = random.Random(config.seed)
    table = build_province_sampler(rng)
    province_by_abbr = {abbr: name for name, abbr, _ in PROVINCES}

    # Resolve each province's city pool (some provinces have multiple cities).
    city_by_province = {abbr: list(cities) for abbr, cities in _CITY_PREFIX.items()}

    records = []
    used_emails: set[str] = set()
    email_counter = 0
    for _ in tqdm(range(config.n), desc="generating people", unit="person"):
        sex = "M" if rng.random() < (MALE / (MALE + FEMALE)) else "F"
        abbr = sample_province(rng, table)
        first, last = pick_name(rng, sex, abbr)
        dob = sample_birth_date(rng)
        city = rng.choice(city_by_province[abbr])
        number = rng.randint(1, 9999)
        street = rng.choice(STREETS)
        kind = rng.choice(STREET_KINDS)
        pc = postal_code(rng, abbr, city)
        address = f"{number} {street} {kind}, {city}, {province_by_abbr[abbr]} {pc}"
        # Safe, name-independent email.
        while True:
            email_counter += 1
            candidate = f"{rng.choice(_EMAIL_WORDS)}{email_counter}@example.com"
            if candidate not in used_emails:
                used_emails.add(candidate)
                break
        records.append({
            "first_name": first,
            "last_name": last,
            "date_of_birth": dob,
            "email": candidate,
            "address": address,
            "sex": sex,
            "province": province_by_abbr[abbr],
            "province_code": abbr,
            "city": city,
            "postal_code": pc,
        })
    return records


def summarize(records: list[dict]) -> None:
    from collections import Counter

    n = len(records)
    prov = Counter(r["province_code"] for r in records)
    sex = Counter(r["sex"] for r in records)
    print(f"total: {n:,}")
    print("sex:")
    for k in ("M", "F"):
        print(f"  {k}: {sex[k]:,} ({sex[k] / n:.2%})")
    print("province (sample vs census):")
    census = {abbr: p / TOTAL_POP for _, abbr, p in PROVINCES}
    for abbr in census:
        got = prov.get(abbr, 0) / n
        print(f"  {abbr}: {got:.2%}  (census {census[abbr]:.2%})")
    # postal-code validity
    for r in records[:0]:
        pass
    bad = sum(1 for r in records if not r["postal_code"][0] in POSTAL_PREFIX[r["province_code"]])
    print(f"postal-code/province mismatches: {bad}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="benchmarks/population.json")
    args = parser.parse_args()

    print(f"Generating {args.n:,} census-distributed people (seed={args.seed})...")
    records = generate(PopulationConfig(n=args.n, seed=args.seed))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(records, fh)
    print(f"Wrote {len(records):,} records to {out}")

    summarize(records)


if __name__ == "__main__":
    main()