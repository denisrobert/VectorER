"""Shared test fixtures: small deterministic person datasets."""

import pytest

from vectorer.comparisons import make_comparison


def person(first, last, dob, email=None, address=None, **extra):
    return {
        "first_name": first,
        "last_name": last,
        "date_of_birth": dob,
        "email": email,
        "address": address,
        **extra,
    }


def base_names():
    return [
        ("john", "smith", "1985-06-15"),
        ("mary", "jones", "1990-11-03"),
        ("robert", "martinez", "1978-02-28"),
        ("susan", "brown", "1988-07-19"),
        ("james", "wilson", "1965-12-01"),
    ]


@pytest.fixture
def person_duplicate_dataset():
    """5 base records + one exact duplicate and one lightly perturbed duplicate."""
    records = []
    for i, (first, last, dob) in enumerate(base_names()):
        records.append(
            person(first, last, dob, f"{first}.{last}@mail.com", f"{i+1} main st")
        )
    # duplicate i shares a base record; exact copy and noisy copy.
    for i in range(5):
        first, last, dob = base_names()[i]
        records.append(person(first, last, dob, f"{first}.{last}@mail.com", f"{i+1} main st"))
    for i in range(5):
        first, last, dob = base_names()[i]
        noisy = last[:-1] + ("q" if last[-1] != "q" else "k")
        records.append(person(first, noisy, dob, f"{first}.{last}@mail.com", f"{i+1} main st"))
    return records


@pytest.fixture
def base_comparisons():
    """Small always-constructible comparison set (native vectorer registry)."""
    return [
        make_comparison("jaro_winkler_at_thresholds", col_name="first_name"),
        make_comparison("jaro_winkler_at_thresholds", col_name="last_name"),
        make_comparison("date_of_birth_comparison", col_name="date_of_birth"),
        make_comparison("email_comparison", col_name="email"),
    ]


@pytest.fixture
def fs_scorer(base_comparisons):
    from vectorer.scoring import FellegiSunterScorer

    return FellegiSunterScorer.from_comparisons(base_comparisons)