"""Tests for the extensible Fellegi-Sunter comparison set (native, vectorized)."""

import pytest

from vectorer.comparisons import (
    Comparison,
    ComparisonSpec,
    REGISTRY,
    available_comparisons,
    comparison_catalog,
    comparison_fields,
    comparison_set,
    comparison_to_dict,
    make_comparison,
    make_comparisons,
    register_comparison,
)
from vectorer.sim import jaro_winkler_similarity

# Comparison names that need explicit non-default constructor params.
REQUIRED_KWARGS = {
    "distance_function_at_thresholds": {
        "distance_function_name": "levenshtein",
        "distance_threshold_or_thresholds": [1, 2],
    },
    "pairwise_string_distance_function_at_thresholds": {
        "distance_function_name": "jaro_winkler",
        "distance_threshold_or_thresholds": [0.9, 0.7],
    },
    "custom_comparison": {
        "output_column_name": "custom",
        "comparison_levels": [
            {"sql_condition": '"a_l" = "a_r"', "label_for_charts": "exact"},
            {"sql_condition": "ELSE", "label_for_charts": "else"},
        ],
    },
    "forename_surname_comparison": {"forename_col_name": "forename", "surname_col_name": "surname"},
    "distance_in_km_at_thresholds": {"lat_col": "lat", "long_col": "long"},
}


def test_catalog_matches_registry():
    """Every registered comparison is catalogued with its fields/description."""
    catalog = comparison_catalog()
    assert set(catalog) == set(REGISTRY.names())
    for name, entry in catalog.items():
        assert "description" in entry
        assert isinstance(entry["fields"], tuple)


def test_registered_names_cover_the_comparison_family():
    """All 19 comparison options are available by name."""
    assert set(REGISTRY.names()) == {
        "absolute_date_difference_at_thresholds",
        "absolute_time_difference_at_thresholds",
        "array_intersect_at_sizes",
        "cosine_similarity_at_thresholds",
        "custom_comparison",
        "damerau_levenshtein_at_thresholds",
        "date_of_birth_comparison",
        "distance_function_at_thresholds",
        "distance_in_km_at_thresholds",
        "email_comparison",
        "exact_match",
        "forename_surname_comparison",
        "jaccard_at_thresholds",
        "jaro_at_thresholds",
        "jaro_winkler_at_thresholds",
        "levenshtein_at_thresholds",
        "name_comparison",
        "pairwise_string_distance_function_at_thresholds",
        "postcode_comparison",
    }


@pytest.mark.parametrize("name", sorted(REGISTRY.names()))
def test_every_registered_comparison_builds(name):
    kwargs = dict(REQUIRED_KWARGS.get(name, {}))
    if "col_name" in REGISTRY.fields_of(name) and "col_name" not in kwargs:
        kwargs["col_name"] = "field"
    comparison = make_comparison(name, **kwargs)
    spec = comparison.spec()
    assert isinstance(spec, ComparisonSpec)
    assert spec.output_column_name
    assert len(spec.levels) >= 2
    assert comparison.output_column_name() == spec.output_column_name


def test_every_built_level_has_default_mu():
    """The default m/u are assigned to every level."""
    comparison = make_comparison(
        "jaro_winkler_at_thresholds", col_name="first_name",
        score_threshold_or_thresholds=[0.9, 0.8, 0.7],
    )
    levels = comparison.spec().levels
    non_null = [lv for lv in levels if not lv.is_null]
    assert len(non_null) == 5
    # First non-null (exact) level keeps m=0.95; u=0.95/2^10.
    exact = non_null[0]
    assert exact.m == pytest.approx(0.95)
    assert exact.u == pytest.approx(0.95 / 1024.0)
    # The bottom (else) level gets the -5 match weight.
    else_level = non_null[-1]
    assert else_level.cvv == 0
    assert else_level.u == pytest.approx(0.0125 / (2 ** -5))
    null_level = levels[0]
    assert null_level.is_null and null_level.m is None


def test_unknown_comparison_raises():
    with pytest.raises(KeyError):
        make_comparison("does_not_exist")


def test_scalar_and_vectorized_paths_agree():
    """Small-batch (scalar) and large-batch (vectorized) paths must agree."""
    import random

    from vectorer.sim import (
        damerau_levenshtein_distance,
        jaro_similarity,
        jaro_winkler_similarity,
        levenshtein_distance,
    )

    rng = random.Random(7)
    pairs = []
    for _ in range(140):  # > _SMALL_BATCH, exercises both paths
        n = rng.randint(0, 14)
        a = "".join(rng.choice("abcdefgijk") for _ in range(n))
        b = "".join(rng.choice("abcdefgijk") for _ in range(rng.randint(0, 14)))
        if rng.random() < 0.2:
            a = None
        pairs.append((a, b))

    left = [p[0] for p in pairs]
    right = [p[1] for p in pairs]

    small_jaro = jaro_similarity(left[:30], right[:30])
    big_jaro = jaro_similarity(left, right)
    for i, (a, b) in enumerate(pairs[:30]):
        assert small_jaro[i] == pytest.approx(big_jaro[i], abs=1e-12)
        assert small_jaro[i] == pytest.approx(jaro_similarity([a], [b])[0], abs=1e-12)

    small_jw = jaro_winkler_similarity(left[:30], right[:30])
    big_jw = jaro_winkler_similarity(left, right)
    for i, (a, b) in enumerate(pairs[:30]):
        assert small_jw[i] == pytest.approx(big_jw[i], abs=1e-12)
        assert small_jw[i] == pytest.approx(jaro_winkler_similarity([a], [b])[0], abs=1e-12)

    small_lev = levenshtein_distance(left[:30], right[:30])
    big_lev = levenshtein_distance(left, right)
    small_dl = damerau_levenshtein_distance(left[:30], right[:30])
    big_dl = damerau_levenshtein_distance(left, right)
    for i, (a, b) in enumerate(pairs[:30]):
        assert small_lev[i] == big_lev[i]
        assert small_dl[i] == big_dl[i]
        assert small_dl[i] == damerau_levenshtein_distance([a], [b])[0]


def test_jaro_winkler_canonical_reference_values():
    """The vectorized JW matches the canonical Jaro-Winkler reference values."""
    pairs = [
        ("MARTHA", "MARHTA"),
        ("", ""),
        ("abc", "xyz"),
        ("dwayne", "duane"),
    ]
    left = [p[0] for p in pairs]
    right = [p[1] for p in pairs]
    scores = jaro_winkler_similarity(left, right)
    assert scores[0] == pytest.approx(0.9611, abs=1e-3)  # famous MARTHA/MARHTA
    assert scores[1] == 1.0  # "" / ""
    assert scores[2] == 0.0  # disjoint
    assert scores[3] > scores[2]


def test_custom_registration_is_extensible():
    from vectorer.comparisons import build_spec, exact_match_spec

    def factory(col_name, **kwargs):
        return exact_match_spec(col_name)

    register_comparison("my_custom", factory, fields=("col_name",), description="test")
    assert "my_custom" in REGISTRY.names()
    comparison = make_comparison("my_custom", col_name="phone")
    assert comparison.output_column_name() == "phone"
    assert comparison.spec().levels[1].label == "Exact match on phone"


def test_make_comparisons_from_objects_and_dicts():
    specs = [
        {"type": "exact_match", "col_name": "first_name"},
        make_comparison("email_comparison", col_name="email"),
    ]
    parsed = make_comparisons(specs)
    assert all(isinstance(c, Comparison) for c in parsed)
    assert [c.name for c in parsed] == ["exact_match", "email_comparison"]


def test_comparison_set_and_fields():
    comparisons = make_comparisons(
        [
            {"type": "exact_match", "col_name": "first_name"},
            {"type": "date_of_birth_comparison", "col_name": "date_of_birth"},
        ]
    )
    assert comparison_fields(comparisons) == ["first_name", "date_of_birth"]
    specs = comparison_set(comparisons)
    assert all(isinstance(s, ComparisonSpec) for s in specs)
    assert [s.output_column_name for s in specs] == ["first_name", "date_of_birth"]


def test_comparison_round_trip_dict():
    c = make_comparison(
        "jaro_winkler_at_thresholds",
        col_name="first_name",
        score_threshold_or_thresholds=[0.9, 0.8, 0.7],
    )
    restored = Comparison.from_dict(c.to_dict())
    assert restored == c
    assert restored.output_column_name() == "first_name"


def test_available_comparisons_lists_everything():
    available = available_comparisons()
    assert "custom_comparison" in available
    assert all(isinstance(desc, str) for desc in available.values())


def test_resolved_round_trips_trained_levels():
    c = make_comparison("jaro_winkler_at_thresholds", col_name="first_name")
    resolved = c.resolved()
    assert resolved["type"] == "jaro_winkler_at_thresholds"
    assert len(resolved["levels"]) == len(c.spec().levels)
    # Mutate a level then rebuild via from_resolved.
    resolved["levels"][1]["m_probability"] = 0.5
    resolved["levels"][1]["u_probability"] = 0.25
    rebuilt = Comparison.from_resolved(resolved)
    assert rebuilt.spec().levels[1].m == 0.5
    assert rebuilt.spec().levels[1].u == 0.25
    assert rebuilt.output_column_name() == "first_name"


def test_custom_comparison_declarative_conditions():
    c = make_comparison(
        "custom_comparison",
        output_column_name="phone",
        comparison_levels=[
            {"sql_condition": '"phone_l" IS NULL OR "phone_r" IS NULL'},
            {"sql_condition": '"phone_l" = "phone_r"', "label_for_charts": "exact"},
            {"sql_condition": "ELSE"},
        ],
    )
    spec = c.spec()
    assert [lv.is_null for lv in spec.levels] == [True, False, False]


def test_custom_comparison_rejects_sql():
    with pytest.raises(ValueError, match="does not run SQL"):
        make_comparison(
            "custom_comparison",
            output_column_name="x",
            comparison_levels=[
                {
                    "sql_condition": 'LEFT("x_l", 3) = LEFT("x_r", 3)',
                    "label_for_charts": "left3",
                }
            ],
        ).spec()