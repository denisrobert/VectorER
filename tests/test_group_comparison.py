"""Tests for the group (composite) comparison over co-dependent fields."""

import numpy as np
import pytest

from vectorer import (
    FellegiSunterScorer,
    available_comparisons,
    group_comparison,
    group_comparison_spec,
    make_comparison,
    replace_with_group,
)
from vectorer.comparisons import PairValues
from vectorer.scoring._levels import _assign_levels


def _two_exact():
    return (
        make_comparison("exact_match", col_name="a"),
        make_comparison("exact_match", col_name="b"),
    )


def _pv(pairs):
    left = {"a": np.array([p[0][0] for p in pairs], dtype=object),
            "b": np.array([p[0][1] for p in pairs], dtype=object)}
    right = {"a": np.array([p[1][0] for p in pairs], dtype=object),
             "b": np.array([p[1][1] for p in pairs], dtype=object)}
    return PairValues(left, right)


def test_default_cross_product_structure():
    a, b = _two_exact()
    g = group_comparison_spec("ab", [a, b])
    la, lb = len(a.spec().levels), len(b.spec().levels)
    assert len(g.levels) == la * lb
    assert g.fields == ("a", "b")
    # first level is the all-null level (both members null) -> is_null
    assert g.levels[0].is_null is True
    # last level is the all-ELSE combination -> the ELSE fallback (test=None)
    assert g.levels[-1].test is None
    assert g.levels[-1].is_null is False


def test_every_pair_maps_to_exactly_one_level():
    a, b = _two_exact()
    g = group_comparison_spec("ab", [a, b])
    n = len(g.levels)
    pairs = [
        (("x", "y"), ("x", "y")),     # both exact
        (("x", "y"), ("x", "z")),     # a exact, b different
        (("x", "y"), ("z", "y")),     # b exact, a different
        (("x", "y"), ("z", "w")),     # neither
        ((None, "y"), ("x", "y")),    # a null
        ((None, None), (None, None)), # both null
    ]
    assigned = _assign_levels(g, _pv(pairs))
    assert assigned.shape == (len(pairs),)
    assert (assigned >= 0).all() and (assigned < n).all()
    # identical "(x, y)" pairs -> the both-exact cell, not null/else
    assert assigned[0] != 0 and assigned[0] != n - 1
    # both-null pair -> the null level
    assert g.levels[assigned[5]].is_null is True


def test_marginal_seed_is_product_of_members():
    a, b = _two_exact()
    la = a.spec().levels
    g = group_comparison_spec("ab", [a, b])
    # level whose cell is (a exact, b exact): dims (3,3) -> cell 1*3+1 = 4,
    # and with the ELSE cell already last the index is unchanged.
    exact_a = la[1].m
    # cell (a null, b exact) = 1: neutral factor for a -> seed ratio == m_a(exact)
    assert g.levels[4].m / g.levels[1].m == pytest.approx(exact_a, rel=0.05)
    # non-null m seeds sum to 1
    total = sum(lv.m for lv in g.levels if not lv.is_null)
    assert total == pytest.approx(1.0, rel=1e-9)


def test_combine_merges_cells_and_orders_else_last():
    a, b = _two_exact()

    def combine(t):
        x, y = t
        if x == 1 and y == 1:
            return 0
        if x == 1 or y == 1:
            return 1
        return 2

    g = group_comparison_spec("ab", [a, b], combine=combine,
                              labels=["both", "one", "neither"])
    assert len(g.levels) == 3
    assert [lv.label for lv in g.levels] == ["both", "one", "neither"]
    assert g.levels[-1].test is None  # ELSE stays last
    # the "both" level fires only for the (exact, exact) cell
    assigned = _assign_levels(g, _pv([(("x", "y"), ("x", "y")), (("x", "y"), ("z", "w"))]))
    assert g.levels[assigned[0]].label == "both"
    assert g.levels[assigned[1]].label == "neither"


def test_combine_negative_index_rejected():
    a, b = _two_exact()
    with pytest.raises(ValueError, match="non-negative"):
        group_comparison_spec("ab", [a, b], combine=lambda t: -1)


def test_needs_two_members():
    a = make_comparison("exact_match", col_name="a")
    with pytest.raises(ValueError, match="at least two"):
        group_comparison_spec("a", [a])


def test_replace_with_group_swaps_members():
    a, b = _two_exact()
    dob = make_comparison("date_of_birth_comparison", col_name="date_of_birth")
    g = group_comparison("ab", [a, b])
    out = replace_with_group([a, b, dob], g, [a, b])
    assert len(out) == 2
    names = {c.output_column_name() if hasattr(c, "output_column_name") else c.output_column_name
             for c in out}
    assert names == {"date_of_birth", "ab"}


def test_registered_and_scorable_and_em_roundtrips():
    a, b = _two_exact()
    assert "group_comparison" in available_comparisons()
    g = group_comparison("ab", [a, b])
    assert isinstance(g.params["members"], list)
    # comparison construction + scoring works through the registry
    scorer = FellegiSunterScorer.from_comparisons([g])
    probs = scorer.score_pairs(
        [{"a": "x", "b": "y"}, {"a": "x", "b": "y"}],
        [{"a": "x", "b": "y"}, {"a": "z", "b": "w"}],
    )
    assert float(probs[0]) > float(probs[1])
