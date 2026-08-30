"""Tests for the Swoosh merge functions (Union / timestamp) and the
Union-Class (existential) lift in the Fellegi-Sunter scorer."""

import numpy as np
import pytest

from vectorer.comparisons import make_comparison
from vectorer.clustering import (
    latest_merge,
    select_representative,
    union_merge,
    SwooshClusterer,
    gswoosh,
)
from vectorer.scoring import FellegiSunterScorer


def person_comparisons():
    return [
        make_comparison("jaro_winkler_at_thresholds", col_name="first_name"),
        make_comparison("jaro_winkler_at_thresholds", col_name="last_name"),
        make_comparison("date_of_birth_comparison", col_name="date_of_birth"),
        make_comparison("email_comparison", col_name="email"),
    ]


def person_scorer(**kwargs):
    return FellegiSunterScorer.from_comparisons(person_comparisons(), **kwargs)


# ---------------------------------------------------------------------------
# Merge functions
# ---------------------------------------------------------------------------


def test_union_merge_collects_all_values():
    records = [
        {"first_name": "john", "last_name": "smith", "email": None},
        {"first_name": "jon", "last_name": "smyth", "email": "j@x.com"},
    ]
    master, pos = union_merge(records, [0, 1])
    assert pos == -1  # synthetic
    assert master["first_name"] == frozenset({"john", "jon"})
    assert master["last_name"] == frozenset({"smith", "smyth"})
    assert master["email"] == frozenset({"j@x.com"})


def test_union_merge_none_when_field_missing_everywhere():
    records = [
        {"first_name": "john", "last_name": "smith", "email": None},
        {"first_name": "jon", "last_name": "smyth", "email": None},
    ]
    master, _ = union_merge(records, [0, 1])
    assert master["email"] is None
    assert master["first_name"] == frozenset({"john", "jon"})


def test_union_merge_flattens_nested_sets():
    records = [
        {"name": frozenset({"john", "jon"}), "age": 30},
        {"name": frozenset({"jon", "jonny"}), "age": 31},
    ]
    master, _ = union_merge(records, [0, 1])
    assert master["name"] == frozenset({"john", "jon", "jonny"})
    assert master["age"] == frozenset({30, 31})


def test_union_merge_unhashable_falls_back_to_list():
    records = [
        {"meta": {"a": 1}},
        {"meta": {"b": 2}},
    ]
    master, _ = union_merge(records, [0, 1])
    assert isinstance(master["meta"], list)
    assert len(master["meta"]) == 2


def test_union_merge_respects_fields_subset():
    records = [
        {"first_name": "john", "last_name": "smith", "email": None},
        {"first_name": "jon", "last_name": "smyth", "email": "j@x.com"},
    ]
    master, _ = union_merge(records, [0, 1], fields=["first_name", "email"])
    assert "last_name" not in master
    assert master["first_name"] == frozenset({"john", "jon"})


def test_latest_merge_uses_newest_non_null_per_field():
    records = [
        {"first_name": "john", "last_name": "smith", "date_of_birth": None,
         "email": None, "timestamp": "2020-01-01"},
        {"first_name": "johnny", "last_name": None, "date_of_birth": "1985-06-15",
         "email": "j@x.com", "timestamp": "2022-06-01"},
        {"first_name": "jon", "last_name": "smyth", "date_of_birth": "1985-06-15",
         "email": "j2@x.com", "timestamp": "2019-03-03"},
    ]
    master, pos = latest_merge(records, [0, 1, 2], timestamp_field="timestamp")
    # last_name only present in the oldest record -> still taken from it.
    assert master["last_name"] == "smith"
    # first_name/email/date_of_birth from the newest record (index 1).
    assert master["first_name"] == "johnny"
    assert master["email"] == "j@x.com"
    assert master["date_of_birth"] == "1985-06-15"
    assert master["timestamp"] == "2022-06-01"
    assert pos == 1  # newest member position is the anchor


def test_latest_merge_handles_epoch_timestamps():
    records = [
        {"name": "a", "timestamp": 100},
        {"name": "b", "timestamp": 200},
    ]
    master, pos = latest_merge(records, [0, 1], timestamp_field="timestamp")
    assert master["name"] == "b"
    assert pos == 1


# ---------------------------------------------------------------------------
# Union-Class (existential) lift in the scorer
# ---------------------------------------------------------------------------


def test_union_lift_returns_max_of_scalar_rows():
    scorer = person_scorer()
    scalar_left = {"first_name": "john", "last_name": "smith",
                   "date_of_birth": "1985-06-15", "email": None}
    scalar_right = {"first_name": "john", "last_name": "smith",
                    "date_of_birth": "1985-06-15", "email": None}
    # Union record offers john AND jon as alternatives on first_name.
    union_left = dict(scalar_left)
    union_left["first_name"] = frozenset({"john", "zonkwxq"})
    expected = scorer.score(dict(scalar_left), dict(scalar_right))
    got = scorer.score(union_left, scalar_right)
    assert got == pytest.approx(expected, abs=1e-9)
    assert got >= scorer.score(
        {**scalar_left, "first_name": "zonkwxq"}, scalar_right
    )


def test_union_record_matches_own_member_value():
    scorer = person_scorer()
    member = {"first_name": "john", "last_name": "smith",
              "date_of_birth": "1985-06-15", "email": "j@x.com"}
    union = dict(member)
    union["email"] = frozenset({"j@x.com", "other@x.com"})
    # A union record scored against a record whose value is one of its
    # alternatives matches (1.0 via idempotence on the matching scalar row).
    assert scorer.score(union, member) == 1.0


def test_list_valued_fields_are_not_treated_as_union():
    from vectorer.comparisons import make_comparison as mc

    scorer = FellegiSunterScorer.from_comparisons(
        [mc("cosine_similarity_at_thresholds", col_name="vec",
            score_threshold_or_thresholds=[0.9, 0.7])],
        idempotent=False,
    )
    # A list is a comparison-column value (one cosine vector), NOT a union of
    # alternatives: it flows to the scalar cosine path unchanged.  With a
    # single comparison the top level scores the default 0.0929.
    a, b = {"vec": [0.1, 0.9]}, {"vec": [0.2, 0.8]}
    assert scorer.score(a, b) == pytest.approx(0.0929, abs=1e-3)
    # An identical list is also a single value (not a set): same posterior.
    assert scorer.score(a, dict(a)) == pytest.approx(0.0929, abs=1e-3)
    # A frozenset IS flagged as union and lifts without error (the existence
    # lift's max row equals the identical-vector scalar score here).
    assert scorer.score({"vec": frozenset({(0.0, 0.9)})}, {"vec": [0.0, 0.9]}) == pytest.approx(
        scorer.score({"vec": [0.0, 0.9]}, {"vec": [0.0, 0.9]}), abs=1e-9
    )


def test_union_lift_in_batch_and_pairs():
    scorer = person_scorer()
    left = {"first_name": frozenset({"john", "mary"}), "last_name": "smith",
            "date_of_birth": "1985-06-15", "email": None}
    c1 = {"first_name": "john", "last_name": "smith", "date_of_birth": "1985-06-15", "email": None}
    c2 = {"first_name": "zoe", "last_name": "khan", "date_of_birth": "1999-01-01", "email": "z@x.com"}
    batch = scorer.score_batch(left, [c1, c2])
    assert batch[0] > batch[1]
    pairs = scorer.score_pairs([left, {**c1, "first_name": frozenset({"x"})}], [c1, c2])
    assert len(pairs) == 2
    weights = scorer.match_weight_pairs([left], [c1])
    assert np.isfinite(weights[0])


# ---------------------------------------------------------------------------
# Swoosh algorithms with the new merge functions
# ---------------------------------------------------------------------------


@pytest.fixture
def merging_scorer():
    return person_scorer()


def test_swoosh_transitive_closure_with_union_merge(merging_scorer):
    from vectorer.clustering import ScoredPair

    records = [
        {"first_name": "john", "last_name": "smith", "date_of_birth": "1985-06-15", "email": "j@x.com"},
        {"first_name": "john", "last_name": "smit", "date_of_birth": "1985-06-15", "email": "j@x.com"},
        {"first_name": "mary", "last_name": "jones", "date_of_birth": "1990-11-03", "email": None},
    ]
    # records 0 and 1 clearly match (exact first+dob+email, fuzzy last).
    assert merging_scorer.score(records[0], records[1]) > 0.9
    scores = [
        ScoredPair(0, 1, probability=merging_scorer.score(records[0], records[1])),
    ]
    assign = SwooshClusterer(tau=0.85, merge=union_merge).cluster(records, scores)
    assert assign.node_cluster[0] == assign.node_cluster[1]
    assert assign.node_cluster[0] != assign.node_cluster[2]
    cluster = assign.clusters[assign.node_cluster[0]]
    assert cluster.representative_position == -1  # synthetic
    assert isinstance(cluster.representative, dict)
    assert cluster.representative["first_name"] == frozenset({"john"})
    assert cluster.representative["last_name"] == frozenset({"smith", "smit"})


def test_gswoosh_union_merge_rematches_and_clusters(merging_scorer):
    records = [
        {"first_name": "john", "last_name": "smith", "date_of_birth": "1985-06-15", "email": "j@x.com"},
        {"first_name": "john", "last_name": "smit", "date_of_birth": "1985-06-15", "email": "j@x.com"},
        {"first_name": "mary", "last_name": "jones", "date_of_birth": "1990-11-03", "email": None},
    ]
    assign = SwooshClusterer(tau=0.85, merge=union_merge).cluster_with_merger(
        records,
        [(0, 1), (0, 2)],
        scorer_match=lambda l, r: merging_scorer.score(l, r),
    )
    assert assign.node_cluster[0] == assign.node_cluster[1]
    assert assign.node_cluster[0] != assign.node_cluster[2]
    # The merged cluster's representative is a synthetic union master record.
    cluster = assign.clusters[assign.node_cluster[0]]
    assert cluster.representative_position == -1
    assert cluster.representative["first_name"] == frozenset({"john"})
    assert assign.n_pairs_evaluated >= 2


def test_swoosh_with_latest_merge(merging_scorer):
    from vectorer.clustering import ScoredPair

    records = [
        {"first_name": "john", "last_name": "smith", "date_of_birth": None,
         "email": "j@x.com", "timestamp": "2020-01-01"},
        {"first_name": "john", "last_name": "smit", "date_of_birth": None,
         "email": "j@x.com", "timestamp": "2022-06-01"},
        {"first_name": "mary", "last_name": "jones", "date_of_birth": "1990-11-03",
         "email": None, "timestamp": "2021-01-01"},
    ]
    scores = [ScoredPair(0, 1, probability=merging_scorer.score(records[0], records[1]))]
    assign = SwooshClusterer(tau=0.85, merge=lambda recs, pos: latest_merge(recs, pos)).cluster(records, scores)
    assert assign.node_cluster[0] == assign.node_cluster[1]
    cluster = assign.clusters[assign.node_cluster[0]]
    # latest_merge anchors to the newest member and copies its non-null fields.
    assert cluster.representative_position == 1
    assert cluster.representative["last_name"] == "smit"
    assert cluster.representative["timestamp"] == "2022-06-01"


def test_batch_pipeline_accepts_union_merge():
    from vectorer.batch import build_batch_pipeline

    records = [
        {"first_name": "john", "last_name": "smith", "date_of_birth": "1985-06-15", "email": "j@x.com"},
        {"first_name": "john", "last_name": "smit", "date_of_birth": "1985-06-15", "email": "j@x.com"},
        {"first_name": "mary", "last_name": "jones", "date_of_birth": "1990-11-03", "email": None},
        {"first_name": "mari", "last_name": "jones", "date_of_birth": "1990-11-03", "email": None},
    ]
    pipeline = build_batch_pipeline(
        comparisons=person_comparisons(),
        n_canopies=2,
        overlap_m=1,
        tau=0.95,
        merge=union_merge,
    )
    result = pipeline.run(records)
    # exact-duplicate pairs merge into the same cluster
    assert result.cluster_of_position(0) == result.cluster_of_position(1)
    assert result.cluster_of_position(2) == result.cluster_of_position(3)
    assert result.n_clusters <= 2


def test_select_representative_still_default():
    records = [{"a": 1, "b": None}, {"a": 1, "b": 2}]
    rep, pos = select_representative(records, [0, 1])
    assert pos == 1
    assert rep == records[1]