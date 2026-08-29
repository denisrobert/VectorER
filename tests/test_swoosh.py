"""Tests for Swoosh clustering."""

import pytest

from vectorer.clustering import (
    ClusterAssignment,
    ScoredPair,
    SwooshClusterer,
    connected_components,
    gswoosh,
    select_representative,
)


def test_gswoosh_merges_matching_pairs():
    records = [
        {"name": "john smith", "id": 0},
        {"name": "jon smyth", "id": 1},
        {"name": "mary jones", "id": 2},
    ]
    pairs = [(0, 1), (0, 2), (1, 2)]
    match = lambda left, right: 0.99 if left["id"] in (0, 1) and right["id"] in (0, 1) else 0.01  # noqa: E731
    assignment = gswoosh(records, pairs, match, tau=0.9)
    assert assignment.n_pairs_evaluated >= 2
    assert assignment.node_cluster[0] == assignment.node_cluster[1]
    assert assignment.node_cluster[0] != assignment.node_cluster[2]


def test_gswoosh_none_below_threshold():
    records = [{"name": "a"}, {"name": "b"}, {"name": "c"}]
    assignment = gswoosh(
        records,
        [(0, 1), (1, 2)],
        lambda left, right: 0.5,
        tau=0.9,
    )
    assert len(assignment.clusters) == 3
    assert set(assignment.node_cluster.values()) == {0, 1, 2}


def test_gswoosh_representative_promotion():
    records = [
        {"first_name": "john", "last_name": "smith", "email": None, "address": None},
        {"first_name": "john", "last_name": "smith", "email": "john@x.com", "address": None},
        {"first_name": "johnathon", "last_name": "smyth", "email": "john@x.com", "address": "1 main st"},
    ]
    assignment = gswoosh(
        records,
        [(0, 1), (1, 2)],
        lambda left, right: 0.99,
        tau=0.9,
    )
    cluster = assignment.clusters[assignment.node_cluster[0]]
    # The richest record (2, with all three fields) must become representative.
    assert cluster.representative_position == 2
    assert cluster.representative["email"] == "john@x.com"


def test_select_representative_prefers_completeness():
    records = [
        {"a": 1, "b": None},
        {"a": 1, "b": 2},
    ]
    rep, pos = select_representative(records, [0, 1])
    assert pos == 1


def test_connected_components():
    components = connected_components(5, [(0, 1), (1, 3), (2, 4)])
    assert components[0] == components[1] == components[3] == 0
    assert components[2] == components[4] == 2


def test_swoosh_clusterer_transitive_closure():
    records = [{"n": i} for i in range(5)]
    scored = [
        ScoredPair(0, 1, probability=0.95),
        ScoredPair(1, 2, probability=0.90),
        ScoredPair(2, 3, probability=0.10),
    ]
    assignment = SwooshClusterer(tau=0.85).cluster(records, scored)
    assert assignment.node_cluster[0] == assignment.node_cluster[1] == assignment.node_cluster[2]
    assert assignment.node_cluster[2] != assignment.node_cluster[3]
    assert assignment.n_pairs_matched == 2


def test_cluster_assignment_to_dict():
    records = [{"n": 0}, {"n": 1}]
    scored = [ScoredPair(0, 1, probability=0.99)]
    assignment = SwooshClusterer(tau=0.85).cluster(records, scored)
    data = assignment.to_dict(records)
    cluster_id = sorted(data)[0]
    assert sorted(data[cluster_id]["members"]) == [0, 1]
    assert data[cluster_id]["representative"]["n"] in (0, 1)