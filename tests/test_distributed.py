"""Tests for the distributed batch ER executor and the streaming/multi-machine
parts (Milestones A-B of the v0.4.0 distribution plan)."""

import numpy as np
import pytest

from vectorer.batch import BatchPipeline
from vectorer.blocking import assign_canopies, train_canopy_centroids
from vectorer.clustering import SwooshClusterer
from vectorer.comparisons import make_comparison
from vectorer.distributed import (
    create_executor,
    distributed_batch_er,
    distributed_closure,
    distributed_closure_reduce,
    distributed_score_pairs,
    hash_pair,
    streaming_distributed_closure,
)
from vectorer.scoring import FellegiSunterScorer


def small_person_comparisons():
    return [
        make_comparison("jaro_winkler_at_thresholds", col_name="first_name"),
        make_comparison("jaro_winkler_at_thresholds", col_name="last_name"),
        make_comparison("date_of_birth_comparison", col_name="date_of_birth"),
        make_comparison("email_comparison", col_name="email"),
    ]


def build_small_dataset():
    base = [
        {"first_name": "john", "last_name": "smith", "date_of_birth": "1985-06-15", "email": "j@x.com", "address": "1 main st"},
        {"first_name": "mary", "last_name": "jones", "date_of_birth": "1990-11-03", "email": "m@x.com", "address": "2 elm st"},
        {"first_name": "robert", "last_name": "martinez", "date_of_birth": "1978-02-28", "email": "r@x.com", "address": "3 oak st"},
        {"first_name": "susan", "last_name": "brown", "date_of_birth": "1988-07-19", "email": "s@x.com", "address": "4 pine st"},
    ]
    records = [dict(base[i % 4]) for i in range(12)]
    records += [dict(base[0]), dict(base[1], last_name="jonez"), dict(base[2], date_of_birth="1978-02-27")]
    return records


@pytest.fixture
def dataset():
    return build_small_dataset()


@pytest.fixture
def scorer():
    return FellegiSunterScorer.from_comparisons(small_person_comparisons(), threshold=0.85)


def _single_process(dataset, scorer):
    from vectorer.embeddings import CharacterHashingEmbedding

    return BatchPipeline(
        embedder=CharacterHashingEmbedding(dimension=384),
        scorer=scorer, n_canopies=3, overlap_m=2, canopy_seed=42, tau=0.85,
    ).run(dataset).assignment


def test_distributed_matches_single_process_threads(dataset, scorer):
    single = _single_process(dataset, scorer)
    dist = distributed_batch_er(
        dataset, scorer=scorer, n_canopies=3, overlap_m=2, tau=0.85,
        n_workers=2, use_threads=True,
    )
    assert dist.node_cluster == single.node_cluster


def test_distributed_matches_single_process_processes(dataset, scorer):
    single = _single_process(dataset, scorer)
    dist = distributed_batch_er(
        dataset, scorer=scorer, n_canopies=3, overlap_m=2, tau=0.85,
        n_workers=2, use_threads=False,
    )
    assert dist.node_cluster == single.node_cluster


def test_distributed_single_worker_equals_serial(dataset, scorer):
    single = _single_process(dataset, scorer)
    dist = distributed_batch_er(
        dataset, scorer=scorer, n_canopies=3, overlap_m=2, tau=0.85,
        n_workers=1, use_threads=True,
    )
    assert dist.node_cluster == single.node_cluster


def test_hash_pair_is_deterministic():
    assert hash_pair(0, 5, 3) == hash_pair(5, 0, 3)
    assert 0 <= hash_pair(0, 5, 3) < 3


def test_train_assign_canopy_roundtrip():
    """train_canopy_centroids + assign_canopies reproduce canopy_blocking."""
    from vectorer.blocking import canopy_blocking
    from vectorer.embeddings import CharacterHashingEmbedding
    from vectorer.records import to_record_dict

    records = build_small_dataset()
    emb = CharacterHashingEmbedding(128)
    vecs = np.asarray(emb.embed_many([
        "\n".join(f"{k}: {v}" for k, v in to_record_dict(r).items() if v is not None)
        for r in records
    ]), dtype="float32")

    local = canopy_blocking(vecs, 3, 2, seed=42)
    centroids = train_canopy_centroids(vecs, 3, seed=42, sample_size=None)
    dist = assign_canopies(vecs, centroids, 2)
    assert set(local.candidate_pairs()) == set(dist.candidate_pairs())


def test_distributed_closure_equals_local_closure(dataset, scorer):
    from vectorer.clustering import SwooshClusterer
    from vectorer.embeddings import CharacterHashingEmbedding

    single = BatchPipeline(
        embedder=CharacterHashingEmbedding(dimension=384),
        scorer=scorer, n_canopies=3, overlap_m=2, canopy_seed=42, tau=0.85,
    ).run(dataset).scored_pairs
    edges = [p for p in single if p.probability >= 0.85]
    local = SwooshClusterer(tau=0.85).cluster(dataset, edges)
    dist = distributed_closure(edges, len(dataset), records=dataset)
    assert local.node_cluster == dist.node_cluster

def _above_tau_edges(dataset, scorer):
    from vectorer.embeddings import CharacterHashingEmbedding

    single = BatchPipeline(
        embedder=CharacterHashingEmbedding(dimension=384),
        scorer=scorer, n_canopies=3, overlap_m=2, canopy_seed=42, tau=0.85,
    ).run(dataset)
    return [p for p in single.scored_pairs if p.probability >= 0.85]


def test_distributed_score_pairs_matches_single(dataset, scorer):
    edges = _above_tau_edges(dataset, scorer)
    for nw in (1, 2, 3):
        rows = distributed_score_pairs(
            scorer,
            [dataset[p.left_position] for p in edges],
            [dataset[p.right_position] for p in edges],
            tau=0.85, n_workers=nw,
        )
        assert len(rows) == len(edges)
        indexes = {i for i, _, _ in rows}
        assert len(indexes) == len(edges)


def test_streaming_distributed_closure_matches_single(dataset, scorer):
    edges = _above_tau_edges(dataset, scorer)
    single = SwooshClusterer(tau=0.85).cluster(dataset, edges)
    streamed = streaming_distributed_closure(
        [edges[:4], edges[4:8], edges[8:]], len(dataset), records=dataset,
    )
    assert streamed.node_cluster == single.node_cluster


@pytest.mark.parametrize("n_workers", [1, 2, 3])
def test_distributed_closure_reduce_matches_single(dataset, scorer, n_workers):
    edges = _above_tau_edges(dataset, scorer)
    single = SwooshClusterer(tau=0.85).cluster(dataset, edges)
    reduced = distributed_closure_reduce(
        edges, len(dataset), n_workers=n_workers, records=dataset,
    )
    assert reduced.node_cluster == single.node_cluster


def test_distributed_closure_reduce_with_thread_executor(dataset, scorer):
    from concurrent.futures import ThreadPoolExecutor

    edges = _above_tau_edges(dataset, scorer)
    single = SwooshClusterer(tau=0.85).cluster(dataset, edges)
    with ThreadPoolExecutor(max_workers=3) as ex:
        reduced = distributed_closure_reduce(
            edges, len(dataset), n_workers=3, executor=ex, records=dataset,
        )
    assert reduced.node_cluster == single.node_cluster


def test_create_executor_kinds(dataset):
    ex = create_executor("thread", n_workers=2)
    assert list(ex.map(lambda x: x + 1, [1, 2, 3])) == [2, 3, 4]
    with pytest.raises(ValueError):
        create_executor("unknown")


def test_gather_canopy_sample_deterministic_and_bounded(dataset):
    import numpy as np
    from vectorer.distributed import gather_canopy_sample

    vec_shards = [np.random.RandomState(i).randn(40, 8).astype("float32") for i in range(4)]
    full = np.vstack(vec_shards)
    # sample_size < total -> bounded, reproducible
    sample1 = gather_canopy_sample(vec_shards, sample_size=60, seed=7)
    sample2 = gather_canopy_sample(vec_shards, sample_size=60, seed=7)
    assert len(sample1) <= 60
    assert np.array_equal(sample1, sample2)
    # sample_size >= total -> returns the full stack
    sample_full = gather_canopy_sample(vec_shards, sample_size=len(full), seed=7)
    assert np.array_equal(sample_full, full)


def test_build_global_tf_tables_merges_shards(dataset):
    from vectorer.distributed import build_global_tf_tables

    shards = [
        [{"surname": "smith", "city": "toronto"}, {"surname": "smith", "city": None}],
        [{"surname": "jones", "city": "toronto"}, {"surname": None, "city": "ottawa"}],
]
    tables = build_global_tf_tables(shards, ["surname", "city"])
    assert tables["surname"]["smith"] == pytest.approx(2 / 3, abs=1e-9)
    assert tables["surname"]["jones"] == pytest.approx(1 / 3, abs=1e-9)
    assert tables["city"]["toronto"] == pytest.approx(2 / 3, abs=1e-9)
    assert tables["city"]["ottawa"] == pytest.approx(1 / 3, abs=1e-9)
