"""Tests for the distributed batch ER executor.

The key contract: ``distributed_batch_er`` must produce the SAME cluster
assignment as the single-process ``BatchPipeline.run`` for the same input.
"""

import numpy as np
import pytest

from vectorer.batch import BatchPipeline
from vectorer.blocking import assign_canopies, train_canopy_centroids
from vectorer.comparisons import make_comparison
from vectorer.distributed import distributed_batch_er, distributed_closure, hash_pair
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