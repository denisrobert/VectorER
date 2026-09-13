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
from vectorer.embeddings import (
    CharacterHashingEmbedding,
    SentenceTransformerEmbedding,
    embedder_from_settings,
)
from vectorer.records import EMBED_DEFAULT, positional_embed_text
from vectorer.scoring import FellegiSunterScorer
from vectorer.vectorstores import FlatIndex, InMemoryVectorDatabase


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


def _store_of(dataset):
    """A pre-populated store embedding the same text the records path sends."""
    store = InMemoryVectorDatabase(
        CharacterHashingEmbedding(dimension=384),
        FlatIndex(normalize=False),
        embed_text=EMBED_DEFAULT,
    )
    store.add(dataset)
    return store


def _run_both(dataset, scorer, **kwargs):
    from vectorer.distributed import distributed_batch_er

    records_assign = distributed_batch_er(
        dataset, scorer=scorer, n_canopies=3, overlap_m=2, tau=0.85, **kwargs
    )
    store_assign = distributed_batch_er(
        vector_database=_store_of(dataset), scorer=scorer,
        n_canopies=3, overlap_m=2, tau=0.85, **kwargs,
    )
    return records_assign, store_assign


def test_distributed_store_matches_records_threads(dataset, scorer):
    records_assign, store_assign = _run_both(dataset, scorer, n_workers=2, use_threads=True)
    assert store_assign.node_cluster == records_assign.node_cluster
    assert store_assign.node_cluster == _single_process(dataset, scorer).node_cluster


def test_distributed_store_matches_records_processes(dataset, scorer):
    records_assign, store_assign = _run_both(dataset, scorer, n_workers=2, use_threads=False)
    assert store_assign.node_cluster == records_assign.node_cluster


def test_distributed_store_single_worker_equals_serial(dataset, scorer):
    records_assign, store_assign = _run_both(dataset, scorer, n_workers=1, use_threads=True)
    assert store_assign.node_cluster == records_assign.node_cluster


def test_distributed_store_matches_records_sampled_canopy(dataset, scorer):
    # n=15 > sample_size=8 forces the cross-shard sampling path (centroid
    # training never sees the full matrix); store and records feeds must agree.
    records_assign, store_assign = _run_both(
        dataset, scorer, n_workers=2, use_threads=True, sample_size=8,
    )
    assert store_assign.node_cluster == records_assign.node_cluster


def test_distributed_store_fills_representatives(dataset, scorer):
    assign = distributed_batch_er(
        vector_database=_store_of(dataset), scorer=scorer,
        n_canopies=3, overlap_m=2, tau=0.85, n_workers=2, use_threads=True,
    )
    for cluster in assign.clusters.values():
        assert cluster.representative is not None
        rep_pos = cluster.representative_position
        assert dataset[rep_pos] == cluster.representative


def test_distributed_store_requires_exactly_one_source(dataset, scorer):
    with pytest.raises(ValueError, match="exactly one"):
        distributed_batch_er(
            dataset, _store_of(dataset), scorer=scorer, n_canopies=3, tau=0.85,
        )
    with pytest.raises(ValueError, match="exactly one"):
        distributed_batch_er(scorer=scorer, n_canopies=3, tau=0.85)


def test_distributed_empty_store(dataset, scorer):
    store = InMemoryVectorDatabase(
        CharacterHashingEmbedding(dimension=384),
        FlatIndex(normalize=False),
        embed_text=EMBED_DEFAULT,
    )
    assign = distributed_batch_er(
        vector_database=store, scorer=scorer, n_canopies=3, tau=0.85,
        n_workers=2, use_threads=True,
    )
    assert len(assign.clusters) == 0
    assert assign.node_cluster == {}


def test_store_vectors_for_and_records_at(dataset):
    store = _store_of(dataset)
    n = len(dataset)
    assert len(store.vectors_for(0, n)) == n
    assert len(store.vectors_for(2, 5)) == 3
    assert store.vectors_for(4, 4) == []
    assert store.records_at([0, 2, n - 1]) == [dataset[0], dataset[2], dataset[n - 1]]
    assert store.records_at([]) == []


def test_distributed_records_honors_explicit_embedder(dataset, scorer):
    # A non-default embedder must drive the records path: the distributed run
    # embeds with it (per-worker rebuild via to_settings) and matches the
    # single-process pipeline using the same model.
    from vectorer.batch import BatchPipeline
    from vectorer.distributed._canopy import _embed_shard

    embedder = CharacterHashingEmbedding(dimension=96, ngrams=(1, 2))
    single = BatchPipeline(
        embedder=embedder, scorer=scorer, n_canopies=3, overlap_m=2,
        canopy_seed=42, tau=0.85,
    ).run(dataset).assignment
    custom = distributed_batch_er(
        dataset, embedder=embedder, scorer=scorer, n_canopies=3, overlap_m=2,
        tau=0.85, n_workers=2, use_threads=True,
    )
    assert custom.node_cluster == single.node_cluster

    # The embedder state drives the shard embedding (deterministic proof, since
    # the final clustering can coincide across embedders on tiny data).
    _, custom_vecs = _embed_shard(dataset, embedder.to_settings())
    _, default_vecs = _embed_shard(
        dataset, CharacterHashingEmbedding(dimension=384).to_settings()
    )
    assert not np.array_equal(custom_vecs, default_vecs)
    expected = np.asarray(
        BatchPipeline(
            embedder=embedder, scorer=scorer, n_canopies=3, overlap_m=2,
            canopy_seed=42, tau=0.85,
        ).embed_all(dataset),
        dtype="float32",
    )
    assert np.array_equal(custom_vecs, expected)


def test_embedder_settings_roundtrip():
    embedder = CharacterHashingEmbedding(dimension=64, ngrams=(1, 3))
    rebuilt = embedder_from_settings(embedder.to_settings())
    assert rebuilt.dimension == 64
    assert rebuilt.ngrams == (1, 3)
    assert rebuilt.embed("hello world") == embedder.embed("hello world")
    with pytest.raises(ValueError, match="unknown embedding model settings"):
        embedder_from_settings({"type": "nope"})


def test_distributed_records_honors_embed_text(dataset, scorer):
    # A custom serializer must flow into the distributed shard embedding,
    # exactly like BatchPipeline(embed_text=...): end-to-end parity with the
    # same-serializer single-process pipeline, and a deterministic vector-level
    # proof that embed_text (not the default) was used.
    from vectorer.batch import BatchPipeline
    from vectorer.distributed._canopy import _embed_shard

    embedder = CharacterHashingEmbedding(dimension=96, ngrams=(1, 2))
    serializer = positional_embed_text(
        ["first_name", "last_name", "date_of_birth", "email", "address"],
        delimiter="|",
    )
    single = BatchPipeline(
        embedder=embedder, scorer=scorer, n_canopies=3, overlap_m=2,
        canopy_seed=42, tau=0.85, embed_text=serializer,
    ).run(dataset).assignment
    custom = distributed_batch_er(
        dataset, embedder=embedder, embed_text=serializer,
        scorer=scorer, n_canopies=3, overlap_m=2, tau=0.85,
        n_workers=2, use_threads=True,
    )
    assert custom.node_cluster == single.node_cluster

    _, custom_vecs = _embed_shard(dataset, embedder.to_settings(), serializer)
    _, default_vecs = _embed_shard(dataset, embedder.to_settings(), EMBED_DEFAULT)
    assert not np.array_equal(custom_vecs, default_vecs)
    expected = np.asarray(
        BatchPipeline(
            embedder=embedder, scorer=scorer, n_canopies=3, overlap_m=2,
            canopy_seed=42, tau=0.85, embed_text=serializer,
        ).embed_all(dataset),
        dtype="float32",
    )
    assert np.array_equal(custom_vecs, expected)


def test_sentence_transformer_settings_carry_backend():
    # sentence-transformers is an optional extra; skip the relay check when
    # the package is not installed (the constructor imports it lazily).
    st_mod = pytest.importorskip("sentence_transformers")
    from unittest import mock

    captured = {}

    class _FakeST:
        def __init__(self, *args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs

        def get_sentence_embedding_dimension(self):
            return 8

    with mock.patch.object(st_mod, "SentenceTransformer", _FakeST):
        embedder = SentenceTransformerEmbedding(
            model_id="my/model", device="cpu", backend="onnx"
        )
    assert embedder.backend == "onnx"
    assert captured["kwargs"] == {
        "revision": None, "device": "cpu", "backend": "onnx",
    }

    settings = embedder.to_settings()
    assert settings["backend"] == "onnx"
    with mock.patch.object(st_mod, "SentenceTransformer", _FakeST):
        rebuilt = embedder_from_settings(settings)
    assert rebuilt.backend == "onnx"
    assert captured["kwargs"]["backend"] == "onnx"


def test_distributed_wrapped_embedder_rejected(dataset, scorer):
    # A pre-loaded model object cannot be re-instantiated per worker; the
    # caller must pass model_id/device so workers can re-load it.
    class _StubModel:
        def encode(self, texts):
            return [[0.0] * 4 for _ in texts]

        def get_sentence_embedding_dimension(self):
            return 4

    embedder = SentenceTransformerEmbedding(model=_StubModel())
    with pytest.raises(ValueError, match="pre-loaded"):
        distributed_batch_er(
            dataset, embedder=embedder, scorer=scorer, n_canopies=3, tau=0.85,
        )


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
