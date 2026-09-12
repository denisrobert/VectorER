"""Tests for the in-memory FAISS indexing strategies (flat + HNSW)."""

import numpy as np
import pytest

faiss = pytest.importorskip(
    "faiss",
    reason="faiss not installed (install via `pip install -e .[test]`)",
)

from vectorer.embeddings import CharacterHashingEmbedding
from vectorer.vectorstores import FlatIndex, HnswIndex, InMemoryVectorDatabase


def _records(n: int) -> list[dict]:
    return [
        {"first_name": f"name{i}", "last_name": "s", "date_of_birth": "2000-01-01"}
        for i in range(n)
    ]


def test_hnsw_add_search_returns_nearest():
    idx = HnswIndex(m=16, ef_construction=64, ef_search=64)
    vectors = np.random.default_rng(0).normal(size=(50, 32)).astype("float32")
    idx.add(vectors)
    assert len(idx) == 50
    # The stored vector itself is its nearest neighbour under cosine.
    query = vectors[7]
    indices, scores = idx.search(list(query), k=5)
    assert indices[0] == 7
    assert scores[0] == pytest.approx(1.0, abs=1e-4)


def test_hnsw_incremental_add():
    idx = HnswIndex()
    idx.add(np.random.default_rng(1).normal(size=(20, 16)).astype("float32"))
    idx.add(np.random.default_rng(2).normal(size=(10, 16)).astype("float32"))
    assert len(idx) == 30


def test_hnsw_save_load_roundtrip(tmp_path):
    idx = HnswIndex(m=16)
    vectors = np.random.default_rng(3).normal(size=(40, 16)).astype("float32")
    idx.add(vectors)
    idx.save(tmp_path / "hnsw.faiss")
    restored = HnswIndex.load(tmp_path / "hnsw.faiss", m=16)
    assert len(restored) == 40
    indices, _ = restored.search(list(vectors[3]), k=1)
    assert indices[0] == 3


def test_hnsw_in_memory_database(tmp_path):
    embedding = CharacterHashingEmbedding(dimension=32)
    db = InMemoryVectorDatabase(embedding, HnswIndex(m=16))
    records = _records(20)
    db.add(records)
    assert len(db) == 20
    vec = db.embedding.embed(
        "first_name: name4\nlast_name: s\ndate_of_birth: 2000-01-01"
    )
    idx, _ = db.index.search(vec, k=1)
    assert idx[0] == 4
    # Persistence round-trip uses write_index/read_index.
    db.save(tmp_path)
    restored = InMemoryVectorDatabase.load(tmp_path, embedding=embedding, index=HnswIndex.load(tmp_path / "index.faiss", m=16))
    assert len(restored) == 20
    assert restored.record_at(4) == records[4]


def test_hnsw_matches_flat_top1_for_tiny_memory():
    """On a small, well-separated set, HNSW's top-1 matches the exact index."""
    rng = np.random.default_rng(5)
    vectors = rng.normal(size=(30, 24)).astype("float32")
    flat = FlatIndex()
    hnsw0 = HnswIndex(m=32, ef_construction=200, ef_search=200)
    flat.add(vectors)
    hnsw0.add(vectors)
    for i in range(0, 30, 5):
        _, fi = flat.search(list(vectors[i]), k=3)
        _, hi = hnsw0.search(list(vectors[i]), k=3)
        assert hi[0] == fi[0]
        assert hi[1] == pytest.approx(fi[1], abs=1e-4)