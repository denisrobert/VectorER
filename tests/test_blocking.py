"""Tests for vector search blocking and canopy blocking."""

import numpy as np
import pytest

from vectorer.blocking import VectorBlocker, canopy_blocking
from vectorer.embeddings import CharacterHashingEmbedding
from vectorer.vectorstores import FlatIndex, InMemoryVectorDatabase


def test_blocker_retrieves_nearest_records():
    embedding = CharacterHashingEmbedding(dimension=128)
    records = [
        {"first_name": "john", "last_name": "smith", "date_of_birth": "1985-06-15"},
        {"first_name": "mary", "last_name": "jones", "date_of_birth": "1990-11-03"},
        {"first_name": "johnathan", "last_name": "smyth", "date_of_birth": "1985-06-15"},
    ]
    blocker = VectorBlocker.build(records, embedding=embedding, k=2)
    query = {"first_name": "johnathan", "last_name": "smyth", "date_of_birth": "1985-06-15"}
    candidates = blocker.block(query, k=2)
    assert len(candidates) == 2
    # The exact query record (position 2) must be retrieved first.
    assert candidates[0].position == 2
    # Blocking scores are descending.
    scores = [c.score for c in candidates]
    assert scores == sorted(scores, reverse=True)


def test_blocker_empty_database():
    embedding = CharacterHashingEmbedding()
    database = InMemoryVectorDatabase(embedding, FlatIndex())
    blocker = VectorBlocker(database, k=5)
    assert blocker.block({"a": 1}) == []


def test_canopy_blocking_produces_overlapping_canopies():
    embedding = CharacterHashingEmbedding(dimension=64)
    records = [
        {"first_name": f"person{i}", "last_name": f"family{i}", "city": "town"} for i in range(40)
    ]
    vectors = embedding.embed_many([f"first_name: {r['first_name']}\nlast_name: {r['last_name']}" for r in records])
    canopy = canopy_blocking(vectors, n_clusters=8, overlap_m=2, seed=42)
    assert canopy.overlap_m == 2
    assert canopy.assignments.shape == (40, 2)
    assert len(canopy.canopies) == 8
    pairs = list(canopy.candidate_pairs())
    assert len(pairs) >= 40
    # Every record belongs to exactly its top-2 canopies.
    total_memberships = sum(len(c) for c in canopy.canopies)
    assert total_memberships == 80


def test_canopy_multiple_assignments():
    vectors = np.random.RandomState(0).randn(50, 16).astype("float32")
    canopy = canopy_blocking(vectors, n_clusters=4, overlap_m=3, seed=1)
    assert canopy.assignments.shape == (50, 3)
    assert len(canopy.canopies) == 4


def test_in_memory_store_persists_and_reloads(tmp_path):
    embedding = CharacterHashingEmbedding(dimension=128)
    records = [
        {"first_name": "john", "last_name": "smith", "email": "j@x.com"},
        {"first_name": "mary", "last_name": "jones", "email": "m@x.com"},
    ]
    database = InMemoryVectorDatabase(embedding, FlatIndex(normalize=True))
    database.add(records)
    assert len(database) == 2
    database.save(tmp_path)

    restored = InMemoryVectorDatabase.load(tmp_path, embedding=embedding)
    assert len(restored) == 2
    assert restored.record_at(0) == records[0]
    # Reloaded store searches identically to the original.
    blocker_orig = VectorBlocker(database, k=2)
    blocker_new = VectorBlocker(restored, k=2)
    query = records[0]
    assert blocker_orig.block(query)[0].position == 0
    assert blocker_new.block(query)[0].position == 0