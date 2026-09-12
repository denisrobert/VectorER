"""Tests for the external distributed vector DB adapter (Qdrant, in-memory mode).

Milestone D of the v0.4.0 distribution plan: the incremental serving path can
scale horizontally by pointing ``VectorDatabase`` at a distributed Qdrant
collection.  Qdrant's in-memory local mode is used so the tests need no
server.
"""

import pytest

qdrant_client = pytest.importorskip(
    "qdrant_client",
    reason="qdrant_client not installed (install via `pip install -e .[qdrant]`)",
)

from vectorer.embeddings import CharacterHashingEmbedding
from vectorer.vectorstore_adapters import QdrantVectorDatabase


@pytest.fixture
def qdrant_db():
    from qdrant_client import QdrantClient

    client = QdrantClient(":memory:")
    embedder = CharacterHashingEmbedding(dimension=64)
    db = QdrantVectorDatabase(
        embedder, client=client, collection="people", vector_size=64,
    )
    yield db
    client.close()


def test_qdrant_add_record_search(qdrant_db):
    records = [
        {"first_name": "john", "last_name": "smith", "date_of_birth": "1985-06-15"},
        {"first_name": "mary", "last_name": "jones", "date_of_birth": "1990-11-03"},
    ]
    qdrant_db.add(records)
    assert len(qdrant_db) == 2

    rec = qdrant_db.record_at(1)
    assert rec["first_name"] == "mary"


def test_qdrant_search_returns_nearest(qdrant_db):
    records = [
        {"first_name": "john", "last_name": "smith", "date_of_birth": "1985-06-15"},
        {"first_name": "mary", "last_name": "jones", "date_of_birth": "1990-11-03"},
    ]
    qdrant_db.add(records)
    vec = qdrant_db.embedding.embed(
        "first_name: john\nlast_name: smith\ndate_of_birth: 1985-06-15"
    )
    idx, _ = qdrant_db.index.search(vec, k=1)
    assert idx[0] == 0  # closest to the john record


def test_qdrant_update_and_delete(qdrant_db):
    qdrant_db.add([{"first_name": "a", "last_name": "x", "date_of_birth": "2000-01-01"},
                   {"first_name": "b", "last_name": "y", "date_of_birth": "2001-01-01"}])
    qdrant_db.update([{"first_name": "zz", "last_name": "z", "date_of_birth": "1999-01-01"}], [1])
    assert qdrant_db.record_at(1)["first_name"] == "zz"
    qdrant_db.delete([0])
    assert len(qdrant_db) == 1
    with pytest.raises(IndexError):
        qdrant_db.record_at(0)


def test_qdrant_search_populates_payload_cache(qdrant_db, monkeypatch):
    """One search round-trip must also serve the subsequent record_at calls
    (k+1 network round-trips -> 1): the search fetches payloads and populates
    the cache, so record_at never needs another retrieve call."""
    qdrant_db.add([
        {"first_name": "john", "last_name": "smith", "date_of_birth": "1985-06-15"},
        {"first_name": "mary", "last_name": "jones", "date_of_birth": "1990-11-03"},
    ])
    qdrant_db._payload_cache.clear()
    vec = qdrant_db.embedding.embed(
        "first_name: john\nlast_name: smith\ndate_of_birth: 1985-06-15"
    )
    idx, _ = qdrant_db.index.search(vec, k=2)

    # The search itself cached the payloads.
    assert {qdrant_db._payload_cache[i]["first_name"] for i in idx} == {"john", "mary"}
    # record_at serves from the cache without a retrieve round-trip.
    assert qdrant_db.record_at(idx[0])["first_name"] in {"john", "mary"}
    assert qdrant_db.record_at(idx[1])["first_name"] in {"john", "mary"}


def test_qdrant_count_cache_tracks_mutations(qdrant_db):
    """add/delete maintain the cached point count so len() never issues a
    fresh count round-trip mid-session."""
    assert len(qdrant_db) == 0
    qdrant_db.add([
        {"first_name": "a", "last_name": "x", "date_of_birth": "2000-01-01"},
        {"first_name": "b", "last_name": "y", "date_of_birth": "2001-01-01"},
        {"first_name": "c", "last_name": "z", "date_of_birth": "2002-01-01"},
    ])
    assert len(qdrant_db) == 3
    qdrant_db.delete([1])
    assert len(qdrant_db) == 2
    qdrant_db.update([{"first_name": "d", "last_name": "w", "date_of_birth": "1998-01-01"}], [0])
    assert len(qdrant_db) == 2  # update doesn't change the count