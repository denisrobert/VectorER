"""Tests for pluggable record → embedding-text serializers and their
consistency across the store / blocker / pipeline."""

import numpy as np
import pytest

from vectorer.comparisons import make_comparison
from vectorer.embeddings import CharacterHashingEmbedding
from vectorer.records import (
    EMBED_DEFAULT,
    embed_text,
    positional_embed_text,
    template_embed_text,
)
from vectorer.vectorstores import FlatIndex, InMemoryVectorDatabase

RECORD = {
    "first_name": "john",
    "last_name": "smith",
    "date_of_birth": None,  # missing on purpose
    "email": "js@example.com",
}


def test_embed_default_is_schema_agnostic():
    text = embed_text(RECORD)
    assert "first_name: john" in text
    assert "last_name: smith" in text
    assert "date_of_birth" not in text  # None skipped
    assert "js@example.com" in text
    assert text == EMBED_DEFAULT(RECORD)
    # Schema-agnostic: no declared field list required.
    assert embed_text({"a": 1, "b": 2}) == "a: 1\nb: 2"


def test_positional_embed_text_fixed_schema():
    render = positional_embed_text(
        ["first_name", "last_name", "date_of_birth", "email"], delimiter="|"
    )
    assert render(RECORD) == "john|smith||js@example.com"


def test_positional_embed_text_missing_token():
    render = positional_embed_text(
        ["first_name", "last_name", "date_of_birth", "email"],
        delimiter="|",
        missing="-",
    )
    assert render(RECORD) == "john|smith|-|js@example.com"


def test_template_embed_text():
    render = template_embed_text("Name: {first_name} {last_name}")
    assert render(RECORD) == "Name: john smith"


def test_store_embeds_with_custom_serializer():
    embedding = CharacterHashingEmbedding(dimension=32)
    render = positional_embed_text(["first_name", "last_name"], delimiter="|")
    db = InMemoryVectorDatabase(embedding, FlatIndex(normalize=True), embed_text=render)
    db.add([RECORD])
    # The store's embed_text is exposed and used for its own embeddings.
    assert db.embed_text is render


def test_blocker_inherits_store_serializer():
    from vectorer.blocking import BlockedCandidate, VectorBlocker

    embedding = CharacterHashingEmbedding(dimension=32)
    render = positional_embed_text(["first_name", "last_name"], delimiter="|")
    db = InMemoryVectorDatabase(embedding, FlatIndex(normalize=True), embed_text=render)
    db.add([
        {"first_name": "john", "last_name": "smith", "date_of_birth": "1980-01-01"},
        {"first_name": "mary", "last_name": "jones", "date_of_birth": "1985-01-01"},
    ])
    blocker = VectorBlocker(db, k=2)
    # The blocker uses the store's serializer (no name/value text leak).
    candidates = blocker.block(
        {"first_name": "john", "last_name": "smth", "date_of_birth": "1980-01-01"}
    )
    assert isinstance(candidates[0], BlockedCandidate)
    assert candidates[0].position == 0  # john is nearest to john


def test_incremental_pipeline_positional_serializer_consistent():
    from vectorer.embeddings import CharacterHashingEmbedding
    from vectorer.incremental import IncrementalPipeline
    from vectorer.scoring import FellegiSunterScorer

    embedding = CharacterHashingEmbedding(dimension=64)
    render = positional_embed_text(
        ["first_name", "last_name", "date_of_birth"], delimiter="|"
    )
    records = [
        {"first_name": "john", "last_name": "smith", "date_of_birth": "1980-01-01",
         "email": "js@example.com"},
        {"first_name": "mary", "last_name": "jones", "date_of_birth": "1985-01-01",
         "email": "mj@example.com"},
    ]
    db = InMemoryVectorDatabase(embedding, FlatIndex(normalize=True), embed_text=render)
    db.add(records)
    scorer = FellegiSunterScorer.from_comparisons([
        make_comparison("jaro_winkler_at_thresholds", col_name="first_name"),
        make_comparison("jaro_winkler_at_thresholds", col_name="last_name"),
        make_comparison("date_of_birth_comparison", col_name="date_of_birth"),
    ])
    pipeline = IncrementalPipeline(vector_database=db, scorer=scorer, k=2)
    # Query is a close variant; would fail to match if the serializer leaked.
    query = {"first_name": "john", "last_name": "smitj", "date_of_birth": "1980-01-01",
             "email": "js@example.com"}
    resolution = pipeline.resolve(query)
    assert resolution.decision.name == "MATCH"
    assert any(m.record["first_name"] == "john" for m in resolution.matches)