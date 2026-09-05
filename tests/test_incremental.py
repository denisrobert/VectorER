"""End-to-end tests for the incremental entity-resolution pipeline."""

import pytest

from vectorer.embeddings import CharacterHashingEmbedding
from vectorer.incremental import IncrementalPipeline, build_incremental_pipeline
from vectorer.classification import Decision
from vectorer.vectorstores import FlatIndex, InMemoryVectorDatabase


def test_from_store_serves_pre_embedded_persisted_store(tmp_path, person_duplicate_dataset, base_comparisons):
    """IncrementalPipeline.from_store is a thin alias for the serving path."""
    from vectorer.scoring import FellegiSunterScorer

    embedding = CharacterHashingEmbedding(dimension=384)
    db = InMemoryVectorDatabase(embedding, FlatIndex())
    db.add(person_duplicate_dataset[:5])
    db.save(tmp_path)
    db2 = InMemoryVectorDatabase.load(tmp_path, embedding=embedding)
    scorer = FellegiSunterScorer.from_comparisons(base_comparisons)
    pipeline = IncrementalPipeline.from_store(db2, scorer, k=10, tau=0.85)
    assert len(pipeline.vector_database) == 5
    result = pipeline.resolve(person_duplicate_dataset[5])
    assert result.decision is Decision.MATCH


def test_build_from_pre_embedded_vector_database(person_duplicate_dataset, base_comparisons):
    """build_incremental_pipeline(vector_database=...) serves a pre-embedded store."""
    import tempfile

    from vectorer.scoring import FellegiSunterScorer
    from vectorer.vectorstores import InMemoryVectorDatabase, FlatIndex

    embedding = CharacterHashingEmbedding(dimension=384)
    db = InMemoryVectorDatabase(embedding, FlatIndex())
    db.add(person_duplicate_dataset[:5])
    pipeline = build_incremental_pipeline(
        vector_database=db, comparisons=base_comparisons, k=10, tau=0.85,
    )
    assert pipeline.vector_database is db  # no re-embed of a copy
    result = pipeline.resolve(person_duplicate_dataset[5])
    assert result.decision is Decision.MATCH


def test_build_rejects_both_or_neither_source(base_comparisons):
    from vectorer.embeddings import CharacterHashingEmbedding
    from vectorer.vectorstores import FlatIndex, InMemoryVectorDatabase

    db = InMemoryVectorDatabase(CharacterHashingEmbedding(64), FlatIndex())
    db.add([{"first_name": "a", "last_name": "b", "date_of_birth": "2000-01-01", "email": None, "address": None}])
    with pytest.raises(ValueError, match="either records"):
        build_incremental_pipeline([{"first_name": "x"}], vector_database=db, comparisons=base_comparisons)
    with pytest.raises(ValueError, match="supply records="):
        build_incremental_pipeline(comparisons=base_comparisons)


def test_build_from_persisted_then_reloaded_store(tmp_path, person_duplicate_dataset, base_comparisons):
    """Full serving path: embed once, persist, reload, build from the store."""
    embedding = CharacterHashingEmbedding(dimension=384)
    db = InMemoryVectorDatabase(embedding, FlatIndex())
    db.add(person_duplicate_dataset[:5])
    db.save(tmp_path)
    db2 = InMemoryVectorDatabase.load(tmp_path, embedding=embedding)
    pipeline = build_incremental_pipeline(vector_database=db2, comparisons=base_comparisons)
    assert len(pipeline.vector_database) == 5
    result = pipeline.resolve(person_duplicate_dataset[5])
    assert result.decision is Decision.MATCH


def test_build_and_resolve_exact_duplicate(person_duplicate_dataset, base_comparisons):
    base = person_duplicate_dataset[:5]
    pipeline = build_incremental_pipeline(
        base,
        comparisons=base_comparisons,
        k=10,
        tau=0.85,
    )
    query = person_duplicate_dataset[5]  # exact duplicate of base[0]
    resolution = pipeline.resolve(query)
    assert resolution.decision is Decision.MATCH
    assert len(resolution.matches) >= 1
    assert resolution.matches[0].candidate_position == 0
    assert resolution.matches[0].match_probability > 0.85


def test_resolve_unrelated_record_is_rejected(person_duplicate_dataset, base_comparisons):
    base = person_duplicate_dataset[:5]
    pipeline = build_incremental_pipeline(base, comparisons=base_comparisons, k=10, tau=0.85)
    unrelated = {
        "first_name": "zoe",
        "last_name": "khan",
        "date_of_birth": "1999-01-01",
        "email": "zoe@nobody.com",
        "address": None,
    }
    resolution = pipeline.resolve(unrelated)
    assert resolution.decision is Decision.NON_MATCH
    assert resolution.matches == []


def test_noisy_duplicate_retrieved_and_matched(person_duplicate_dataset, base_comparisons):
    base = person_duplicate_dataset[:5]
    pipeline = build_incremental_pipeline(base, comparisons=base_comparisons, k=20, tau=0.7)
    noisy = person_duplicate_dataset[10]  # noisy duplicate of base[0]
    resolution = pipeline.resolve(noisy)
    assert resolution.decision is Decision.MATCH
    positions = [m.candidate_position for m in resolution.matches]
    assert 0 in positions


def test_ingest_adds_record_to_store(person_duplicate_dataset, base_comparisons):
    base = person_duplicate_dataset[:5]
    pipeline = build_incremental_pipeline(base, comparisons=base_comparisons, k=10, tau=0.85)
    assert len(pipeline.vector_database) == 5
    position = pipeline.ingest(person_duplicate_dataset[5])
    assert len(pipeline.vector_database) == 6
    assert pipeline.vector_database.record_at(position)["first_name"] == "john"


def test_ingest_novel_skips_duplicates_and_adds_novel(person_duplicate_dataset, base_comparisons):
    base = person_duplicate_dataset[:5]
    pipeline = build_incremental_pipeline(base, comparisons=base_comparisons, k=10, tau=0.85)
    n0 = len(pipeline.vector_database)

    # Exact duplicate of reference[0] is resolved to a match -> skipped.
    assert pipeline.ingest_novel(person_duplicate_dataset[5]) is None
    assert len(pipeline.vector_database) == n0

    # Genuinely novel record (no match at/above tau) is ingested.
    novel = {
        "first_name": "zoe",
        "last_name": "khan",
        "date_of_birth": "1999-01-01",
        "email": "zoe@nowhere.com",
        "address": None,
    }
    position = pipeline.ingest_novel(novel)
    assert position == n0
    assert len(pipeline.vector_database) == n0 + 1
    assert pipeline.vector_database.record_at(position)["first_name"] == "zoe"


def test_ingest_novel_many_aligned_positions(person_duplicate_dataset, base_comparisons):
    base = person_duplicate_dataset[:5]
    pipeline = build_incremental_pipeline(base, comparisons=base_comparisons, k=10, tau=0.85)
    deck = [
        person_duplicate_dataset[5],          # duplicate of base[0] -> skipped
        {"first_name": "zoe", "last_name": "khan", "date_of_birth": "1999-01-01",
         "email": "z@nowhere.com", "address": None},   # novel -> ingested
        {"first_name": "ray", "last_name": "paul", "date_of_birth": "1970-04-04",
         "email": None, "address": "99 elm st toronto"},  # novel -> ingested
    ]
    positions = pipeline.ingest_novel_many(deck)
    assert positions[0] is None
    assert positions[1] == 5
    assert positions[2] == 6
    assert len(pipeline.vector_database) == 7


def test_ingest_novel_respects_novelty_threshold(person_duplicate_dataset, base_comparisons):
    base = person_duplicate_dataset[:5]
    pipeline = build_incremental_pipeline(base, comparisons=base_comparisons, k=10, tau=0.85)
    n0 = len(pipeline.vector_database)

    # Partial match (posterior ~0.77) is below the default match threshold, so
    # the record counts as novel and is ingested.
    partial = {
        "first_name": "john", "last_name": "smith", "date_of_birth": "1999-01-01",
        "email": None, "address": None,
    }
    assert pipeline.ingest_novel(partial) == n0
    assert len(pipeline.vector_database) == n0 + 1

    # Raising the novelty bar above that posterior treats the record as an
    # existing entity, so the twin is skipped.
    twin = {
        "first_name": "john", "last_name": "smith", "date_of_birth": "1999-02-02",
        "email": None, "address": None,
    }
    assert pipeline.ingest_novel(twin, novelty_threshold=0.9) is None
    assert len(pipeline.vector_database) == n0 + 1


def test_pipeline_stages_are_exposed(person_duplicate_dataset, base_comparisons):
    from vectorer.comparisons import comparison_set

    base = person_duplicate_dataset[:5]
    pipeline = build_incremental_pipeline(base, comparisons=base_comparisons, k=10, tau=0.85)
    record = pipeline.parse(person_duplicate_dataset[5])
    assert record["first_name"] == "john"
    candidates = pipeline.block(record)
    scored = pipeline.score(record, candidates)
    assert len(scored) == min(10, len(base))
    matches = pipeline.classify(record, scored)
    assert all(m.match_probability >= 0.85 for m in matches)


def test_custom_embedder_used_by_pipeline(person_duplicate_dataset, base_comparisons):
    from vectorer.scoring import FellegiSunterScorer
    from vectorer.vectorstores import FlatIndex, InMemoryVectorDatabase

    embedding = CharacterHashingEmbedding(dimension=96)
    database = InMemoryVectorDatabase(embedding, FlatIndex(normalize=True))
    base = person_duplicate_dataset[:5]
    database.add(base)
    scorer = FellegiSunterScorer.from_comparisons(base_comparisons)
    pipeline = IncrementalPipeline(database, scorer, k=10, tau=0.85)
    query = person_duplicate_dataset[5]
    resolution = pipeline.resolve(query)
    assert resolution.decision is Decision.MATCH
    assert resolution.matches[0].candidate_position == 0