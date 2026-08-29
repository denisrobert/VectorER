"""End-to-end tests for the incremental entity-resolution pipeline."""

import pytest

from vectorer.embeddings import CharacterHashingEmbedding
from vectorer.incremental import IncrementalPipeline, build_incremental_pipeline
from vectorer.classification import Decision


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