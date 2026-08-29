"""End-to-end tests for the batch entity-resolution pipeline."""

import pytest

from vectorer.batch import BatchPipeline, build_batch_pipeline
from vectorer.records import RecordSchema


def test_batch_pipeline_merges_exact_duplicates(person_duplicate_dataset, base_comparisons):
    records = person_duplicate_dataset  # 5 base + 5 exact + 5 noisy duplicates
    pipeline = build_batch_pipeline(
        comparisons=base_comparisons,
        n_canopies=3,
        overlap_m=2,
        tau=0.85,
    )
    result = pipeline.run(records)

    assert len(result.records) == 15
    assert result.n_clusters < 15
    # Exact duplicate pairs (position i and i + 5) must be in one cluster.
    for i in range(5):
        assert result.cluster_of_position(i) == result.cluster_of_position(i + 5)
    assert result.n_candidate_pairs > 0
    assert set(result.timing) >= {"parse", "embed", "canopy", "fellegi_sunter", "swoosh"}


def test_batch_pipeline_singletons_for_distinct_records(person_duplicate_dataset, base_comparisons):
    records = person_duplicate_dataset[:5]  # all distinct
    pipeline = build_batch_pipeline(
        comparisons=base_comparisons,
        n_canopies=2,
        overlap_m=1,
        tau=0.85,
    )
    result = pipeline.run(records)
    # Hard-partition canopies; distinct records should not merge.
    assert result.n_clusters == 5
    assert result.n_singletons == 5


def test_batch_cluster_ids_via_schema(person_duplicate_dataset, base_comparisons):
    records = []
    for i, record in enumerate(person_duplicate_dataset):
        records.append({"record_id": f"r{i:03d}", **record})
    pipeline = build_batch_pipeline(
        comparisons=base_comparisons,
        n_canopies=3,
        overlap_m=2,
        tau=0.85,
    )
    result = pipeline.run(records)
    schema = RecordSchema(("first_name", "last_name", "date_of_birth", "email"), id_column="record_id")
    cluster_ids = result.cluster_ids_of(schema)
    assert cluster_ids["r000"] == cluster_ids["r005"]
    assert cluster_ids["r000"] != cluster_ids["r001"]


def test_batch_pipeline_stage_hooks(person_duplicate_dataset, base_comparisons):
    from vectorer.embeddings import CharacterHashingEmbedding
    from vectorer.scoring import FellegiSunterScorer

    embedding = CharacterHashingEmbedding(dimension=96)
    scorer = FellegiSunterScorer.from_comparisons(base_comparisons)
    pipeline = BatchPipeline(
        embedder=embedding,
        scorer=scorer,
        n_canopies=3,
        overlap_m=2,
        tau=0.85,
    )
    records = person_duplicate_dataset
    vectors = pipeline.embed_all(records)
    assert len(vectors) == 15
    canopy = pipeline.block(vectors)
    pairs = list(canopy.candidate_pairs())
    scored = pipeline.score(records, pairs)
    assert all(len(records) > p.left_position >= 0 for p in scored)
    assignment = pipeline.cluster(records, scored)
    assert len(assignment.clusters) <= len(records)


def test_batch_empty_dataset(base_comparisons):
    pipeline = build_batch_pipeline(comparisons=base_comparisons, n_canopies=2, tau=0.85)
    result = pipeline.run([])
    assert len(result.records) == 0
    assert result.n_clusters == 0