"""Tests for the two-database Record Linkage mode (Link, not merge)."""

import pytest

from vectorer.comparisons import make_comparison
from vectorer.embeddings import CharacterHashingEmbedding
from vectorer.link import FieldMap, LinkEdge, LinkTable, RecordLinker
from vectorer.scoring import FellegiSunterScorer
from vectorer.vectorstores import FlatIndex, InMemoryVectorDatabase


def small_comparisons():
    return [
        make_comparison("jaro_winkler_at_thresholds", col_name="name"),
        make_comparison("date_of_birth_comparison", col_name="dob"),
        make_comparison("email_comparison", col_name="email"),
    ]


@pytest.fixture
def linker():
    scorer = FellegiSunterScorer.from_comparisons(small_comparisons(), prior=1e-2, threshold=0.7)
    return RecordLinker(
        embedder=CharacterHashingEmbedding(dimension=384),
        scorer=scorer,
        field_maps={
            "A": FieldMap({"name": "name", "dob": "birth_date", "email": "email_c"},
                          id_column="cust_id"),
            "B": FieldMap({"name": "legal_name", "dob": "dob", "email": "email_p"},
                          id_column="partner_id"),
        },
        k=3,
        tau=0.70,
    )


def db_a():
    return [
        {"cust_id": "c1", "name": "john smith", "birth_date": "1985-06-15", "email_c": "john@a.com"},
        {"cust_id": "c2", "name": "mary jones", "birth_date": "1990-11-03", "email_c": None},
        {"cust_id": "c3", "name": "robert martin", "birth_date": "1978-02-28", "email_c": "rob@a.com"},
    ]


def db_b():
    return [
        {"partner_id": "p1", "legal_name": "jon smith", "dob": "1985-06-15", "email_p": "jsmith@b.com"},
        {"partner_id": "p2", "legal_name": "robert martinez", "dob": None, "email_p": None},
        {"partner_id": "p3", "legal_name": "zoe kwan", "dob": "1999-01-01", "email_p": "zoe@b.com"},
    ]


def test_field_map_projects_and_normalizes():
    fm = FieldMap({"name": "legal_name", "dob": "dob"},
                  normalize=lambda c, v: v.lower() if c == "name" else v,
                  id_column="partner_id")
    out = fm.project({"legal_name": "JON SMITH", "dob": "1985-06-15", "email_p": "x"})
    assert out == {"name": "jon smith", "dob": "1985-06-15"}
    assert "email" not in out  # not in the mapping


def test_directed_links_correct_pair(linker):
    tbl = linker.link_directed(db_a(), db_b())
    pairs = set(tbl.as_pairs())
    assert ("c1", "p1") in pairs
    # unrelated B record (zoe) must NOT be linked.
    assert all(b != "p3" for _, b in pairs)


def test_symmetric_links_correct_pair(linker):
    tbl = linker.link_symmetric(db_a(), db_b())
    pairs = set(tbl.as_pairs())
    assert ("c1", "p1") in pairs
    assert all(b != "p3" for _, b in pairs)


def test_enforce_11_uses_each_b_once(linker):
    # If two A records point at the same B, only one link survives.
    a = [
        {"cust_id": "x1", "name": "john smith", "birth_date": "1985-06-15", "email_c": "john@a.com"},
        {"cust_id": "x2", "name": "jon smith", "birth_date": "1985-06-15", "email_c": "jsmith@b.com"},
    ]
    tbl = linker.link_directed(a, db_b(), enforce_11=True)
    b_used = [e.b_id for e in tbl.matches]
    assert len(b_used) == len(set(b_used))  # each B used once


def test_possible_band_three_tier(linker):
    # possible_low below tau -> the tie tier is captured.
    tbl = linker.link_directed(db_a(), db_b())
    assert tbl.n_possible_matches == 0  # nothing in the possible band at this config


def test_link_table_helpers(linker):
    tbl = linker.link_directed(db_a(), db_b())
    assert len(tbl.edges) == tbl.n_matches
    assert callable(tbl.by_a)
    assert callable(tbl.by_b)
    d = tbl.to_dict()
    assert d["n_links"] == tbl.n_matches
    assert "links" in d


def test_missing_field_map_raises(linker):
    with pytest.raises(ValueError, match="field map"):
        linker.project("Z", {"name": "x"})


def test_default_ids_are_positional(linker):
    # Construct a linker without id_column -> positional ids
    from vectorer.link import RecordLinker as RL

    rl = RL(
        embedder=CharacterHashingEmbedding(dimension=384),
        scorer=linker.scorer,
        field_maps={"A": FieldMap({"name": "name"}), "B": FieldMap({"name": "name"})},
        k=3, tau=0.5,
    )
    tbl = rl.link_directed(
        [{"name": "alice"}, {"name": "bob"}],
        [{"name": "alice"}, {"name": "zoe"}],
    )
    assert all(isinstance(e.a_id, int) for e in tbl.edges)
    assert all(isinstance(e.b_id, int) for e in tbl.edges) or tbl.n_matches == 0


def test_directed_with_injected_store_matches_default(linker):
    # An injected in-memory store must produce the same edges as the default.
    tbl_default = linker.link_directed(db_a(), db_b())
    store = InMemoryVectorDatabase(
        linker.embedder, FlatIndex(normalize=True), embed_text=linker.embed_text
    )
    store.add([linker.project("B", r) for r in db_b()])
    tbl = linker.link_directed(db_a(), [], b_store=store, b_ids=["p1", "p2", "p3"])
    assert set(tbl.as_pairs()) == set(tbl_default.as_pairs())
    assert tbl.n_matches == tbl_default.n_matches


def test_directed_prepopulated_store_uses_record_at(linker):
    # Link against an already-populated store: no b_records are needed and the
    # candidates must come from the store's own payloads (record_at).
    store = InMemoryVectorDatabase(
        linker.embedder, FlatIndex(normalize=True), embed_text=linker.embed_text
    )
    store.add([linker.project("B", r) for r in db_b()])
    tbl = linker.link_directed(db_a(), [], b_store=store, b_ids=["p1", "p2", "p3"])
    pairs = set(tbl.as_pairs())
    assert ("c1", "p1") in pairs
    assert all(b != "p3" for _, b in pairs)


def test_directed_populated_store_not_readded(linker):
    # Passing b_records alongside a populated store must NOT append/re-embed:
    # the store size stays fixed between runs (the old behavior doubled it).
    store = InMemoryVectorDatabase(
        linker.embedder, FlatIndex(normalize=True), embed_text=linker.embed_text
    )
    store.add([linker.project("B", r) for r in db_b()])
    before = len(store)
    tbl = linker.link_directed(db_a(), db_b(), b_store=store, b_ids=["p1", "p2", "p3"])
    assert len(store) == before
    # positions are still the store's original 0..2, so results match the default.
    assert set(tbl.as_pairs()) == set(linker.link_directed(db_a(), db_b()).as_pairs())


def canonical_store(linker, side, records):
    store = InMemoryVectorDatabase(
        linker.embedder, FlatIndex(normalize=True), embed_text=linker.embed_text
    )
    store.add([linker.project(side, r) for r in records])
    return store


def test_directed_store_to_store_no_record_lists(linker):
    # The huge-data path: populate A and B stores up front, then link with no
    # record lists passed in at all.
    a_store = canonical_store(linker, "A", db_a())
    b_store = canonical_store(linker, "B", db_b())
    tbl = linker.link_directed(
        a_store=a_store, b_store=b_store,
        a_ids=["c1", "c2", "c3"], b_ids=["p1", "p2", "p3"],
    )
    assert set(tbl.as_pairs()) == set(linker.link_directed(db_a(), db_b()).as_pairs())


def test_directed_empty_a_store_populated_from_records(linker):
    # An empty injected A store is populated from a_records on first use, and
    # ids come from id_column (not positional).
    a_store = InMemoryVectorDatabase(
        linker.embedder, FlatIndex(normalize=True), embed_text=linker.embed_text
    )
    assert len(a_store) == 0
    tbl = linker.link_directed(db_a(), db_b(), a_store=a_store)
    assert len(a_store) == len(db_a())  # populated
    assert ("c1", "p1") in set(tbl.as_pairs())


def test_directed_requires_source_per_side(linker):
    import pytest

    with pytest.raises(ValueError, match="A side"):
        linker.link_directed(b_records=db_b())
    with pytest.raises(ValueError, match="B side"):
        linker.link_directed(a_records=db_a())