"""Record Linkage mode: link records across **two** databases.

Where the incremental and bulk modes deduplicate or cluster *one* dataset, this
mode takes two separately-managed databases (different schemas, overlapping
compared fields) and emits **link edges** -- ``(a_id, b_id, posterior, ...)``
pairs -- without ever merging the two stores.  The common use cases are
mergers and cross-enterprise collaborations: each party keeps its own records
and identifiers; the framework only says which of *their* records refer to the
same entity.

Architecture
------------
The stage chain mirrors the single-DB modes, with a canonical field projection
up front so the two schemas can overlap: project each side into canonical
compared fields, then block (top-k ANN of A against indexed B for directed, or
cross-DB canopy pairs for symmetric), then Fellegi-Sunter score the candidates
over the canonical fields, then classify and emit ``LinkEdge`` rows.

The heavy machinery (embedding, FAISS blocking, FS scoring, calibration,
three-band classification) is the framework's existing, unchanged machinery: a
canonical comparison field whose value is ``None`` on one side degrades to a
null level (no evidence), so schema overlap *within* the compared fields is
handled for free.

Usage::

    from vectorer.link import RecordLinker, FieldMap

    linker = RecordLinker(
        embedder=embedder,
        comparisons=[...],              # declared on the canonical field names
        field_maps={
            "A": FieldMap({"full_name": "name", "dob": "birth_date", "email": "email"}),
            "B": FieldMap({"full_name": "legal_name", "dob": "dob", "email": "em"}),
        },
        k=20, tau=0.85,
    )
    links = linker.link(records_a, records_b, enforce_11=False)
    for edge in links:
        print(edge.a_id, edge.b_id, edge.probability)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence

from .classification import Decision, ThresholdClassifier
from .clustering import SwooshClusterer, select_representative
from .embeddings import EmbeddingModel
from .scoring import DEFAULT_THRESHOLD, FellegiSunterScorer
from .vectorstores import FlatIndex, InMemoryVectorDatabase, VectorDatabase


@dataclass(frozen=True)
class FieldMap:
    """Maps a database's columns onto the linker's canonical compared fields.

    ``mapping``: canonical_name -> source column name in that database.
    Each canonical field defaults to the source column of the same name when
    absent (identity).  ``normalize`` is an optional (canonical_name, value)
    transformer applied to the projected value (e.g. lowercase, strip
    abbreviations) so the two independently-produced sources align better.
    ``id_column`` optionally names the primary key the linker should use as the
    database's record identifier (falls back to positional ids).
    """

    mapping: Mapping[str, str] = field(default_factory=dict)
    normalize: Optional[Callable[[str, Any], Any]] = None
    id_column: Optional[str] = None

    def project(self, record: Mapping[str, Any]) -> dict:
        """Return a canonical record: ``{canonical: value or None}``."""
        output: dict[str, Any] = {}
        for canonical, source in self.mapping.items():
            value = record.get(source)
            if self.normalize is not None and value is not None:
                value = self.normalize(canonical, value)
            output[canonical] = value
        return output

    def canonical_fields(self) -> tuple[str, ...]:
        return tuple(self.mapping)


@dataclass(frozen=True)
class LinkEdge:
    """A scored cross-database link between one A record and one B record."""

    a_id: Any
    b_id: Any
    probability: float
    match_weight: float = 0.0
    blocking_score: float = 0.0
    decision: str = "non_match"


class LinkTable:
    """The result of a two-database linkage run."""

    def __init__(self, edges: Sequence[LinkEdge], a_ids: Sequence[Any], b_ids: Sequence[Any]) -> None:
        self.edges = list(edges)
        self.a_ids = list(a_ids)
        self.b_ids = list(b_ids)

    @property
    def matches(self) -> list[LinkEdge]:
        return [e for e in self.edges if e.decision == "match"]

    @property
    def possible_matches(self) -> list[LinkEdge]:
        return [e for e in self.edges if e.decision == "possible_match"]

    @property
    def n_matches(self) -> int:
        return len(self.matches)

    @property
    def n_possible_matches(self) -> int:
        return len(self.possible_matches)

    def by_a(self) -> dict[Any, list[LinkEdge]]:
        out: dict[Any, list[LinkEdge]] = {}
        for edge in self.edges:
            out.setdefault(edge.a_id, []).append(edge)
        return out

    def by_b(self) -> dict[Any, list[LinkEdge]]:
        out: dict[Any, list[LinkEdge]] = {}
        for edge in self.edges:
            out.setdefault(edge.b_id, []).append(edge)
        return out

    def as_pairs(self) -> list[tuple[Any, Any]]:
        return [(e.a_id, e.b_id) for e in self.matches]

    def to_dict(self) -> dict:
        return {
            "a_count": len(self.a_ids),
            "b_count": len(self.b_ids),
            "n_links": self.n_matches,
            "n_possible_links": self.n_possible_matches,
            "links": [
                {
                    "a_id": e.a_id,
                    "b_id": e.b_id,
                    "probability": round(e.probability, 6),
                    "match_weight": round(e.match_weight, 4),
                    "decision": e.decision,
                }
                for e in self.edges
            ],
        }

    def __len__(self) -> int:
        return len(self.edges)

    def __iter__(self):
        return iter(self.edges)


def _default_embed_text(canonical_record: dict) -> str:
    return "\n".join(f"{k}: {v}" for k, v in canonical_record.items() if v is not None)


class RecordLinker:
    """Links records across two databases along matching entities.

    Parameters
    ----------
    embedder:
        Embedding model used on the **canonical** record text (both DBs are
        projected first, so blocking compares aligned embeddings).
    comparisons:
        The canonical comparison set (declared on ``field_maps``' canonical
        names) via ``make_comparison`` / ``Comparison`` objects.
    field_maps:
        ``{"A": FieldMap(...), "B": FieldMap(...)}`` -- per-database column ->
        canonical-field projection.
    k:
        Top-k ANN blocking (directed mode).
    tau:
        Link threshold on the posterior.
    possible_low:
        Optional lower band for a "possible match" review tier.
    scorer:
        Optional pre-calibrated scorer (overrides building from ``comparisons``).
    embed_text:
        Optional text serializer for canonical records (defaults to
        ``"field: value"`` lines).
    """

    def __init__(
        self,
        *,
        embedder: EmbeddingModel,
        comparisons: Optional[Sequence[Any]] = None,
        field_maps: Mapping[str, FieldMap] = None,
        k: int = 20,
        tau: float = DEFAULT_THRESHOLD,
        possible_low: Optional[float] = None,
        scorer: Optional[FellegiSunterScorer] = None,
        embed_text: Optional[Callable[[dict], str]] = None,
    ) -> None:
        if not field_maps:
            raise ValueError("field_maps is required (per-DB canonical projection)")
        self.embedder = embedder
        self.field_maps = dict(field_maps)
        self.k = int(k)
        self.tau = float(tau)
        self.embed_text = embed_text or _default_embed_text
        if scorer is None:
            if not comparisons:
                raise ValueError("supply comparisons or a calibrated scorer")
            scorer = FellegiSunterScorer.from_comparisons(comparisons, threshold=tau)
        self.scorer = scorer
        self.classifier = ThresholdClassifier(tau=tau, possible_low=possible_low)

    # -- projection helpers -------------------------------------------------

    def project(self, side: str, record: Mapping[str, Any]) -> dict:
        return self._field_map(side).project(record)

    def _field_map(self, side: str) -> FieldMap:
        fm = self.field_maps.get(side)
        if fm is None:
            raise ValueError(f"no field map registered for side {side!r}; have {sorted(self.field_maps)}")
        return fm

    def _ids_of(self, side: str, records: Sequence[Mapping[str, Any]]) -> list[Any]:
        id_col = self._field_map(side).id_column
        if id_col:
            ids: list[Any] = []
            for rec in records:
                value = rec.get(id_col) if hasattr(rec, "get") else None
                ids.append(value if value is not None else len(ids))
            return ids
        return list(range(len(records)))

    # -- directed link ------------------------------------------------------

    def link_directed(
        self,
        a_records: Sequence[Mapping[str, Any]],
        b_records: Sequence[Mapping[str, Any]],
        *,
        a_ids: Optional[Sequence[Any]] = None,
        b_ids: Optional[Sequence[Any]] = None,
        enforce_11: bool = False,
    ) -> LinkTable:
        """Index B, resolve every A record against it; return link edges.

        Each A record is embedded, its top-k B candidates are found by ANN, and
        FS scores them; edges at/above ``tau`` are emitted.  With
        ``enforce_11=True`` each B record is used at most once (the best A
        match wins).
        """
        if a_ids is None:
            a_ids = self._ids_of("A", a_records)
        if b_ids is None:
            b_ids = self._ids_of("B", b_records)
        b_canonical = [self.project("B", r) for r in b_records]

        # Index canonicalized B: the stored records are the canonical view so
        # that embeddings, comparisons and positions all align.
        b_store = InMemoryVectorDatabase(
            self.embedder, FlatIndex(normalize=True), embed_text=self.embed_text
        )
        b_store.add(b_canonical)

        edges: list[LinkEdge] = []
        used_b: set[int] = set()  # index positions, not b_ids
        for a_i, a_rec in enumerate(a_records):
            canonical_a = self.project("A", a_rec)
            query = self.embedder.embed(self.embed_text(canonical_a))
            indices, scores = b_store.index.search(query, min(self.k, len(b_store)))
            best_b_pos = -1
            best_prob = 0.0
            for b_pos, block_score in zip(indices.tolist() if hasattr(indices, "tolist") else indices,
                                          scores.tolist() if hasattr(scores, "tolist") else scores):
                b_pos = int(b_pos)
                if b_pos < 0:
                    continue
                if enforce_11 and b_pos in used_b:
                    continue
                prob = self.scorer.score(canonical_a, b_canonical[b_pos])
                if prob >= self.tau:
                    weight = float(self.scorer.match_weight_batch(
                        canonical_a, [b_canonical[b_pos]]
                    )[0])
                    decision = self.classifier.decide(float(prob))
                    edges.append(LinkEdge(
                        a_id=a_ids[a_i],
                        b_id=b_ids[b_pos],
                        probability=float(prob),
                        match_weight=weight,
                        blocking_score=float(block_score),
                        decision=decision.value,
                    ))
                    if enforce_11 and float(prob) > best_prob:
                        best_prob = float(prob)
                        best_b_pos = b_pos
            if enforce_11 and best_b_pos >= 0:
                # drop the b_id we linked so it's not reused; keep an index
                used_b.add(best_b_pos)
        return LinkTable(edges, a_ids, b_ids)

    # -- symmetric link -----------------------------------------------------

    def link_symmetric(
        self,
        a_records: Sequence[Mapping[str, Any]],
        b_records: Sequence[Mapping[str, Any]],
        *,
        a_ids: Optional[Sequence[Any]] = None,
        b_ids: Optional[Sequence[Any]] = None,
        n_canopies: int = 256,
        overlap_m: int = 2,
        seed: int = 42,
        batch_size: int = 4096,
    ) -> LinkTable:
        """Canopy-block A and B together, score only **cross-DB** pairs, emit edges.

        Both sides are canonicalized, concatenated with an origin tag, canopy
        clustered, and only pairs spanning the two origins are FS-scored (the
        solver never merges either database).
        """
        if a_ids is None:
            a_ids = self._ids_of("A", a_records)
        if b_ids is None:
            b_ids = self._ids_of("B", b_records)
        from .blocking import canopy_blocking
        from .vectorstores import embed_text_of

        canonical_a = [self.project("A", r) for r in a_records]
        canonical_b = [self.project("B", r) for r in b_records]
        n_a, n_b = len(canonical_a), len(canonical_b)

        union_vectors = []
        for rec in canonical_a:
            union_vectors.append(self.embedder.embed(self.embed_text(rec)))
        for rec in canonical_b:
            union_vectors.append(self.embedder.embed(self.embed_text(rec)))

        # Clamp the canopy grid to the (tiny) data size: FAISS k-means needs
        # ~39x the number of training points vs clusters.
        n_canopies = min(int(n_canopies), max(1, len(union_vectors) // 39))
        overlap_m = min(int(overlap_m), n_canopies)
        canopy = canopy_blocking(union_vectors, n_canopies, overlap_m, seed=seed)
        edges: list[LinkEdge] = []
        # Evaluate cross-origin canopy pairs in batches (symmetric 1-1, 1-N by default).
        a_of = lambda pos: pos < n_a  # noqa: E731
        pairs = [(i, j) for (i, j) in canopy.candidate_pairs() if a_of(i) != a_of(j)]
        for start in range(0, len(pairs), batch_size):
            chunk = pairs[start:start + batch_size]
            left = [union_records(i, canonical_a, canonical_b) for i, _ in chunk]
            right = [union_records(j, canonical_a, canonical_b) for _, j in chunk]
            probs = self.scorer.score_pairs(left, right)
            weights = self.scorer.match_weight_pairs(left, right)
            for (i, j), prob, weight in zip(chunk, probs, weights):
                if prob < self.tau:
                    continue
                decision = self.classifier.decide(float(prob))
                if i < n_a:
                    a_id, b_id = a_ids[i], b_ids[j - n_a]
                else:
                    a_id, b_id = a_ids[j], b_ids[i - n_a]
                edges.append(LinkEdge(
                    a_id=a_id, b_id=b_id,
                    probability=float(prob),
                    match_weight=float(weight),
                    decision=decision.value,
                ))
        return LinkTable(edges, a_ids, b_ids)

    # -- unified entry ------------------------------------------------------

    def link(
        self,
        a_records: Sequence[Mapping[str, Any]],
        b_records: Sequence[Mapping[str, Any]],
        *,
        mode: str = "directed",
        a_ids: Optional[Sequence[Any]] = None,
        b_ids: Optional[Sequence[Any]] = None,
        enforce_11: bool = False,
        **kwargs,
    ) -> LinkTable:
        """Link the two databases; ``mode`` is ``"directed"`` or ``"symmetric"``."""
        if mode == "directed":
            return self.link_directed(a_records, b_records, a_ids=a_ids, b_ids=b_ids,
                                      enforce_11=enforce_11, **kwargs)
        if mode == "symmetric":
            if enforce_11:
                raise ValueError("enforce_11 is only supported in directed mode")
            return self.link_symmetric(a_records, b_records, a_ids=a_ids, b_ids=b_ids, **kwargs)
        raise ValueError(f"unknown mode {mode!r}; expected 'directed' or 'symmetric'")


def union_records(pos: int, a_records, b_records) -> dict:
    """Return the canonical record at ``pos`` in the concatenated A+B list."""
    n_a = len(a_records)
    if pos < n_a:
        return a_records[pos]
    return b_records[pos - n_a]