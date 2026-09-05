"""Incremental entity-resolution pipeline.

Stage chain (per incoming record)::

    parse -> embed -> vector search blocking (top-k) -> Fellegi-Sunter
    scoring on the top-k candidates -> classify

The pipeline resolves *one* streaming record at a time against an existing
:class:`~vectorer.vectorstores.VectorDatabase` reference population and can
optionally ingest the accepted record back into the store (growing the index).
Ingestion supports a *novelty-only* switch (:meth:`ingest_novel`,
:meth:`ingest_novel_many`) so only records with no match in the reference
population are added.

Every stage is a public method so subclasses can override a single step (e.g. a
custom parser or a tuned blocker): :meth:`parse`, :meth:`block`,
:meth:`score`, :meth:`classify`, :meth:`embed`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

from .blocking import BlockedCandidate, VectorBlocker
from .classification import (
    Decision,
    MatchResult,
    ScoredCandidate,
    ThresholdClassifier,
)
from .embeddings import EmbeddingModel
from .records import to_record_dict
from .scoring import DEFAULT_THRESHOLD, FellegiSunterScorer
from .vectorstores import VectorDatabase


@dataclass
class Resolution:
    """Result of resolving one input record against the reference population."""

    input_record: dict
    retrieved: list[ScoredCandidate]
    matches: list[MatchResult]
    decision: Decision
    embedding: Optional[list[float]] = None


@dataclass
class IncrementalPipeline:
    """Incremental (streaming/online) entity resolution.

    Parameters
    ----------
    vector_database:
        Reference store (embedding model + index + record payloads).
    scorer:
        Calibrated Fellegi-Sunter scorer over the comparison set.
    k:
        Number of candidates retrieved by vector search blocking.
    tau:
        Match threshold on the FS posterior.
    possible_low:
        Optional lower threshold for a "possible match" band.
    """

    vector_database: VectorDatabase
    scorer: FellegiSunterScorer
    k: int = 20
    tau: Optional[float] = None

    @classmethod
    def from_store(
        cls,
        vector_database: VectorDatabase,
        scorer: FellegiSunterScorer,
        *,
        k: int = 20,
        tau: Optional[float] = None,
    ) -> "IncrementalPipeline":
        """Serve incremental queries against an **already-embedded** store.

        This is the production modality: the reference population was embedded
        into the vector store separately (previously) — e.g. built once,
        persisted with :meth:`InMemoryVectorDatabase.save`, and reloaded with
        :meth:`InMemoryVectorDatabase.load`, or sourced from an external
        distributed vector DB — and only the *queries* are new.  Unlike
        :func:`build_incremental_pipeline`, this does **not** embed the
        reference records again; it wires the store straight into the pipeline.

        Conveniently mirrors constructing ``IncrementalPipeline(vector_database=,
        scorer=, k=, tau=)`` directly, but names the intent and is the
        discoverable shortcut for the serving use case.
        """
        return cls(
            vector_database=vector_database,
            scorer=scorer,
            k=k,
            tau=tau,
        )

    def __post_init__(self) -> None:
        self.blocker = VectorBlocker(self.vector_database, k=self.k)
        tau = self.tau if self.tau is not None else self.scorer.threshold
        self.classifier = ThresholdClassifier(tau=tau)
        self._embed_cache: Optional[list[float]] = None

    # -- stage hooks --------------------------------------------------------

    def parse(self, payload: Any) -> dict:
        """Stage 1: coerce the inbound payload into a record mapping."""
        return to_record_dict(payload)

    def embed(self, record: dict) -> Optional[list[float]]:
        """Stage 2: embed the parsed record (result cached on the resolution)."""
        text = self._embed_text(record)
        vector = self.vector_database.embedding.embed(text)
        self._embed_cache = [float(x) for x in vector]
        return self._embed_cache

    def block(
        self,
        record: dict,
        k: Optional[int] = None,
    ) -> list[BlockedCandidate]:
        """Stage 3: vector-search blocking (top-k candidates)."""
        return self.blocker.block(record, k=k)

    def score(
        self,
        record: dict,
        candidates: Sequence[BlockedCandidate],
    ) -> list[ScoredCandidate]:
        """Stage 4: Fellegi-Sunter scoring of every candidate (single evaluation)."""
        candidate_records = [
            to_record_dict(candidate.record) for candidate in candidates
        ]
        if not candidate_records:
            return []
        posteriors, weights = self.scorer.score_and_weight_batch(record, candidate_records)
        return [
            ScoredCandidate(
                record=candidate.record,
                probability=float(p),
                match_weight=float(w),
                blocking_score=candidate.score,
                position=candidate.position,
            )
            for candidate, p, w in zip(candidates, posteriors, weights)
        ]

    def classify(
        self,
        record: dict,
        scored: Sequence[ScoredCandidate],
    ) -> list[MatchResult]:
        """Stage 5: classification (matches at/above the threshold)."""
        matches: list[MatchResult] = []
        for candidate in scored:
            if self.classifier.decide(candidate.probability) is Decision.MATCH:
                matches.append(
                    MatchResult(
                        record=candidate.record,
                        match_probability=candidate.probability,
                        match_weight=candidate.match_weight,
                        blocking_score=candidate.blocking_score,
                        candidate_position=candidate.position,
                    )
                )
        matches.sort(key=lambda m: m.match_probability, reverse=True)
        return matches

    # -- main entry point ---------------------------------------------------

    def resolve(self, payload: Any, k: Optional[int] = None) -> Resolution:
        """Resolve one record: parse -> embed -> block -> score -> classify."""
        record = self.parse(payload)
        vector = self.embed(record)
        candidates = self.blocker.block(record, k=k, query_vector=self._embed_cache)
        scored = self.score(record, candidates)
        matches = self.classify(record, scored)
        decision = Decision.MATCH if matches else Decision.NON_MATCH
        return Resolution(
            input_record=record,
            retrieved=scored,
            matches=matches,
            decision=decision,
            embedding=vector,
        )

    # -- ingestion ----------------------------------------------------------

    def add(self, records: Sequence[Any]) -> None:
        """Ingest parsed records into the reference store (vector DB)."""
        self.vector_database.add(records)

    def ingest(self, payload: Any) -> int:
        """Parse, embed and append one record; return its new position."""
        record = self.parse(payload)
        self.vector_database.add([record])
        return len(self.vector_database) - 1

    def ingest_novel(
        self,
        payload: Any,
        k: Optional[int] = None,
        novelty_threshold: Optional[float] = None,
    ) -> Optional[int]:
        """Ingest ``payload`` only if it is *novel*: no reference record scores
        at or above the (novelty) threshold.

        The record is resolved against the reference store first.  When the
        best candidate posterior is strictly below the threshold (``tau`` by
        default, or ``novelty_threshold`` when given) the record is treated as
        new, appended to the store, and its new position is returned.  When a
        match exists, ``None`` is returned and nothing is ingested.

        This is the "ingest novel records only" switch for the incremental
        path: exact and near-duplicates of the reference population are
        skipped, so the store grows only with genuinely new entities.
        """
        resolution = self.resolve(payload, k=k)
        threshold = (
            novelty_threshold
            if novelty_threshold is not None
            else self.classifier.tau
        )
        if resolution.retrieved and max(
            candidate.probability for candidate in resolution.retrieved
        ) >= threshold:
            return None
        self.vector_database.add([resolution.input_record])
        return len(self.vector_database) - 1

    def ingest_novel_many(
        self,
        payloads: Sequence[Any],
        k: Optional[int] = None,
        novelty_threshold: Optional[float] = None,
    ) -> list[Optional[int]]:
        """Apply :meth:`ingest_novel` to each payload.

        Returns one position per payload: the new store position when the
        record was novel and ingested, ``None`` when it matched an existing
        record.  Positions are aligned with ``payloads`` in order.
        """
        return [
            self.ingest_novel(payload, k=k, novelty_threshold=novelty_threshold)
            for payload in payloads
        ]

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _embed_text(record: dict) -> str:
        return "\n".join(f"{k}: {v}" for k, v in record.items() if v is not None)


def build_incremental_pipeline(
    records: Optional[Sequence[Any]] = None,
    *,
    embedder: Optional[EmbeddingModel] = None,
    scorer: Optional[FellegiSunterScorer] = None,
    comparisons: Optional[Sequence[Any]] = None,
    vector_database: Optional[VectorDatabase] = None,
    k: int = 20,
    tau: float = DEFAULT_THRESHOLD,
) -> IncrementalPipeline:
    """Convenience constructor supporting both population modalities.

    Two ways to configure the reference store:

    * **From raw records** (illustration / small setups): pass ``records`` —
      they are embedded and added to a new :class:`InMemoryVectorDatabase`.
      ``embedder`` (default deterministic hashing) is used for that embedding.
    * **From an already-embedded store** (production / serving): pass
      ``vector_database=`` — a pre-built store that already carries its own
      embedder, e.g. one loaded from disk or from a distributed vector DB.
      The records are **not** re-embedded; only queries are embedded at
      ``resolve`` time.

    The scorer is built from ``comparisons`` (declared ``Comparison`` objects)
    unless a calibrated ``scorer`` is given.
    """
    from .embeddings import CharacterHashingEmbedding
    from .vectorstores import FlatIndex, InMemoryVectorDatabase

    if vector_database is not None:
        if records is not None or embedder is not None:
            raise ValueError(
                "supply either records (+ optional embedder) OR a pre-built "
                "vector_database, not both"
            )
        database = vector_database
    else:
        if records is None:
            raise ValueError("supply records= or vector_database=")
        embedding = embedder or CharacterHashingEmbedding()
        database = InMemoryVectorDatabase(embedding, FlatIndex(normalize=True))
        database.add(records)
    if scorer is None:
        if not comparisons:
            raise ValueError("supply comparisons or a calibrated scorer")
        scorer = FellegiSunterScorer.from_comparisons(comparisons, threshold=tau)
    return IncrementalPipeline(
        vector_database=database,
        scorer=scorer,
        k=k,
        tau=tau,
    )