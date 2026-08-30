"""Batch entity-resolution pipeline.

Stage chain (over the whole dataset at once)::

    parse -> embed -> index -> canopy blocking on the embedded dataset ->
    Fellegi-Sunter scoring of every canopy candidate pair -> Swoosh
    clustering on the scored results

Canopy blocking produces overlapping candidate pair sets (k-means
multi-assignment); Fellegi-Sunter scores the pairs; Swoosh merges the
above-threshold pairs into equivalence classes.  The output is a
:class:`~vectorer.clustering.ClusterAssignment` mapping every record to a
cluster and identifying a representative record per entity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable, Optional, Sequence

from .blocking import CanopyIndex, canopy_blocking
from .clustering import ClusterAssignment, ScoredPair, SwooshClusterer
from .embeddings import EmbeddingModel
from .records import RecordSchema, to_record_dict
from .scoring import DEFAULT_THRESHOLD, FellegiSunterScorer


@dataclass
class BatchResult:
    """Output of a batch clustering run plus its stage statistics."""

    assignment: ClusterAssignment
    canopy: CanopyIndex
    records: list[dict]
    scored_pairs: list[ScoredPair]
    n_candidate_pairs: int
    timing: dict[str, float] = field(default_factory=dict)

    @property
    def n_clusters(self) -> int:
        return len(self.assignment.clusters)

    @property
    def n_singletons(self) -> int:
        return sum(
            1 for cluster in self.assignment.clusters.values()
            if len(cluster.member_positions) == 1
        )

    @property
    def n_non_singletons(self) -> int:
        return self.n_clusters - self.n_singletons

    def cluster_of_position(self, position: int) -> int:
        return self.assignment.cluster_of(position)

    def cluster_ids_of(self, schema: RecordSchema) -> dict[Any, int]:
        """Map user-facing record ids (schema.id_column) to cluster ids."""
        out: dict[Any, int] = {}
        for position, record in enumerate(self.records):
            record_id = schema.id_of(record)
            if record_id is not None:
                out[record_id] = self.assignment.cluster_of(position)
        return out


class BatchPipeline:
    """Batch (offline) deduplication of an embedded dataset.

    Parameters
    ----------
    embedder:
        Embedding model over the serialized records.
    scorer:
        Calibrated Fellegi-Sunter scorer over the comparison set.
    n_canopies:
        Number of canopy centroids (k-means clusters) for the cheap phase.
    overlap_m:
        Top-m centroids per record (multi-assignment overlap; 1 = hard
        partition).
    canopy_seed:
        Seed for the k-means training.
    tau:
        Swoosh merge threshold on the FS posterior.
    """

    def __init__(
        self,
        *,
        embedder: EmbeddingModel,
        scorer: FellegiSunterScorer,
        n_canopies: int = 512,
        overlap_m: int = 3,
        canopy_seed: int = 42,
        tau: Optional[float] = None,
        merge: Callable[[Sequence, Sequence], tuple[Any, int]] = None,
    ) -> None:
        self.embedder = embedder
        self.scorer = scorer
        self.n_canopies = int(n_canopies)
        self.overlap_m = int(max(1, overlap_m))
        self.canopy_seed = int(canopy_seed)
        self.tau = float(tau) if tau is not None else scorer.threshold
        from .clustering import select_representative

        self.merge = merge if merge is not None else select_representative
        self.swoosh = SwooshClusterer(tau=self.tau, merge=self.merge)

    # -- stage hooks --------------------------------------------------------

    def embed_all(self, records: Sequence[dict]) -> list[list[float]]:
        """Stage 2: embed every record into a dense vector."""
        texts = [self._embed_text(record) for record in records]
        return [list(v) for v in self.embedder.embed_many(texts)]

    def block(self, vectors: Sequence[Sequence[float]]) -> CanopyIndex:
        """Stage: canopy-block the embedded dataset."""
        return canopy_blocking(
            vectors,
            self.n_canopies,
            self.overlap_m,
            seed=self.canopy_seed,
        )

    def score(
        self,
        records: Sequence[dict],
        pairs: Sequence[tuple[int, int]],
        batch_size: int = 4096,
    ) -> list[ScoredPair]:
        """Stage: score every canopy candidate pair with Fellegi-Sunter."""
        scored: list[ScoredPair] = []
        left_records = [records[i] for i, _ in pairs]
        right_records = [records[j] for _, j in pairs]
        for start in range(0, len(pairs), batch_size):
            left_slice = left_records[start : start + batch_size]
            right_slice = right_records[start : start + batch_size]
            probs = self.scorer.score_pairs(left_slice, right_slice)
            weights = self.scorer.match_weight_pairs(left_slice, right_slice)
            for (i, j), prob, weight in zip(
                pairs[start : start + batch_size], probs, weights
            ):
                scored.append(
                    ScoredPair(
                        left_position=i,
                        right_position=j,
                        probability=float(prob),
                        match_weight=float(weight),
                    )
                )
        return scored

    def cluster(
        self,
        records: Sequence[dict],
        scored_pairs: Sequence[ScoredPair],
    ) -> ClusterAssignment:
        """Stage: Swoosh-cluster the scored pairs."""
        return self.swoosh.cluster(records, scored_pairs)

    # -- main entry point ---------------------------------------------------

    def run(
        self,
        records: Sequence[Any],
        schema: Optional[RecordSchema] = None,
    ) -> BatchResult:
        """Cluster the whole dataset: parse -> embed -> canopy -> FS -> Swoosh."""
        del schema  # reserved for id reporting
        timing: dict[str, float] = {}
        t0 = perf_counter()
        parsed = [to_record_dict(r) for r in records]
        timing["parse"] = perf_counter() - t0

        t0 = perf_counter()
        vectors = self.embed_all(parsed)
        timing["embed"] = perf_counter() - t0

        t0 = perf_counter()
        canopy = self.block(vectors)
        timing["canopy"] = perf_counter() - t0

        pairs = list(canopy.candidate_pairs())
        n_candidate_pairs = len(pairs)

        t0 = perf_counter()
        scored_pairs = self.score(parsed, pairs)
        timing["fellegi_sunter"] = perf_counter() - t0

        t0 = perf_counter()
        assignment = self.cluster(parsed, scored_pairs)
        timing["swoosh"] = perf_counter() - t0

        return BatchResult(
            assignment=assignment,
            canopy=canopy,
            records=parsed,
            scored_pairs=scored_pairs,
            n_candidate_pairs=n_candidate_pairs,
            timing=timing,
        )

    @staticmethod
    def _embed_text(record: dict) -> str:
        return "\n".join(f"{k}: {v}" for k, v in record.items() if v is not None)


def build_batch_pipeline(
    *,
    embedder: Optional[EmbeddingModel] = None,
    scorer: Optional[FellegiSunterScorer] = None,
    comparisons: Optional[Sequence[Any]] = None,
    n_canopies: int = 512,
    overlap_m: int = 3,
    canopy_seed: int = 42,
    tau: float = DEFAULT_THRESHOLD,
    merge: Callable[[Sequence, Sequence], tuple[Any, int]] = None,
) -> BatchPipeline:
    """Convenience constructor: default embedder + scorer from ``comparisons``."""
    from .embeddings import CharacterHashingEmbedding

    embedding = embedder or CharacterHashingEmbedding()
    if scorer is None:
        if not comparisons:
            raise ValueError("supply comparisons or a calibrated scorer")
        scorer = FellegiSunterScorer.from_comparisons(comparisons, threshold=tau)
    return BatchPipeline(
        embedder=embedding,
        scorer=scorer,
        n_canopies=n_canopies,
        overlap_m=overlap_m,
        canopy_seed=canopy_seed,
        tau=tau,
        merge=merge,
    )