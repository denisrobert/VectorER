"""Candidate blocking: vector k-ANN search and canopy clustering.

Two blocking strategies share this module:

* :class:`VectorBlocker` - vector *search* blocking: embed the query record
  and retrieve its top-k nearest neighbours from a :class:`VectorDatabase`.
  This is the blocking stage of the incremental pipeline.
* :class:`CanopyIndex` - coarser *canopy* blocking used by the batch
  pipeline: cluster the embedded dataset (FAISS k-means), assign every record
  to its top-``m`` centroids so canopies overlap, and enumerate the candidate
  record pairs as pairs co-occurring in a canopy.  A pair that never shares a
  canopy cannot be declared a match by the expensive stage (blocking recall).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, Iterator, Optional, Sequence, TypeVar

import numpy as np

from .records import Record, to_record_dict
from .vectorstores import VectorDatabase

T = TypeVar("T")


@dataclass
class BlockedCandidate(Generic[T]):
    """A retrieved reference record together with its blocking score."""

    record: T
    score: float
    position: int


class VectorBlocker:
    """Embeds an input record and retrieves its top-k nearest neighbours.

    Parameters
    ----------
    vector_database:
        The store (embedding + index + record payloads) to search against.
    k:
        Default number of candidates to retrieve.
    """

    def __init__(self, vector_database: VectorDatabase, k: int = 20) -> None:
        if not isinstance(vector_database, VectorDatabase):
            raise TypeError("vector_database must be a VectorDatabase")
        self.vector_database = vector_database
        self.k = int(k)

    @classmethod
    def build(
        cls,
        records: Sequence[T],
        embedding: Any = None,
        index: Any = None,
        k: int = 20,
    ) -> "VectorBlocker":
        """Build a blocker over ``records`` using sensible defaults."""
        from .embeddings import CharacterHashingEmbedding
        from .vectorstores import FlatIndex, InMemoryVectorDatabase, VectorDatabase

        embedding = embedding or CharacterHashingEmbedding()
        index = index or FlatIndex(normalize=True)
        database = InMemoryVectorDatabase(embedding, index)
        database.add(records)
        return cls(database, k)

    def block(
        self,
        input_record: Any,
        k: Optional[int] = None,
        query_vector: Optional[Sequence[float]] = None,
    ) -> list[BlockedCandidate[T]]:
        """Embed ``input_record`` and return its k-ANN candidate records.

        Candidates are returned in blocking-score descending order; invalid
        (negative) index entries are dropped.  ``query_vector`` may be passed
        to skip re-embedding when the caller already embedded the record.
        """
        kk = min(self.k if k is None else int(k), len(self.vector_database))
        if kk <= 0:
            return []
        if query_vector is None:
            text = self._embed_text(input_record)
            query_vector = self.vector_database.embedding.embed(text)
        indices, scores = self.vector_database.index.search(list(query_vector), kk)
        candidates: list[BlockedCandidate[T]] = []
        for i, score in zip(indices, scores):
            if i < 0:
                continue
            candidates.append(
                BlockedCandidate(
                    record=self.vector_database.record_at(int(i)),
                    score=float(score),
                    position=int(i),
                )
            )
        return candidates

    @staticmethod
    def _embed_text(value: Any) -> str:
        record = to_record_dict(value)
        return "\n".join(f"{k}: {v}" for k, v in record.items() if v is not None)


@dataclass
class CanopyIndex:
    """An overlapping canopy partition over an embedded dataset.

    Attributes
    ----------
    assignments:
        ``(n, m)`` int matrix: the centroid ids each record was assigned to.
    canopies:
        ``list[set[int]]`` of record positions per canopy.
    n_clusters:
        Number of canopies (k-means centroids).
    overlap_m:
        Canopies assigned per record.
    """

    assignments: np.ndarray
    canopies: list[set[int]]
    n_clusters: int
    overlap_m: int

    def candidate_pairs(self) -> Iterator[tuple[int, int]]:
        """Yield unique unordered candidate pairs (positions) in canopies."""
        seen: set[tuple[int, int]] = set()
        for canopy in self.canopies:
            members = sorted(canopy)
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    key = (members[i], members[j])
                    if key not in seen:
                        seen.add(key)
                        yield key

    def n_candidate_pairs(self) -> int:
        return sum(1 for _ in self.candidate_pairs())

    def overlaps(self, a: int, b: int) -> bool:
        return bool(set(self.assignments[a]) & set(self.assignments[b]))

    def to_dict(self) -> dict:
        return {
            "n_records": int(self.assignments.shape[0]),
            "n_canopies": int(self.n_clusters),
            "overlap_m": int(self.overlap_m),
            "n_candidate_pairs": self.n_candidate_pairs(),
        }


def canopy_blocking(
    vectors: Sequence[Sequence[float]],
    n_clusters: int,
    overlap_m: int,
    seed: int = 42,
    max_iter: int = 50,
    nredo: int = 2,
) -> CanopyIndex:
    """Cluster ``vectors`` into overlapping canopies via FAISS k-means.

    Each record is assigned to its top-``overlap_m`` centroids (multi-
    assignment), producing an overlapping canopy around every centroid.
    ``vectors`` are L2-normalized so cosine similarity == inner product.
    """
    import faiss

    vectors = np.asarray(vectors, dtype="float32")
    if vectors.size == 0:
        return CanopyIndex(
            assignments=np.empty((0, int(overlap_m)), dtype=np.int64),
            canopies=[],
            n_clusters=int(n_clusters),
            overlap_m=int(overlap_m),
        )
    faiss.normalize_L2(vectors)

    kmeans = faiss.Kmeans(
        int(vectors.shape[1]),
        int(n_clusters),
        niter=int(max_iter),
        nredo=int(nredo),
        seed=int(seed),
        verbose=False,
    )
    kmeans.train(vectors)

    centroid_index = faiss.IndexFlatIP(int(vectors.shape[1]))
    faiss.normalize_L2(kmeans.centroids)
    centroid_index.add(kmeans.centroids)

    kk = min(int(overlap_m), int(n_clusters))
    _, assignments = centroid_index.search(vectors, kk)

    canopies: list[set[int]] = [set() for _ in range(int(n_clusters))]
    for record_i, row in enumerate(assignments):
        for centroid in row:
            if centroid >= 0:
                canopies[int(centroid)].add(int(record_i))

    return CanopyIndex(
        assignments=np.asarray(assignments, dtype=np.int64),
        canopies=canopies,
        n_clusters=int(n_clusters),
        overlap_m=int(overlap_m),
    )